"""rl/replay_obs.py -- forensic replay of a recorded fly_rl flight (no sim needed).

Re-parses a session's ``mavlink.tlog``, replicates fly_rl's control loop on the recorded
telemetry (the SAME ``build_obs`` + ``policy_step`` code objects), and dumps a per-tick
JSONL of everything the policy saw and would have commanded: raw telemetry fields, the
17-dim obs (labeled), raw actor mean, post-tanh, post-rescale action, and the wire
command. Built for the 2026-06-11 live transfer failure (collective=0 + yaw rail from
step 0 on BOTH inc4/inc5 while offline_rollout finishes 6/6 from the same handoff).

Self-consistency checks that localize telemetry corruption using only the recording:
  * v_fd        central-difference of ODOMETRY world position -- artifact-free ground
                truth for velocity (position needs no frame/sign assumptions).
  * v_client    world_vec_from_body_quat(v_body, q_raw) -- what MavlinkClient stores in
                velocity_ned and fly_rl consumes. CORRECT: the twist is expressed in the
                same reported frame as the raw quat (FRAME-AUDIT 2026-06-12).
  * v_artifact  R_true @ v_body (TRUE attitude applied to the reported-frame twist) --
                wrong-by-model on purpose: a live canary for the twist-frame convention.
  Per-tick |v_client - v_fd| must stay small at every attitude; |v_artifact - v_fd|
  must GROW with |vE| at bank. If that flips, the sim's telemetry convention changed.

Offline reference (--offline-compare): rolls the rl_plant aero twin from the SAME
artifact-undone handoff state with the same actor and prints obs/action side by side --
the first diverging dim is the corrupted one.

Usage:
  .venv\\Scripts\\python.exe rl\\replay_obs.py --session data/runs/20260611_184326_rl_inc5_live_f1 \\
      --checkpoint rl/checkpoints/stage1_inc5_actor.pth --ticks 60 --offline-compare
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from racer.frames import euler_from_quat_wxyz, world_vec_from_body_quat
from racer.mavlink_client import parse_race_status
import fly_rl
from fly_rl import (N_GATES, OBS_LABELS, _GATE_POS_ZUP, _ODO_RATE_SIGN, build_obs,
                    load_actor, policy_step)


# ---------------------------------------------------------------------------
# tlog -> event streams
# ---------------------------------------------------------------------------
def parse_session(session: Path) -> dict:
    """One pass over mavlink.tlog -> time-stamped event streams (epoch seconds)."""
    from pymavlink import mavutil

    odo, race, coll, imu = [], [], [], []
    conn = mavutil.mavlink_connection(str(session / "mavlink.tlog"))
    try:
        while True:
            msg = conn.recv_match(blocking=False)
            if msg is None:
                break
            t = float(getattr(msg, "_timestamp", 0.0))
            typ = msg.get_type()
            if typ == "ODOMETRY":
                odo.append({
                    "t": t,
                    "pos": np.array([msg.x, msg.y, msg.z], dtype=np.float64),
                    "v_body": np.array([msg.vx, msg.vy, msg.vz], dtype=np.float64),
                    "q_raw": np.array([float(v) for v in msg.q], dtype=np.float64),
                    "w_raw": np.array([msg.rollspeed, msg.pitchspeed, msg.yawspeed],
                                      dtype=np.float64),
                    "reset_counter": int(getattr(msg, "reset_counter", 0)),
                })
            elif typ == "ENCAPSULATED_DATA":
                raw = bytes(msg.data)
                if raw and raw[0] == 1:
                    rs = parse_race_status(raw)
                    if rs is not None:
                        rs["t"] = t
                        race.append(rs)
            elif typ == "COLLISION":
                coll.append({"t": t, "id": int(msg.id),
                             "threat": int(getattr(msg, "threat_level", 0))})
            elif typ == "HIGHRES_IMU":
                imu.append({"t": t, "sim_time_ns": int(msg.time_usec) * 1_000})
    finally:
        conn.close()
    return {"odo": odo, "race": race, "coll": coll, "imu": imu}


def v_client_of(o: dict) -> np.ndarray:
    """velocity_ned exactly as MavlinkClient computes it (raw-quat rotation)."""
    return world_vec_from_body_quat(o["v_body"], o["q_raw"])


def find_handoff(odo: list[dict], dist: float, speed_min: float, skip_s: float) -> int:
    """Index of the first ODOMETRY sample crossing INTO along<dist at speed>speed_min
    (requires the previous sample at along>=dist, so prior-race residue can't match)."""
    gate0_x = float(_GATE_POS_ZUP[0, 0])
    t0 = odo[0]["t"] if odo else 0.0
    prev_along = None
    for i, o in enumerate(odo):
        along = float(o["pos"][0]) - gate0_x
        if o["t"] - t0 >= skip_s and prev_along is not None and prev_along >= dist > along:
            if float(np.linalg.norm(v_client_of(o))) > speed_min:
                return i
        prev_along = along
    raise SystemExit(f"no handoff crossing (along<{dist} at speed>{speed_min}) in the tlog")


# ---------------------------------------------------------------------------
# artifact-undone TRUE state (what build_obs believes the telemetry means)
# ---------------------------------------------------------------------------
def true_state_of(o: dict) -> dict:
    """Undo the ODOMETRY reporting artifacts the same way build_obs does.
    FRAME-AUDIT convention (2026-06-12): true quat = raw * [1,-1,1,-1] (R_y(pi)
    telemetry conjugation); true FRD rate = -w_raw. The TWIST stays in the
    REPORTED body frame -- the RAW-quat rotation (v_client) recovers the true
    world velocity; rotating it by the TRUE attitude (vel_artifact below) is
    deliberately wrong-by-model and serves as a live canary: it must DISAGREE
    with v_fd whenever the East velocity is significant at bank."""
    from scipy.spatial.transform import Rotation as _Rot
    roll, pitch, yaw = euler_from_quat_wxyz(o["q_raw"])
    q = o["q_raw"] * fly_rl._ODO_QUAT_TRUE_CONJ
    R_true = _Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    return {
        "pos": o["pos"].copy(),
        "R_frd2ned": R_true,
        "vel_artifact": R_true @ o["v_body"],              # canary: wrong frame on purpose
        "w_frd": o["w_raw"] * _ODO_RATE_SIGN,              # raw -> true FRD ([-1,-1,-1])
        "rpy_raw": (roll, pitch, yaw),
    }


def v_fd_of(odo: list[dict], i: int) -> np.ndarray | None:
    """Central-difference world velocity at odo sample i (artifact-free reference)."""
    if i < 1 or i + 1 >= len(odo):
        return None
    a, b = odo[i - 1], odo[i + 1]
    dt = b["t"] - a["t"]
    if dt <= 1e-6:
        return None
    return (b["pos"] - a["pos"]) / dt


# ---------------------------------------------------------------------------
# replay loop
# ---------------------------------------------------------------------------
def replay(session: Path, actor, args) -> list[dict]:
    ev = parse_session(session)
    odo, race = ev["odo"], ev["race"]
    if not odo:
        raise SystemExit("no ODOMETRY in tlog")
    hi = find_handoff(odo, args.handoff_dist, args.handoff_speed_min, args.skip_s)
    t0 = odo[hi]["t"]
    print(f"[replay] {session.name}: {len(odo)} odo, {len(race)} race_status, "
          f"{len(ev['coll'])} collisions; handoff at odo[{hi}] t0={t0:.3f} "
          f"pos={np.round(odo[hi]['pos'], 2).tolist()}")

    tick = 1.0 / args.rate
    rows: list[dict] = []
    oi = hi          # latest odo index <= tick time
    ri = -1          # latest race_status index <= tick time
    ci = 0           # collision cursor
    last_normed = 0.0
    gate_index = 0
    for k in range(args.ticks):
        tk = t0 + k * tick
        while oi + 1 < len(odo) and odo[oi + 1]["t"] <= tk:
            oi += 1
        while ri + 1 < len(race) and race[ri + 1]["t"] <= tk:
            ri += 1
        n_coll = 0
        max_threat = 0
        while ci < len(ev["coll"]) and ev["coll"][ci]["t"] <= tk:
            n_coll += 1
            max_threat = max(max_threat, ev["coll"][ci]["threat"])
            ci += 1
        if odo[oi]["t"] > tk:       # before the first sample (k=0 edge): hold
            continue
        o = odo[oi]
        rs = race[ri] if ri >= 0 else None
        if rs is not None and rs.get("active_gate_index") is not None:
            gate_index = min(int(rs["active_gate_index"]), N_GATES - 1)

        v_client = v_client_of(o)
        st = SimpleTelemetry(position_ned=o["pos"], velocity_ned=v_client,
                             orientation_ned_wxyz=o["q_raw"],
                             angular_rate_body=o["w_raw"])
        obs = build_obs(st, gate_index, last_normed, virtual_flip=args.virtual_flip)
        with torch.no_grad():
            mean = actor(torch.as_tensor(obs[None], dtype=torch.float32))[0].numpy()
        rate_frd, collective, last_normed = policy_step(
            actor, obs, args.max_rate, virtual_flip=args.virtual_flip)

        tru = true_state_of(o)
        v_fd = v_fd_of(odo, oi)
        rows.append({
            "tick": k, "t_rel": round(tk - t0, 4), "odo_age_ms": round((tk - o["t"]) * 1e3, 1),
            "gate_index": gate_index,
            "race_started": (bool(rs["started"]) if rs else None),
            "reset_counter": o["reset_counter"],
            "n_coll_cum": ci, "coll_in_tick": n_coll, "coll_max_threat": max_threat,
            "pos_ned": o["pos"].tolist(), "v_body_raw": o["v_body"].tolist(),
            "q_raw": o["q_raw"].tolist(), "w_raw": o["w_raw"].tolist(),
            "rpy_raw_deg": np.degrees(tru["rpy_raw"]).tolist(),
            "v_client": v_client.tolist(),
            "v_artifact": tru["vel_artifact"].tolist(),
            "v_fd": (v_fd.tolist() if v_fd is not None else None),
            "v_client_err": (float(np.linalg.norm(v_client - v_fd)) if v_fd is not None else None),
            "v_artifact_err": (float(np.linalg.norm(tru["vel_artifact"] - v_fd))
                               if v_fd is not None else None),
            "obs": obs.tolist(),
            "actor_mean": mean.tolist(), "tanh": np.tanh(mean).tolist(),
            "act_rescaled": (fly_rl._ACT_MIN + (fly_rl._ACT_MAX - fly_rl._ACT_MIN)
                             * (np.tanh(mean.astype(np.float64)) + 1.0) / 2.0).tolist(),
            "rate_frd": rate_frd.tolist(), "collective": collective,
            "normed_thrust": last_normed,
        })
    return rows


class SimpleTelemetry:
    """Duck-typed DroneState carrying exactly the fields build_obs reads."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


# ---------------------------------------------------------------------------
# offline twin reference from the same handoff state
# ---------------------------------------------------------------------------
def offline_reference(rows: list[dict], actor, args) -> list[dict]:
    from racer.rl_plant import (ALPHA_MAX_RPS2_MEASURED, COLL_MAP_ACCEL_MEASURED,
                                COLL_MAP_THR_MEASURED, PlantParams, PlantState,
                                QUAD_DRAG_C2_MEASURED, SUPER_RATE_S_MEASURED,
                                step as plant_step)

    r0 = rows[0]
    q0 = np.array(r0["q_raw"], dtype=np.float64) * fly_rl._ODO_QUAT_TRUE_CONJ
    quat = q0 / np.linalg.norm(q0)          # telemetry conjugation -> true attitude
    st = PlantState(
        pos=np.array(r0["pos_ned"]),
        vel=np.array(r0["v_client"]),       # raw-quat twist rotation = true world velocity
        quat=quat,
        omega=np.array(r0["w_raw"]) * _ODO_RATE_SIGN,
        thrust=np.float64(fly_rl._HOVER_THRUST),
    )
    from offline_rollout import _RATE_SIGN_LIVE
    params = PlantParams(transport_delay_steps=args.latency_steps,
                         rate_sign=_RATE_SIGN_LIVE.copy(),
                         super_rate_s=SUPER_RATE_S_MEASURED,
                         alpha_max_rps2=ALPHA_MAX_RPS2_MEASURED.copy(),
                         linear_drag=0.0,
                         quad_drag_c2=QUAD_DRAG_C2_MEASURED.copy(),
                         coll_map_thr=COLL_MAP_THR_MEASURED.copy(),
                         coll_map_accel=COLL_MAP_ACCEL_MEASURED.copy())
    from offline_rollout import obs_from_truth   # same repo, same conventions
    out: list[dict] = []
    last_normed = 0.0
    gate = rows[0]["gate_index"]
    dt = 1.0 / args.rate
    for k in range(args.ticks):
        obs = obs_from_truth(st, gate, last_normed, args.virtual_flip)
        with torch.no_grad():
            mean = actor(torch.as_tensor(obs[None], dtype=torch.float32))[0].numpy()
        rate_frd, collective, last_normed = policy_step(
            actor, obs, args.max_rate, virtual_flip=args.virtual_flip)
        out.append({
            "tick": k, "pos_ned": st.pos.tolist(), "vel_ned": st.vel.tolist(),
            "obs": obs.tolist(), "actor_mean": mean.tolist(),
            "rate_frd": rate_frd.tolist(), "collective": collective,
            "normed_thrust": last_normed, "gate_index": gate,
        })
        st = plant_step(st, np.concatenate([rate_frd, [collective]]), dt, params)
        # advance the target gate on plane crossings (offline_rollout's gate_event)
        from offline_rollout import gate_event
        prev = np.array(out[-1]["pos_ned"])
        ev = gate_event(prev, st.pos, gate)
        if ev == "pass" and gate < N_GATES - 1:
            gate += 1
    return out


# ---------------------------------------------------------------------------
# table printing
# ---------------------------------------------------------------------------
def _fmt3(v) -> str:
    return "[" + ",".join(f"{x:+7.2f}" for x in v) + "]"


def print_table(rows: list[dict], n: int, title: str, offline: bool = False) -> None:
    print(f"\n---- {title} ----")
    print(f"{'tk':>3} {'age':>5} {'gi':>2} {'pos_g (obs0:3)':>26} {'vel_g (obs3:6)':>26} "
          f"{'rpy_g (obs6:9)':>26} {'w_flu (obs9:12)':>26} {'o12':>6} "
          f"{'thr':>6} {'rate_frd':>26} {'vErrCli':>7} {'vErrArt':>7} {'col':>3}")
    for r in rows[:n]:
        o = r["obs"]
        ec = r.get("v_client_err")
        ea = r.get("v_artifact_err")
        ec_s = f"{ec:.2f}" if ec is not None else "-"
        ea_s = f"{ea:.2f}" if ea is not None else "-"
        print(f"{r['tick']:>3} "
              f"{r.get('odo_age_ms', 0):>5} "
              f"{r['gate_index']:>2} "
              f"{_fmt3(o[0:3]):>26} {_fmt3(o[3:6]):>26} {_fmt3(o[6:9]):>26} "
              f"{_fmt3(o[9:12]):>26} {o[12]:>6.2f} "
              f"{r['collective']:>6.3f} {_fmt3(r['rate_frd']):>26} "
              f"{ec_s:>7} {ea_s:>7} "
              f"{r.get('coll_in_tick', 0):>3}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--checkpoint",
                    default=str(Path(__file__).resolve().parent / "checkpoints"
                                / "stage1_inc5_actor.pth"))
    ap.add_argument("--rate", type=float, default=30.0)
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--max-rate", type=float, default=0.0)
    ap.add_argument("--virtual-flip", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--handoff-dist", type=float, default=3.0)
    ap.add_argument("--handoff-speed-min", type=float, default=4.0)
    ap.add_argument("--skip-s", type=float, default=0.0,
                    help="ignore the first N seconds of the tlog (prior-race residue)")
    ap.add_argument("--latency-steps", type=int, default=0)
    ap.add_argument("--offline-compare", action="store_true")
    ap.add_argument("--table-n", type=int, default=20)
    ap.add_argument("--out", default="", help="JSONL dump path ('' = <session>/replay_obs.jsonl)")
    args = ap.parse_args()

    session = Path(args.session)
    actor = load_actor(args.checkpoint)
    rows = replay(session, actor, args)
    out = Path(args.out) if args.out else session / "replay_obs.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "header", "session": str(session),
                            "checkpoint": str(args.checkpoint),
                            "obs_labels": OBS_LABELS,
                            "act_min": fly_rl._ACT_MIN.tolist(),
                            "act_max": fly_rl._ACT_MAX.tolist()}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[replay] {len(rows)} ticks -> {out}")

    print_table(rows, args.table_n, f"LIVE-RECONSTRUCTED ({session.name})")

    bad = [r for r in rows if not np.all(np.isfinite(r["obs"]))]
    if bad:
        print(f"\n!! NON-FINITE OBS at ticks {[r['tick'] for r in bad[:20]]}")
    stale = [r for r in rows if r["odo_age_ms"] > 100.0]
    if stale:
        print(f"!! ODOMETRY staleness >100 ms at ticks {[r['tick'] for r in stale[:20]]}")
    rc = {r["reset_counter"] for r in rows}
    if len(rc) > 1:
        print(f"!! reset_counter changes during replay: {sorted(rc)}")

    if args.offline_compare:
        ref = offline_reference(rows, actor, args)
        print_table(ref, args.table_n, "OFFLINE TWIN from the same handoff state")
        ref_out = out.with_name(out.stem + "_offline.jsonl")
        with open(ref_out, "w", encoding="utf-8") as f:
            for r in ref:
                f.write(json.dumps(r) + "\n")
        print(f"[offline] {len(ref)} steps -> {ref_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
