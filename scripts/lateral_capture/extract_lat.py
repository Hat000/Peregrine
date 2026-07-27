"""Extract per-tick lateral-capture geometry for gates 1-3 across the wire corpus.

FRAME CONTRACT (the part that kills analyses):
  logged rel_flu     : TRUE body FLU [fwd, left, up], UNFLIPPED.
  obs[3:5] roll,pitch: VIRTUAL-FLIPPED  -> true roll = -obs[3], true pitch = -obs[4]
  obs[11:14] rel_pos : VIRTUAL-FLIPPED  -> rel_flu = [-o11, -o12, +o13]   <-- WHAT THE POLICY SEES
  obs[5:8] rates     : VIRTUAL-FLIPPED  -> true w_flu = [-o5, -o6, +o7]

Gravity-levelling uses the SAME R = Ry(pitch) @ Rx(roll) as scripts/failure_profile/vg.py, so
    L_lat  = rel[1]*cos(roll) - rel[2]*sin(roll)        (the parent's formula -- verified identical)
    L_vert = -sin(pitch)*rel[0] + cos(pitch)*sin(roll)*rel[1] + cos(pitch)*cos(roll)*rel[2]
    L_fwd  =  cos(pitch)*rel[0] + sin(pitch)*sin(roll)*rel[1] + sin(pitch)*cos(roll)*rel[2]

B_lat = rel[1] is the RAW BODY lateral -- the only fine-grained lateral the policy is fed.
"""
import glob
import json
import os
import pickle
import sys

import numpy as np

RUNS = r"C:\Users\Fengy\Downloads\Projects\wt-arrest\data\runs"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lat_ticks.pkl")


def lineage_of(ck, label):
    s = f"{ck or ''}|{label or ''}".lower()
    for tag in ("v20", "v19", "v18", "v17", "v16", "v15"):
        if tag in s:
            return tag
    if "vpef" in s:
        return "vpef"
    return "other"


def do_session(p):
    try:
        meta = json.load(open(os.path.join(p, "meta.json"), encoding="utf-8"))
    except Exception:
        return None
    fs = meta.get("final_state")
    recs = []
    try:
        with open(os.path.join(p, "ego_obs.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                try:
                    recs.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        return None
    n = len(recs)
    if n < 20:
        return None
    obs = np.array([(r["obs"] if r.get("obs") and len(r["obs"]) == 21 else [np.nan] * 21)
                    for r in recs], dtype=float)
    t = np.array([r["sim_time_ns"] for r in recs], dtype=np.float64) / 1e9
    t -= t[0]
    gi = np.array([r.get("gate_index", 0) for r in recs], dtype=int)
    rel = np.full((n, 3), np.nan)
    for i, r in enumerate(recs):
        v = r.get("rel_flu")
        if v is not None:
            rel[i] = v
        else:
            o = obs[i]
            if np.isfinite(o[11:15]).all() and o[14] > 0.0:
                rel[i] = (-o[11], -o[12], o[13])
    roll = -obs[:, 3]
    pitch = -obs[:, 4]
    seen = np.array([bool(r.get("pose_seen", False)) for r in recs])
    age = np.array([float(r.get("age_s") or 0.0) for r in recs])
    conf = np.array([float(r.get("conf") or 0.0) for r in recs])
    rate = np.array([(r.get("rate_frd") or [np.nan] * 3) for r in recs], dtype=float)
    am = np.array([(r.get("actor_mean") or [np.nan] * 4) for r in recs], dtype=float)
    coll = np.array([float(r.get("collective") or np.nan) for r in recs])
    aim = np.zeros(n, dtype=bool)
    for i, r in enumerate(recs):
        a = r.get("aim_off", None)
        if a is not None:
            try:
                aim[i] = (abs(float(a[0])) > 1e-6) or (abs(float(a[1])) > 1e-6)
            except Exception:
                aim[i] = True
    rho = np.linalg.norm(rel, axis=1)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    L_lat = rel[:, 1] * cr - rel[:, 2] * sr
    L_vert = -sp * rel[:, 0] + cp * sr * rel[:, 1] + cp * cr * rel[:, 2]
    L_fwd = cp * rel[:, 0] + sp * sr * rel[:, 1] + sp * cr * rel[:, 2]
    zb = float(meta.get("ego_gate_z_bias") or 0.0)
    L_vert = L_vert + zb * float(np.cos(np.radians(20.0)))
    # closure rate from RANGE only (vision-only; never obs[0:3])
    drho = np.full(n, np.nan)
    dt = np.diff(t, prepend=t[0])
    ok = np.isfinite(rho)
    for i in range(2, n):
        if ok[i] and ok[i - 2] and (t[i] - t[i - 2]) > 1e-3:
            drho[i] = (rho[i] - rho[i - 2]) / (t[i] - t[i - 2])
    gmax = int(gi.max())
    rows = []
    for g in (1, 2, 3):
        m = gi == g
        if not m.any():
            continue
        idx = np.where(m)[0]
        died = (fs == "CRASH") and (gmax == g) and (int(gi[-1]) == g)
        passed = gmax > g
        if not (died or passed):
            continue
        if aim[idx].any():
            continue
        keep = idx[np.isfinite(rho[idx]) & (rho[idx] >= 0.8) & (rho[idx] <= 14.0)]
        if len(keep) < 4:
            continue
        for i in keep:
            rows.append((g, 1 if died else 0, t[i], rho[i],
                         rel[i, 0], rel[i, 1], rel[i, 2], roll[i], pitch[i],
                         L_lat[i], L_vert[i], L_fwd[i],
                         rate[i, 0], rate[i, 1], rate[i, 2],
                         -obs[i, 5], -obs[i, 6], obs[i, 7],
                         am[i, 1], coll[i], drho[i],
                         1.0 if seen[i] else 0.0, age[i], conf[i], obs[i, 15],
                         float(i - idx[0]), float(t[idx[-1]] - t[i])))
    if not rows:
        return None
    A = np.array(rows, dtype=float)
    return dict(run=os.path.basename(p), ckpt=meta.get("ego_ckpt"), label=meta.get("label"),
                lineage=lineage_of(meta.get("ego_ckpt"), meta.get("label")),
                final_state=fs, collisions=meta.get("collisions"),
                zbias=zb, pitch_clamp=meta.get("ego_pitch_clamp"),
                roll_clamp=meta.get("ego_roll_clamp"), rate_scale=meta.get("ego_rate_scale"),
                A=A)


COLS = ["gate", "died", "t", "rho", "relx", "rely", "relz", "roll", "pitch",
        "L_lat", "L_vert", "L_fwd", "cmd_roll", "cmd_pitch", "cmd_yaw",
        "w_roll", "w_pitch", "w_yaw", "am_roll", "coll", "drho",
        "seen", "age", "conf", "area", "k_in_gate", "t_to_end"]


def main():
    paths = sorted(glob.glob(os.path.join(RUNS, "*")))
    paths = [p for p in paths if os.path.isdir(p)]
    out = []
    for i, p in enumerate(paths):
        try:
            r = do_session(p)
        except Exception as ex:
            r = None
            print(f"  ERR {os.path.basename(p)}: {ex}", flush=True)
        if r is not None:
            out.append(r)
        if i % 100 == 0:
            print(f"  [{i}/{len(paths)}] kept {len(out)}", flush=True)
    pickle.dump(dict(cols=COLS, sessions=out), open(OUT, "wb"))
    tot = sum(len(r["A"]) for r in out)
    print(f"sessions with usable gate1-3 approaches: {len(out)}   ticks: {tot}")


if __name__ == "__main__":
    sys.exit(main())
