"""Offline A/B of the ego obs-builder FIX GAIN (``--ego-fix-gain``) over real flight logs.

THE QUESTION. Until 2026-07-25 the deploy obs builder SNAPPED the held gate lever to every accepted
vision fix (K=1). Training did not: its estimator low-passed each fix into the ego-propagated belief
with K = 1/N_eff, N_eff ~ U[4,9] (rl/ego_estimator.py:722-734), i.e. K ~ 0.154. Inside 1-2 m of a
gate 96% of fixes carry <4 corners and 23% fall back to bbox range, so the snap writes that noise
straight into the vertical target the policy chases. Does the training gain actually damp the
terminal vertical jitter -- and does the belief stay glued to vision at long range, or drift?

METHOD. This does NOT re-implement the blend: it drives the REAL ``EgoObsBuilder`` (twice, at two
gains) over the logged fix sequence, so what is measured is the shipped code path.

  * ``ego_obs.jsonl`` records, per control tick: ``obs`` (the 21-dim vector as flown), ``rel_flu``
    (the HELD slot0 lever, body FLU, full precision), ``pose_seen`` (a fresh slot0 fix landed this
    tick), the slot1 mirror (``rel_flu1`` / ``pose_seen1``), ``gate_index`` and ``sim_time_ns``.
  * The flights replayed here flew K=1, so at a ``pose_seen`` tick the logged ``rel_flu`` IS the raw
    fix -- that is what makes an honest replay possible at all. Those become the ``pose=``/
    ``next_pose=`` inputs (``t_cam_gate`` is the exact inverse of
    ``ego_obs.rel_pos_body_frd_from_gatepose``, so the builder recovers the logged lever bit-for-bit).
  * The drone's own motion (which drives the between-fix ego-propagation) comes from the obs itself:
    obs[0:3] is the virtual-flipped body-FLU velocity and obs[5:8] the virtual-flipped body rates,
    and the flip matrix diag(-1,-1,1) is its own inverse. Feeding R_frd2ned = I with vel_ned /
    gyro_frd back-solved from those reproduces the builder's v_flu / w_flu exactly. NOTE obs is
    logged ROUNDED TO 5 DECIMALS, so the replayed propagation carries ~1e-5 m/s of velocity
    quantisation -- identical in both arms, and reported as the driver-fidelity residual below.
  * AIM-OFFSET TICKS ARE UNREPLAYABLE AND ARE CUT. 8 of these 50 sessions carry a non-null ``aim_off``
    ([lateral, vertical] metres) on some ticks -- e.g. [0.0, 10.0], which lands in the log's rel_flu as
    a pure +10 m on the vertical while forward/lateral match the seeker emission exactly. NOTHING in
    this repo writes that field (``git log -S aim_off`` is empty on every branch), so those ticks were
    flown by ShadowPC-side code that is not here and their lever cannot be reconstructed. Each session
    is therefore split into CLEAN SEGMENTS at those ticks and each segment is replayed with a fresh
    builder; the driver-fidelity line below is the proof that what remains replays exactly.

WHAT IS REPORTED.
  * ``--check-identical``: the SAME driver through the PRE-CHANGE builder (loaded straight out of
    git) vs the new one at fix_gain=1.0, asserting every emitted 21-dim obs is BITWISE equal. This
    is the empirical byte-identity proof for the default.
  * Tick-to-tick |delta vertical| of the fed gate lever (median / p90 / p99), bucketed by forward
    distance -- the jitter the policy's vertical target actually carries. Only pairs where BOTH
    ticks are FED (the training keep-mask: det_proxy AND det_geom AND conf>0) and the gate index is unchanged
    count, because a masked tick feeds a structural zero, not a noisy target.
  * Belief-vs-fix tracking at long range: median |blended belief - raw fix| at fix ticks in the 7-9 m
    band. A low gain that had gone stale would show up here as a growing offset.

Usage:
  python scripts/replay_fix_gain.py --glob "20260724_2[23]*v19pick_Ws0*" --gains 1.0 0.154 \
      --check-identical
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "data" / "runs"
sys.path.insert(0, str(ROOT / "src"))

from racer import frames                                              # noqa: E402
from racer.contracts import GatePose                                  # noqa: E402
from racer.ego_obs import (EgoObsBuilder, EgoObsBuilderConfig,        # noqa: E402
                           _FLIP_FRD_FLU, _RZ_PI_BODY)

# Forward-distance bands (metres, half-open [lo, hi)) -- the terminal approach split the way the
# close-in blackout forensics slice it: coarse far out, 1 m steps once the gate fills the frame.
BANDS = [(7.0, 9.0), (5.0, 7.0), (3.0, 5.0), (2.0, 3.0), (1.0, 2.0), (0.0, 1.0)]
BAND_LABELS = ["9-7 m", "7-5 m", "5-3 m", "3-2 m", "2-1 m", "1-0 m"]


# --------------------------------------------------------------------------------------------
# log -> builder inputs
# --------------------------------------------------------------------------------------------
def _pose_from_rel_flu(rel_flu, sim_time_ns: int, area: float | None) -> GatePose:
    """A GatePose whose lever, put through ``rel_pos_body_frd_from_gatepose``, returns exactly the
    logged ``rel_flu``. Inverts that function: rel_frd = R_cb^T t + [0,0,voff]  =>
    t = R_cb (rel_frd - [0,0,voff]), with rel_frd = _FLIP_FRD_FLU * rel_flu.

    ``visible_area_meas`` is set from the logged area so the area channel replays as flown instead
    of being re-projected through a fabricated rotation (R_cam_gate is then never read)."""
    rel_frd = _FLIP_FRD_FLU * np.asarray(rel_flu, dtype=np.float64)
    voff = frames.BORESIGHT.vert_offset_m
    t_cam = frames.R_camera_from_body() @ (rel_frd - np.array([0.0, 0.0, voff]))
    return GatePose(frame_id=0, sim_time_ns=int(sim_time_ns), R_cam_gate=np.eye(3),
                    t_cam_gate=t_cam, reproj_error_px=0.0, n_corners=4,
                    visible_area_meas=(None if area is None else float(area)))


def load_ticks(session: Path) -> list[dict]:
    """Per-tick builder inputs for one session, from ego_obs.jsonl."""
    ticks = []
    for line in (session / "ego_obs.jsonl").open():
        if not line.strip():
            continue
        r = json.loads(line)
        obs = np.asarray(r["obs"], dtype=np.float64)
        # obs[0:3] = _RZ_PI_BODY @ v_flu, v_flu = _FLIP_FRD_FLU * (R^T v_ned); obs[5:8] likewise for
        # the rates. R = I and _RZ_PI_BODY / _FLIP_FRD_FLU are their own inverses, so:
        v_flu = _RZ_PI_BODY @ obs[0:3]
        w_flu = _RZ_PI_BODY @ obs[5:8]
        ticks.append(dict(
            sim_time_ns=int(r["sim_time_ns"]),
            gate_index=int(r["gate_index"]),
            vel_ned=_FLIP_FRD_FLU * v_flu,          # R = I  =>  v_ned = _FLIP * v_flu
            gyro_frd=_FLIP_FRD_FLU * w_flu,         # w_flu  = _FLIP * w_frd
            last_normed_thrust=float(r.get("normed_thrust") or 0.0),
            pose=(_pose_from_rel_flu(r["rel_flu"], r["sim_time_ns"], r.get("area"))
                  if (r.get("pose_seen") and r.get("rel_flu") is not None) else None),
            next_pose=(_pose_from_rel_flu(r["rel_flu1"], r["sim_time_ns"], r.get("area1"))
                       if (r.get("pose_seen1") and r.get("rel_flu1") is not None) else None),
            logged_rel_flu=(None if r.get("rel_flu") is None
                            else np.asarray(r["rel_flu"], dtype=np.float64)),
            logged_pose_seen=bool(r.get("pose_seen")),
            # non-null aim_off => the flown lever carries an offset this repo cannot reproduce
            dirty=(r.get("aim_off") is not None),
        ))
    return ticks


def clean_segments(ticks: list[dict], min_len: int = 8) -> list[list[dict]]:
    """Split a session at the unreplayable aim-offset ticks; keep runs of at least ``min_len``."""
    segs, cur = [], []
    for t in ticks:
        if t["dirty"]:
            if len(cur) >= min_len:
                segs.append(cur)
            cur = []
        else:
            cur.append(t)
    if len(cur) >= min_len:
        segs.append(cur)
    return segs


def session_cfg(session: Path, **over) -> EgoObsBuilderConfig:
    """The builder config this session FLEW (from meta.json), with overrides applied."""
    m = json.loads((session / "meta.json").read_text())
    return EgoObsBuilderConfig(
        stale_horizon_s=float(m.get("ego_stale_horizon", 0.5)),
        det_hold_s=float(m.get("ego_det_hold", 0.2)),
        obs_coast=bool(m.get("ego_obs_coast", False)),
        virtual_flip=bool(m.get("virtual_flip", True)),
        slot1_enabled=bool(m.get("ego_slot1", False)),
        sector_mode="zero",   # the flown 'map' rows are a CONSTANT per gate: they never touch the
                              # gate lever, and pinning them here keeps the driver map-file-free.
        **over)


def run_builder(builder, ticks: list[dict]) -> list[dict]:
    """Drive one builder over one session; return per-tick outputs."""
    out = []
    for t in ticks:
        obs = builder.update(
            sim_time_ns=t["sim_time_ns"], gate_index=t["gate_index"], R_frd2ned=np.eye(3),
            vel_ned=t["vel_ned"], gyro_frd=t["gyro_frd"], pose=t["pose"],
            next_pose=t["next_pose"], last_normed_thrust=t["last_normed_thrust"])
        d = builder.last_diag
        rel = d.get("rel_flu")
        out.append(dict(
            obs=obs,
            gate_index=t["gate_index"],
            rel_flu=(None if rel is None else np.asarray(rel, dtype=np.float64)),
            # FED == the training keep-mask the builder just applied: only then does the lever
            # reach the policy rather than a structural zero. The det is det_proxy AND det_geom --
            # ``det_geom`` (--ego-det-geometric, 2026-07-27) is ABSENT from every log written with
            # the knob off, so the True default leaves this expression identical on the whole
            # historical corpus while keeping it honest on an ARMED flight, where reading det_proxy
            # alone would call a geometrically-masked tick "fed".
            fed=(bool(d.get("det_proxy")) and bool(d.get("det_geom", True))
                 and float(d.get("conf") or 0.0) > 0.0),
            pose_seen=bool(d.get("pose_seen")),
        ))
    return out


# --------------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------------
def _band(fwd: float) -> int | None:
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= fwd < hi:
            return i
    return None


def jitter_by_band(rows: list[dict], ref_rows: list[dict]) -> list[list[float]]:
    """|delta vertical| of the FED gate lever, per forward-distance band.

    Bucketing uses ``ref_rows`` (the K=1 arm) for BOTH arms so a tick lands in the same band
    whichever gain produced it -- otherwise the gains would be compared over different tick sets."""
    buckets: list[list[float]] = [[] for _ in BANDS]
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        if a["rel_flu"] is None or b["rel_flu"] is None:
            continue
        if not (a["fed"] and b["fed"]) or a["gate_index"] != b["gate_index"]:
            continue
        ref = ref_rows[i]["rel_flu"]
        if ref is None:
            continue
        bi = _band(float(ref[0]))                       # rel_flu[0] = FORWARD (body FLU)
        if bi is not None:
            buckets[bi].append(abs(float(b["rel_flu"][2]) - float(a["rel_flu"][2])))
    return buckets


def belief_vs_fix(rows: list[dict], ticks: list[dict], ref_rows: list[dict],
                  lo: float, hi: float) -> list[float]:
    """SIGNED (blended belief - raw fix), vertical, at FIX ticks with forward distance in [lo, hi).

    The staleness test. A low gain that had stopped tracking shows up two ways: a large |value|, and
    -- the real tell -- a one-signed MEAN, since a lagging filter sits consistently behind a moving
    target. A belief that is merely SMOOTHING sits scattered around the fixes with a ~zero mean."""
    out = []
    for i, (row, t) in enumerate(zip(rows, ticks)):
        if not t["logged_pose_seen"] or t["logged_rel_flu"] is None or row["rel_flu"] is None:
            continue
        ref = ref_rows[i]["rel_flu"]
        if ref is None or not (lo <= float(ref[0]) < hi):
            continue
        out.append(float(row["rel_flu"][2]) - float(t["logged_rel_flu"][2]))
    return out


def fix_to_fix(ticks: list[dict], ref_rows: list[dict], lo: float, hi: float) -> list[float]:
    """SIGNED delta vertical between CONSECUTIVE RAW FIXES in [lo, hi). |.| is the per-fix noise
    scale the belief-vs-fix offset above must be judged against; the MEAN is the target's real
    drift, which is what any low-pass necessarily lags behind."""
    out, prev = [], None
    for i, t in enumerate(ticks):
        if not t["logged_pose_seen"] or t["logged_rel_flu"] is None:
            continue
        ref = ref_rows[i]["rel_flu"]
        if prev is not None and ref is not None and lo <= float(ref[0]) < hi:
            out.append(float(t["logged_rel_flu"][2]) - prev)
        prev = float(t["logged_rel_flu"][2])
    return out


def prop_gap_error(rows: list[dict], ticks: list[dict], lo: float, hi: float) -> list[float]:
    """SIGNED vertical (ego-propagated prior - raw fix) over ONE detection gap, at K=1.

    WHY THIS EXISTS. Whether a low fix gain helps or hurts turns entirely on whether the thing it
    blends toward -- the ego-propagated prior -- is UNBIASED against vision. Feed it a prior that
    drifts one way and the gain does not smooth, it BIASES. At K=1 the held lever IS the previous
    fix, so propagating it to the next fix tick and differencing measures the propagation alone,
    with no filter in the loop. Requires ``rows`` to be the K=1 arm."""
    from scipy.spatial.transform import Rotation
    out = []
    for i in range(1, len(ticks)):
        t = ticks[i]
        prev = rows[i - 1]["rel_flu"]
        if not t["logged_pose_seen"] or t["logged_rel_flu"] is None or prev is None:
            continue
        if rows[i]["gate_index"] != rows[i - 1]["gate_index"]:
            continue
        fix = t["logged_rel_flu"]
        if not (lo <= float(fix[0]) < hi):
            continue
        dt = (t["sim_time_ns"] - ticks[i - 1]["sim_time_ns"]) / 1e9
        if dt <= 0.0:
            continue
        v_flu = _FLIP_FRD_FLU * t["vel_ned"]        # R = I in the driver, so this inverts exactly
        w_flu = _FLIP_FRD_FLU * t["gyro_frd"]
        prior = Rotation.from_rotvec(-w_flu * dt).as_matrix() @ prev - v_flu * dt
        out.append(float(prior[2]) - float(fix[2]))
    return out


def _q(v: list[float], p: float) -> float:
    return float(np.percentile(np.asarray(v), p)) if v else float("nan")


# --------------------------------------------------------------------------------------------
# byte-identity: the PRE-CHANGE builder, loaded straight out of git
# --------------------------------------------------------------------------------------------
def load_baseline_builder(rev: str):
    """Import ``src/racer/ego_obs.py`` AS OF ``rev`` under a private module name."""
    src = subprocess.run(["git", "show", f"{rev}:src/racer/ego_obs.py"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    tmp = Path(tempfile.mkdtemp()) / "ego_obs_baseline.py"
    tmp.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("ego_obs_baseline", tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ego_obs_baseline"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="20260724_2[23]*v19pick_Ws0*",
                    help="session-directory glob under data/runs.")
    ap.add_argument("--gains", type=float, nargs="+", default=[1.0, 0.154],
                    help="fix_gain values to compare; the FIRST is the reference arm.")
    ap.add_argument("--check-identical", action="store_true",
                    help="also replay the PRE-CHANGE builder (--baseline-rev) and assert every "
                         "emitted obs is bitwise equal to the new builder at fix_gain=1.0.")
    ap.add_argument("--baseline-rev", default="ratchet-arrestor-2026-07-18",
                    help="git rev to read the pre-change src/racer/ego_obs.py from.")
    args = ap.parse_args()

    sessions = [d for d in sorted(RUNS.glob(args.glob))
                if (d / "ego_obs.jsonl").exists() and (d / "meta.json").exists()]
    if not sessions:
        print(f"no sessions matched {args.glob!r} under {RUNS}", file=sys.stderr)
        return 2

    baseline_mod = load_baseline_builder(args.baseline_rev) if args.check_identical else None
    # belief-vs-fix is reported in two bands: 7-9 m (the REQUIRED staleness check -- is the low gain
    # still tracking where vision is good?) and 0-2 m (the terminal band the defect lives in).
    TRACK_BANDS = [(7.0, 9.0), (0.0, 2.0)]
    buckets = {g: [[] for _ in BANDS] for g in args.gains}
    track = {g: [[] for _ in TRACK_BANDS] for g in args.gains}
    noise_scale: list[list[float]] = [[] for _ in TRACK_BANDS]
    prop_err: list[list[float]] = [[] for _ in TRACK_BANDS]
    n_ticks = n_dirty = n_used = 0
    drive_resid = []            # replayed vs LOGGED rel_flu at the reference gain (driver fidelity)
    identical, mismatch = 0, 0

    for sess in sessions:
        ticks = load_ticks(sess)
        if not ticks:
            continue
        n_ticks += len(ticks)
        n_dirty += sum(1 for t in ticks if t["dirty"])
        for seg in clean_segments(ticks):
            n_used += len(seg)
            runs = {}
            for g in args.gains:
                runs[g] = run_builder(EgoObsBuilder(session_cfg(sess, fix_gain=g)), seg)
            ref = runs[args.gains[0]]

            for g in args.gains:
                for bi, vals in enumerate(jitter_by_band(runs[g], ref)):
                    buckets[g][bi].extend(vals)
                for ti, (lo, hi) in enumerate(TRACK_BANDS):
                    track[g][ti].extend(belief_vs_fix(runs[g], seg, ref, lo, hi))
            for ti, (lo, hi) in enumerate(TRACK_BANDS):
                noise_scale[ti].extend(fix_to_fix(seg, ref, lo, hi))
                if 1.0 in runs:
                    prop_err[ti].extend(prop_gap_error(runs[1.0], seg, lo, hi))

            # driver fidelity: how close the reference replay lands to what actually flew
            for row, t in zip(ref, seg):
                if row["rel_flu"] is not None and t["logged_rel_flu"] is not None:
                    drive_resid.append(float(np.max(np.abs(row["rel_flu"] - t["logged_rel_flu"]))))

            if baseline_mod is not None:
                m = json.loads((sess / "meta.json").read_text())
                base_cfg = baseline_mod.EgoObsBuilderConfig(
                    stale_horizon_s=float(m.get("ego_stale_horizon", 0.5)),
                    det_hold_s=float(m.get("ego_det_hold", 0.2)),
                    obs_coast=bool(m.get("ego_obs_coast", False)),
                    virtual_flip=bool(m.get("virtual_flip", True)),
                    slot1_enabled=bool(m.get("ego_slot1", False)), sector_mode="zero")
                base = run_builder(baseline_mod.EgoObsBuilder(base_cfg), seg)
                new1 = (runs.get(1.0)
                        or run_builder(EgoObsBuilder(session_cfg(sess, fix_gain=1.0)), seg))
                for rb, rn in zip(base, new1):
                    if rb["obs"].tobytes() == rn["obs"].tobytes():
                        identical += 1
                    else:
                        mismatch += 1

    ref_g = args.gains[0]
    print(f"sessions {len(sessions)}   logged ticks {n_ticks}   "
          f"aim_off (unreplayable, CUT) {n_dirty}   replayed {n_used}")
    if drive_resid:
        print(f"driver fidelity (replay@K={ref_g:g} vs LOGGED rel_flu, max-abs per tick): "
              f"median {np.median(drive_resid):.3e} m  p99 {_q(drive_resid, 99):.3e} m  "
              f"max {max(drive_resid):.3e} m")
    if baseline_mod is not None:
        print(f"BYTE-IDENTITY vs {args.baseline_rev} @ fix_gain=1.0: "
              f"{identical} obs bitwise-equal, {mismatch} differing "
              f"-> {'IDENTICAL' if mismatch == 0 else 'DIVERGED'}")

    print("\nTICK-TO-TICK |delta vertical| of the FED gate lever, metres, by forward distance")
    print("  (>0.25 = the fraction of tick pairs jumping more than a quarter-metre -- the tail that")
    print("   kills a flight; a strong median with a fat tail is still a dead flight.)")
    head = f"{'band':>7} {'n':>7} " + " ".join(
        f"{f'K={g:g} med':>11} {'p90':>7} {'p99':>7} {'>0.25':>7}" for g in args.gains)
    print(head)
    print("-" * len(head))

    def _row(label: str, per_gain: dict, n: int) -> None:
        cells = []
        for g in args.gains:
            v = per_gain[g]
            frac = (sum(1 for x in v if x > 0.25) / len(v)) if v else float("nan")
            cells.append(f"{_q(v, 50):>11.4f} {_q(v, 90):>7.4f} {_q(v, 99):>7.4f} {frac:>7.3f}")
        print(f"{label:>7} {n:>7} " + " ".join(cells))

    for bi, label in enumerate(BAND_LABELS):
        _row(label, {g: buckets[g][bi] for g in args.gains}, len(buckets[ref_g][bi]))
    inside2 = {g: buckets[g][4] + buckets[g][5] for g in args.gains}      # the 2-1 and 1-0 bands
    print("-" * len(head))
    _row("<2 m", inside2, len(inside2[ref_g]))

    print("\nBELIEF-vs-FIX (SIGNED vertical belief - raw fix, at fix ticks). A SMOOTHING belief")
    print("scatters around the fixes with a ~zero mean; a STALE one sits one-signed behind them --")
    print("and a low-pass necessarily lags by ~(1/K - 1) ticks of the target's own drift.")
    for ti, (lo, hi) in enumerate(TRACK_BANDS):
        ns = np.asarray(noise_scale[ti]) if noise_scale[ti] else np.zeros(0)
        print(f"  [{lo:g}-{hi:g} m]  raw fix(k)-fix(k-1): median|.| "
              f"{_q(list(np.abs(ns)), 50):.4f} m   MEAN(signed, = the target's drift/fix) "
              f"{(float(ns.mean()) if len(ns) else float('nan')):+.4f} m   n={len(ns)}")
        for g in args.gains:
            v = np.asarray(track[g][ti]) if track[g][ti] else np.zeros(0)
            a = list(np.abs(v))
            print(f"        K={g:<6g} n={len(v):<6} median|.| {_q(a, 50):.4f} m   p90|.| "
                  f"{_q(a, 90):.4f} m   MEAN(signed) "
                  f"{(float(v.mean()) if len(v) else float('nan')):+.4f} m")
        pe = np.asarray(prop_err[ti]) if prop_err[ti] else np.zeros(0)
        print(f"        PROPAGATION alone (ego-propagated prior - fix, one gap, K=1): n={len(pe)} "
              f"median {_q(list(pe), 50):+.4f} m   MEAN(signed) "
              f"{(float(pe.mean()) if len(pe) else float('nan')):+.4f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
