"""Per-frame ANGULAR error decomposition of the world-fix errors.

The lever-arm model says a chain rotation error delta (world frame) moves the fix by
off = delta x L (perpendicular to L). The component of delta orthogonal to L is therefore
directly observable per frame:

    delta_perp_i = (L_hat_i x off_i) / |L_i|        [2 DOF; the along-L component is unobservable]

and the along-L component of off is the (sign-flipped) PnP depth error. This script:
  1. reports the distribution of delta_perp (world + body frames) -- the DIRECT measurement of
     the attitude-equivalent error the lever-arm covariance term must cover;
  2. tests the per-gate means (is the 'calibration' actually leg-dependent?);
  3. regresses delta_perp(body) against the body angular rate (odo_angular_rate) -- a consistent
     image-vs-attitude timing skew dt makes delta ~= omega * dt (slope = effective skew);
  4. checks the depth channel: relative range error vs range (scale bias / long-range noise).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from racer import frames as F                      # noqa: E402
from racer.navigator import load_track_map         # noqa: E402

CHAR_DIR = ROOT / "handoff/shadowpc-assoc-flipfix-2026-06-09"
PG_DIR = ROOT / "handoff/perception-char-2026-06-08/pg"
MAP = ROOT / "handoff/shadowpc-firstcontact-2026-06-02/track_map.json"


def load_fixes():
    gates = {g.gate_id: g for g in load_track_map(MAP, corner_to_center=True)}
    out = []
    for gi in range(6):
        char = json.loads((CHAR_DIR / f"char_g{gi}_robust_k1.0.json").read_text())
        meta = json.loads((PG_DIR / f"course_g{gi}/frames.json").read_text())
        by_fid = {fr["frame_id"]: fr for fr in meta["frames"]}
        for r in char["rows"]:
            if not (r.get("associated") and "world_fix_err_m" in r):
                continue
            fr = by_fid[r["frame_id"]]
            p = np.asarray(fr["drone_position_ned"], float)
            R_wb = F.R_world_from_body(*F.euler_from_quat_wxyz(np.asarray(fr["odo_q_wxyz"], float)))
            off = np.asarray(r["off_ned"], float)
            L = gates[r["gate_id"]].position_ned - (p + off)
            out.append(dict(bundle=gi, gate_id=r["gate_id"], n_corners=r["n_corners"],
                            err=r["world_fix_err_m"], off=off, L=L, R_wb=R_wb,
                            range_m=r["true_range_m"], range_err=r["range_err_m"],
                            omega=np.asarray(fr["odo_angular_rate"], float),
                            pair_gap_ms=fr["pair_gap_ms"], odo_gap_ms=fr["odo_gap_ms"],
                            speed=fr["speed_mps"]))
    return out


def main():
    fixes = load_fixes()
    good = [r for r in fixes if r["err"] < 1.5 and r["n_corners"] == 4]
    print(f"good 4-corner fixes: {len(good)}")

    # per-frame observable angular error (world + body frames), deg
    for r in good:
        L = r["L"]; nL = np.linalg.norm(L); Lh = L / nL
        r["d_w"] = np.degrees(np.cross(Lh, r["off"]) / nL)        # world-frame delta_perp (deg)
        r["d_b"] = r["R_wb"].T @ r["d_w"]                          # body frame
        r["depth_rel"] = r["range_err"] / r["range_m"]             # relative depth error

    dw = np.array([r["d_w"] for r in good])
    db = np.array([r["d_b"] for r in good])
    mag = np.linalg.norm(dw, axis=1)
    print("\n== delta_perp (attitude-equivalent angular error), deg ==")
    print(f"  |delta| p50 {np.percentile(mag,50):.2f}  p90 {np.percentile(mag,90):.2f}  "
          f"p95 {np.percentile(mag,95):.2f}  max {mag.max():.2f}")
    for nm, d in (("world", dw), ("body ", db)):
        print(f"  {nm} mean [{d[:,0].mean():+5.2f} {d[:,1].mean():+5.2f} {d[:,2].mean():+5.2f}]  "
              f"std [{d[:,0].std():4.2f} {d[:,1].std():4.2f} {d[:,2].std():4.2f}]  "
              f"rms-per-axis {np.sqrt((d**2).mean()):.2f}")

    print("\n== per-gate mean +- std of body-frame delta_perp (deg) ==")
    for g in sorted({r["gate_id"] for r in good}):
        sub = db[[i for i, r in enumerate(good) if r["gate_id"] == g]]
        rng = [r["range_m"] for r in good if r["gate_id"] == g]
        print(f"  gate {g}: N={len(sub):3d}  range {min(rng):4.1f}-{max(rng):4.1f} m  "
              f"mean [{sub[:,0].mean():+5.2f} {sub[:,1].mean():+5.2f} {sub[:,2].mean():+5.2f}]  "
              f"std [{sub[:,0].std():4.2f} {sub[:,1].std():4.2f} {sub[:,2].std():4.2f}]")

    # timing-skew test: delta_b ~ omega * dt ; fit per-axis slope (ms) + full 3x3
    om = np.array([r["omega"] for r in good])           # rad/s (odo sign conventions apply)
    print("\n== timing-skew regression: delta_b[deg] vs omega[rad/s] ==")
    print(f"  omega magnitude: p50 {np.percentile(np.linalg.norm(om,axis=1),50):.2f}  "
          f"p90 {np.percentile(np.linalg.norm(om,axis=1),90):.2f} rad/s")
    for ax, nm in enumerate(("roll", "pitch", "yaw")):
        x, y = om[:, ax], np.radians(db[:, ax])
        if x.std() < 1e-6:
            continue
        slope = float(np.polyfit(x, y, 1)[0])
        r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
        print(f"  {nm}: slope {slope*1000:+6.2f} ms  r^2 {r2:.3f}")
    # full least-squares matrix M (delta = M omega): reveals cross-axis / sign structure
    M, *_ = np.linalg.lstsq(om, np.radians(db), rcond=None)
    print("  full 3x3 (delta_b = M^T omega), ms:")
    for i, nm in enumerate(("roll", "pitch", "yaw")):
        print(f"    d_{nm}: [{M[0,i]*1000:+6.2f} {M[1,i]*1000:+6.2f} {M[2,i]*1000:+6.2f}]")
    pred = om @ M
    resid = np.radians(db) - pred
    print(f"  delta_b rms {np.degrees(np.sqrt((np.radians(db)**2).mean())):.2f} deg -> "
          f"residual rms {np.degrees(np.sqrt((resid**2).mean())):.2f} deg after omega fit")

    print("\n== depth channel ==")
    rel = np.array([r["depth_rel"] for r in good]); rngs = np.array([r["range_m"] for r in good])
    print(f"  relative depth err: mean {rel.mean():+.3f}  std {rel.std():.3f}")
    sl, ic = np.polyfit(rngs, rel, 1)
    print(f"  vs range: intercept {ic:+.4f}  slope {sl*100:+.3f} %/m")
    for lo, hi in ((0, 8), (8, 14), (14, 20), (20, 30)):
        b = rel[(rngs >= lo) & (rngs < hi)]
        if len(b):
            print(f"   {lo:2d}-{hi:2d} m: N={len(b):3d}  mean {b.mean():+.3f}  std {b.std():.3f}")

    # how big is the angular error vs what 1.0 deg assumes, split by rate
    print("\n== |delta_perp| vs |omega| (is the attitude error rate-driven?) ==")
    omag = np.linalg.norm(om, axis=1)
    for lo, hi in ((0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 10.0)):
        sel = (omag >= lo) & (omag < hi)
        if sel.sum() >= 5:
            print(f"  |omega| {lo:3.1f}-{hi:3.1f} rad/s: N={sel.sum():3d}  "
                  f"|delta| p50 {np.percentile(mag[sel],50):.2f}  p90 {np.percentile(mag[sel],90):.2f} deg")


if __name__ == "__main__":
    main()
