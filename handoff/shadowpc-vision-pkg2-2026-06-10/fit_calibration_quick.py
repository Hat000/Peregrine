"""Quick calibration fit from EXISTING per-frame dumps (no detector re-run).

Joins handoff/shadowpc-assoc-flipfix-2026-06-09/char_g{N}_robust_k1.0.json (world-fix error
``off_ned`` per solved fix) with handoff/perception-char-2026-06-08/pg/course_g{N}/frames.json
(given pose: ``drone_position_ned`` + ``odo_q_wxyz``) and the track map (gate world positions,
corner_to_center) to reconstruct, per fix:

    p_fix = p_given + off                 (the fix the chain produced)
    L     = gate_pos - p_fix              (world lever arm; == R_wc @ t_cam_solved exactly)

A small constant rotation error e in the vision->world chain predicts (to first order)

    off ~= [L]x (R e)
      world-frame error (R = I):     off = [L]x e_w     (e.g. a world-yaw bias in the given att)
      body-frame error  (R = R_wb):  off = [L]x R_wb e_b (camera-mount / attitude-decode error;
                                     a camera-frame error is the SAME family: e_c = R_cb e_b)

Fits both, each with and without a constant world offset c (absorbs the range-independent
bias: map offset / PnP depth bias), via linear LS + one 3-sigma trim pass, on GOOD 4-corner
fixes only. Also fits PER-GATE rotations to test whether one fixed calibration explains all
legs, and reports per-axis slope-vs-range before/after (connects to the published ~3.6 deg).
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
GOOD_MAX_M = 1.5          # fit on |fix| below this (sensitivity-checked below)


def _skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def load_fixes() -> list[dict]:
    gates = {g.gate_id: g for g in load_track_map(MAP, corner_to_center=True)}
    fixes = []
    for gi in range(6):
        char = json.loads((CHAR_DIR / f"char_g{gi}_robust_k1.0.json").read_text())
        frames_meta = json.loads((PG_DIR / f"course_g{gi}/frames.json").read_text())
        by_fid = {fr["frame_id"]: fr for fr in frames_meta["frames"]}
        for r in char["rows"]:
            if not (r.get("associated") and "world_fix_err_m" in r):
                continue
            fr = by_fid[r["frame_id"]]
            p = np.asarray(fr["drone_position_ned"], float)
            R_wb = F.R_world_from_body(*F.euler_from_quat_wxyz(np.asarray(fr["odo_q_wxyz"], float)))
            off = np.asarray(r["off_ned"], float)
            G = gates[r["gate_id"]].position_ned
            L = G - (p + off)
            # join sanity: |L| must equal the solved PnP range exactly
            assert abs(np.linalg.norm(L) - r["pose_range_m"]) < 1e-6, (gi, r["frame_id"])
            fixes.append(dict(bundle=gi, frame_id=r["frame_id"], gate_id=r["gate_id"],
                              n_corners=r["n_corners"], err=r["world_fix_err_m"],
                              off=off, L=L, R_wb=R_wb, range_m=r["true_range_m"],
                              range_ok=r.get("range_ok", True), maha=r["maha"]))
    return fixes


def fit(rows, mode: str, const: str | None, trim_sigma: float = 3.0):
    """LS fit of off = [L]x R e (+ c). mode: world|body. const: None|'global'|'pergate'.

    Returns dict with e (rad, 3), its 1-sigma, const(s), rms before/after, n_used.
    """
    gids = sorted({r["gate_id"] for r in rows}) if const == "pergate" else []
    npar = 3 + (3 if const == "global" else 0) + 3 * len(gids)

    def design(r):
        A = np.zeros((3, npar))
        Rm = np.eye(3) if mode == "world" else r["R_wb"]
        A[:, 0:3] = _skew(r["L"]) @ Rm
        if const == "global":
            A[:, 3:6] = np.eye(3)
        elif const == "pergate":
            j = 3 + 3 * gids.index(r["gate_id"])
            A[:, j:j + 3] = np.eye(3)
        return A

    use = list(rows)
    for _pass in range(2):                       # plain LS, then one 3-sigma trim
        A = np.vstack([design(r) for r in use])
        b = np.concatenate([r["off"] for r in use])
        theta, *_ = np.linalg.lstsq(A, b, rcond=None)
        res = (b - A @ theta).reshape(-1, 3)
        rn = np.linalg.norm(res, axis=1)
        keep = rn < trim_sigma * max(np.std(rn), 1e-9) + np.mean(rn)
        if keep.all():
            break
        use = [r for r, k in zip(use, keep) if k]
    dof = max(len(use) * 3 - npar, 1)
    sigma2 = float(res.ravel() @ res.ravel()) / dof
    cov = sigma2 * np.linalg.inv(A.T @ A)
    rms0 = float(np.sqrt(np.mean(np.concatenate([r["off"] for r in use]) ** 2)))
    rms1 = float(np.sqrt(np.mean(res.ravel() ** 2)))
    out = dict(mode=mode, const=const or "none", n_used=len(use), n_in=len(rows),
               e_rad=theta[:3], e_sig=np.sqrt(np.diag(cov)[:3]),
               rms_before=rms0, rms_after=rms1, cond=float(np.linalg.cond(A)))
    if const == "global":
        out["c"] = theta[3:6]
    elif const == "pergate":
        out["c_per_gate"] = {g: theta[3 + 3 * i:6 + 3 * i] for i, g in enumerate(gids)}
    return out


def show(f):
    e_deg = np.degrees(f["e_rad"])
    s_deg = np.degrees(f["e_sig"])
    line = (f"{f['mode']:>5}+{f['const']:<7} N={f['n_used']:3d}/{f['n_in']:3d} "
            f"e=[{e_deg[0]:+6.2f}+-{s_deg[0]:4.2f}, {e_deg[1]:+6.2f}+-{s_deg[1]:4.2f}, "
            f"{e_deg[2]:+6.2f}+-{s_deg[2]:4.2f}] deg  |e|={np.linalg.norm(e_deg):5.2f}  "
            f"rms {f['rms_before']:.3f}->{f['rms_after']:.3f} m  cond={f['cond']:.0f}")
    if "c" in f:
        line += f"  c=[{f['c'][0]:+.2f},{f['c'][1]:+.2f},{f['c'][2]:+.2f}] m"
    print(line)


def slope_report(rows, label, e_rad=None, mode="body"):
    """Per-axis off vs range linear fits (the published-diagnostic view), opt. after removing e."""
    rng = np.array([r["range_m"] for r in rows])
    print(f"  {label}:")
    for ax, nm in enumerate("N E D".split()):
        o = []
        for r in rows:
            v = r["off"].copy()
            if e_rad is not None:
                Rm = np.eye(3) if mode == "world" else r["R_wb"]
                v -= _skew(r["L"]) @ Rm @ e_rad
            o.append(v[ax])
        slope, intc = np.polyfit(rng, np.asarray(o), 1)
        print(f"    {nm}: intercept {intc:+.2f} m  slope {slope:+.4f} m/m (= {np.degrees(np.arctan(slope)):+.2f} deg)")


def main():
    fixes = load_fixes()
    print(f"reconstructed {len(fixes)} solved fixes from 6 bundles")
    good = [r for r in fixes if r["err"] < GOOD_MAX_M and r["n_corners"] == 4]
    print(f"good 4-corner fixes (|fix|<{GOOD_MAX_M} m): {len(good)}  "
          f"(ranges {min(r['range_m'] for r in good):.1f}-{max(r['range_m'] for r in good):.1f} m)\n")

    print("== model comparison (fit on good fixes) ==")
    fits = {}
    for mode in ("world", "body"):
        for const in (None, "global"):
            f = fit(good, mode, const)
            fits[(mode, f["const"])] = f
            show(f)
    f_pg = fit(good, "body", "pergate")
    fits[("body", "pergate")] = f_pg
    show(f_pg)

    # camera-frame equivalent of the body fit (e_c = R_cb e_b)
    fb = fits[("body", "global")]
    e_cam = F.R_camera_from_body() @ fb["e_rad"]
    print(f"\nbody e as CAMERA-frame rotvec: [{np.degrees(e_cam[0]):+.2f}, "
          f"{np.degrees(e_cam[1]):+.2f}, {np.degrees(e_cam[2]):+.2f}] deg "
          f"(cam X=right/pitch-ish, Y=down/yaw-ish, Z=fwd/roll)")

    print("\n== per-gate body-frame rotation fits (stability of ONE fixed calibration) ==")
    for gi in sorted({r["gate_id"] for r in good}):
        sub = [r for r in good if r["gate_id"] == gi]
        if len(sub) < 6:
            print(f"  gate {gi}: N={len(sub)} (too few)")
            continue
        f = fit(sub, "body", None)
        e_deg = np.degrees(f["e_rad"])
        print(f"  gate {gi}: N={f['n_used']:3d}  e_b=[{e_deg[0]:+6.2f},{e_deg[1]:+6.2f},{e_deg[2]:+6.2f}] deg  "
              f"rms {f['rms_before']:.3f}->{f['rms_after']:.3f} m")

    print("\n== per-axis slope-vs-range (the published ~3.6 deg view) ==")
    slope_report(good, "raw")
    slope_report(good, "after body-frame e removed", fits[("body", "global")]["e_rad"], "body")
    slope_report(good, "after world-frame e removed", fits[("world", "global")]["e_rad"], "world")

    # sensitivity to the good-fix threshold
    print("\n== threshold sensitivity (body+global) ==")
    for thr in (1.0, 1.5, 2.0, 3.0):
        sub = [r for r in fixes if r["err"] < thr and r["n_corners"] == 4]
        if len(sub) >= 12:
            f = fit(sub, "body", "global")
            e_deg = np.degrees(f["e_rad"])
            print(f"  |fix|<{thr}: N={f['n_used']:3d}  e_b=[{e_deg[0]:+6.2f},{e_deg[1]:+6.2f},{e_deg[2]:+6.2f}] deg")


if __name__ == "__main__":
    main()
