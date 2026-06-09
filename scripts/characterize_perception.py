"""Perception-noise characterization: run the NAVIGATOR's actual vision chain on a real FPV
frame bundle and measure its error vs the GIVEN (ground-truth) pose, as DISTRIBUTIONS over range.

This is the perception-noise model the RL twin needs (the speed ceiling -- master-plan C1) AND the
VQ2-vision-threshold de-risk. It is NOT Task 2: `task2_gate_pnp.py` answered a geometry question
(is the map anchor a corner or centre?) with a rotation-FIXED solver to isolate the gate centre.
Here we run the SAME chain that flies -- `estimate_gate_pose` (full IPPE/P3P + weighted refine) ->
`gate_pose_to_world_position` -- exactly as `navigator._maybe_run_vision`, and ask: how big is the
position fix error, how often does the detector drop the gate, how does it grow with range?

Faithful to the in-loop chain:
  * attitude R_wb = R_world_from_body(euler_from_quat_wxyz(odo_q)) -- the RAW ODOMETRY quat, no roll
    correction (mavlink_client derives ds.roll/pitch/yaw this way; the odo sign-undo lives in the
    CONTROLLER, not the nav R_wb). So this measures what the navigator's PnP actually sees.
  * gates from the saved map with corner_to_center=True (opening centre = the PnP gate origin), so a
    perfect fix returns the given drone position; the residual IS the chain error.
  * predict-in-camera + nearest-centre association + predicted-pose prior == navigator's logic, using
    the GIVEN position as the prior anchor (in VQ1 the KF holds the pristine given pos anyway).

Per frame we record range/off-axis-bearing/speed and, for the associated gate: detected? score,
n_corners, reproj_px; pose depth (range) error; gate-in-camera translation error; and the headline
WORLD-POSITION-FIX error per axis (N/E/D) + magnitude vs the given pos. Aggregates: detect/assoc rate
by range band, world-fix error percentiles by band, and a per-axis linear fit offset = intercept +
slope*range (intercept = the true noise floor; slope = an angular/calibration bias -- the ~3.6 deg/range
yaw bias Task 2 flagged, now measured through the FULL PnP chain). `--json PATH` dumps per-frame rows +
the aggregate (the seed of the RL perception model).

Usage: .venv\\Scripts\\python scripts/characterize_perception.py
       [--bundle DIR] [--weights PT] [--map JSON] [--json OUT.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from racer import frames as F
from racer.contracts import Frame, GatePose
from racer.localization import (
    PNP_FIX_COV_INFLATION,
    P3P_FIX_COV_INFLATION,
    gate_pose_to_world_position,
)
from racer.navigator import load_track_map
from racer.vision.detector import GateDetector
from racer.vision.gate_pose import estimate_gate_pose

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "handoff/shadowpc-followups-2026-06-05/task2_frames"
WEIGHTS = ROOT / "models/gate_yolo11s_curriculum_v2.pt"
MAP = ROOT / "handoff/shadowpc-firstcontact-2026-06-02/track_map.json"
_RCB = F.R_camera_from_body()
_K = np.asarray(F.CAMERA_INTRINSICS_K, float)
ASSOC_MAX_PX = 150.0
CHI2_GATE = 16.27          # navigator.NavigatorConfig.vision_gate_chi2 (chi2_0.999, 3 DOF)


def _predict_gates(drone_pos: np.ndarray, R_wb: np.ndarray, gates) -> dict:
    """navigator._predict_gates_in_camera, verbatim: gate_id -> (R_pred, t_pred, center_px)."""
    R_world_camera = R_wb @ _RCB.T
    R_camera_world = R_world_camera.T
    out = {}
    for g in gates:
        t_pred = R_camera_world @ (g.position_ned - drone_pos)
        if t_pred[2] <= 0:                                  # behind the camera
            continue
        uv = _K @ t_pred
        out[g.gate_id] = (R_camera_world @ g.R_world_gate, t_pred, uv[:2] / uv[2])
    return out


def _associate(center_px: np.ndarray, predicted: dict) -> int | None:
    best_id, best_d = None, ASSOC_MAX_PX
    for gid, (_, _, c) in predicted.items():
        d = float(np.linalg.norm(c - center_px))
        if d < best_d:
            best_id, best_d = gid, d
    return best_id


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", default=str(BUNDLE))
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--map", default=str(MAP))
    ap.add_argument("--json", default=None)
    ap.add_argument("--cov-inflation", type=float, default=PNP_FIX_COV_INFLATION,
                    help="analytic 4-corner PnP world-fix cov multiplier fed to the chi2 gate "
                         "(pass 1.0 for the pre-change baseline; default = production "
                         f"PNP_FIX_COV_INFLATION={PNP_FIX_COV_INFLATION}). Sweep to trade good-fix "
                         "yield against catastrophic leak.")
    args = ap.parse_args()

    bundle = Path(args.bundle)
    d = json.loads((bundle / "frames.json").read_text())
    gates = load_track_map(args.map, corner_to_center=True)
    gates_by_id = {g.gate_id: g for g in gates}
    det = GateDetector.load(Path(args.weights), score_thresh=0.25, kpt_conf_thresh=0.5)
    print(f"bundle={bundle.name}  run={d.get('run')}  n_frames={d['n_frames']}  "
          f"range {d['range_span_m']} m  weights={Path(args.weights).name}  gates={len(gates)}")

    rows = []
    for fr in d["frames"]:
        img = cv2.imread(str(bundle / fr["png"]))
        frame = Frame(frame_id=fr["frame_id"], sim_time_ns=fr["sim_time_ns"], image_bgr=img,
                      recv_monotonic_ns=0, jpeg_bytes=None)
        drone = np.asarray(fr["drone_position_ned"], float)
        R_wb = F.R_world_from_body(*F.euler_from_quat_wxyz(np.asarray(fr["odo_q_wxyz"], float)))
        predicted = _predict_gates(drone, R_wb, gates)
        obs_list = det.detect(frame)
        row = {"frame_id": fr["frame_id"], "range_m": fr["range_m"], "speed_mps": fr["speed_mps"],
               "n_det": len(obs_list), "detected": False, "associated": False}

        # associate each detection to the nearest predicted gate centre (navigator._associate),
        # then keep the one nearest its own predicted centre = the gate this frame is ranged to.
        best = None
        for o in obs_list:
            center = np.mean(np.asarray(o.corners_px, float), axis=0)
            gid = _associate(center, predicted)
            if gid is None:
                continue
            R_pred, t_pred, cpx = predicted[gid]
            d_px = float(np.linalg.norm(cpx - center))
            if best is None or d_px < best[0]:
                best = (d_px, o, gid)
        row["detected"] = len(obs_list) > 0
        if best is not None:
            d_px, o, gid = best
            row["associated"] = True
            gate = gates_by_id[gid]
            R_pred, t_pred, _ = predicted[gid]
            prior = GatePose(o.frame_id, o.sim_time_ns, R_pred, t_pred, 0.0, gate_id=gid)
            pose = estimate_gate_pose(o, prior=prior, compute_covariance=True)
            if pose is not None:
                # range to the associated gate from the given pos = the true depth
                true_rng = float(np.linalg.norm(gate.position_ned - drone))
                # off-axis bearing: angle of the gate dir from the camera optical axis (+Z cam)
                bearing = float(np.degrees(np.arctan2(np.hypot(t_pred[0], t_pred[1]), t_pred[2])))
                # gate-in-camera translation error vs the expected (from map + given pose)
                t_err = float(np.linalg.norm(pose.t_cam_gate - t_pred))
                pos_fix, cov = gate_pose_to_world_position(
                    pose, gate, R_wb, pnp_cov_inflation=args.cov_inflation)
                if pose.n_corners < 4:
                    cov = cov * P3P_FIX_COV_INFLATION
                off = pos_fix - drone                         # world-fix error (N/E/D)
                # is the fix's own covariance consistent with its error? (Mahalanobis self-check)
                try:
                    maha = float(off @ np.linalg.solve(cov, off))
                except np.linalg.LinAlgError:
                    maha = float("nan")
                row.update(gate_id=gid, score=float(o.score), n_corners=int(pose.n_corners),
                           reproj_px=float(pose.reproj_error_px), pose_range_m=float(pose.range_m),
                           true_range_m=true_rng, range_err_m=float(pose.range_m - true_rng),
                           bearing_deg=bearing, t_cam_err_m=t_err,
                           off_ned=[float(x) for x in off], world_fix_err_m=float(np.linalg.norm(off)),
                           maha=maha)
        rows.append(row)

    _report(rows, cov_inflation=args.cov_inflation)
    if args.json:
        Path(args.json).write_text(json.dumps({"bundle": d.get("run"), "weights": Path(args.weights).name,
                                               "rows": rows}, indent=2))
        print(f"\nwrote per-frame rows -> {args.json}")
    return 0


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def _report(rows: list[dict], cov_inflation: float = 1.0) -> None:
    n = len(rows)
    det = [r for r in rows if r["detected"]]
    assoc = [r for r in rows if r.get("associated") and "world_fix_err_m" in r]
    print(f"\nDETECTION: {len(det)}/{n} frames had >=1 detection; "
          f"{len(assoc)}/{n} associated + solved a pose.")

    print(f"\n{'range':>6} {'gid':>3} {'bear':>5} {'spd':>5} {'sc':>5} {'nc':>3} {'reproj':>6} "
          f"{'rngErr':>7} {'offN':>7} {'offE':>7} {'offD':>7} {'|fix|':>6} {'maha':>7}")
    for r in sorted(assoc, key=lambda x: x["true_range_m"]):
        o = r["off_ned"]
        print(f"{r['true_range_m']:6.1f} {r['gate_id']:3d} {r['bearing_deg']:5.1f} {r['speed_mps']:5.1f} "
              f"{r['score']:5.2f} {r['n_corners']:3d} {r['reproj_px']:6.2f} {r['range_err_m']:+7.2f} "
              f"{o[0]:+7.2f} {o[1]:+7.2f} {o[2]:+7.2f} {r['world_fix_err_m']:6.2f} {r['maha']:7.1f}")

    # detection rate + world-fix error by range band
    print("\nBY RANGE BAND:")
    print(f"  {'band':>10} {'frames':>6} {'det%':>5} {'assoc%':>6} "
          f"{'fix_p50':>7} {'fix_p90':>7} {'fix_max':>7} {'rngErr_p50':>10}")
    for lab, lo, hi in [("1.8-5m", 0, 5), ("5-10m", 5, 10), ("10-15m", 10, 15), ("15-24m", 15, 24)]:
        band = [r for r in rows if lo <= r["range_m"] < hi]
        if not band:
            continue
        ba = [r for r in band if r in assoc]
        fixes = np.array([r["world_fix_err_m"] for r in ba])
        rngerr = np.abs([r["range_err_m"] for r in ba])
        print(f"  {lab:>10} {len(band):6d} {100*sum(r['detected'] for r in band)/len(band):4.0f}% "
              f"{100*len(ba)/len(band):5.0f}% {_pct(fixes,50):7.2f} {_pct(fixes,90):7.2f} "
              f"{_pct(fixes,100):7.2f} {_pct(rngerr,50):10.2f}")

    # ROBUST subset: the gate this bundle is ranged to (the modal associated gate_id) with gross
    # outliers dropped (|fix|<3 m) -- separates wrong-gate associations + frontal-PnP 2-fold flips
    # (the heavy tail) from the true per-axis noise floor + calibration bias (Task-2 method).
    gids = [r["gate_id"] for r in assoc]
    primary = max(set(gids), key=gids.count) if gids else None
    clean = [r for r in assoc if r["gate_id"] == primary and r["world_fix_err_m"] < 3.0]
    tail = [r for r in assoc if r not in clean]
    print(f"\nTAIL: {len(tail)}/{len(assoc)} fixes are wrong-gate or |fix|>=3 m "
          f"(frontal-PnP depth flip / background gate). Clean subset = gate {primary}, |fix|<3 m, N={len(clean)}.")
    if len(clean) >= 3:
        rng = np.array([r["true_range_m"] for r in clean])
        print("ROBUST WORLD-FIX OFFSET FIT  off = intercept + slope*range  "
              "(intercept = noise floor; slope -> angular/calibration bias):")
        for ax, nm in enumerate("N E D".split()):
            o = np.array([r["off_ned"][ax] for r in clean])
            slope, intc = np.polyfit(rng, o, 1)
            print(f"  {nm}:  intercept {intc:+.2f} m   slope {slope:+.3f} m/m  "
                  f"(= {np.degrees(np.arctan(slope)):+.1f} deg)")
        cfix = np.array([r["world_fix_err_m"] for r in clean])
        print(f"  clean |fix|  p50 {_pct(cfix,50):.2f}  p90 {_pct(cfix,90):.2f}  max {_pct(cfix,100):.2f} m")
    allfix = np.array([r["world_fix_err_m"] for r in assoc])
    reproj = np.array([r["reproj_px"] for r in assoc])
    maha = np.array([r["maha"] for r in assoc if np.isfinite(r["maha"])])
    print(f"\nFULL SET (N={len(assoc)}, tail included):")
    print(f"  WORLD-FIX |error|  p50 {_pct(allfix,50):.2f}  p90 {_pct(allfix,90):.2f}  max {_pct(allfix,100):.2f} m")
    print(f"  REPROJ px          p50 {_pct(reproj,50):.2f}  p90 {_pct(reproj,90):.2f}  max {_pct(reproj,100):.2f}")
    print(f"  fix-cov Mahalanobis (off^T cov^-1 off, dof=3, chi2.999=16.27): "
          f"p50 {_pct(maha,50):.0f}  -> the analytic PnP cov UNDERSTATES the true fix error if >>16.")

    # -- GATE TRADE-OFF: how the navigator's chi2 innovation gate classifies these fixes at this
    # cov_inflation. GOOD = accurate fix we WANT to keep; CATASTROPHIC = wrong-gate / depth-flip we
    # MUST reject. ``maha`` was computed with the inflated cov, so this is what the live gate sees (in
    # VQ1 the KF is anchored to the given pos, so nu~=off and P<<cov -> d2~=maha). Inflating cov by K
    # scales maha by ~1/K (PnP-dominated near/mid range; less at long range where the attitude lever-
    # arm term dominates). Thresholds match the TAIL line (cat>=3 m); reconcile the absolute %s against
    # handoff/perception-char-2026-06-08 -- the before/after DELTA at a fixed threshold is the metric.
    GOOD_MAX_M, CAT_MIN_M = 1.0, 3.0
    finite = [r for r in assoc if np.isfinite(r["maha"])]
    good = [r for r in finite if r["world_fix_err_m"] < GOOD_MAX_M]
    cat = [r for r in finite if r["world_fix_err_m"] >= CAT_MIN_M]
    n_good, n_cat = len(good), len(cat)
    good_rej = sum(r["maha"] > CHI2_GATE for r in good)
    cat_leak = sum(r["maha"] <= CHI2_GATE for r in cat)
    gr = 100.0 * good_rej / n_good if n_good else float("nan")
    cl = 100.0 * cat_leak / n_cat if n_cat else float("nan")
    catch = 100.0 * (n_cat - cat_leak) / n_cat if n_cat else float("nan")
    print(f"\nGATE TRADE-OFF  (chi2_0.999={CHI2_GATE}, cov_inflation={cov_inflation:.2f}, "
          f"good<{GOOD_MAX_M:.0f} m, catastrophic>={CAT_MIN_M:.0f} m):")
    print(f"  GOOD fixes (keep)    N={n_good:3d}   rejected {good_rej:3d}  ({gr:4.0f}%)"
          f"   <- minimise (good-fix yield loss; target <5%)")
    print(f"  CATASTROPHIC (drop)  N={n_cat:3d}   leaked   {cat_leak:3d}  ({cl:4.0f}%)"
          f"   <- keep <=~2%   (bad-fix catch {catch:.0f}%)")


if __name__ == "__main__":
    raise SystemExit(main())
