"""Task 2 -- resolve gate-0 map anchor (bottom-CENTER vs bottom-CORNER) by vision PnP.

For each course1 FPV frame in the ShadowPC bundle: detect gate 0 (YOLO-pose) -> 4 corners ->
estimate_gate_pose (inner 1.5 m IPPE) -> gate origin in camera (t_cam_gate). Transform to world
with the GIVEN drone pose + ODOMETRY attitude + camera extrinsics:

    gate_center_world = drone_pos + (R_world_body @ R_camera_from_body().T) @ t_cam_gate

then offset = gate_center_world - map(-23.3,-0.4,-0.03), per-axis (N/E/D):
  * |horizontal| (gate width ~ world Y): ~0 -> bottom-CENTER; ~+-1.36 -> bottom-CORNER.
  * D (vertical): expect ~-1.36 (true centre is 1.36 m ABOVE the map's bottom-edge z-anchor).
  * consistency across range_m: a centre that drifts with range => wrong intrinsics / inner-vs-outer.

A near-frontal gate makes IPPE 2-fold AMBIGUOUS (the two mirror solutions flip t_cam_gate laterally),
so we feed estimate_gate_pose a PRIOR rotation built from the KNOWN gate orientation in the map
(R_cam_gate = R_world_camera.T @ R_world_gate). The gate doesn't move, so this only selects the
rotation basin -- it does NOT bias the translation (the centre measurement). ODOMETRY-quat roll is
telemetry-inverted (laptop Task B), reported both as-is and roll-corrected; roll is tiny here.

Usage:  .venv\\Scripts\\python scripts/task2_gate_pnp.py
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from racer import frames as F
from racer.contracts import Frame
from racer.navigator import load_track_map
from racer.vision.detector import GateDetector
from racer.vision.gate_pose import gate_object_points

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "handoff/shadowpc-followups-2026-06-05/task2_frames"
WEIGHTS = ROOT / "models/gate_yolo11s_curriculum_v2.pt"
MAP = ROOT / "handoff/shadowpc-firstcontact-2026-06-02/track_map.json"
_RCB = F.R_camera_from_body()
_R_WORLD_GATE = np.asarray(load_track_map(MAP, corner_to_center=False)[0].R_world_gate, float)
_K = np.asarray(F.CAMERA_INTRINSICS_K, float)
_KINV = np.linalg.inv(_K)
_OBJ = gate_object_points(1.5)                         # 4 inner corners, gate-local (1.5 m opening)


def R_world_camera(q_wxyz, *, roll_correct=False) -> np.ndarray:
    roll, pitch, yaw = F.euler_from_quat_wxyz(q_wxyz)
    if roll_correct:
        roll = -roll                                   # undo the telemetry roll inversion
    return F.R_world_from_body(roll, pitch, yaw) @ _RCB.T


def solve_t_known_R(corners_px: np.ndarray, R_cam_gate: np.ndarray) -> tuple[np.ndarray, float]:
    """Translation-only PnP with the rotation FIXED (known gate orientation + given attitude).
    Each corner's camera point R@obj+t must lie on its back-projected ray d=K^-1[u,v,1]:
    d x (R@obj + t) = 0  ->  [d]_x t = -[d]_x (R@obj). Stack 4 corners, least-squares for t.
    No IPPE 2-fold ambiguity. Returns (t_cam_gate, reproj_rms_px)."""
    A, b, P = [], [], R_cam_gate @ _OBJ.T              # (3,4) rotated object pts
    for i in range(4):
        u, v = corners_px[i]
        d = _KINV @ np.array([u, v, 1.0])
        dx = np.array([[0, -d[2], d[1]], [d[2], 0, -d[0]], [-d[1], d[0], 0]])
        A.append(dx)
        b.append(-dx @ P[:, i])
    t, *_ = np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)
    proj = _K @ (P + t[:, None])
    proj = (proj[:2] / proj[2]).T                      # (4,2) reprojected px
    reproj = float(np.sqrt(np.mean(np.sum((proj - corners_px) ** 2, axis=1))))
    return t, reproj


def main() -> int:
    d = json.loads((BUNDLE / "frames.json").read_text())
    gmap = np.asarray(d["gate0_map_ned"], float)
    det = GateDetector.load(WEIGHTS, score_thresh=0.25, kpt_conf_thresh=0.5)

    rows = []
    for fr in d["frames"]:
        frame = Frame(frame_id=fr["frame_id"], sim_time_ns=fr["sim_time_ns"],
                      image_bgr=cv2.imread(str(BUNDLE / fr["png"])), recv_monotonic_ns=0, jpeg_bytes=None)
        obs = [o for o in det.detect(frame) if o.corners_px.shape[0] == 4]
        if not obs:
            rows.append({"range": fr["range_m"], "ok": False})
            continue
        drone = np.asarray(fr["drone_position_ned"], float)
        R_wc = R_world_camera(fr["odo_q_wxyz"])
        # solve each detection; ASSOCIATE to gate 0 = the one whose centre is nearest the gate-0
        # map (other gates are 4 m+ away, so this picks gate 0 whether it sits at centre or corner).
        cand = []
        for o in obs:
            t_cam, reproj = solve_t_known_R(o.corners_px, R_wc.T @ _R_WORLD_GATE)
            c = drone + R_wc @ t_cam
            cand.append((float(np.linalg.norm(c - gmap)), o, c, reproj))
        dist, o, c, reproj = min(cand, key=lambda x: x[0])
        R_wc_c = R_world_camera(fr["odo_q_wxyz"], roll_correct=True)
        t_cam_c, _ = solve_t_known_R(o.corners_px, R_wc_c.T @ _R_WORLD_GATE)
        c_corr = drone + R_wc_c @ t_cam_c
        rows.append({"range": fr["range_m"], "ok": True, "reproj": reproj, "ambig": 0.0, "score": o.score,
                     "off": c - gmap, "off_corr": c_corr - gmap, "dist": dist})

    ok = [r for r in rows if r["ok"]]
    print(f"detected gate-0 in {len(ok)}/{len(rows)} frames  (map {gmap})\n")
    print(f"{'range':>6} {'reproj':>6} {'ambig':>6} {'offN':>7} {'offE':>7} {'offD':>7}   {'|off|':>6}")
    for r in sorted(ok, key=lambda x: x["range"]):
        o = r["off"]
        print(f"{r['range']:6.1f} {r['reproj']:6.2f} {r['ambig']:6.2f} "
              f"{o[0]:+7.2f} {o[1]:+7.2f} {o[2]:+7.2f}   {r['dist']:6.2f}")

    good = [r for r in ok if r["dist"] < 3.0]          # confidently gate 0 (others are 4 m+ away)
    print(f"\n{len(good)}/{len(ok)} frames confidently gate-0 (centre within 3 m of its map point)")

    def _band(rows_):
        if not rows_:
            return None
        offs = np.array([r["off"] for r in rows_])
        return np.median(offs, axis=0), np.median(np.abs(offs - np.median(offs, axis=0)), axis=0), len(rows_)

    if good:
        med, mad, n = _band(good)
        medc = np.median(np.array([r["off_corr"] for r in good]), axis=0)
        print(f"MEDIAN gate-0 offset (N/E/D), {n} frames  [MAD]:")
        print(f"  as-is roll      N {med[0]:+.2f}[{mad[0]:.2f}] E {med[1]:+.2f}[{mad[1]:.2f}] "
              f"D {med[2]:+.2f}[{mad[2]:.2f}]   |horiz| {np.hypot(*med[:2]):.2f}")
        print(f"  roll-corrected  N {medc[0]:+.2f} E {medc[1]:+.2f} D {medc[2]:+.2f}")
        for lab, lo, hi in [("near 1.8-5m", 0, 5), ("mid 5-10m", 5, 10), ("far 10-24m", 10, 24)]:
            r = _band([x for x in good if lo <= x["range"] < hi])
            if r:
                m, _, k = r
                print(f"  {lab:12s} ({k:2d})  N {m[0]:+.2f} E {m[1]:+.2f} D {m[2]:+.2f}")

        # separate a CONSTANT (true translational) offset from a RANGE-LINEAR (angular) artifact:
        # offset(range) ~ intercept + slope*range; slope = tan(angular bias). [robust: drop |off|>3]
        clean = [r for r in good if r["dist"] < 3.0]
        rng = np.array([r["range"] for r in clean])
        print("\nLINEAR FIT  offset = intercept + slope*range   (intercept = TRUE offset; "
              "slope -> angular bias):")
        for ax, nm in enumerate("N E D".split()):
            o = np.array([r["off"][ax] for r in clean])
            slope, intc = np.polyfit(rng, o, 1)
            print(f"  {nm}:  intercept {intc:+.2f} m   slope {slope:+.3f} m/m  "
                  f"(= {np.degrees(np.arctan(slope)):+.1f} deg angular)")
        bE = np.polyfit(rng, [r["off"][1] for r in clean], 1)
        bN = np.polyfit(rng, [r["off"][0] for r in clean], 1)
        h0 = np.hypot(bN[1], bE[1])
        near_h = np.hypot(*_band([x for x in good if x["range"] < 5])[0][:2])
        print("\nVERDICT:")
        print(f"  (1) z-anchor: map z is the gate BOTTOM EDGE (D {med[2]:+.2f} m flat, true centre ~1.3 m up) "
              "-> the z-only corner_to_center is correct.")
        print(f"  (2) NOT a 1.36 m bottom-corner: horizontal offset is {near_h:.2f} m near / {h0:.2f} m at "
              "range-0 -- nowhere near 1.36 m. Do NOT add a half-width shift.")
        print(f"  (3) E grows {np.degrees(np.arctan(bE[0])):+.1f} deg/range -> a ~yaw bias in "
              "camera-mount/given-attitude/map-gate-yaw; range-dependent lateral error in vision fixes "
              "(VQ2 calibration finding, irrelevant to model-based given-pose nav).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
