"""Boresight-calibration estimator demo: prove the static head-on/level procedure recovers an
injected camera-pitch boresight and drives the post-correction vertical residual to ~0.

Simulates the P3 lock test against the REAL chain: inject a known PHYSICAL boresight delta_true
into the render mount; "measure" the mean gate-vertical fix-vs-GT residual at g2/g4 head-on level
with the decode boresight = 0; solve delta_est = -atan(residual/range); re-measure with the decode
boresight = delta_est and confirm residual -> 0.

This is the calibration math that will live behind frames.BORESIGHT_PITCH_RAD (default 0.0).
Run:  py -3.13 handoff/p1-vision-accuracy-2026-06-14/scratch-audit/calib_estimator_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from racer import frames as F                                            # noqa: E402
from racer.contracts import Gate, GateObservation, GatePose             # noqa: E402
from racer.localization import gate_pose_to_world_position              # noqa: E402
from racer.vision.gate_pose import (                                    # noqa: E402
    GATE_INNER_SIZE_M, estimate_gate_pose, project_gate_corners,
)

DEG = np.pi / 180.0


def mount(pitch_deg: float) -> np.ndarray:
    return F._R_CAMERA_FROM_TILTED_BODY @ Rotation.from_euler("Y", -np.deg2rad(pitch_deg)).as_matrix()


def make_head_on_gate(range_m: float, drone_pos=np.zeros(3), gate_id: int = 0) -> Gate:
    pos = np.asarray(drone_pos, float) + np.array([range_m, 0.0, 0.0])   # due north, same height
    R_world_gate = np.column_stack([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).astype(float)  # X=E,Y=down,Z=N
    return Gate(gate_id=gate_id, position_ned=pos, R_world_gate=R_world_gate, inner_size_m=GATE_INNER_SIZE_M)


def measure_vert_residual(gate: Gate, drone_pos, R_wb, *, phys_pitch_deg: float,
                          decode_pitch_deg: float, n: int = 60, sigma_px: float = 0.7,
                          seed: int = 0) -> float:
    """Mean gate-vertical (down) fix-vs-GT residual over n noisy frames. Render at the PHYSICAL
    mount, decode lever at decode_pitch (the calibrated model). Averages out pixel noise."""
    rng = np.random.default_rng(seed)
    R_cw_phys = (R_wb @ mount(phys_pitch_deg).T).T
    t_cam = R_cw_phys @ (gate.position_ned - drone_pos)
    R_cg = R_cw_phys @ gate.R_world_gate
    clean = project_gate_corners(R_cg, t_cam, gate.inner_size_m)
    R_cw_dec = (R_wb @ mount(decode_pitch_deg).T).T
    prior = GatePose(0, 0, R_cw_dec @ gate.R_world_gate, R_cw_dec @ (gate.position_ned - drone_pos), 0.0)
    vds = []
    for _ in range(n):
        corners = clean + rng.normal(0.0, sigma_px, size=clean.shape)
        obs = GateObservation(0, 0, corners_px=corners, corner_confidence=np.ones(4))
        gp = estimate_gate_pose(obs, prior=prior, weighted_refine=False)
        if gp is None or not np.all(np.isfinite(gp.t_cam_gate)):
            continue
        # decode lever at decode_pitch (NOT the shipped 20deg): mimic frames after BORESIGHT is set
        R_wc_dec = R_wb @ mount(decode_pitch_deg).T
        lever = R_wc_dec @ gp.t_cam_gate
        pos_rec = gate.position_ned - lever
        vds.append(float((pos_rec - drone_pos) @ gate.R_world_gate[:, 1]))
    return float(np.mean(vds))


def solve_boresight_pitch_rad(residual_m: float, range_m: float) -> float:
    """THE calibration estimator. residual_m = mean gate-vertical (down +) fix-vs-GT residual
    measured with BORESIGHT_PITCH_RAD = 0; range_m = head-on range. Returns the boresight (rad)
    to write into frames.BORESIGHT_PITCH_RAD so the decode mount matches the physical camera."""
    return float(-np.arctan(residual_m / range_m))


# ---------------------------------------------------------------------------
print("=" * 78)
print("BORESIGHT CALIBRATION ESTIMATOR DEMO  (recover an injected physical boresight)")
print("=" * 78)
LEVEL = np.eye(3)
DELTA_TRUE = 0.56          # deg, the unknown physical render-vs-model boresight to recover
R2, R4 = 18.0, 22.0       # g2 / g4 head-on lock-test ranges (m)

print(f"injected PHYSICAL boresight delta_true = {DELTA_TRUE:+.3f} deg "
      f"(camera pitched up {DELTA_TRUE} deg more than the 20deg model)\n")

# --- step 1: measure residual at decode boresight = 0 (the uncalibrated stack) ---
ests = []
for gid, rng_m in (("g2", R2), ("g4", R4)):
    g = make_head_on_gate(rng_m, gate_id=int(gid[1]))
    m_v = measure_vert_residual(g, np.zeros(3), LEVEL, phys_pitch_deg=20.0 + DELTA_TRUE,
                                decode_pitch_deg=20.0, seed=hash(gid) % 1000)
    d_est = solve_boresight_pitch_rad(m_v, rng_m)
    ests.append(d_est)
    print(f"  {gid} @ {rng_m:.0f} m : mean vert residual (decode=20) = {m_v:+.4f} m  "
          f"-> delta_est = {np.rad2deg(d_est):+.4f} deg")

boresight_rad = float(np.mean(ests))     # joint (mean over the 2 head-on gates)
print(f"\n  CALIBRATED  BORESIGHT_PITCH_RAD = {boresight_rad:+.6f} rad = {np.rad2deg(boresight_rad):+.4f} deg")
print(f"  recovery error vs delta_true    = {np.rad2deg(boresight_rad) - DELTA_TRUE:+.4f} deg")

# --- step 2: re-measure residual WITH the calibrated decode boresight applied ---
print("\n  post-calibration check (decode boresight = calibrated):")
worst = 0.0
for gid, rng_m in (("g2", R2), ("g4", R4), ("g4_far", 30.0)):
    g = make_head_on_gate(rng_m, gate_id=4)
    m_v = measure_vert_residual(g, np.zeros(3), LEVEL, phys_pitch_deg=20.0 + DELTA_TRUE,
                                decode_pitch_deg=20.0 + np.rad2deg(boresight_rad), seed=42)
    worst = max(worst, abs(m_v))
    print(f"    {gid:7s} @ {rng_m:.0f} m : residual = {m_v:+.4f} m")
print(f"\n  ACCEPTANCE (<= 0.05 m mean |vert residual|): worst = {worst:.4f} m  -> "
      f"{'PASS' if worst <= 0.05 else 'FAIL'}")
print(f"  boresight recovery (<= 0.05 deg): {'PASS' if abs(np.rad2deg(boresight_rad)-DELTA_TRUE) <= 0.05 else 'FAIL'}")
