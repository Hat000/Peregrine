"""A36 offline proof — reproduce run 20260704_042616's ACTUAL pass geometry and show that the
OLD blind turn target latches ~-1.91 rad (LEFT, the bug) while the NEW refine-to-real-gate logic
re-aims to ~+0.66 rad (RIGHT, the true gate 2) and settles there. Run with the repo venv:

    .venv/Scripts/python.exe handoff/a36_offline_turn_proof.py

Geometry (from the run's nav_estimate.jsonl):
  * pass commits at gate 1, range 2.8 m, pass_heading ~ -0.31 rad; gate-1 apparent az at pass -0.31
    (drifting LEFT as passed) -> the OLD blind target = pass_heading + sign(-0.31)*1.6 = -1.91.
  * the TRUE gate 2 is at world bearing +0.66 rad (RIGHT), range ~18 m, first seen right after the pass.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racer.contracts import Frame, Gate, GateObservation, NavState  # noqa: E402
from racer.deploy_profile import vq2_case_c  # noqa: E402
from racer.frames import (  # noqa: E402
    CAMERA_INTRINSICS_K, R_camera_from_body, R_world_from_body,
)
from racer.gate_seeker import GateSeeker, GateSeekerConfig, make_seeker_controller  # noqa: E402

_NS = 1_000_000_000
PASS_HEADING = -0.31          # rad (drone yaw at the pass, from the run)
# gate 1 placed LEFT of the drone's heading so the pass-time apparent az reads ~-0.31 (the run's
# value) -> the OLD blind target = pass_heading + sign(-0.31)*1.6 = -1.91 rad (LEFT), the bug.
GATE1_POS = np.array([2.3, -1.63, -2.5])
GATE2_BEARING = 0.66          # rad, world bearing to gate 2 (RIGHT), from the run's post-pass yaw_des
GATE2_RANGE = 18.0


def _u(v):
    return np.asarray(v, float) / np.linalg.norm(v)


def _gate(pos, normal, gid):
    n = _u(normal)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = _u(np.cross(a, n))
    y = np.cross(n, x)
    return Gate(gate_id=gid, position_ned=np.asarray(pos, float),
                R_world_gate=np.column_stack([x, y, n]))


class _Det:
    """Projects whatever gates are in view from the true pose (the seeker solves its own PnP)."""

    def __init__(self, gates):
        self.gates = gates
        self.pos = np.zeros(3)
        self.R_wb = np.eye(3)
        self._R_cb = R_camera_from_body()

    def _proj(self, g):
        half = g.inner_size_m / 2.0
        cg = np.array([[-half, half, 0.0], [half, half, 0.0],
                       [half, -half, 0.0], [-half, -half, 0.0]])
        R_wg = np.asarray(g.R_world_gate, float)
        K = CAMERA_INTRINSICS_K
        px = []
        for c in cg:
            pc = self._R_cb @ (self.R_wb.T @ (g.position_ned + R_wg @ c - self.pos))
            if pc[2] <= 0.05:
                return None
            px.append([K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]])
        return np.asarray(px, float)

    def detect(self, fr):
        out = []
        for g in self.gates:
            px = self._proj(g)
            if px is None:
                continue
            out.append(GateObservation(frame_id=fr.frame_id, sim_time_ns=fr.sim_time_ns,
                                       corners_px=px, corner_ids=np.array([0, 1, 2, 3]),
                                       corner_confidence=np.ones(4)))
        return out


def _seeker(refine: bool):
    prof = vq2_case_c()
    so = dict(prof.seeker_overrides)
    so["pass_turn_refine"] = refine          # the A/B lever
    # zero launch timing so we are past egress in pursuit; keep everything else = the shipped profile.
    so.update(dict(settle_s=0.0, launch_ramp_s=0.0, use_spawn_egress=False,
                   anchor_release_detections=1, pursuit_ramp_s=0.0, forward_ramp_s=0.0,
                   vertical_align_ramp_s=0.0))
    det = _Det([])
    s = GateSeeker(config=GateSeekerConfig(**so),
                   controller=make_seeker_controller(**(prof.controller_overrides or {})),
                   detector=det)
    return s, det


def _run(refine: bool, n=120, dt=0.033):
    """Approach gate 1 (drone flies +X), pass it, then gate 2 appears at +0.66 rad / 18 m.
    Integrate the INVERTING plant (realized yaw = -2.1 x wire). Return the settled heading + trace."""
    s, det = _seeker(refine)
    g1 = _gate(GATE1_POS, normal=[1, 0, 0], gid=0)
    g2pos = np.array([GATE2_RANGE * np.cos(GATE2_BEARING),
                      GATE2_RANGE * np.sin(GATE2_BEARING), -2.5])
    g2 = _gate(g2pos, normal=[np.cos(GATE2_BEARING + np.pi), np.sin(GATE2_BEARING + np.pi), 0.0], gid=1)
    psi = PASS_HEADING
    pos = np.array([0.0, 0.0, -2.5])
    s._last_yaw = psi
    gate_idx = 0
    traj, latched = [], []
    for k in range(n):
        t = int(k * dt * _NS)
        # gate 1 visible until we pass it (~x>2.3); gate 2 visible once we are past + it's ahead-ish.
        vis = []
        if pos[0] < GATE1_POS[0] + 0.4:
            vis.append(g1)
        if pos[0] >= 1.2:                      # gate 2 comes into view shortly before/after the pass
            vis.append(g2)
        det.gates = vis
        det.R_wb = R_world_from_body(0.0, 0.0, psi)
        det.pos = pos.copy()
        # advance the wire index once we're through gate 1 (the RACE_STATUS pass signal)
        if pos[0] >= GATE1_POS[0] and gate_idx == 0:
            gate_idx = 1
        nav = NavState(sim_time_ns=t, position_ned=pos.copy(), velocity_ned=np.zeros(3),
                       yaw=-psi, time_since_vision_update_s=float("inf"))
        cmd = s.command_visual(nav, Frame(frame_id=k, sim_time_ns=t,
                                          image_bgr=np.ones((360, 640, 3), np.uint8)), gate_idx)
        wire = float(cmd.body_rate[2]) * 0.4
        psi += -2.1 * wire * dt                 # inverting plant
        pos = pos + np.array([np.cos(psi), np.sin(psi), 0.0]) * 0.65 * dt * 3.0
        traj.append(np.degrees(psi))
        latched.append(None if s._pass_turn_yaw is None else np.degrees(s._pass_turn_yaw))
    return traj, latched


def main():
    true_deg = np.degrees(GATE2_BEARING)
    print(f"TRUE gate-2 bearing = {true_deg:+.1f} deg (RIGHT)\n")
    for refine, lbl in [(False, "OLD blind target"), (True, "NEW refine-to-real-gate")]:
        traj, latched = _run(refine)
        settled = traj[-1]
        latched_vals = [x for x in latched if x is not None]
        latched_last = latched_vals[-1] if latched_vals else None
        # overshoot past the true bearing (max excursion beyond it, in the turn direction)
        err = abs(((settled - true_deg + 180) % 360) - 180)
        print(f"{lbl:24s}: latched_turn_target = "
              f"{('%.1f' % latched_last) if latched_last is not None else 'None':>8} deg | "
              f"settled heading = {settled:+7.1f} deg | |err vs true| = {err:5.1f} deg")
    print("\nPASS if NEW settles within ~15 deg of +37.8 (turns RIGHT to the real gate) while "
          "OLD latches ~-110 deg (LEFT, the bug).")


if __name__ == "__main__":
    main()
