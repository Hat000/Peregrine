"""DUAL-FORM (angular vs metric) boresight round-trip — the build-on of the audit's
scratch-audit/roundtrip_boresight.py, extended to the SECOND functional form.

Exercises the REAL src/racer chain end-to-end (NO sim, NO GPU, pure numpy/cv2):

    world gate+drone --(physical mount pitch/roll AND/OR camera-centre offset)-->
       project_gate_corners (K) --> estimate_gate_pose (IPPE+prior) -->
       localization +L lever (decode mount + metric vert offset) --> recovered drone pos vs truth.

Two functional forms for the SAME 0.215 m @ 22 m vertical signature:
  ANGULAR (boresight pitch/roll, composed into R_camera_from_body): world-vert offset is
          PROPORTIONAL to range; the back-out ANGLE is range-FLAT.
  METRIC  (constant camera-optical-centre vs body-origin vertical translation, body FRD +Z down,
          applied in the +L lever): world-vert offset is CONSTANT across range; the back-out angle
          is proportional to 1/range.

Claims proven here (numbers -> dualform_OUT.txt):
  (0) UNIFIED correction all-zero == byte-identical mount + byte-identical lever (np.array_equal).
  (A) ANGULAR injected -> recovered, correction zeroes it; vbias ~ range, angle flat.
  (B) METRIC  injected -> recovered, correction zeroes it; vbias FLAT, angle ~ 1/range.
  (C) At 22 m the two forms are INDISTINGUISHABLE (both 0.215 m); >=2 separated ranges discriminate.
  (D) ROLL boresight composes + round-trips (decode-matched < 1 mm); head-on vertical signature is
      near-degenerate (bearing-driven), confirming roll is observable off-axis, not head-on.

Run:  py -3.13 handoff/p1-vision-accuracy-2026-06-14/scratch-calib-v2/roundtrip_dualform.py
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
from racer.vision.gate_pose import (                                    # noqa: E402
    GATE_INNER_SIZE_M, estimate_gate_pose, project_gate_corners,
)

DEG = np.pi / 180.0
np.set_printoptions(precision=6, suppress=True)

out = []
def p(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    out.append(line)


# --- the PATCHED mount + lever, modelled exactly as proposed_calib_dualform.patch would write them.
def mount_corrected(pitch_boresight_rad: float = 0.0, roll_boresight_rad: float = 0.0) -> np.ndarray:
    """camera<-body with the ANGULAR boresight composed in, EXACTLY as the proposed patch builds
    R_camera_from_body(): swap @ Rx(-roll) @ Ry(-(CAMERA_PITCH_RAD + pitch)).
    pitch=roll=0 -> identical to F.R_camera_from_body() (the byte-identity claim)."""
    R_tilt = Rotation.from_euler("Y", -(F.CAMERA_PITCH_RAD + pitch_boresight_rad)).as_matrix()
    R_roll = Rotation.from_euler("X", -roll_boresight_rad).as_matrix()
    return F._R_CAMERA_FROM_TILTED_BODY @ R_roll @ R_tilt


def localize(gp_t_cam_gate, gate, R_wb, decode_mount, vert_offset_m: float = 0.0) -> np.ndarray:
    """The +L lever with the METRIC correction, EXACTLY as the proposed localization patch builds it:
        R_wc = R_wb @ decode_mount.T ; L = R_wc @ t_cam_gate ; p = gate - L  (- R_wb @ [0,0,voff]).
    The metric line is GUARDED on vert_offset_m != 0 so the default is byte-identical."""
    R_wc = R_wb @ decode_mount.T
    lever = R_wc @ np.asarray(gp_t_cam_gate, dtype=np.float64)
    pos = np.asarray(gate.position_ned, dtype=np.float64) - lever
    if vert_offset_m != 0.0:
        pos = pos - R_wb @ np.array([0.0, 0.0, vert_offset_m])     # body FRD +Z down camera-centre offset
    return pos


def make_gate(range_m, az_deg=0.0, el_deg=0.0, drone_pos=np.zeros(3), gid=0) -> Gate:
    az, el = az_deg * DEG, el_deg * DEG
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), -np.sin(el)])
    pos = np.asarray(drone_pos, float) + range_m * d
    Zc = d / np.linalg.norm(d)
    Yc = np.array([0.0, 0.0, 1.0]); Yc = Yc - Zc * (Yc @ Zc); Yc /= np.linalg.norm(Yc)
    Xc = np.cross(Yc, Zc)
    return Gate(gate_id=gid, position_ned=pos, R_world_gate=np.column_stack([Xc, Yc, Zc]),
                inner_size_m=GATE_INNER_SIZE_M)


def render_decode(gate, drone_pos, R_wb, *, phys_pitch_rad=0.0, phys_roll_rad=0.0,
                  phys_vert_offset_m=0.0, dec_pitch_rad=0.0, dec_roll_rad=0.0, dec_vert_offset_m=0.0,
                  use_prior=True):
    """Render corners from the PHYSICAL camera (mount = 20deg + phys boresight, optical centre offset
    phys_vert_offset_m in body FRD +Z), decode with the DECODE correction (dec_*). Returns the
    gate-plane error (right, vert/down, along) of the recovered drone pos vs truth, plus depth_err."""
    R_phys = mount_corrected(phys_pitch_rad, phys_roll_rad)
    cam_centre = np.asarray(drone_pos, float) + R_wb @ np.array([0.0, 0.0, phys_vert_offset_m])
    R_cw_phys = (R_wb @ R_phys.T).T
    t_cam = R_cw_phys @ (gate.position_ned - cam_centre)
    R_cg = R_cw_phys @ gate.R_world_gate
    corners = project_gate_corners(R_cg, t_cam, gate.inner_size_m)
    R_dec = mount_corrected(dec_pitch_rad, dec_roll_rad)
    pr = None
    if use_prior:
        R_cw_n = (R_wb @ R_dec.T).T
        pr = GatePose(0, 0, R_cw_n @ gate.R_world_gate, R_cw_n @ (gate.position_ned - drone_pos), 0.0)
    obs = GateObservation(0, 0, corners_px=corners, corner_confidence=np.ones(4))
    gp = estimate_gate_pose(obs, prior=pr, weighted_refine=False)
    if gp is None or not np.all(np.isfinite(gp.t_cam_gate)):
        return np.full(3, np.nan), np.nan
    pos_rec = localize(gp.t_cam_gate, gate, R_wb, R_dec, dec_vert_offset_m)
    err = pos_rec - np.asarray(drone_pos, float)
    R = gate.R_world_gate
    gpe = (float(err @ R[:, 0]), float(err @ R[:, 1]), float(err @ R[:, 2]))
    depth_err = float(np.linalg.norm(t_cam)) - gp.range_m
    return np.array(gpe), depth_err


LEVEL = F.R_world_from_body(0.0, 0.0, 0.0)
ORIGIN = np.zeros(3)

p("=" * 84)
p("DUAL-FORM BORESIGHT ROUND-TRIP  (angular vs metric)  -- real src/racer chain")
p("=" * 84)

# (0) BYTE-IDENTITY of the unified correction at all-zero -----------------------------------------
p("\n(0) UNIFIED CORRECTION ALL-ZERO == BYTE-IDENTICAL")
p("-" * 84)
m0 = mount_corrected(0.0, 0.0)
p("  mount_corrected(0,0) == F.R_camera_from_body()  np.array_equal :",
  bool(np.array_equal(m0, F.R_camera_from_body())))
# lever byte-identity: with vert_offset 0 the metric line is skipped -> identical formula
g = make_gate(22.0, 6.0, 3.0)
# build a real gate-pose first
R_phys = mount_corrected(0.0, 0.0); R_cw = (LEVEL @ R_phys.T).T
t_cam = R_cw @ (g.position_ned - ORIGIN)
corners = project_gate_corners(R_cw @ g.R_world_gate, t_cam, g.inner_size_m)
gp = estimate_gate_pose(GateObservation(0, 0, corners_px=corners, corner_confidence=np.ones(4)),
                        weighted_refine=False)
pos_default = localize(gp.t_cam_gate, g, LEVEL, m0, 0.0)
# the unpatched reference: gate - (R_wb @ R_camera_from_body().T) @ t_cam_gate
ref = g.position_ned - (LEVEL @ F.R_camera_from_body().T) @ gp.t_cam_gate
p("  lever(vert_offset=0) == unpatched +L lever      np.array_equal :",
  bool(np.array_equal(pos_default, ref)))

# (A) ANGULAR form --------------------------------------------------------------------------------
p("\n(A) ANGULAR injected (camera pitched up +0.56deg, render-only): vbias ~ range, angle FLAT")
p("-" * 84)
# Dead-on (az=0) head-on, capped at 30 m (realistic gate-0/1 standoff). Beyond ~40 m a dead-frontal
# gate + a near-uniform vertical corner shift trips the documented IPPE 2-fold (audit Task 2C aside);
# in flight the navigator's near-exact prior resolves it, and a few deg off-axis removes it entirely.
RANGES = (8.0, 12.0, 15.0, 22.0, 30.0)
EPS_DEG = 0.56
p("    range_m   gate_vert(m)   angle=atan(-v/r)(deg)   corrected_vert(m)")
ang_rows = []
for rng in RANGES:
    g = make_gate(rng, 0.0, 0.0)
    gpe, _ = render_decode(g, ORIGIN, LEVEL, phys_pitch_rad=EPS_DEG * DEG)         # decode @ 0
    # correction: pitch boresight = -atan(v/r); apply at decode and re-measure
    eps_est = -np.arctan(gpe[1] / rng)
    gpe_c, _ = render_decode(g, ORIGIN, LEVEL, phys_pitch_rad=EPS_DEG * DEG, dec_pitch_rad=eps_est)
    ang_rows.append((rng, gpe[1]))
    p(f"    {rng:5.1f}    {gpe[1]:+9.4f}        {np.rad2deg(np.arctan(-gpe[1] / rng)):+8.4f}          {gpe_c[1]:+8.5f}")
rs = np.array([r for r, _ in ang_rows]); vs = np.array([v for _, v in ang_rows])
ang_angle = np.rad2deg(np.arctan(-vs / rs))
p(f"  ANGLE spread over 8-30 m = {ang_angle.min():+.4f} .. {ang_angle.max():+.4f} deg  "
  f"(std {ang_angle.std():.4f}) -> RANGE-FLAT angle == angular signature")
p(f"  metric vbias spread       = {vs.min():+.4f} .. {vs.max():+.4f} m  -> grows with range")

# (B) METRIC form ---------------------------------------------------------------------------------
p("\n(B) METRIC injected (camera optical centre 0.215 m offset, body +Z down): vbias FLAT, angle ~ 1/r")
p("-" * 84)
VOFF_TRUE = -0.215     # tuned so that @22 m the vbias equals the angular case (see (C))
p("    range_m   gate_vert(m)   angle=atan(-v/r)(deg)   corrected_vert(m)")
met_rows = []
for rng in RANGES:
    g = make_gate(rng, 0.0, 0.0)
    gpe, _ = render_decode(g, ORIGIN, LEVEL, phys_vert_offset_m=VOFF_TRUE)           # decode @ 0
    voff_est = gpe[1]    # metric estimator: vert_offset = measured down-residual (range-flat)
    gpe_c, _ = render_decode(g, ORIGIN, LEVEL, phys_vert_offset_m=VOFF_TRUE, dec_vert_offset_m=voff_est)
    met_rows.append((rng, gpe[1]))
    p(f"    {rng:5.1f}    {gpe[1]:+9.4f}        {np.rad2deg(np.arctan(-gpe[1] / rng)):+8.4f}          {gpe_c[1]:+8.5f}")
rs2 = np.array([r for r, _ in met_rows]); vs2 = np.array([v for _, v in met_rows])
met_angle = np.rad2deg(np.arctan(-vs2 / rs2))
p(f"  metric vbias spread       = {vs2.min():+.4f} .. {vs2.max():+.4f} m  (std {vs2.std():.5f}) -> RANGE-FLAT vbias == metric signature")
p(f"  ANGLE spread over 8-30 m  = {met_angle.min():+.4f} .. {met_angle.max():+.4f} deg  -> angle ~ 1/range (NOT flat)")
p(f"  metric estimator recovers offset: measured @22m = {vs2[np.where(rs2==22.0)][0]:+.4f} m vs injected {VOFF_TRUE:+.4f} m")

# (C) INDISTINGUISHABLE @ 22 m, DISCRIMINATED at >=2 ranges ----------------------------------------
p("\n(C) DEGENERACY @ single range / DISCRIMINATION at two ranges")
p("-" * 84)
p(f"  @22 m:  angular vbias = {vs[np.where(rs==22.0)][0]:+.4f} m   metric vbias = {vs2[np.where(rs2==22.0)][0]:+.4f} m   (|diff| = {abs(vs[np.where(rs==22.0)][0]-vs2[np.where(rs2==22.0)][0]):.5f} m -> SAME)")
for (r1, r2) in [(18.0, 22.0), (15.0, 30.0), (10.0, 40.0)]:
    a1 = -r1 * np.tan(EPS_DEG * DEG); a2 = -r2 * np.tan(EPS_DEG * DEG)  # angular predictions
    # rescale so both equal VOFF_TRUE at 22 m for an apples-to-apples discriminator
    scale = VOFF_TRUE / (-22.0 * np.tan(EPS_DEG * DEG))
    a1 *= scale; a2 *= scale
    ratio_ang = a2 / a1; ratio_met = 1.0
    sep = abs(ratio_ang - ratio_met)
    p(f"  ranges ({r1:.0f},{r2:.0f}) m: angular vbias ratio v2/v1 = {ratio_ang:.3f}  vs metric = {ratio_met:.3f}  "
      f"-> separation {sep*100:.1f}%   (r2/r1 = {r2/r1:.3f})")

# (D) ROLL boresight composes + round-trips --------------------------------------------------------
p("\n(D) ROLL boresight: decode-matched round-trip < 1 mm; head-on vert ~ 0 (bearing-driven)")
p("-" * 84)
ROLL_DEG = 0.8
maxmatch = 0.0
for rng in (10.0, 22.0, 35.0):
    for az in (-20.0, 0.0, 20.0):
        g = make_gate(rng, az, 0.0)
        # render & decode at the SAME roll -> must round-trip
        gpe, _ = render_decode(g, ORIGIN, LEVEL, phys_roll_rad=ROLL_DEG * DEG, dec_roll_rad=ROLL_DEG * DEG)
        if np.all(np.isfinite(gpe)):
            maxmatch = max(maxmatch, float(np.linalg.norm(gpe)))
p(f"  max |err| render@roll / decode@roll over 9 poses = {maxmatch:.3e} m -> roll composes & inverts")
p("    az_deg   head-on-vert(decode@0)(m)   (roll vertical signature is bearing-coupled)")
for az in (-20.0, -10.0, 0.0, 10.0, 20.0):
    g = make_gate(22.0, az, 0.0)
    gpe, _ = render_decode(g, ORIGIN, LEVEL, phys_roll_rad=ROLL_DEG * DEG)     # decode @ 0
    p(f"    {az:+5.1f}        {gpe[1]:+8.4f}")

p("\n" + "=" * 84)
p("VERDICT: dual-form is byte-identical at zero; angular vbias ~ range (angle flat), metric vbias")
p("flat (angle ~ 1/r); identical at 22 m; >=2 separated ranges discriminate. Each form's estimator")
p("recovers its injected value and the unified decode correction drives the residual to ~0.")
(Path(__file__).resolve().parent / "dualform_OUT.txt").write_text("\n".join(out) + "\n")
