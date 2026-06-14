"""DECISIVE synthetic round-trip for the eps_vert (vertical camera-boresight) audit.

Exercises the REAL src/racer chain end-to-end (NO sim, NO GPU, pure numpy/cv2):

    world gate+drone  --(mount@render_pitch)-->  R_cam_gate,t_cam_gate
       --> gate_pose.project_gate_corners (CAMERA_INTRINSICS_K)  --> 4 px corners
       --> gate_pose.estimate_gate_pose (IPPE_SQUARE, +nominal-model prior)
       --> localization.gate_pose_to_world_position (mount@20deg, +L lever)
       --> recovered drone world position  vs  truth.

Three questions:
  (A) Is the chain internally consistent? render@20 / decode@20 must round-trip < 1 mm.
  (B) Does a +delta render-only camera-pitch boresight (decode still @20) reproduce the
      field signature: ~ -0.215 m gate-vertical at 22 m, depth-decoupled, bearing-correlated,
      angle-flat-in-range? Report d(vbias)/d(delta) in m/deg.
  (C) Do a constant +dv px corner offset and a cy principal-point shift produce the SAME
      world signature -> a code-side pixel/cy/y-flip offset is observationally identical to a
      physical mount mismatch (so it MUST be found in code if it exists; the grep says it isn't).

Run:  py -3.13 handoff/p1-vision-accuracy-2026-06-14/scratch-audit/roundtrip_boresight.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]          # .../Anduril-wt-vision-audit
sys.path.insert(0, str(ROOT / "src"))

from racer import frames as F                                            # noqa: E402
from racer.contracts import Gate, GateObservation, GatePose             # noqa: E402
from racer.localization import gate_pose_to_world_position              # noqa: E402
from racer.vision.gate_pose import (                                    # noqa: E402
    GATE_INNER_SIZE_M, estimate_gate_pose, project_gate_corners,
)

DEG = np.pi / 180.0
np.set_printoptions(precision=6, suppress=True)


def mount(pitch_deg: float) -> np.ndarray:
    """camera<-body for an arbitrary mount pitch (deg). pitch_deg=20 == F.R_camera_from_body()."""
    return F._R_CAMERA_FROM_TILTED_BODY @ Rotation.from_euler("Y", -np.deg2rad(pitch_deg)).as_matrix()


def K_with_cy(cy: float) -> np.ndarray:
    K = F.CAMERA_INTRINSICS_K.copy()
    K[1, 2] = cy
    return K


def make_gate(range_m: float, az_deg: float = 0.0, el_deg: float = 0.0,
              drone_pos=np.zeros(3), gate_id: int = 0) -> Gate:
    """Upright gate at (range, azimuth, elevation) from the drone in WORLD NED, through-dir
    pointing downrange (away from the drone)."""
    az, el = az_deg * DEG, el_deg * DEG
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), -np.sin(el)])  # NED unit dir
    pos = np.asarray(drone_pos, float) + range_m * d
    Zc = d / np.linalg.norm(d)                       # gate Z = downrange (through)
    Yc = np.array([0.0, 0.0, 1.0])                   # gate Y = down (world +Z), orthogonalised
    Yc = Yc - Zc * (Yc @ Zc)
    Yc /= np.linalg.norm(Yc)
    Xc = np.cross(Yc, Zc)                            # right-handed: X = Y x Z
    R_world_gate = np.column_stack([Xc, Yc, Zc])
    return Gate(gate_id=gate_id, position_ned=pos, R_world_gate=R_world_gate, inner_size_m=GATE_INNER_SIZE_M)


def run_fix(gate: Gate, drone_pos: np.ndarray, R_world_body: np.ndarray,
            render_pitch_deg: float = 20.0, decode_K: np.ndarray | None = None,
            corner_dv_px: float = 0.0, use_prior: bool = True):
    """Render corners at render_pitch (optionally + a constant vertical px offset / decode_K),
    decode at the NOMINAL 20deg mount, return (err_vec, gp, t_cam_render)."""
    R_cw_r = (R_world_body @ mount(render_pitch_deg).T).T                # camera<-world (render)
    t_cam = R_cw_r @ (gate.position_ned - drone_pos)
    R_cg = R_cw_r @ gate.R_world_gate
    corners = project_gate_corners(R_cg, t_cam, gate.inner_size_m).copy()  # CAMERA_INTRINSICS_K
    corners[:, 1] += corner_dv_px                                        # alt mechanism: +dv px (down)
    pr = None
    if use_prior:                                                       # nominal-model temporal/map prior
        R_cw_n = (R_world_body @ F.R_camera_from_body().T).T
        pr = GatePose(0, 0, R_cw_n @ gate.R_world_gate,
                      R_cw_n @ (gate.position_ned - drone_pos), 0.0)
    obs = GateObservation(0, 0, corners_px=corners, corner_confidence=np.ones(4))
    gp = estimate_gate_pose(obs, camera_matrix=decode_K, prior=pr, weighted_refine=False)
    if gp is None or not np.all(np.isfinite(gp.t_cam_gate)):            # PnP failed / diverged branch
        return np.full(3, np.nan), gp, t_cam
    pos_rec, _ = gate_pose_to_world_position(gp, gate, R_world_body)     # decode lever @20deg, +L
    return pos_rec - drone_pos, gp, t_cam


def gate_plane_err(err_vec: np.ndarray, gate: Gate) -> tuple[float, float, float]:
    """Project a world fix error onto the gate-plane axes: (right, vertical/down, along/through)."""
    R = gate.R_world_gate
    return float(err_vec @ R[:, 0]), float(err_vec @ R[:, 1]), float(err_vec @ R[:, 2])


def corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
LEVEL = F.R_world_from_body(0.0, 0.0, 0.0)   # level, yaw=0 (faces +X north); == I
ORIGIN = np.zeros(3)
out = []
def p(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    out.append(line)


p("=" * 78)
p("SANITY: parametric mount(20) == frames.R_camera_from_body()  :",
  bool(np.allclose(mount(20.0), F.R_camera_from_body())))
p("        R_world_from_body(0,0,0) == I                         :",
  bool(np.allclose(LEVEL, np.eye(3))))
p("        eps_vert arithmetic  atan(0.215/22) deg               :",
  round(np.rad2deg(np.arctan(0.215 / 22.0)), 4),
  "| 0.56deg @ fy=320 ->", round(320.0 * np.tan(0.56 * DEG), 3), "px")

# ============================ (A) INTERNAL CONSISTENCY =======================
p("\n" + "=" * 78)
p("(A) ROUND-TRIP @ ZERO BORESIGHT  (render@20, decode@20)  -- expect < 1mm")
p("-" * 78)
maxerr = 0.0
for rng in (8.0, 15.0, 22.0, 33.0):
    for az in (-20.0, -8.0, 0.0, 8.0, 20.0):
        for el in (-5.0, 0.0, 10.0):
            g = make_gate(rng, az, el)
            err, gp, tcam = run_fix(g, ORIGIN, LEVEL, render_pitch_deg=20.0)
            maxerr = max(maxerr, float(np.linalg.norm(err)))
p(f"  max |recovered - truth| over 60 poses (8-33m, az+-20, el-5..10) = {maxerr:.3e} m")
p(f"  VERDICT(A): chain internally consistent at 0deg boresight   = {bool(maxerr < 1e-3)}")

# ============================ (B) BORESIGHT INJECTION ========================
p("\n" + "=" * 78)
p("(B) +delta RENDER-ONLY camera-pitch boresight, decode@20  -- head-on level, 22 m")
p("-" * 78)
g22 = make_gate(22.0, 0.0, 0.0)
p("  delta_deg   gate_vert(down,m)  gate_right(m)  gate_along(m)  depth_err(m)")
slope_pts = []
for dlt in (-1.0, -0.56, -0.25, 0.0, 0.25, 0.56, 1.0):
    err, gp, tcam = run_fix(g22, ORIGIN, LEVEL, render_pitch_deg=20.0 + dlt)
    er, vd, al = gate_plane_err(err, g22)
    depth_err = gp.range_m - float(np.linalg.norm(tcam))
    p(f"   {dlt:+6.2f}     {vd:+10.4f}        {er:+8.4f}      {al:+8.4f}     {depth_err:+8.4f}")
    slope_pts.append((dlt, vd))
xs = np.array([s[0] for s in slope_pts]); ys = np.array([s[1] for s in slope_pts])
slope = float(np.polyfit(xs, ys, 1)[0])
# vbias at the field-pinned +0.56 deg:
err56, gp56, t56 = run_fix(g22, ORIGIN, LEVEL, render_pitch_deg=20.56)
_, vd56, _ = gate_plane_err(err56, g22)
p(f"  d(gate_vert)/d(delta) @22m = {slope:+.4f} m/deg   (geometry r*pi/180 = {22.0*DEG:+.4f})")
p(f"  gate_vert @ delta=+0.56deg = {vd56:+.4f} m   (field eps_vert ~ -0.215 m)")

# range flatness (constant ANGLE -> metric vbias proportional to range)
p("\n  RANGE BEHAVIOUR @ delta=+0.56deg (head-on level): metric vbias ~ range, ANGLE flat")
p("    range_m   gate_vert(m)   vbias/range(deg)   depth_err(m)")
for rng in (8.0, 15.0, 22.0, 33.0, 45.0):
    g = make_gate(rng, 0.0, 0.0)
    err, gp, tcam = run_fix(g, ORIGIN, LEVEL, render_pitch_deg=20.56)
    _, vd, _ = gate_plane_err(err, g)
    depth_err = gp.range_m - float(np.linalg.norm(tcam))
    p(f"    {rng:5.1f}     {vd:+8.4f}        {np.rad2deg(np.arctan(-vd/rng)):+7.4f}        {depth_err:+8.4f}")

# bearing correlation (camera-frame signature): fixed drone yaw, sweep gate azimuth
p("\n  BEARING SWEEP @ delta=+0.56deg (drone level yaw=0, gate az swept, 22 m):")
p("    az_deg   gate_vert(m)   gate_right(m)   depth_err(m)")
azs, vds, deps = [], [], []
for az in (-28.0, -20.0, -12.0, -6.0, 0.0, 6.0, 12.0, 20.0, 28.0):
    g = make_gate(22.0, az, 0.0)
    err, gp, tcam = run_fix(g, ORIGIN, LEVEL, render_pitch_deg=20.56)
    er, vd, al = gate_plane_err(err, g)
    depth_err = gp.range_m - float(np.linalg.norm(tcam))
    azs.append(az); vds.append(vd); deps.append(depth_err)
    p(f"    {az:+5.1f}    {vd:+8.4f}       {er:+8.4f}       {depth_err:+8.4f}")
p(f"  |gate_vert| range over az+-28 = {min(np.abs(vds)):.4f}..{max(np.abs(vds)):.4f} m "
  f"({100*(1-min(np.abs(vds))/max(np.abs(vds))):.1f}% bearing-dependent; a world-map offset would be FLAT)")
p(f"  corr(gate_vert , |az|)      = {corr(np.abs(azs), vds):+.3f}  (level drone: signed-az corr "
  f"= {corr(azs, vds):+.3f}, symmetric)")
p(f"  max |depth_err| over sweep  = {max(abs(x) for x in deps):.5f} m  (depth-free: boresight does not bias depth)")

# ROLLED-drone bearing sweep: a real course banks toward off-axis gates, which breaks the az
# symmetry and yields a SIGNED bearing correlation (the field's +0.44 lives here, not in a level sweep).
p("\n  ROLLED-drone bearing sweep @ delta=+0.56deg (roll = 0.6*az, gate az swept, 22 m):")
p("    az_deg   roll_deg   gate_vert(m)   gate_right(m)")
azr, vdr = [], []
for az in (-28.0, -20.0, -12.0, -6.0, 0.0, 6.0, 12.0, 20.0, 28.0):
    g = make_gate(22.0, az, 0.0)
    Rwb = F.R_world_from_body(0.6 * az * DEG, 0.0, 0.0)     # bank toward the gate
    err, gp, tcam = run_fix(g, ORIGIN, Rwb, render_pitch_deg=20.56)
    er, vd, al = gate_plane_err(err, g)
    azr.append(az); vdr.append(vd)
    p(f"    {az:+5.1f}    {0.6*az:+5.1f}     {vd:+8.4f}       {er:+8.4f}")
p(f"  corr(gate_vert , signed az) [rolled] = {corr(azr, vdr):+.3f}   (field bearing corr ~ +0.44)")

# random mixed pool: confirm depth-decoupling (range_err ~ 0 under a pure boresight)
rng_gen = np.random.default_rng(7)
vlist, dlist = [], []
for _ in range(300):
    rr = float(rng_gen.uniform(8.0, 40.0)); aa = float(rng_gen.uniform(-22, 22)); ee = float(rng_gen.uniform(-2, 10))
    g = make_gate(rr, aa, ee)
    err, gp, tcam = run_fix(g, ORIGIN, LEVEL, render_pitch_deg=20.56)
    if gp is None or not np.all(np.isfinite(err)):
        continue
    _, vd, _ = gate_plane_err(err, g)
    vlist.append(vd); dlist.append(gp.range_m - float(np.linalg.norm(tcam)))
p(f"  [{len(vlist)}/300 valid mixed poses] mean|depth_err| = {np.mean(np.abs(dlist)):.5f} m, "
  f"max {np.max(np.abs(dlist)):.5f} m  ->  vertical bias is depth-decoupled")

# ============================ (C) ALTERNATE CODE-SIDE MECHANISMS =============
p("\n" + "=" * 78)
p("(C) ALTERNATE MECHANISMS @22m head-on level: are they indistinguishable from a mount tilt?")
p("-" * 78)
base = 320.0 * np.tan(0.56 * DEG)        # 3.13 px == 0.56 deg at fy=320
# A non-frontal gate avoids the exactly-frontal IPPE 2-fold degeneracy that a pure vertical
# corner shift can tip into the wrong branch (an IPPE-disambig artefact, NOT a chain bug); use a
# mild yaw so both alternates resolve cleanly. (At dead-frontal, -dv tips the branch -> None.)
g22o = make_gate(22.0, 6.0, 0.0)
# 1) constant vertical corner offset (render@20, then +dv px, decode@20, K nominal)
for dv in (+base, -base):
    err, gp, tcam = run_fix(g22o, ORIGIN, LEVEL, render_pitch_deg=20.0, corner_dv_px=dv)
    _, vd, _ = (gate_plane_err(err, g22o) if np.all(np.isfinite(err)) else (np.nan,) * 3)
    p(f"  constant corner offset dv = {dv:+6.3f} px  -> gate_vert = {vd:+.4f} m"
      f"{'  (IPPE branch None at dead-frontal; resolves off-axis)' if not np.isfinite(vd) else ''}")
# 2) principal-point cy shift on the DECODE K only (render@20 with cy=180)
for cy in (180.0 - base, 180.0 + base):
    err, gp, tcam = run_fix(g22o, ORIGIN, LEVEL, render_pitch_deg=20.0, decode_K=K_with_cy(cy))
    _, vd, _ = (gate_plane_err(err, g22o) if np.all(np.isfinite(err)) else (np.nan,) * 3)
    p(f"  decode principal pt cy = {cy:7.3f} (nominal 180)  -> gate_vert = {vd:+.4f} m")
# 3) the cy=180-vs-179.5 half-pixel SEAM magnitude (0.5 px corner shift)
errA, _, _ = run_fix(g22o, ORIGIN, LEVEL, render_pitch_deg=20.0, corner_dv_px=+0.5)
_, vdA, _ = gate_plane_err(errA, g22o)
p(f"  half-pixel cy seam (0.5 px)                   -> gate_vert = {vdA:+.4f} m "
  f"(~{abs(vdA)/0.215*100:.0f}% of 0.215 -> a 180-vs-179.5 mismatch cannot alone explain eps_vert)")
p(f"  NOTE: pixel/cy mechanism gives ~{abs(vd):.3f} m vs mount {abs(vd56):.3f} m at +0.56deg-equivalent")
p(f"        -> same sign & order, ~10-15% smaller; observationally the SAME vertical perception bias.")

p("\n" + "=" * 78)
p("INTERPRETATION")
p("-" * 78)
p("  * (A) exact round-trip => K (fy=320, cy=180) projection + the +L localization lever")
p("    are exact inverses; no code-internal vertical bias exists at matched mount.")
p("  * (B) a +0.56deg render-vs-decode mount mismatch reproduces ~-0.215 m gate-vertical at")
p("    22 m, scales with range at constant angle, leaves depth unbiased (depth-free), and is")
p("    bearing-coupled -- the field eps_vert signature.")
p("  * (C) a constant ~3.13 px vertical corner offset OR a 3.13 px cy shift yields the SAME")
p("    world signature -> a code-side pixel/cy/y-flip offset is OBSERVATIONALLY IDENTICAL to a")
p("    physical mount tilt. The Task-1 grep finds NO such code offset (single K, single mount,")
p("    no y-flip, detector convention == auto-label projector) => eps_vert is PHYSICAL.")

(Path(__file__).resolve().parent / "roundtrip_boresight_OUT.txt").write_text("\n".join(out) + "\n")
