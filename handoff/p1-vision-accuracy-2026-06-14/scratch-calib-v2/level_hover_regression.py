r"""#16 Level-hover vertical regression -- the METRIC-form estimator AND the angular-vs-metric
DISCRIMINATOR (complements P3's gate-0/1 epsilon(range) arbiter).

From a static/LEVEL-hover recording that varies RANGE to a head-on vertical reference (by altitude or
standoff), regress the fix-z (gate-vertical, down +) residual against range:

        m_v(r)  =  intercept  +  slope * r  +  noise
                   \________/    \________/
                    METRIC        ANGULAR
        intercept  = range-INVARIANT constant   -> camera optical-centre / CoM vertical offset (Delta z)
        slope      = range-PROPORTIONAL term     -> boresight pitch:  pitch = -atan(slope) (slope=-tan eps)

The intercept-vs-slope split IS the functional-form discriminator. Weighted LS (per-range m_v noise
grows with the lever arm). Validated on the REAL src/racer chain: inject pure-angular, pure-metric, and
mixed; confirm recovery + correct classification. IDENTIFIABILITY: the slope/intercept separation needs
RANGE SPREAD -- we measure sigma_mv from the chain and derive the minimum (r2-r1) for 3-sigma angular
detection, and show a too-close pair is DEGENERATE (the escape-hatch finding for P3's >=2-range test).

Run:  py -3.13 handoff/p1-vision-accuracy-2026-06-14/scratch-calib-v2/level_hover_regression.py
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
out = []
def p(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    out.append(line)


def mount_corrected(pitch_rad=0.0, roll_rad=0.0) -> np.ndarray:
    R_tilt = Rotation.from_euler("Y", -(F.CAMERA_PITCH_RAD + pitch_rad)).as_matrix()
    R_roll = Rotation.from_euler("X", -roll_rad).as_matrix()
    return F._R_CAMERA_FROM_TILTED_BODY @ R_roll @ R_tilt


def head_on_gate(range_m: float) -> Gate:
    return Gate(gate_id=0, position_ned=np.array([range_m, 0.0, 0.0]),
                R_world_gate=np.column_stack([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).astype(float),
                inner_size_m=GATE_INNER_SIZE_M)


def measure_mv(range_m, *, phys_pitch_rad=0.0, phys_voff_m=0.0, n=120, sigma_px=0.7, seed=0):
    """Mean gate-vertical (down +) fix-vs-GT residual + std-of-mean over n noisy level head-on frames.
    Render through the PHYSICAL camera (mount pitch + optical-centre offset), decode at the UNCALIBRATED
    20deg / zero-offset model (the lock-test measurement). Mirrors calib_estimator_demo."""
    g = head_on_gate(range_m); R_wb = np.eye(3); drone = np.zeros(3)
    R_phys = mount_corrected(phys_pitch_rad, 0.0)
    cam_c = drone + R_wb @ np.array([0.0, 0.0, phys_voff_m])
    R_cw = (R_wb @ R_phys.T).T
    clean = project_gate_corners(R_cw @ g.R_world_gate, R_cw @ (g.position_ned - cam_c), g.inner_size_m)
    R_dec = mount_corrected(0.0, 0.0)
    prior = GatePose(0, 0, (R_wb @ R_dec.T).T @ g.R_world_gate,
                     (R_wb @ R_dec.T).T @ (g.position_ned - drone), 0.0)
    rng = np.random.default_rng(seed)
    vds = []
    for _ in range(n):
        obs = GateObservation(0, 0, corners_px=clean + rng.normal(0, sigma_px, clean.shape),
                              corner_confidence=np.ones(4))
        gp = estimate_gate_pose(obs, prior=prior, weighted_refine=False)
        if gp is None or not np.all(np.isfinite(gp.t_cam_gate)):
            continue
        pos = g.position_ned - (R_wb @ R_dec.T) @ gp.t_cam_gate          # decode @0, +L lever
        vds.append(float((pos - drone) @ g.R_world_gate[:, 1]))
    vds = np.array(vds)
    return float(vds.mean()), float(vds.std(ddof=1) / np.sqrt(len(vds))), float(vds.std(ddof=1))


def wls(ranges, mv, mv_sem):
    """Weighted LS  mv ~ intercept + slope*range. Returns (beta[2], cov[2,2])."""
    X = np.column_stack([np.ones_like(ranges), ranges])
    W = np.diag(1.0 / np.asarray(mv_sem) ** 2)
    XtWX = X.T @ W @ X
    cov = np.linalg.inv(XtWX)
    beta = cov @ X.T @ W @ mv
    return beta, cov


# ===================================================================================================
p("=" * 88)
p("#16  LEVEL-HOVER VERTICAL REGRESSION  (metric estimator + angular-vs-metric discriminator)")
p("=" * 88)

# --- 0. measure the chain's per-frame fix-z noise vs range (n=120 frames, 0.7 px corner noise) ----
p("\n0. CHAIN fix-z noise vs range (level head-on, decode@0, 0.7 px corner noise, n=120):")
p("    range_m   per-frame sigma_mv(m)   std-of-mean(120)(m)")
RANGES = np.array([10.0, 14.0, 18.0, 22.0, 26.0, 30.0])
sems = {}
for r in RANGES:
    _, sem, sd = measure_mv(r, seed=11)
    sems[r] = sem
    p(f"    {r:5.1f}      {sd:8.4f}              {sem:8.5f}")
sigma_mv_typ = float(np.median(list(sems.values())))
p(f"  -> per-MEAN sigma_mv (median over ranges, 120 frames) ~ {sigma_mv_typ:.5f} m "
  f"(grows with range = lever arm)")

EPS = 0.56 * DEG     # angular boresight to inject
VOFF = -0.215        # metric offset to inject
scenarios = [
    ("PURE ANGULAR (eps=0.56deg, voff=0)   ", EPS, 0.0),
    ("PURE METRIC  (eps=0, voff=-0.215 m)  ", 0.0, VOFF),
    ("MIXED        (eps=0.30deg, voff=-0.10)", 0.30 * DEG, -0.10),
]
# Physical significance floors (NOT pure statistical): a term needs correcting only if it is BOTH
# statistically resolved (3-sigma) AND physically above the acceptance floor (metric 0.05 m / angle
# 0.05 deg). This rejects the sigma_px^2 PnP NOISE BIAS (~0.01 m intercept, ~0.03 deg slope at 0.7 px,
# pinned by the noiseless control in 1a) from spuriously flagging a form.
FLOOR_M = 0.05
FLOOR_SLOPE = np.tan(0.05 * DEG)

def classify(icpt, slope, icpt_sd, slope_sd):
    m_sig = abs(icpt) > max(3 * icpt_sd, FLOOR_M)
    a_sig = abs(slope) > max(3 * slope_sd, FLOOR_SLOPE)
    if a_sig and not m_sig: return "ANGULAR"
    if m_sig and not a_sig: return "METRIC"
    if a_sig and m_sig:     return "MIXED"
    return "NEGLIGIBLE"

# --- 1a. NOISELESS control: prove the chain is exactly linear (intercept=0/slope=-tan eps) so the
#         small biases in 1b are NOISE-induced (a PnP-bias floor), not a method error ---------------
p("\n1a. NOISELESS control (sigma_px=0, OLS): biases vanish -> 1b's residual bias is PnP noise bias")
for tag, pe, vo in scenarios:
    mv = np.array([measure_mv(r, phys_pitch_rad=pe, phys_voff_m=vo, n=1, sigma_px=0.0)[0] for r in RANGES])
    X = np.column_stack([np.ones_like(RANGES), RANGES])
    icpt, slope = np.linalg.lstsq(X, mv, rcond=None)[0]
    p(f"  [{tag}] intercept = {icpt:+.5f} m (inj {vo:+.3f})   pitch = {np.rad2deg(-np.arctan(slope)):+.4f} deg (inj {np.rad2deg(pe):+.3f})")

# --- 1b. realistic noise (0.7 px, 120 frames, WLS) ------------------------------------------------
p("\n1b. REGRESSION on the REAL chain (6 ranges 10-30 m, 0.7 px, 120 frames each, WLS):")
for tag, pe, vo in scenarios:
    mv = np.array([measure_mv(r, phys_pitch_rad=pe, phys_voff_m=vo, seed=int(r) + 7)[0] for r in RANGES])
    sem = np.array([sems[r] for r in RANGES])
    beta, cov = wls(RANGES, mv, sem)
    icpt, slope = beta
    icpt_sd, slope_sd = np.sqrt(np.diag(cov))
    pitch_est = -np.arctan(slope)
    p(f"\n  [{tag}]")
    p(f"    intercept (METRIC voff)  = {icpt:+.4f} +- {icpt_sd:.4f} m     (injected {vo:+.4f} m)")
    p(f"    slope                    = {slope:+.5f} +- {slope_sd:.5f}      -> pitch = {np.rad2deg(pitch_est):+.4f} deg "
      f"(injected {np.rad2deg(pe):+.4f})")
    p(f"    classify (3-sigma + physical floor {FLOOR_M} m / 0.05 deg): {classify(icpt, slope, icpt_sd, slope_sd)}")

# --- 2. identifiability: minimum range separation for 3-sigma angular detection -------------------
p("\n2. IDENTIFIABILITY -- minimum range spread to separate slope (angular) from intercept (metric)")
p("-" * 88)
p("  Two-range lock test at (r1,r2), per-mean noise sigma_mv: slope_sd = sigma_mv*sqrt(2)/(r2-r1).")
p("  The eps_vert angular signal is slope = -tan(0.56deg) = %.5f. Detect at 3-sigma => " % (-np.tan(EPS)))
p("  (r2-r1) > 3*sqrt(2)*sigma_mv/|slope|.")
for sg in (0.005, 0.010, 0.020, 0.030):
    dmin = 3 * np.sqrt(2) * sg / np.tan(EPS)
    p(f"    sigma_mv = {sg:.3f} m  ->  min (r2-r1) = {dmin:5.1f} m")
p(f"  At the MEASURED sigma_mv ~ {sigma_mv_typ:.4f} m (120 frames): min (r2-r1) = "
  f"{3*np.sqrt(2)*sigma_mv_typ/np.tan(EPS):.1f} m")

# concrete: discrimination SNR at candidate gate-0/1 standoff pairs (chain-measured, pure angular)
p("\n  Discrimination at candidate gate-0/1 standoff pairs (inject eps=0.56deg, chain-measured):")
p("    (r1,r2) m    slope_est +- sd        slope/sd     verdict")
for (r1, r2) in [(18.0, 22.0), (12.0, 22.0), (10.0, 28.0), (10.0, 30.0)]:
    mv = np.array([measure_mv(r, phys_pitch_rad=EPS, seed=int(r) + 7)[0] for r in (r1, r2)])
    sem = np.array([measure_mv(r, phys_pitch_rad=EPS, seed=int(r) + 7)[1] for r in (r1, r2)])
    b, c = wls(np.array([r1, r2]), mv, sem)
    ssd = np.sqrt(c[1, 1])
    snr = abs(b[1]) / ssd
    p(f"    ({r1:4.0f},{r2:4.0f})   {b[1]:+.5f} +- {ssd:.5f}    {snr:6.1f}     "
      f"{'CLEAN (>3)' if snr > 3 else 'DEGENERATE (<3) -> cannot split intercept/slope'}")

# --- 3. degeneracy at a SINGLE range (escape-hatch) ----------------------------------------------
p("\n3. SINGLE-RANGE DEGENERACY (the escape-hatch): 1 equation, 2 unknowns -> intercept/slope confounded")
p("-" * 88)
r0 = 22.0
mv0, sem0, _ = measure_mv(r0, phys_pitch_rad=EPS, seed=29)
p(f"  @22 m only: m_v = {mv0:+.4f} m. Consistent with BOTH (intercept={mv0:+.4f},slope=0) METRIC")
p(f"  AND (intercept=0, slope={mv0/r0:+.5f} -> pitch {np.rad2deg(-np.arctan(mv0/r0)):+.3f}deg) ANGULAR.")
p("  => A single range CANNOT discriminate. P3's gate-0/1 test MUST supply >=2 WELL-SEPARATED ranges;")
p("     if its two ranges are too close (e.g. 18/22 m), the >=2-range test pins MAGNITUDE but the")
p("     FORM stays degenerate -- elevate the range spread to load-bearing (report UP).")

p("\n" + "=" * 88)
p("VERDICT: WLS m_v ~ intercept + slope*r recovers the metric (intercept) and angular (slope=-tan eps)")
p("terms and classifies the form. Discrimination needs range SPREAD; at the measured sigma_mv a >= ~10-15")
p("m separation gives a clean (>3-sigma) split, while 18/22 m is marginal/degenerate.")
(Path(__file__).resolve().parent / "level_hover_OUT.txt").write_text("\n".join(out) + "\n")
