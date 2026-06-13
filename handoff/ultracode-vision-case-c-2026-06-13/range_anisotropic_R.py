"""Range-ANISOTROPIC world-fix measurement covariance R(range, bearing) for the case-C
vision->KF chain.

CONTEXT (ultracode vision case-C, 2026-06-13). The current ``localization.gate_pose_to_world_position``
fix covariance is

    Cov(p) = K2 * R_wc Sigma_pnp R_wc^T            # analytic PnP block (pixel noise, INFLATED x2)
             + sigma_theta^2 (|L|^2 I - L L^T)     # attitude lever (TANGENTIAL, perp to LOS, grows |L|)
             + sigma_floor^2 I                     # isotropic constant-systematics floor (0.40 m)

The attitude lever already models BEARING (tangential) uncertainty anisotropically and growing with
range (std ~ |L| = r). The GAP this module closes: the RADIAL (depth, along-line-of-sight)
uncertainty of a known-size planar square scales ~ r^2 (std) / r^4 (variance) with range, and today
it is carried ONLY by the inflated analytic ``Sigma_pnp`` plus the isotropic 0.40 m floor -- the floor
is range-FLAT and the analytic block is shape-correct only if the PnP Fisher block is trusted (it is
inflated x2 ad-hoc and understates the gate-size-model mismatch). So the radial axis is mis-SHAPED vs
range. This module replaces the radial channel with the DERIVED+CALIBRATED depth law while keeping the
attitude-lever (tangential) and floor terms exactly where they physically belong.

DERIVATION (single planar square, side s, range r, focal f, N corners, per-corner pixel noise sigma_px;
camera optical frame, gate head-on so t_cam_gate ~ [0,0,r]):

  A corner at gate position (x,y) (|x|,|y| = s/2) projects to u = f*X/Z, v = f*Y/Z with
  (X,Y,Z) = (x, y, r) for a frontal gate. The depth r is recovered from the APPARENT SIZE of the
  square: the pixel half-width is w_px = f*(s/2)/r, so r = f*s/(2 w_px) and

        dr/dw_px = -f*s/(2 w_px^2) = -2 r^2 /(f*s).

  Each corner pixel carries independent noise sigma_px; the apparent-size estimate w_px averages the
  4 corner offsets from the centroid, so its variance is var(w_px) = sigma_px^2 / N_eff with N_eff the
  effective number of size-constraining corners (4 for a full square; the two opposing-corner
  differences give 2 independent size measurements per axis -> N_eff ~ 4). Hence

        var(r)  = (dr/dw_px)^2 var(w_px) = (2 r^2/(f s))^2 * sigma_px^2 / N_eff
                = 4 r^4 sigma_px^2 / (f^2 s^2 N_eff).
        std(r)  = (2 r^2 sigma_px) / (f s sqrt(N_eff))         ==> std ~ r^2  (DEPTH, RADIAL).

  The LATERAL (cross-LOS) translation t_x = X is recovered from the centroid u_c = f*X/r, X = r u_c/f,
  dX/du_c = r/f; the centroid averages N corners so var(u_c) = sigma_px^2/N:

        var(t_x) = (r/f)^2 sigma_px^2 / N
        std(t_x) = (r sigma_px) / (f sqrt(N))                  ==> std ~ r^1  (LATERAL, TANGENTIAL).

  This is the textbook monocular-known-size anisotropy: depth uncertainty grows ONE POWER OF r FASTER
  than lateral. The MC self-check (validate_R.py / __main__) confirms std(depth) ~ r^2 and
  std(lat) ~ r^1 with coefficients matching to ~10%.

CALIBRATION (against the MEASURED perception-char-2026-06-08 profile). The IDEAL pixel-noise depth law
above is the SHAPE; the real depth error over the VQ1 regime (0-24 m) is dominated by a gate-size-model
mismatch (assumed 1.5 m inner vs true) + sub-pixel corner bias, which add an approximately
RANGE-PROPORTIONAL depth bias on top of the r^2 pixel term, AND a range-flat systematics floor. We
therefore fit a compact 3-coefficient radial model

        var_radial(r) = (c2 * r^2)^2 + (c1 * r)^2 + sig0_rad^2

and 2-coefficient tangential / lateral PnP model on top of the (physically calibrated, UNCHANGED)
attitude-lever term

        var_tangential_pnp(r) = (a1 * r)^2 + sig0_tan^2

fit by matching the per-range-band radial/tangential error variance of the GOOD (|off|<3 m) fixes.
fit_range_R.py writes the calibrated coefficients to ``range_R_coeffs.json`` next to this file; this
module loads them if present and otherwise falls back to PHYSICS defaults (derived coefficients with
sigma_px=1.5, plus the shipped 0.40 m floor) so it is usable with no calibration file.

OUTPUT FRAME. The covariance is assembled in world NED: the radial axis is L_hat L_hat^T (rank-1 along
LOS), the tangential/lateral axes are (|L|^2 I - L L^T)/|L|^2-style perpendicular projectors. All three
physical channels (PnP depth, PnP lateral, attitude lever, floor) are summed -> a full 3x3 PSD NED cov,
a drop-in replacement for the cov returned by ``gate_pose_to_world_position``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Reuse the live constants so this module tracks the shipped model exactly.
import sys as _sys
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in _sys.path:
    _sys.path.insert(0, str(_REPO / "src"))

from racer.frames import ATTITUDE_NOISE_STD_RAD, CAMERA_INTRINSICS_K  # noqa: E402
from racer.localization import FIX_COV_FLOOR_STD  # noqa: E402
from racer.vision.gate_pose import GATE_INNER_SIZE_M  # noqa: E402

# --- PHYSICS defaults (used when no calibration file is present) ----------------------------------
_F = float(CAMERA_INTRINSICS_K[0, 0])      # focal length px (=320)
_S = float(GATE_INNER_SIZE_M)              # inner-square side m (=1.5)
_SIGMA_PX_DEFAULT = 1.5                    # WEIGHTED_SIGMA_PX (per-corner pixel noise, gate_pose.py)
_N_CORNERS_FULL = 4

# Derived depth coefficient c2 such that std_depth = c2 * r^2  (the Fisher law above, N_eff=4):
#   c2 = 2 sigma_px / (f s sqrt(N_eff))
def _c2_from_pixels(sigma_px: float, n_eff: float = 4.0) -> float:
    return 2.0 * sigma_px / (_F * _S * np.sqrt(n_eff))

# Derived lateral coefficient a1 such that std_lat = a1 * r  (N=4 corners averaged):
#   a1 = sigma_px / (f sqrt(N))
def _a1_from_pixels(sigma_px: float, n: float = 4.0) -> float:
    return sigma_px / (_F * np.sqrt(n))

# Default coefficient bundle (PHYSICS, no calibration). c1 (range-proportional depth bias from
# gate-size mismatch) defaults to 0 -- pure pixel-noise model -- until calibration fits it.
_DEFAULT_COEFFS = {
    "c2": _c2_from_pixels(_SIGMA_PX_DEFAULT),     # r^2 depth std coefficient (Fisher)
    "c1": 0.0,                                    # r^1 depth-bias coefficient (gate-size mismatch; fit)
    "sig0_rad": FIX_COV_FLOOR_STD,                # range-flat radial floor (m)
    "a1": _a1_from_pixels(_SIGMA_PX_DEFAULT),     # r^1 lateral PnP std coefficient (Fisher)
    "sig0_tan": 0.0,                              # extra range-flat tangential floor (m); fit
    "sigma_theta": float(ATTITUDE_NOISE_STD_RAD), # attitude-lever 1-sigma (rad); UNCHANGED from frames
    "sig0_iso": 0.0,                              # extra isotropic floor beyond the directional terms
    "p3p_inflation": 9.0,                         # variance multiplier for 3-corner fixes (matches live)
    "source": "PHYSICS_DEFAULT (no calibration file)",
}

_COEFFS_PATH = Path(__file__).resolve().parent / "range_R_coeffs.json"


def load_coeffs(path: Path | str | None = None) -> dict:
    """Load calibrated coefficients (range_R_coeffs.json) or fall back to PHYSICS defaults."""
    p = Path(path) if path is not None else _COEFFS_PATH
    coeffs = dict(_DEFAULT_COEFFS)
    if p.exists():
        try:
            disk = json.loads(p.read_text())
            coeffs.update({k: disk[k] for k in disk if k in coeffs or k == "source"})
        except (json.JSONDecodeError, OSError):
            pass
    return coeffs


def _perp_projector(Lhat: np.ndarray) -> np.ndarray:
    """3x3 projector onto the plane perpendicular to unit LOS Lhat (rank 2)."""
    return np.eye(3) - np.outer(Lhat, Lhat)


def R_aniso(
    t_cam_gate: np.ndarray,
    R_wc: np.ndarray,
    conf: float | np.ndarray | None = None,
    reproj_px: float | None = None,
    range_m: float | None = None,
    n_corners: int = 4,
    coeffs: dict | None = None,
) -> np.ndarray:
    """Range-ANISOTROPIC 3x3 world-NED fix covariance from a single gate sighting.

    Parameters
    ----------
    t_cam_gate : (3,) gate origin in the camera optical frame (the PnP translation). Its norm is
                 the PnP depth/range; its rotation by R_wc gives the world lever L (gate rel. drone).
    R_wc       : (3,3) world<-camera rotation = R_world_body @ R_camera_from_body().T (the SAME R_wc
                 ``localization.gate_pose_to_world_position`` builds; pass it in).
    conf       : optional per-corner confidence (mean is used to scale sigma_px up when low); None ->
                 nominal pixel noise.
    reproj_px  : optional RMS reprojection error; if it EXCEEDS the nominal pixel noise, it inflates
                 the radial+lateral PnP terms (a poorly-fitting fix is depth-untrustworthy). None ->
                 no reproj inflation.
    range_m    : optional explicit range; defaults to |t_cam_gate| (the PnP depth). Using the SOLVED
                 range (not true range) is correct at runtime -- it is all the filter has.
    n_corners  : 3 or 4. A 3-corner (P3P) fix inflates the whole cov by coeffs['p3p_inflation'].
    coeffs     : coefficient dict (load_coeffs()); None -> load from disk / PHYSICS defaults.

    Returns
    -------
    (3,3) PSD covariance in world NED. Radial axis (along LOS) carries the r^2 depth law; the two
    tangential axes carry the r^1 lateral-PnP term + the attitude lever (|L|^2-growing) + isotropic
    floor. Drop-in replacement for the cov from ``gate_pose_to_world_position``.
    """
    if coeffs is None:
        coeffs = load_coeffs()
    t = np.asarray(t_cam_gate, dtype=np.float64)
    R_wc = np.asarray(R_wc, dtype=np.float64)
    L = R_wc @ t                                    # world lever (gate rel. drone, NED)
    nL = float(np.linalg.norm(L))
    r = float(range_m) if range_m is not None else nL
    if nL < 1e-9:
        # Degenerate (drone on the gate): isotropic floor only, never singular.
        sig0 = max(coeffs["sig0_rad"], coeffs["sig0_iso"], 1e-3)
        return (sig0**2) * np.eye(3)
    Lhat = L / nL

    # Pixel-noise scaling from confidence + reprojection. The Fisher coefficients were derived at the
    # nominal sigma_px; a low-confidence corner set or a high reproj fit means the effective pixel
    # noise is larger -> scale c2 (depth r^2) and a1 (lateral r^1) which are both LINEAR in sigma_px.
    px_scale = 1.0
    if conf is not None:
        cmean = float(np.mean(np.clip(np.asarray(conf, dtype=np.float64), 0.1, 1.0)))
        px_scale *= 1.0 / cmean                     # less confident -> looser
    if reproj_px is not None and reproj_px > _SIGMA_PX_DEFAULT:
        px_scale *= float(reproj_px) / _SIGMA_PX_DEFAULT

    # --- RADIAL (depth, along LOS): (c2 r^2)^2 pixel-Fisher + (c1 r)^2 size-mismatch bias + floor.
    sig_rad_pnp = px_scale * coeffs["c2"] * r * r   # r^2 term (scaled by pixel-noise)
    sig_rad_bias = coeffs["c1"] * r                 # r^1 term (gate-size mismatch; NOT pixel-scaled)
    var_radial = sig_rad_pnp**2 + sig_rad_bias**2 + coeffs["sig0_rad"] ** 2
    cov = var_radial * np.outer(Lhat, Lhat)

    # --- TANGENTIAL (lateral, perp to LOS): PnP lateral (a1 r)^2 + tangential floor, isotropic in the
    # perp plane (rank-2 projector). std ~ r^1 (one power below depth).
    var_tan_pnp = (px_scale * coeffs["a1"] * r) ** 2 + coeffs["sig0_tan"] ** 2
    cov = cov + var_tan_pnp * _perp_projector(Lhat)

    # --- ATTITUDE LEVER (perp to L, grows |L|^2): UNCHANGED physical term from the live model. A small
    # attitude error rotates the lever; induced cov sigma_theta^2 (|L|^2 I - L L^T). Already tangential.
    st = coeffs["sigma_theta"]
    if st > 0.0:
        cov = cov + (st**2) * (nL * nL * np.eye(3) - np.outer(L, L))

    # --- ISOTROPIC floor (extra range-flat systematics not captured directionally).
    if coeffs["sig0_iso"] > 0.0:
        cov = cov + (coeffs["sig0_iso"] ** 2) * np.eye(3)

    # --- 3-corner (P3P) inflation: a clipped-gate fix is geometrically weaker; match the live policy.
    if int(n_corners) < 4:
        cov = cov * coeffs["p3p_inflation"]

    # Symmetrise (guard tiny asymmetry from outer products) and return.
    return 0.5 * (cov + cov.T)


# Convenience: the CURRENT (baseline) model, re-implemented here so validate_R.py can compare the two
# WITHOUT importing the live gate_pose covariance (which needs a full GatePose / detector). This mirrors
# localization.gate_pose_to_world_position's cov branch with an explicit Sigma_pnp (radial+lateral) so
# the two models see the same inputs. The live model's Sigma_pnp is the analytic Fisher block; here we
# feed the SAME derived radial/lateral Fisher variances (un-anisotropic-ised: it puts the r^2 depth in
# Sigma_pnp too) inflated by K2, so the contrast is purely SHAPE (isotropic-floor vs anisotropic-axis),
# not coefficient choice.
def R_baseline_k2_lever_floor(
    t_cam_gate: np.ndarray,
    R_wc: np.ndarray,
    range_m: float | None = None,
    n_corners: int = 4,
    sigma_px: float = _SIGMA_PX_DEFAULT,
    k2: float = 2.0,
    p3p_inflation: float = 9.0,
    sigma_theta: float = float(ATTITUDE_NOISE_STD_RAD),
    sigma_floor: float = FIX_COV_FLOOR_STD,
) -> np.ndarray:
    """Baseline K2 * R_wc Sigma_pnp R_wc^T + sigma_theta^2(|L|^2 I - L L^T) + sigma_floor^2 I.

    Sigma_pnp is built in the CAMERA frame from the SAME derived Fisher variances (depth r^2 on the
    optical-Z axis, lateral r^1 on X/Y), so the only difference from R_aniso is that the baseline
    routes the depth variance through R_wc Sigma_pnp R_wc^T (correct axis!) but then adds an ISOTROPIC
    floor -- i.e. baseline ALSO has a depth axis. The REAL contrast the validation exposes is whether
    the K2-inflation + isotropic floor is correctly CALIBRATED vs the anisotropic per-axis floors.
    This is the fair, apples-to-apples baseline.
    """
    t = np.asarray(t_cam_gate, dtype=np.float64)
    R_wc = np.asarray(R_wc, dtype=np.float64)
    L = R_wc @ t
    nL = float(np.linalg.norm(L))
    r = float(range_m) if range_m is not None else nL
    # Analytic camera-frame PnP cov: depth on optical Z (=r^2 law), lateral on X,Y (=r^1 law).
    c2 = _c2_from_pixels(sigma_px)
    a1 = _a1_from_pixels(sigma_px)
    sigma_pnp_cam = np.diag([(a1 * r) ** 2, (a1 * r) ** 2, (c2 * r * r) ** 2])
    cov = k2 * (R_wc @ sigma_pnp_cam @ R_wc.T)
    if sigma_theta > 0.0 and nL > 1e-9:
        cov = cov + (sigma_theta**2) * (nL * nL * np.eye(3) - np.outer(L, L))
    if sigma_floor > 0.0:
        cov = cov + (sigma_floor**2) * np.eye(3)
    if int(n_corners) < 4:
        cov = cov * p3p_inflation
    return 0.5 * (cov + cov.T)


if __name__ == "__main__":
    # Quick self-print: physics coefficients + a sample cov at 5/15/25 m head-on.
    import numpy as _np
    c = load_coeffs()
    print("coeffs:", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in c.items()})
    R_wc = _np.eye(3)
    for rr in (5.0, 15.0, 25.0):
        cov = R_aniso(_np.array([0, 0, rr]), R_wc, range_m=rr)
        ev = _np.sqrt(_np.linalg.eigvalsh(cov))
        print(f" r={rr:4.0f} m  cov eig-std (m) = {_np.round(ev, 3)}  "
              f"(largest = depth axis, should grow ~r^2)")
