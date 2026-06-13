"""VISION-CAL (piece F): per-gate track-map registration re-survey + offset estimator.

WHAT THIS IS
------------
The case-C estimator world-fixes against ``track_map.json``: a sighting of gate g gives a
drone-position fix ``p_drone = map_centre[g] - lever``. If ``map_centre[g]`` is mis-registered
by delta_g, EVERY fix near gate g inherits that delta_g as a bias the KF cannot reject (it is
consistent across sightings, so it looks like signal). This module RE-SURVEYS each gate's true
world opening-centre from the calibration-lap fixes and compares to the map, so we can (a) detect
a mis-registered gate and (b) quantify the RESIDUAL per-gate registration sigma the case-C KF
must carry in R for gates with no live cross-check (the inc8 gate-4 provisional-margin de-risk).

THE RE-SURVEY (prototyped + run here)
-------------------------------------
On the calibration lap the drone pose is GIVEN (cases A/B). Each sighting implies a world
gate-centre measurement ``gate_meas = drone_pos + R_wc @ t_cam_gate``. We do NOT have the raw
``t_cam_gate`` on the laptop (no detector weights / no live frames), but the shipped perception
characterization (handoff/perception-char-2026-06-08) recorded, per sighting, the world-fix
error ``off_ned = p_fix - drone_given`` where ``p_fix = map_centre[g] - R_wc @ t_cam_gate``
(characterize_perception.py:189, against the GIVEN==GT pose in the VQ1 regime). Algebra:

    off_ned = (map_centre[g] - lever) - drone_given,   lever = R_wc @ t_cam_gate
    gate_meas = drone_given + lever = map_centre[g] - off_ned          (drone cancels)

so the MAP-INDEPENDENT gate-centre measurement is recovered as ``map_centre[g] - off_ned``: the
map enters only as a known additive constant we subtract straight back out. The registration
offset estimate is therefore ``estimated_true_centre - map_centre = -robust_mean(off_ned)``.

We feed those reconstructed gate-centre measurements into the SHIPPED robust averager
``racer.gate_mapper.estimate_map_pose_aided`` (median/MAD rejection, the exact code the live
mapper runs for cases A/B) and read each gate's estimated centre + statistical covariance, then
compare to the map opening-centre that the live navigator actually localizes against
(``navigator.gates_from_track_records(corner_to_center=True)`` lift: record bottom-centre minus
half-height along the down axis).

SIGN HANDLING (the footgun)
---------------------------
``MEASURED_FIX_BIAS_NED = [-0.42, +0.06, -0.28]`` (N,E,D) is the mean FIX error. A gate
MEASUREMENT error is MINUS the fix error (gate_mapper docstring), so the registration offset is
``-mean(off_ned)``, and a positive D offset means the true gate centre is BELOW (larger D, NED
down-positive) the map. We assert the recovered global bias matches MEASURED_FIX_BIAS_NED to
prove the sign chain.

2-CORNER PnP + BAYESIAN-IoU REFINEMENT (designed, not prototyped -- see ``design_two_corner_iou``)
------------------------------------------------------------------------------------------------
The robust-average re-survey above is the prototyped, validated path. A stronger extrinsic recovery
that does not need the full 4-corner detector -- a 2-corner-PnP + silhouette-IoU Bayesian refine --
is specified in ``design_two_corner_iou`` for the case where the detector emits only the two
high-confidence vertical posts at long range; it is OUT OF SCOPE to prototype here (no detector / no
live frames on the laptop) and is documented as the next build.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Live-code reuse (editable install; "import racer" works from repo root).
from racer.gate_mapper import (
    GateSighting,
    MapperConfig,
    estimate_map_pose_aided,
    robust_position_estimate,
    MEASURED_FIX_BIAS_NED,
    MEASURED_FIX_SIGMA_NED,
)

REPO = Path(__file__).resolve().parents[2]
CHAR_DIR = REPO / "handoff/perception-char-2026-06-08"
MAP_PATH = REPO / "handoff/shadowpc-firstcontact-2026-06-02/track_map.json"
BUNDLE_FRAMES = CHAR_DIR / "course_bundle/frames.json"


# ---------------------------------------------------------------------------
# Map: record bottom-centre -> opening centre (EXACTLY navigator.gates_from_track_records)
# ---------------------------------------------------------------------------
def map_opening_centres(map_path: str | Path = MAP_PATH) -> dict[int, np.ndarray]:
    """The world opening-centre per gate that the live navigator localizes against.

    navigator.gates_from_track_records(corner_to_center=True): centre = record.position_ned
    - 0.5*height * col2, where col2 is the quaternion's down-pointing (height) axis. This is
    the SAME lift gate_mapper._parse_prior_records uses; we replicate it standalone so the
    comparison reference is bit-identical to what flies."""
    mp = json.loads(Path(map_path).read_text())
    out: dict[int, np.ndarray] = {}
    for rec in mp["gates"]:
        h = float(rec.get("height_m") or 2.72)
        q = rec.get("orientation_ned_wxyz")
        pos = np.asarray(rec["position_ned"], dtype=np.float64)
        if q is not None:
            R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            col2 = R[:, 2] if R[:, 2][2] >= 0.0 else -R[:, 2]
            pos = pos - 0.5 * h * col2
        else:
            pos = pos - np.array([0.0, 0.0, 0.5 * h])
        out[int(rec["gate_id"])] = pos
    return out


# ---------------------------------------------------------------------------
# Load + pool the characterization rows into reconstructed gate-centre measurements
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CharRow:
    frame_id: int
    gate_id: int
    off_ned: np.ndarray
    range_err_m: float
    true_range_m: float
    n_corners: int
    world_fix_err_m: float
    reproj_px: float
    maha: float


def load_char_rows(char_dir: str | Path = CHAR_DIR) -> list[CharRow]:
    """Pool characterize_g{0..5}.json, dedup by frame_id (the 6 windows are disjoint but we
    dedup defensively), keep detected+associated rows. The ``gate_id`` is the ACTUALLY
    ASSOCIATED gate (often != the file's nominal gate), which is exactly what we want: each
    row is a measurement of whichever gate the chain locked onto."""
    char_dir = Path(char_dir)
    by_fid: dict[int, CharRow] = {}
    for g in range(6):
        d = json.loads((char_dir / f"characterize_g{g}.json").read_text())
        for r in d["rows"]:
            if not (r.get("associated") and r.get("gate_id") is not None):
                continue
            if "off_ned" not in r:
                continue
            by_fid[int(r["frame_id"])] = CharRow(
                frame_id=int(r["frame_id"]),
                gate_id=int(r["gate_id"]),
                off_ned=np.asarray(r["off_ned"], dtype=np.float64),
                range_err_m=float(r.get("range_err_m", np.nan)),
                true_range_m=float(r.get("true_range_m", np.nan)),
                n_corners=int(r.get("n_corners", 0)),
                world_fix_err_m=float(r.get("world_fix_err_m", np.nan)),
                reproj_px=float(r.get("reproj_px", np.nan)),
                maha=float(r.get("maha", np.nan)),
            )
    return list(by_fid.values())


CHI2_GATE = 16.27   # navigator.NavigatorConfig.vision_gate_chi2 (chi2_0.999, 3 DOF)


def rows_to_sightings(rows: list[CharRow],
                      centres: dict[int, np.ndarray],
                      four_corner_only: bool = True,
                      maha_gate: float | None = CHI2_GATE) -> list[GateSighting]:
    """Reconstruct the MAP-INDEPENDENT world gate-centre measurement per row.

    gate_meas = map_centre[g] - off_ned   (derivation in the module docstring). gate_id is the
    associated gate so estimate_map_pose_aided averages per-gate; the wrong-gate / depth-flip
    tail lands ~an inter-gate spacing off and dies in the robust rejection (same screen the live
    mapper uses). 4-corner-only by default (P3P fixes are geometrically weaker; characterize
    inflates their cov 9x live, and they are a small minority here).

    ``maha_gate`` (default chi2_0.999=16.27, the navigator's live innovation gate) pre-filters to
    the KF-ACCEPTED set -- the fixes the case-C KF would actually fuse. Without it the raw
    pre-gate set carries the wrong-gate/depth-flip tail (e.g. gate-3 has 13 fixes at 55-114 m
    true range mis-locked from far gates, N error +36..+99 m); the robust median/MAD trims the
    extreme tail but an ASYMMETRIC depth-flip tail (all negative range_err -> positive N) still
    biases the median by ~+1.5 m. The live chi2 gate rejects exactly that tail (maha >> 16.27),
    so re-surveying the KF-accepted set is the faithful estimate of the registration the live
    case-C KF experiences. Set ``maha_gate=None`` to reproduce the raw pre-gate behaviour."""
    out: list[GateSighting] = []
    for r in rows:
        if four_corner_only and r.n_corners != 4:
            continue
        if r.gate_id not in centres:
            continue
        if maha_gate is not None and np.isfinite(r.maha) and r.maha > maha_gate:
            continue
        gate_meas = centres[r.gate_id] - r.off_ned
        out.append(GateSighting(gate_world_ned=gate_meas, gate_id=r.gate_id,
                                t=float(r.frame_id), frame_id=r.frame_id,
                                range_m=r.true_range_m))
    return out


# ---------------------------------------------------------------------------
# The re-survey: run the SHIPPED robust averager, compare to map
# ---------------------------------------------------------------------------
@dataclass
class GateRegistration:
    gate_id: int
    map_centre: np.ndarray          # (3,) navigator opening-centre reference
    est_centre: np.ndarray          # (3,) re-surveyed world opening-centre
    offset_ned: np.ndarray          # (3,) est - map  (registration error; +D = true gate lower)
    offset_std_ned: np.ndarray      # (3,) 1-sigma uncertainty of the OFFSET (mean uncertainty)
    n_sightings: int
    n_used: int
    flags: tuple[str, ...]
    # residual registration sigma the case-C KF must carry in R for this gate (see resolve below)
    residual_reg_sigma_ned: np.ndarray | None = None


def resurvey(char_dir: str | Path = CHAR_DIR,
             map_path: str | Path = MAP_PATH,
             config: MapperConfig | None = None,
             four_corner_only: bool = True,
             maha_gate: float | None = CHI2_GATE) -> tuple[list[GateRegistration], dict]:
    """Re-survey every gate from the calibration-lap fixes; compare to the live map.

    Returns ``(registrations, diagnostics)``. Uses ``estimate_map_pose_aided`` verbatim so the
    robust median/MAD rejection is the live mapper's. The per-gate ``pos_cov`` it returns is the
    statistical covariance of the inlier MEAN (sigma_hat^2 / n_in), i.e. the uncertainty of the
    re-surveyed centre == the uncertainty of the offset estimate.

    ``maha_gate`` pre-filters to the KF-accepted set before averaging (see rows_to_sightings)."""
    cfg = config or MapperConfig()
    centres = map_opening_centres(map_path)
    rows = load_char_rows(char_dir)
    sightings = rows_to_sightings(rows, centres, four_corner_only=four_corner_only,
                                  maha_gate=maha_gate)
    est = estimate_map_pose_aided(sightings, prior_records=None, config=cfg)
    est_by_id = {g.gate_id: g for g in est.gates}

    regs: list[GateRegistration] = []
    for gid in sorted(centres):
        mc = centres[gid]
        if gid not in est_by_id:
            regs.append(GateRegistration(
                gate_id=gid, map_centre=mc, est_centre=mc.copy(),
                offset_ned=np.zeros(3), offset_std_ned=np.full(3, np.nan),
                n_sightings=0, n_used=0, flags=("no_sightings",)))
            continue
        g = est_by_id[gid]
        offset = g.position_ned - mc
        off_std = np.sqrt(np.clip(np.diag(g.pos_cov), 0.0, None))
        regs.append(GateRegistration(
            gate_id=gid, map_centre=mc, est_centre=g.position_ned,
            offset_ned=offset, offset_std_ned=off_std,
            n_sightings=g.n_sightings, n_used=g.n_used, flags=g.flags))

    diag = {
        "n_rows_pooled": len(rows),
        "n_sightings_used": len(sightings),
        "four_corner_only": four_corner_only,
        "maha_gate": maha_gate,
        "mapper_diag": est.diagnostics,
    }
    return regs, diag


# ---------------------------------------------------------------------------
# Depth-bias vs true-registration decomposition (adversarial check 2)
# ---------------------------------------------------------------------------
def depth_decomposition(char_dir: str | Path = CHAR_DIR,
                        map_path: str | Path = MAP_PATH) -> dict[int, dict]:
    """Split each gate's mean fix offset into a RADIAL (depth-bias-explained) part and a
    residual (true-registration) part, using GT pose from course_bundle/frames.json.

    PnP depth error (``range_err_m`` = pose_range - true_range) lies along the sighting RAY.
    Project mean(off_ned) onto the mean world ray direction (gate_centre - drone)/||.||: the
    radial component is what a depth bias could account for; the perpendicular residual is
    genuine registration (or non-radial systematics). On this course the rays are ~horizontal
    (mostly -N), so depth bleeds into N, NOT D -- which is why the D offset cannot be a depth
    artifact."""
    centres = map_opening_centres(map_path)
    frames = {f["frame_id"]: f for f in
              json.loads(Path(BUNDLE_FRAMES).read_text())["frames"]}
    rows = load_char_rows(char_dir)
    out: dict[int, dict] = {}
    for gid in sorted(centres):
        sel = [r for r in rows if r.gate_id == gid and r.n_corners == 4
               and r.world_fix_err_m < 3.0 and r.frame_id in frames]
        if len(sel) < 3:
            out[gid] = {"n": len(sel), "note": "insufficient GT-pose fixes"}
            continue
        offs, drones = [], []
        for r in sel:
            offs.append(r.off_ned)
            drones.append(np.asarray(frames[r.frame_id]["drone_position_ned"], dtype=np.float64))
        offs = np.array(offs)
        mean_off = offs.mean(axis=0)
        ray = centres[gid] - np.array(drones).mean(axis=0)
        ray = ray / np.linalg.norm(ray)
        # per-fix radial component, then mean (more honest than projecting the mean onto a mean ray)
        radial = float(np.mean([float(o @ ray) for o in offs]))
        depth_explained = radial * ray
        residual = mean_off - depth_explained
        out[gid] = {
            "n": len(sel),
            "mean_off_ned": mean_off,
            "ray_world": ray,
            "mean_radial_m": radial,
            "depth_explained_ned": depth_explained,
            "residual_reg_ned": residual,
            "residual_reg_norm_m": float(np.linalg.norm(residual)),
            "residual_D_m": float(residual[2]),
        }
    return out


# ---------------------------------------------------------------------------
# Lateral offset: constant (true map registration) vs range-slope (bearing calibration)
# ---------------------------------------------------------------------------
def lateral_range_decomposition(gid: int,
                                axis_ned: int,
                                char_dir: str | Path = CHAR_DIR,
                                map_path: str | Path = MAP_PATH,
                                maha_gate: float | None = CHI2_GATE) -> dict:
    """Fit per-gate lateral offset vs range: ``off_axis = intercept + slope * range``.

    A CONSTANT map mis-registration is range-INDEPENDENT (slope ~ 0, intercept = the offset).
    A BEARING/yaw calibration bias produces a lateral error that GROWS with range
    (lateral ~ theta * range, so a non-zero slope; slope in rad ~ the bearing bias). Separating
    them matters for the inc8 gate-4 margin: the drone TRANSITS gate-4 at SHORT range, so a
    pure bearing-slope offset is small at the margin-relevant moment, whereas a constant
    registration error bites at every range. ``axis_ned``: 0=N,1=E,2=D (use the gate's lateral
    axis -- E for the -X-facing course gates)."""
    centres = map_opening_centres(map_path)
    rows = load_char_rows(char_dir)
    # Gate on a small depth error too: beyond ~lock-loss range the PnP depth FLIPS (range_err
    # large negative), which also corrupts the lateral component (the flipped pose mislocates
    # laterally). A clean bearing-slope fit needs depth-consistent fixes only.
    sel = [r for r in rows if r.gate_id == gid and r.n_corners == 4
           and (maha_gate is None or not np.isfinite(r.maha) or r.maha <= maha_gate)
           and abs(r.range_err_m) < 1.5]
    if len(sel) < 4:
        return {"gid": gid, "n": len(sel), "note": "insufficient depth-consistent fixes"}
    rng = np.array([r.true_range_m for r in sel])
    off = np.array([(centres[gid][axis_ned] - r.off_ned[axis_ned]) - centres[gid][axis_ned]
                    for r in sel])   # est_centre_axis - map_centre_axis = -off_ned[axis]
    slope, intc = np.polyfit(rng, off, 1)
    # short-range (transit-relevant) prediction at 5 m vs the intercept
    off_at_5m = intc + slope * 5.0
    return {
        "gid": gid, "n": len(sel), "axis_ned": axis_ned,
        "intercept_m": float(intc),                 # range-extrapolated CONSTANT (map-reg candidate)
        "slope_m_per_m": float(slope),              # bearing-bias signature
        "slope_deg": float(np.degrees(np.arctan(slope))),
        "off_at_5m_m": float(off_at_5m),            # offset at the transit-relevant short range
        "mean_off_m": float(off.mean()),
        "range_min_max": (float(rng.min()), float(rng.max())),
    }


# ---------------------------------------------------------------------------
# Residual registration sigma the case-C KF must carry in R
# ---------------------------------------------------------------------------
def residual_registration_sigma(regs: list[GateRegistration]) -> dict:
    """Per-gate residual registration sigma the case-C KF must add (per axis) to R.

    Two regimes:
      * CALIBRATED gates (we have a re-survey AND a live cross-check): residual = the offset
        ESTIMATION uncertainty (offset_std_ned). After applying the calibrated offset, what
        remains is how well we KNOW the offset -- small.
      * UNCALIBRATED / no-live-cross-check gates (inc8 gate-4, gate-5): we cannot trust the
        re-survey as ground truth (it shares the chain's systematics), so the case-C KF must
        carry the FULL apparent per-gate offset spread as a registration sigma -- i.e. the
        across-gate scatter of the offsets (the part that does NOT cancel as a global bias)
        added in quadrature to the per-gate estimation uncertainty.

    We report both, plus the global-bias-removed per-gate residual (the genuinely per-gate,
    non-cancellable registration component)."""
    have = [r for r in regs if r.n_used >= 1 and np.all(np.isfinite(r.offset_std_ned))]
    offsets = np.array([r.offset_ned for r in have])         # (G,3)
    global_bias = offsets.mean(axis=0)                       # cancels in a global frame shift
    per_gate_resid = offsets - global_bias                   # genuinely per-gate
    across_gate_sigma = per_gate_resid.std(axis=0, ddof=1)   # (3,) scatter that does NOT cancel
    out = {
        "global_bias_ned": global_bias,
        "across_gate_residual_sigma_ned": across_gate_sigma,
        "per_gate": {},
    }
    for r in have:
        est_unc = r.offset_std_ned
        # uncalibrated residual: cannot remove the per-gate part with confidence -> carry it
        uncal = np.sqrt(per_gate_resid[[x.gate_id for x in have].index(r.gate_id)]**2 + est_unc**2)
        out["per_gate"][r.gate_id] = {
            "offset_ned": r.offset_ned,
            "offset_est_sigma_ned": est_unc,            # calibrated regime residual
            "per_gate_resid_ned": per_gate_resid[[x.gate_id for x in have].index(r.gate_id)],
            "uncalibrated_residual_sigma_ned": uncal,   # no-live-cross-check regime
        }
    return out


# ---------------------------------------------------------------------------
# DESIGN ONLY: 2-corner PnP + Bayesian-IoU extrinsic refinement
# ---------------------------------------------------------------------------
def design_two_corner_iou() -> str:
    """Specification for the stronger extrinsic recovery (NOT prototyped -- no detector/frames
    on laptop). Returned as a string so it travels with the module.

    PROBLEM. At long range / oblique view the detector reliably emits only the two vertical
    posts (4 corners collapse to 2 high-confidence + 2 low). A 4-corner IPPE/P3P is then
    ill-conditioned in depth. But the gate is a known-size planar square with a known world
    orientation (all gates face -X, from the map quat), and on the CALIBRATION lap the drone
    pose+attitude are GIVEN -- so the only unknown for re-survey is the gate's 3-DoF world
    translation (its yaw is the map's; its size is the known 1.5 m inner square).

    STEP 1 -- 2-corner metric PnP for an initial centre. With attitude given, two image points
    of two known 3D gate points (e.g. top-left & bottom-left posts at +-0.75 m in the gate
    plane) over-determine a 2-DoF bearing + give a weak 1-DoF range from their pixel separation
    s_px ~ f * S_post / depth (S_post = 1.5 m vertical post). depth = f * S_post / s_px;
    centre0 = drone + depth * R_wc @ K^-1 @ [u_mid, v_mid, 1]_normalised. Depth from a 2-corner
    baseline is noisy (sigma_depth/depth ~ sigma_px * depth / (f * S_post)), so this is only a
    SEED.

    STEP 2 -- Bayesian-IoU refine over the 3-DoF translation. Render the known gate square
    (1.5 m, map yaw, candidate centre) through the FULL projector (frames.project_gate_corners
    / project_camera_point -- the exact inverse of the live PnP) to a predicted silhouette;
    score against the detector's mask/box by IoU. Posterior:
        log p(centre | obs) = w_iou * log IoU(render(centre), det_mask)
                              + w_reproj * (-sum ||proj(corner_i) - det_corner_i||^2 / sigma_px^2)
                              + log N(centre ; centre0, Sigma_2corner)        # PnP-seed prior
                              + log N(centre ; map_centre, Sigma_map)         # map prior (case B)
    Maximise over centre (3 DoF) by a few Gauss-Newton steps on the reproj+prior terms with the
    IoU term as a coarse-to-fine bracket (IoU is non-smooth; use it to pick the basin, then GN).
    The IoU term disambiguates the depth that 2 corners leave weak: a too-near/too-far centre
    renders a too-big/too-small square -> IoU drops sharply even when the 2 visible corners still
    reproject well. Marginal covariance from the GN Hessian at the optimum gives the per-gate
    re-survey sigma directly (replaces the statistical MAD covariance used in the prototyped
    path), and the radial (depth) direction gets its OWN sigma from the IoU curvature -- exactly
    the range-anisotropy piece C wants.

    WHY BAYESIAN. The PnP seed, the map prior, and the IoU likelihood are fused by their
    covariances, so a gate seen only twice (weak depth) is pulled toward the map prior, while a
    gate seen 50x at varied range is pulled toward the data -- no hand-tuned switch. The output
    is a centre + full 3x3 covariance per gate, anisotropic in range, feeding R for the case-C KF.

    VALIDATION PLAN (ShadowPC, has detector+frames): run on the course_bundle PNGs, compare the
    IoU-refined centres to (a) the prototyped off_ned re-survey here and (b) the map; expect the
    radial sigma to shrink vs the pure-statistical estimate and the centres to agree within the
    reported sigma. Acceptance: gate-3 D offset stays < 0.4 m (no spurious 1.46 m), and the
    range-anisotropic sigma is < the isotropic FIX_COV_FLOOR (0.40 m) tangentially at mid-range.
    """
    return design_two_corner_iou.__doc__ or ""


if __name__ == "__main__":
    regs, diag = resurvey()
    print("diag:", json.dumps({k: v for k, v in diag.items() if k != "mapper_diag"}, default=str))
    for r in regs:
        print(f"g{r.gate_id}: offset_NED={np.round(r.offset_ned,3)} "
              f"+-{np.round(r.offset_std_ned,3)}  n_used={r.n_used} flags={r.flags}")
