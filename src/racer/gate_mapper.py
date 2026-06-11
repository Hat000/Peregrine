"""Offline gate mapper — build or refine the gate map from logged detections, between attempts.

VQ2 may hand us (A) pose + exact map, (B) pose + ROUGH map, or (C) no pose at all
(vision-only). This module is the offline infrastructure for B and C: after a conservative
exploration lap, turn the logged gate sightings into a track map the navigator can consume
exactly like the given one (``to_track_records`` emits ``capture_track_map.py``-schema
records, so ``navigator.load_track_map`` reads our output unchanged). Pure offline: numpy +
scipy only, no live deps, no imports from the live navigator/localization/state-estimator
(those are owned by the live loop; this module sits behind the same frozen map schema).

Three estimation modes
----------------------
* **Pose-aided (cases A/B)** — ``estimate_map_pose_aided``. Drone pose per detection is
  trusted, so each sighting is an implied world gate-centre measurement
  ``gate_meas = drone_pos + R_world_camera @ t_cam_gate``. Per gate: iterative
  median/MAD-rejection robust averaging (sized to the measured ~1.1% catastrophic leak AND
  the 5-15% wrong-gate association rate, which appears as huge structured outliers ~the
  inter-gate spacing), then an inlier mean + statistical covariance. With a prior map
  (case B) the robust estimate is fused per-axis with configurable prior trust
  (``prior_sigma_pos`` / ``prior_sigma_yaw_rad``), and prior-only gates pass through flagged.
* **No-pose (case C)** — ``estimate_map_no_pose``. Drone positions are unknown; attitude is
  GIVEN (IMU/sim quat), so each sighting is a world-frame lever ``L = R(q) @ rel_frd`` and
  the residual ``p_gate - x_pose - L`` is LINEAR in all unknowns. The whole adjustment is a
  sparse linear least-squares (exact constant Jacobian) with a robust loss — no SO(3)
  manifold, no relinearisation. Metric scale is inherent (PnP against the KNOWN 1.5 m inner
  square makes ``rel_frd`` metric); the gauge is 3 translation DoF, anchored by fixing the
  first pose. Gate YAW decouples completely (attitude given => each rel-yaw is a direct
  gate-yaw measurement) and is solved by the same robust circular estimator as cases A/B.
  Association is geometry-driven (incremental spatial chaining); upstream labels, when
  present, are used only to NAME the output gates (majority vote) — never to attach
  geometry, so a wrongly-labeled sighting still lands on the gate it actually saw.

Conditioning (case C) — read before trusting an output map
----------------------------------------------------------
Without odometry, poses link to each other ONLY through co-visible gates. On the VQ1 course
the inter-gate spacing is 23.7-38.5 m vs a ~24-32 m usable detection range, so consecutive
gates are co-visible only briefly (and the 38.5 m g2->g3 leg possibly never). If the
frame<->gate graph splits, the relative placement of the components is UNOBSERVABLE; the
optimiser still returns numbers (each component anchored at its dead-reckoned init), so the
output flags every gate outside the anchor component (``disconnected``) and the diagnostics
carry the component count + the normal-matrix conditioning. Two mitigations are built in:
an optional constant-acceleration smoothness prior between consecutive poses
(``accel_prior_sigma``, motion-honest: we know the platform's accel envelope; it bridges
short co-visibility gaps at the cost of soft, drift-prone linkage) and multi-lap sightings
(more sightings per gate, but the SAME co-visibility pattern — laps do not fix a broken
graph). If attitude ever has to float (real-IMU yaw drift), the problem stops being linear
and a proper factor-graph engine (GTSAM: SO(3) manifold + IMU preintegration + iSAM2) is
the upgrade; at <=10 landmarks and a few hundred poses, scipy is comfortably sufficient.

Noise model provenance
----------------------
All defaults come from the measured perception characterization on the canonical 6/6 VQ1
course recording (handoff/perception-char-2026-06-08, N=312 solved fixes): KF-accepted
world-fix error bias [-0.42, +0.06, -0.28] m (N,E,D), sigma [0.73, 0.47, 0.29] m,
range-flat to ~24 m, ~+-3deg angular term, ~1.1-1.6% catastrophic leak (3-5+ m) after the
chi2 gate, association reliability ~85-95% at 5-15 m. NB the SIGN: those are FIX-error
stats (fix = gate_true - lever), and a gate measurement is ``drone_pos + lever``, so the
gate-measurement error is MINUS the fix error — ``bias_correction_ned`` therefore ADDS the
measured fix bias to gate measurements / levers (see ``MEASURED_FIX_BIAS_NED``). The bias
was measured on one course flown in one direction; treat correction as an opt-in
calibration (default OFF), not a constant of nature.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# --- measured perception-noise constants [handoff/perception-char-2026-06-08, KF-accepted set] ---
MEASURED_FIX_SIGMA_NED = np.array([0.73, 0.47, 0.29])    # m, 1-sigma world-fix error (N,E,D)
MEASURED_FIX_BIAS_NED = np.array([-0.42, +0.06, -0.28])  # m, mean world-fix error (N,E,D)
MEASURED_YAW_SIGMA_RAD = float(np.deg2rad(3.0))          # the ~+-3deg angular/calibration term
MEASURED_CATASTROPHIC_LEAK = 0.011                       # residual bad-fix rate after the chi2 gate

GATE_OUTER_SIZE_M = 2.72   # outer square; track records' width_m/height_m (capture 2026-06-02)


def wrap_pi(a):
    """Wrap angle(s) to (-pi, pi]."""
    return -(np.mod(-np.asarray(a, dtype=np.float64) + np.pi, 2.0 * np.pi) - np.pi)


def _yaw_from_quat_wxyz(q_wxyz) -> float:
    """Body->world (w,x,y,z) quaternion -> aerospace ZYX yaw (rad)."""
    w, x, y, z = (float(v) for v in q_wxyz)
    R = Rotation.from_quat([x, y, z, w]).as_matrix()
    return float(np.arctan2(R[1, 0], R[0, 0]))


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class GateSighting:
    """One pose-aided gate sighting (cases A/B): the implied world gate-centre measurement.

    ``gate_world_ned = drone_position_ned + R_world_camera @ t_cam_gate`` — i.e. the trusted
    pose plus the PnP lever, computed by the caller (the live chain already has both sides).
    ``yaw_world`` is the measured gate through-yaw (rad, world NED, optional); ``gate_id`` is
    the upstream association label (optional — clustering mode ignores/replaces labels).
    """

    gate_world_ned: np.ndarray
    gate_id: int | None = None
    yaw_world: float | None = None
    t: float = 0.0
    frame_id: int | None = None
    range_m: float | None = None


@dataclass(frozen=True, eq=False)
class RelativeGateSighting:
    """One no-pose gate sighting (case C): the gate centre RELATIVE to the drone.

    ``rel_position_frd`` is the gate centre in the body FRD frame at sighting time (the PnP
    camera-frame translation rotated through the camera mount — metric, thanks to the known
    1.5 m inner square). ``quat_wxyz`` is the GIVEN body->world attitude. ``rel_yaw`` is
    ``wrap(gate_yaw_world - drone_yaw)`` (optional). ``gate_id`` is an upstream tracking
    label if one exists (used for output NAMING only); geometry association is the mapper's.
    """

    frame_id: int
    rel_position_frd: np.ndarray
    quat_wxyz: np.ndarray
    gate_id: int | None = None
    rel_yaw: float | None = None
    t: float = 0.0

    def lever_world(self) -> np.ndarray:
        """Gate centre relative to the drone, world NED (rotate FRD lever via the given quat)."""
        q = np.asarray(self.quat_wxyz, dtype=np.float64)
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        return R @ np.asarray(self.rel_position_frd, dtype=np.float64)


# ---------------------------------------------------------------------------
# Config / outputs
# ---------------------------------------------------------------------------
@dataclass
class MapperConfig:
    """Knobs, defaulted to the MEASURED noise model (see module docstring for provenance)."""

    sigma_ned: np.ndarray = field(default_factory=lambda: MEASURED_FIX_SIGMA_NED.copy())
    yaw_sigma_rad: float = MEASURED_YAW_SIGMA_RAD
    # Outlier rejection: normalized 3D residual norm threshold. 4.0 ~ sqrt(chi2_3,0.999) —
    # the same 99.9% convention as the live KF innovation gate; catches the >=3 m leak tail
    # (a 3 m N-only error is 4.1 sigma_N) and the wrong-gate ~24 m structured outliers.
    reject_norm_k: float = 4.0
    reject_yaw_k: float = 4.0
    robust_iters: int = 4
    min_sightings: int = 3
    # OPT-IN measured-calibration bias removal: ADDS this (the measured FIX-error bias, e.g.
    # MEASURED_FIX_BIAS_NED) to every gate measurement / lever. Off by default: the bias was
    # measured on one course+direction and is not established as a constant of nature.
    bias_correction_ned: np.ndarray | None = None
    # Residual per-gate systematics no averaging removes (per-gate offsets ~0.1-0.2 m, the
    # uncorrected course bias, yaw roll-wander): added in quadrature to the statistical cov
    # for B-mode fusion weights and reported. Kin to localization.FIX_COV_FLOOR_STD (0.40 m
    # per FIX); after averaging the per-gate constant part remains -> 0.25 m default.
    map_systematic_std: float = 0.25
    # Unlabeled pose-aided clustering: new cluster if no existing centre within this radius.
    # Must sit between the noise tail (~3 sigma ~ 2.2 m) and the min inter-gate spacing (23.7 m).
    cluster_radius_m: float = 5.0
    # B-mode prior trust (1-sigma). 2 m / ~15 deg ~ a "rough map" per the competition FAQ risk.
    prior_sigma_pos: float = 2.0
    prior_sigma_yaw_rad: float = float(np.deg2rad(15.0))
    # --- case C ---
    # Constant-acceleration smoothness prior between consecutive poses (m/s^2, 1-sigma).
    # This is the link that carries the map across blind transit gaps (true co-visibility
    # of consecutive gates is geometrically IMPOSSIBLE on the VQ1 course: both inside
    # detection range only between them, where one is behind the forward camera). 1.0 is
    # honest FOR THE EXPLORATION LAP — we fly it deliberately smooth at near-constant
    # velocity — while the corner impulses at gates land where observations dominate and
    # the robust loss absorbs them. None disables (pure observation graph; expect
    # disconnected components on this course).
    accel_prior_sigma: float | None = 1.0
    # Post-solve duplicate-gate merge radius. A gate first seen far out can be re-FOUNDED
    # after a blind bridge gap (dead-reckoned prediction misses assoc_radius_m); after the
    # adjustment the two copies coincide while real gates sit >=23.7 m apart, so anything
    # inside this radius is one physical gate (merged, then one re-solve).
    dup_merge_radius_m: float = 8.0
    # Virtual coast poses inserted into detection gaps (> 2.5x this spacing) so the
    # smoothness prior chains THROUGH the gap instead of extrapolating one noise-amplified
    # instantaneous velocity across it (a raw second-difference factor spanning a 2 s gap
    # differentiates two 0.07 s-apart noisy positions and projects that velocity over the
    # whole gap — measured ~10-16 m/leg drift; chained virtuals average the entry velocity
    # over the whole pre-gap window). Only active when the smoothness prior is enabled.
    virtual_frame_dt: float | None = 0.25
    assoc_radius_m: float = 5.0     # case-C incremental association / implied-position sanity
    robust_loss: str = "soft_l1"    # scipy least_squares loss; residuals are whitened (sigma units)
    f_scale: float = 2.0
    max_poses: int = 800            # uniform frame decimation cap (keeps the dense solve bounded)
    compute_covariance: bool = True  # case C: dense (J^T J)^-1 gate marginals (skipped if huge)


@dataclass
class MappedGate:
    """One estimated gate: opening-centre position + through-yaw, with honesty attached."""

    gate_id: int
    position_ned: np.ndarray          # (3,) OPENING CENTRE, world NED
    yaw: float | None                 # through-direction yaw (rad, world), None if unobserved
    pos_cov: np.ndarray               # (3,3) statistical covariance of position_ned
    yaw_std: float | None
    n_sightings: int
    n_used: int
    flags: tuple[str, ...] = ()


@dataclass
class GateMapEstimate:
    """The mapper's output: ordered gates + diagnostics, serialisable to the live map schema."""

    gates: list[MappedGate]
    diagnostics: dict

    def to_track_records(self, width_m: float = GATE_OUTER_SIZE_M,
                         height_m: float = GATE_OUTER_SIZE_M) -> list[dict]:
        """Capture-schema records (``capture_track_map.py``): gates assumed upright, so the
        record position is the BOTTOM-centre (opening centre pushed down-axis by height/2)
        and the quaternion is authored with col0=+width, col1=normal, col2=+height(down) —
        the exact convention ``navigator.gates_from_track_records(corner_to_center=True)``
        inverts, so our output round-trips through the live map loader unchanged."""
        records = []
        for g in self.gates:
            psi = 0.0 if g.yaw is None else float(g.yaw)
            n = np.array([np.cos(psi), np.sin(psi), 0.0])      # col1: gate normal (through-dir)
            d = np.array([0.0, 0.0, 1.0])                      # col2: height axis, down
            R = np.column_stack([np.cross(n, d), n, d])        # col0 = col1 x col2 (right-handed)
            q_xyzw = Rotation.from_matrix(R).as_quat()
            records.append({
                "gate_id": int(g.gate_id),
                "position_ned": [float(v) for v in (g.position_ned + 0.5 * height_m * d)],
                "orientation_ned_wxyz": [float(q_xyzw[3]), float(q_xyzw[0]),
                                         float(q_xyzw[1]), float(q_xyzw[2])],
                "width_m": float(width_m),
                "height_m": float(height_m),
            })
        return records

    def save_track_map_json(self, path: str | Path, note: str = "") -> Path:
        """Write a ``track_map_*.json``-compatible file (plus a mapper provenance block)."""
        records = self.to_track_records()
        payload = {
            "schema": "racer.track_map/mapper",
            "num_gates": len(records),
            "dims_note": "width/height are the OUTER gate square (~2.72 m); "
                         "inner opening ~1.5 m is what PnP uses",
            "mapper": {"note": note, "diagnostics": _json_safe(self.diagnostics),
                       "per_gate": [{
                           "gate_id": g.gate_id,
                           "pos_std_ned": [float(v) for v in np.sqrt(np.diag(g.pos_cov))],
                           "yaw_std_deg": None if g.yaw_std is None else float(np.rad2deg(g.yaw_std)),
                           "n_sightings": g.n_sightings, "n_used": g.n_used,
                           "flags": list(g.flags)} for g in self.gates]},
            "gates": records,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2))
        return p


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


# ---------------------------------------------------------------------------
# Robust primitives (shared by all cases)
# ---------------------------------------------------------------------------
def robust_position_estimate(points: np.ndarray, sigma_ned: np.ndarray,
                             k: float = 4.0, iters: int = 4,
                             seed: np.ndarray | None = None,
                             seed_extra_var: float = 0.0,
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Iterative median/MAD-rejection mean of 3D points.

    ``seed`` (e.g. a prior gate position) replaces the iteration-0 centre; its trust enters
    the first rejection radius via ``seed_extra_var`` (added per-axis, m^2) so an offset
    prior cannot reject all the good sightings. Rejection is on the NORMALIZED 3D residual
    norm vs ``k`` (chi2-style, matching the live gate convention). The per-axis scale is the
    1.4826*MAD estimate clipped to [0.3, 3]x the supplied sigma (adapts to a noisier-than-
    expected flight without collapsing on tiny clusters).

    Returns ``(mean, cov_stat, inlier_mask, sigma_hat)``; cov_stat = diag(sigma_hat^2)/n_in.
    For n < 3 the median is returned with the supplied sigma as the (no-averaging) scale.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    sigma_ned = np.asarray(sigma_ned, dtype=np.float64)
    n = len(pts)
    if n == 0:
        raise ValueError("robust_position_estimate needs at least one point")
    if n < 3:
        centre = np.median(pts, axis=0)
        return centre, np.diag(sigma_ned**2), np.ones(n, dtype=bool), sigma_ned.copy()
    centre = seed.astype(np.float64).copy() if seed is not None else np.median(pts, axis=0)
    sigma_hat = sigma_ned.copy()
    mask = np.ones(n, dtype=bool)
    for it in range(max(1, iters)):
        extra = seed_extra_var if (seed is not None and it == 0) else 0.0
        scale = np.sqrt(sigma_hat**2 + extra)
        r = (pts - centre) / scale
        new_mask = np.linalg.norm(r, axis=1) <= k
        if not new_mask.any():                       # pathological: keep the closest point
            new_mask[np.argmin(np.linalg.norm(r, axis=1))] = True
        sub = pts[new_mask]
        centre = np.median(sub, axis=0)
        mad = np.median(np.abs(sub - centre), axis=0) * 1.4826
        sigma_hat = np.clip(mad, 0.3 * sigma_ned, 3.0 * sigma_ned)
        if np.array_equal(new_mask, mask) and it > 0:
            break
        mask = new_mask
    inliers = pts[mask]
    mean = inliers.mean(axis=0)
    # Empirical per-axis std of the inliers, floored by the (clipped) MAD scale so a lucky
    # tight cluster doesn't report near-zero uncertainty.
    emp = inliers.std(axis=0, ddof=1) if len(inliers) >= 3 else sigma_hat
    sigma_used = np.maximum(emp, 0.5 * sigma_hat)
    cov = np.diag(sigma_used**2) / max(len(inliers), 1)
    return mean, cov, mask, sigma_used


def robust_yaw_estimate(yaws: np.ndarray, sigma_rad: float, k: float = 4.0,
                        ) -> tuple[float | None, float | None, np.ndarray]:
    """Robust circular mean: circular median seed -> MAD rejection -> mean direction.

    Handles wrap at +-pi and kills the catastrophic-flip tail (90/180 deg PnP face/branch
    errors are >> k*sigma for any plausible sigma). Returns ``(yaw, yaw_std, inlier_mask)``;
    (None, None, empty mask) when no finite yaw measurements were supplied.
    """
    a = np.asarray([y for y in np.atleast_1d(yaws) if y is not None and np.isfinite(y)],
                   dtype=np.float64)
    if a.size == 0:
        return None, None, np.zeros(0, dtype=bool)
    if a.size == 1:
        return float(wrap_pi(a[0])), sigma_rad, np.ones(1, dtype=bool)
    # circular median: the sample minimizing the summed wrapped absolute deviation
    dev = np.abs(wrap_pi(a[:, None] - a[None, :])).sum(axis=0)
    med = a[int(np.argmin(dev))]
    r = wrap_pi(a - med)
    mad = np.median(np.abs(r)) * 1.4826
    scale = float(np.clip(mad, 0.3 * sigma_rad, 3.0 * sigma_rad))
    mask = np.abs(r) <= k * scale
    if not mask.any():
        mask[np.argmin(np.abs(r))] = True
    inl = a[mask]
    mean = med + float(np.arctan2(np.mean(np.sin(wrap_pi(inl - med))),
                                  np.mean(np.cos(wrap_pi(inl - med)))))
    resid = wrap_pi(inl - mean)
    emp = float(resid.std(ddof=1)) if inl.size >= 3 else scale
    std = max(emp, 0.5 * scale) / np.sqrt(max(inl.size, 1))
    return float(wrap_pi(mean)), float(std), mask


def cluster_sightings(points: np.ndarray, order: np.ndarray, radius_m: float) -> np.ndarray:
    """Greedy online clustering of implied gate positions (unlabeled pose-aided mode).

    Processes points in ``order`` (ascending time, so clusters get course-ordered ids);
    a point joins the nearest existing cluster MEDIAN within ``radius_m``, else starts a new
    cluster. Outliers (leak tail, 3-8 m) spawn small spurious clusters by design — the
    caller drops clusters below ``min_sightings``. Valid because the noise tail (~3 sigma
    ~2 m) and the min inter-gate spacing (23.7 m) are separated by an order of magnitude.
    Returns per-point integer labels (cluster id in first-seen order).
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    labels = np.full(len(pts), -1, dtype=int)
    centres: list[np.ndarray] = []
    members: list[list[int]] = []
    for i in order:
        p = pts[i]
        if centres:
            d = np.linalg.norm(np.asarray(centres) - p, axis=1)
            j = int(np.argmin(d))
            if d[j] <= radius_m:
                labels[i] = j
                members[j].append(i)
                if len(members[j]) <= 50 or len(members[j]) % 10 == 0:  # cheap running median
                    centres[j] = np.median(pts[members[j]], axis=0)
                continue
        centres.append(p.copy())
        members.append([i])
        labels[i] = len(centres) - 1
    return labels


# ---------------------------------------------------------------------------
# Cases A / B — pose-aided robust averaging (+ optional prior fusion)
# ---------------------------------------------------------------------------
def estimate_map_pose_aided(sightings: list[GateSighting],
                            prior_records: list[dict] | None = None,
                            config: MapperConfig | None = None) -> GateMapEstimate:
    """Robust per-gate averaging of pose-aided sightings; optional rough-prior fusion (B).

    Association: if every sighting carries a ``gate_id`` those labels are used (wrong-label
    sightings land ~an inter-gate spacing away and die in rejection); otherwise sightings
    are clustered spatially and clusters take course order (first-seen time). With a prior,
    clusters/labels are matched to prior gates and gates the lap never saw pass through
    prior-only (flagged); the fusion is per-axis information weighting with the measurement
    variance = statistical + ``map_systematic_std``^2.
    """
    cfg = config or MapperConfig()
    if not sightings:
        if prior_records is None:
            return GateMapEstimate(gates=[], diagnostics={"n_sightings": 0, "mode": "pose_aided"})
        prior = _parse_prior_records(prior_records)
        gates = [_prior_only_gate_from_parsed(prior, i, cfg) for i in range(len(prior["ids"]))]
        return GateMapEstimate(gates=gates, diagnostics={"n_sightings": 0, "mode": "pose_aided",
                                                         "note": "no sightings: prior passthrough"})

    pts = np.array([np.asarray(s.gate_world_ned, dtype=np.float64) for s in sightings])
    if cfg.bias_correction_ned is not None:
        pts = pts + np.asarray(cfg.bias_correction_ned, dtype=np.float64)
    times = np.array([s.t for s in sightings], dtype=np.float64)

    labeled = all(s.gate_id is not None for s in sightings)
    n_label_groups_merged = 0
    if labeled:
        labels = np.array([int(s.gate_id) for s in sightings])
        association = "labels"
        # Geometry cross-check: a label group whose robust centre coincides with a BIGGER
        # group's is not a gate — it is that gate's mislabel debris. With partial coverage
        # a yet-unseen gate's label can consist 100% of neighbour mislabels: a tight
        # phantom at the wrong gate that no outlier screen can catch from the inside
        # (measured 24-40 m map errors on quarter-lap sweeps). Real gates are >=23.7 m
        # apart, so coincidence within the merge radius identifies one physical gate.
        meds = {int(u): np.median(pts[labels == u], axis=0) for u in np.unique(labels)}
        sizes = {u: int((labels == u).sum()) for u in meds}
        for u in sorted(meds, key=lambda k: sizes[k]):
            others = [v for v in meds if v != u and sizes[v] >= sizes[u]]
            if not others:
                continue
            d = [np.linalg.norm(meds[u] - meds[v]) for v in others]
            j = int(np.argmin(d))
            if d[j] <= cfg.dup_merge_radius_m:
                labels[labels == u] = others[j]
                sizes[others[j]] += sizes.pop(u)
                meds.pop(u)
                n_label_groups_merged += 1
    else:
        labels = cluster_sightings(pts, np.argsort(times, kind="stable"), cfg.cluster_radius_m)
        association = "cluster"

    prior = _parse_prior_records(prior_records) if prior_records else None
    label_to_prior: dict[int, int] = {}
    if prior is not None:
        label_to_prior = _match_labels_to_prior(labels, pts, prior, cfg, by_id=labeled)

    gates: list[MappedGate] = []
    n_rejected = 0
    dropped_clusters = 0
    uniq = _ordered_labels(labels, times)
    next_free_id = (max(prior["ids"]) + 1) if prior is not None else 0
    used_prior_idx: set[int] = set()
    for lab in uniq:
        idx = np.flatnonzero(labels == lab)
        pi = label_to_prior.get(int(lab))
        if association == "cluster" and len(idx) < cfg.min_sightings and pi is None:
            dropped_clusters += 1          # leak-debris speck matched to nothing
            continue
        seed = prior["pos"][pi] if pi is not None else None
        mean, cov, mask, _sig = robust_position_estimate(
            pts[idx], cfg.sigma_ned, k=cfg.reject_norm_k, iters=cfg.robust_iters,
            seed=seed, seed_extra_var=cfg.prior_sigma_pos**2)
        yaw, yaw_std, ymask = robust_yaw_estimate(
            np.array([sightings[i].yaw_world for i in idx
                      if sightings[i].yaw_world is not None], dtype=np.float64),
            cfg.yaw_sigma_rad, k=cfg.reject_yaw_k)
        n_used = int(mask.sum())
        n_rejected += int((~mask).sum()) + (int((~ymask).sum()) if ymask.size else 0)
        flags: list[str] = []
        if n_used < cfg.min_sightings:
            flags.append("low_confidence_n")
        if n_used and (1.0 - n_used / len(idx)) > 0.30:
            flags.append("high_outlier_fraction")
        if yaw is None:
            flags.append("yaw_unobserved")

        if pi is not None:
            used_prior_idx.add(pi)
            gid = prior["ids"][pi]
            mean, cov, yaw, yaw_std, fused_flags = _fuse_with_prior(
                mean, cov, yaw, yaw_std, prior, pi, cfg, n_used)
            flags += fused_flags
        elif labeled:
            gid = int(lab)
        else:
            gid = next_free_id
            next_free_id += 1
            if prior is not None:
                flags.append("unmatched_cluster")
        gates.append(MappedGate(gate_id=gid, position_ned=mean, yaw=yaw,
                                pos_cov=cov, yaw_std=yaw_std,
                                n_sightings=len(idx), n_used=n_used, flags=tuple(flags)))

    if prior is not None:                  # gates the lap never observed: prior passthrough
        for pi in range(len(prior["ids"])):
            if pi not in used_prior_idx:
                gates.append(_prior_only_gate_from_parsed(prior, pi, cfg))

    gates.sort(key=lambda g: g.gate_id)
    if association == "cluster" and prior is None:
        # course order == first-seen order == cluster id order; renumber 0..G-1 for cleanliness
        gates = [MappedGate(gate_id=i, position_ned=g.position_ned, yaw=g.yaw,
                            pos_cov=g.pos_cov, yaw_std=g.yaw_std, n_sightings=g.n_sightings,
                            n_used=g.n_used, flags=g.flags) for i, g in enumerate(gates)]
    diag = {"mode": "pose_aided", "association": association,
            "n_sightings": len(sightings), "n_rejected": n_rejected,
            "n_dropped_clusters": dropped_clusters,
            "n_label_groups_merged": n_label_groups_merged,
            "bias_corrected": cfg.bias_correction_ned is not None,
            "prior_used": prior is not None}
    return GateMapEstimate(gates=gates, diagnostics=diag)


def _parse_prior_records(records: list[dict]) -> dict:
    """Prior map records -> opening centres + through-yaws (same lift/convention as the
    live loader: bottom-centre + half-height along the quat height axis; through-direction
    from the quat normal, oriented down-course by the segment geometry)."""
    ids = [int(r["gate_id"]) for r in records]
    pos_raw = [np.asarray(r["position_ned"], dtype=np.float64) for r in records]
    centres, yaws = [], []
    n = len(records)
    for i, r in enumerate(records):
        h = float(r.get("height_m") or GATE_OUTER_SIZE_M)
        q = r.get("orientation_ned_wxyz")
        if n >= 2:
            j = i + 1 if i + 1 < n else i
            seg = pos_raw[j] - pos_raw[i - 1 if j == i else i]
        else:
            seg = np.array([-1.0, 0.0, 0.0])
        if q is not None:
            Rq = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            nrm = Rq[:, 1]
            if nrm @ seg < 0.0:
                nrm = -nrm
            col2 = Rq[:, 2] if Rq[:, 2][2] >= 0.0 else -Rq[:, 2]
            centres.append(pos_raw[i] - 0.5 * h * col2)
            yaws.append(float(np.arctan2(nrm[1], nrm[0])))
        else:
            centres.append(pos_raw[i] - np.array([0.0, 0.0, 0.5 * h]))
            yaws.append(float(np.arctan2(seg[1], seg[0])))
    return {"ids": ids, "pos": np.array(centres), "yaw": np.array(yaws)}


def _match_labels_to_prior(labels, pts, prior, cfg: MapperConfig, by_id: bool) -> dict:
    """label -> prior index. Labeled mode matches by gate_id; cluster mode greedily by
    distance (cluster median -> nearest unclaimed prior gate within a generous radius)."""
    out: dict[int, int] = {}
    if by_id:
        id_to_idx = {gid: i for i, gid in enumerate(prior["ids"])}
        for lab in np.unique(labels):
            if int(lab) in id_to_idx:
                out[int(lab)] = id_to_idx[int(lab)]
        return out
    radius = max(4.0 * cfg.prior_sigma_pos, 8.0)
    pairs = []
    for lab in np.unique(labels):
        med = np.median(pts[labels == lab], axis=0)
        d = np.linalg.norm(prior["pos"] - med, axis=1)
        pairs.append((float(d.min()), int(lab), int(np.argmin(d))))
    claimed: set[int] = set()
    for d, lab, pi in sorted(pairs):
        if d <= radius and pi not in claimed:
            out[lab] = pi
            claimed.add(pi)
    return out


def _fuse_with_prior(mean, cov, yaw, yaw_std, prior, pi: int, cfg: MapperConfig,
                     n_used: int):
    """Per-axis information fusion of the robust estimate with the prior gate.

    Below ``min_sightings`` the robust screen cannot tell a leak outlier from signal (a
    2-sighting median can sit metres off), so the measurement is hard-downweighted instead
    of letting its optimistic statistical cov dominate a trusted prior."""
    flags = ["prior_fused"]
    meas_var = np.diag(cov) + cfg.map_systematic_std**2
    if n_used < cfg.min_sightings:
        meas_var = meas_var + 4.0 * cfg.prior_sigma_pos**2
        flags.append("low_n_downweighted")
    prior_var = np.full(3, cfg.prior_sigma_pos**2)
    # chi2 consistency gate (same convention as everywhere else): a measurement that
    # contradicts the prior beyond k-sigma of the COMBINED uncertainty is contamination
    # (e.g. a label group that is 100% neighbour-gate mislabels sits ~24 m off) — keep the
    # prior outright rather than dragging it part-way toward garbage.
    delta = mean - prior["pos"][pi]
    if float(np.sqrt(np.sum(delta**2 / (prior_var + meas_var)))) > cfg.reject_norm_k:
        flags.append("measurement_rejected_vs_prior")
        return (prior["pos"][pi].copy(), np.diag(prior_var),
                float(prior["yaw"][pi]), cfg.prior_sigma_yaw_rad, flags)
    w = prior_var / (prior_var + meas_var)            # weight on the measurement
    fused = prior["pos"][pi] + w * delta
    fused_var = (prior_var * meas_var) / (prior_var + meas_var)
    if yaw is not None:
        # gate-plane 180deg fold: a sighting set from the far side reports yaw+pi
        dyaw = float(wrap_pi(yaw - prior["yaw"][pi]))
        if abs(dyaw) > np.pi / 2:
            dyaw = float(wrap_pi(dyaw + np.pi))
            flags.append("yaw_folded_pi")
        mv = (yaw_std or cfg.yaw_sigma_rad)**2 + (0.5 * cfg.yaw_sigma_rad)**2
        if n_used < cfg.min_sightings:        # an unscreenable lone flip must not win
            mv = mv + 4.0 * cfg.prior_sigma_yaw_rad**2
        pv = cfg.prior_sigma_yaw_rad**2
        wy = pv / (pv + mv)
        fused_yaw = float(wrap_pi(prior["yaw"][pi] + wy * dyaw))
        fused_yaw_std = float(np.sqrt(pv * mv / (pv + mv)))
    else:
        fused_yaw, fused_yaw_std = float(prior["yaw"][pi]), cfg.prior_sigma_yaw_rad
    return fused, np.diag(fused_var), fused_yaw, fused_yaw_std, flags


def _prior_only_gate_from_parsed(prior, pi: int, cfg: MapperConfig) -> MappedGate:
    return MappedGate(gate_id=prior["ids"][pi], position_ned=prior["pos"][pi].copy(),
                      yaw=float(prior["yaw"][pi]),
                      pos_cov=np.eye(3) * cfg.prior_sigma_pos**2,
                      yaw_std=cfg.prior_sigma_yaw_rad,
                      n_sightings=0, n_used=0, flags=("prior_only",))


def _ordered_labels(labels: np.ndarray, times: np.ndarray) -> list[int]:
    """Unique labels ordered by median sighting time (course order for an exploration lap)."""
    uniq = np.unique(labels)
    med_t = [float(np.median(times[labels == u])) for u in uniq]
    return [int(u) for _, u in sorted(zip(med_t, uniq))]


# ---------------------------------------------------------------------------
# Case C — no pose: bounded linear gate-landmark adjustment
# ---------------------------------------------------------------------------
def estimate_map_no_pose(rel_sightings: list[RelativeGateSighting],
                         config: MapperConfig | None = None,
                         n_gates_hint: int | None = None) -> GateMapEstimate:
    """Gate-landmark adjustment from relative observations only (no drone positions).

    See the module docstring for the formulation (linear residuals, translation gauge
    anchored on the first pose, yaw solved separately) and the conditioning discussion
    (co-visibility components, smoothness prior). The output frame is the ANCHOR frame:
    world up to the (unobservable) translation of the first pose — consumers localize
    against the map, so only the relative geometry matters.
    """
    cfg = config or MapperConfig()
    if not rel_sightings:
        return GateMapEstimate(gates=[], diagnostics={"mode": "no_pose", "n_sightings": 0})

    # --- group into frames (poses), time-ordered, decimated to the pose cap ---
    by_frame: dict[int, list[RelativeGateSighting]] = {}
    for s in rel_sightings:
        by_frame.setdefault(int(s.frame_id), []).append(s)
    frame_ids = sorted(by_frame, key=lambda f: (min(s.t for s in by_frame[f]), f))
    if len(frame_ids) > cfg.max_poses:
        keep = np.unique(np.linspace(0, len(frame_ids) - 1, cfg.max_poses).round().astype(int))
        frame_ids = [frame_ids[i] for i in keep]
    frames = [by_frame[f] for f in frame_ids]
    frame_t = np.array([float(np.median([s.t for s in fr])) for fr in frames])

    levers = [[s.lever_world() for s in fr] for fr in frames]
    if cfg.bias_correction_ned is not None:
        b = np.asarray(cfg.bias_correction_ned, dtype=np.float64)
        levers = [[L + b for L in fr] for fr in levers]

    # --- geometry-driven incremental association + chained dead-reckoned init ---
    obs4, gate_init, pose_init, n_assoc_rejected, first_seen = _chain_init(
        frames, levers, frame_t, cfg)
    obs = [(f, g, L) for (f, g, L, _s) in obs4]
    G, P = len(gate_init), len(pose_init)
    if G == 0:
        return GateMapEstimate(gates=[], diagnostics={
            "mode": "no_pose", "n_sightings": len(rel_sightings),
            "note": "no gates survived association"})

    # --- virtual coast poses: chain the smoothness prior THROUGH detection gaps ---
    n_virtual = 0
    if cfg.accel_prior_sigma is not None and cfg.virtual_frame_dt and P >= 2:
        vt, vinit, idx_map = [], [], {}
        for i in range(P):
            if i > 0:
                gap = float(frame_t[i] - frame_t[i - 1])
                if gap > 2.5 * cfg.virtual_frame_dt:
                    n_ins = min(int(gap / cfg.virtual_frame_dt) - 1, 400)
                    for k in range(1, n_ins + 1):
                        f = k / (n_ins + 1)
                        vt.append(frame_t[i - 1] + gap * f)
                        vinit.append(pose_init[i - 1] * (1.0 - f) + pose_init[i] * f)
            idx_map[i] = len(vt)
            vt.append(float(frame_t[i]))
            vinit.append(pose_init[i])
        n_virtual = len(vt) - P
        if n_virtual:
            obs = [(idx_map[f], g, L) for (f, g, L) in obs]
            obs4 = [(idx_map[f], g, L, s) for (f, g, L, s) in obs4]
            frame_t = np.asarray(vt)
            pose_init = vinit
            P = len(pose_init)

    # --- co-visibility components (frames<->gates union-find) ---
    comp_of_gate, n_components = _covis_components(obs, G, P)

    # --- assemble + solve the (linear) robust least-squares ---
    sol = _solve_no_pose(obs, gate_init, pose_init, frame_t, cfg)

    # --- duplicate merge: a gate re-founded across a blind gap coincides post-solve ---
    n_merged = 0
    remap = _duplicate_merge_map(sol["gate_pos"], cfg.dup_merge_radius_m)
    if remap is not None:
        n_merged = G - len(set(remap.values()))
        obs4 = [(f, remap[g], L, s) for (f, g, L, s) in obs4]
        obs = [(f, remap[g], L) for (f, g, L) in obs]
        new_first: dict[int, int] = {}
        new_init: dict[int, np.ndarray] = {}
        for old, new in remap.items():
            new_first[new] = min(new_first.get(new, 1 << 60), first_seen.get(old, 0))
            new_init.setdefault(new, np.asarray(sol["gate_pos"][old]))
        G = len(set(remap.values()))
        gate_init = [new_init[g] for g in range(G)]
        first_seen = new_first
        comp_of_gate, n_components = _covis_components(obs, G, P)
        sol = _solve_no_pose(obs, gate_init, pose_init, frame_t, cfg)

    # --- yaw: direct measurements (attitude given), robust circular per gate ---
    yaw_meas: dict[int, list[float]] = {g: [] for g in range(G)}
    for k, (fi, gi, _L, s) in enumerate(obs4):
        if s.rel_yaw is not None and sol["obs_mask"][k]:
            yaw_meas[gi].append(float(wrap_pi(_yaw_from_quat_wxyz(s.quat_wxyz)
                                              + float(s.rel_yaw))))

    # --- output naming: upstream labels (majority vote) when unanimous+unique, else order ---
    have_labels = all(s.gate_id is not None for fr in frames for s in fr)
    majority_label: dict[int, int] = {}
    if have_labels:
        for g in range(G):
            labs = [int(s.gate_id) for (_f, gi, _L, s) in obs4 if gi == g]
            if labs:
                vals, counts = np.unique(labs, return_counts=True)
                majority_label[g] = int(vals[np.argmax(counts)])
        if len(set(majority_label.values())) != len(majority_label):
            majority_label = {}            # collision: two spatial gates claim one label

    anchor_comp = comp_of_gate[0] if G else 0
    n_per_gate = [sum(1 for (_f, gi, _L) in obs if gi == g) for g in range(G)]
    order = sorted(range(G), key=lambda g: first_seen.get(g, 0))
    gates: list[MappedGate] = []
    n_dropped_gates = 0
    for out_seq, g in enumerate(order):
        if n_per_gate[g] < cfg.min_sightings:
            n_dropped_gates += 1           # leak-spawned speck: not a real gate
            continue
        yaw, yaw_std, _ = robust_yaw_estimate(np.array(yaw_meas[g], dtype=np.float64),
                                              cfg.yaw_sigma_rad, k=cfg.reject_yaw_k)
        flags: list[str] = []
        n_used = sol["n_used_per_gate"][g]
        if comp_of_gate[g] != anchor_comp:
            flags.append("disconnected")
        if n_used < cfg.min_sightings:
            flags.append("low_confidence_n")
        if yaw is None:
            flags.append("yaw_unobserved")
        gid = majority_label.get(g, len(gates))
        gates.append(MappedGate(
            gate_id=gid, position_ned=sol["gate_pos"][g], yaw=yaw,
            pos_cov=sol["gate_cov"][g], yaw_std=yaw_std,
            n_sightings=n_per_gate[g], n_used=n_used, flags=tuple(flags)))
    gates.sort(key=lambda g: g.gate_id)

    diag = {"mode": "no_pose", "n_sightings": len(rel_sightings),
            "n_frames": P - n_virtual, "n_virtual_poses": n_virtual,
            "n_gates": len(gates), "n_dropped_gates": n_dropped_gates,
            "n_merged_gates": n_merged,
            "labels": "given(naming-only)" if have_labels else "incremental",
            "n_assoc_rejected": n_assoc_rejected,
            "n_residual_rejected": sol["n_rejected"],
            "covis_components": n_components,
            "frames_multi_gate_frac": float(np.mean([len(fr) >= 2 for fr in frames])),
            "smoothness_prior_mps2": cfg.accel_prior_sigma,
            "jtj_cond": sol["jtj_cond"], "jtj_min_eig": sol["jtj_min_eig"],
            "final_cost": sol["cost"],
            "anchor": "first pose fixed at its dead-reckoned init (translation gauge)"}
    if n_gates_hint is not None and len(gates) != n_gates_hint:
        diag["warning_gate_count"] = f"expected {n_gates_hint}, mapped {len(gates)}"
    if n_components > 1:
        diag["warning_connectivity"] = (
            f"{n_components} co-visibility components: relative placement BETWEEN components "
            "is unobservable" + ("" if cfg.accel_prior_sigma is None
                                 else " (bridged only by the soft smoothness prior)"))
    return GateMapEstimate(gates=gates, diagnostics=diag)


def _chain_init(frames, levers, frame_t_chain, cfg: MapperConfig):
    """Time-sequential chaining: pose from already-known gates, new gates from the pose.

    Association is purely spatial (labels are never used for geometry): each sighting's
    implied gate position (predicted pose + lever) matches the nearest known gate within
    ``assoc_radius_m`` or founds a new one. Inter-frame motion at ~15 Hz is ~0.3-0.5 m,
    far inside the radius, so prev-pose prediction suffices. A sighting whose implied
    position contradicts its matched gate by more than the radius is dropped (leak tail /
    impossible geometry), counted in the returned rejection tally.

    Two passes: pass 1 discovers gates with a running-median init (frozen after 12
    sightings, so an early outlier cannot poison a young gate permanently); pass 2
    re-chains every pose against the frozen robust gate seeds for clean, consistent inits.

    Returns ``(obs, gate_init, pose_init, n_rejected, first_seen)`` with obs entries
    ``(pose_idx, gate_idx, lever_world, sighting)``.
    """

    def run_pass(seeds: list[np.ndarray] | None):
        gate_pos: list[np.ndarray] = [s.copy() for s in seeds] if seeds is not None else []
        gate_pts: list[list[np.ndarray]] = [[] for _ in gate_pos]
        first_seen: dict[int, int] = {}
        obs, pose_init = [], []
        prev_pose = np.zeros(3)
        prev_t = None
        recent: list[tuple[float, np.ndarray]] = []   # last chained (t, pose) for velocity fit
        for fi, (fr, Ls) in enumerate(zip(frames, levers)):
            t_f = float(frame_t_chain[fi])
            # constant-velocity dead-reckoned prediction (NOT a stale hold: across a blind
            # transit gap a held pose is ~speed*gap wrong, which founded every post-gap gate
            # ~15 m off as one rigid block — measured before this fix)
            if prev_t is not None and len(recent) >= 2 and t_f > prev_t:
                (t0, p0), (t1, p1) = recent[0], recent[-1]
                vel = (p1 - p0) / max(t1 - t0, 1e-6)
                speed = float(np.linalg.norm(vel))
                if speed > 20.0:                       # physically impossible: distrust
                    vel = vel * (20.0 / speed)
                pred = prev_pose + vel * (t_f - prev_t)
            else:
                pred = prev_pose
            # 1) pose votes from sightings matching KNOWN gates (median = robust to one outlier)
            votes = []
            for L in Ls:
                if gate_pos:
                    implied = pred + L
                    d = np.linalg.norm(np.asarray(gate_pos) - implied, axis=1)
                    j = int(np.argmin(d))
                    if d[j] <= cfg.assoc_radius_m:
                        votes.append(gate_pos[j] - L)
            pose = np.median(np.asarray(votes), axis=0) if votes else pred.copy()
            pose_init.append(pose)
            recent.append((t_f, pose))
            if len(recent) > 12:
                recent.pop(0)
            # 2) attach sightings to gates (re-matched against the refined pose); found new gates
            for s, L in zip(fr, Ls):
                implied = pose + L
                gi = None
                if gate_pos:
                    d = np.linalg.norm(np.asarray(gate_pos) - implied, axis=1)
                    j = int(np.argmin(d))
                    if d[j] <= cfg.assoc_radius_m:
                        gi = j
                if gi is None:
                    gate_pos.append(implied.copy())
                    gate_pts.append([])
                    gi = len(gate_pos) - 1
                first_seen.setdefault(gi, fi)
                gate_pts[gi].append(implied)
                if seeds is None and len(gate_pts[gi]) <= 12:
                    gate_pos[gi] = np.median(np.asarray(gate_pts[gi]), axis=0)
                obs.append((fi, gi, L, s))
            prev_pose = pose
            prev_t = t_f
        return obs, gate_pos, pose_init, first_seen

    _obs1, gp1, _pi1, _fs1 = run_pass(seeds=None)
    obs, gate_pos, pose_init, first_seen = run_pass(seeds=gp1)
    # final sanity sweep: drop sightings contradicting their (frozen-seed) gate
    kept = []
    n_rejected = 0
    for (fi, gi, L, s) in obs:
        if np.linalg.norm((pose_init[fi] + L) - gate_pos[gi]) > cfg.assoc_radius_m:
            n_rejected += 1
            continue
        kept.append((fi, gi, L, s))
    return (kept, [np.asarray(p, dtype=np.float64) for p in gate_pos],
            [np.asarray(p, dtype=np.float64) for p in pose_init], n_rejected, first_seen)


def _duplicate_merge_map(gate_pos, radius_m: float) -> dict[int, int] | None:
    """old gate index -> compact merged index, or None when nothing merges.

    Union-find over pairs closer than ``radius_m`` (real gates are >=23.7 m apart, so a
    pair inside 8 m is one physical gate founded twice). Survivor indices keep first-found
    order, so course ordering is preserved."""
    G = len(gate_pos)
    parent = list(range(G))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    any_merge = False
    for i in range(G):
        for j in range(i + 1, G):
            if np.linalg.norm(np.asarray(gate_pos[i]) - np.asarray(gate_pos[j])) <= radius_m:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
                    any_merge = True
    if not any_merge:
        return None
    roots: dict[int, int] = {}
    remap: dict[int, int] = {}
    for g in range(G):
        r = find(g)
        roots.setdefault(r, len(roots))
        remap[g] = roots[r]
    return remap


def _covis_components(obs, G: int, P: int) -> tuple[list[int], int]:
    """Connected components of the bipartite frame<->gate observation graph (gates only)."""
    parent = list(range(G + P))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for (pi, gi, _L) in obs:
        union(gi, G + pi)
    roots: dict[int, int] = {}
    comp_of_gate = []
    for g in range(G):
        r = find(g)
        roots.setdefault(r, len(roots))
        comp_of_gate.append(roots[r])
    return comp_of_gate, len(roots)


def _solve_no_pose(obs, gate_init, pose_init, frame_t, cfg: MapperConfig) -> dict:
    """Assemble + solve the whitened linear system; one hard-rejection re-solve; marginals.

    Parameters: ``x = [gates (3G), poses 1..P-1 (3(P-1))]`` — pose 0 is the gauge anchor,
    fixed at its init. Residual blocks: per sighting ``W (p_g - x_i - L)`` and (optionally)
    per interior pose the whitened discrete acceleration. Everything is linear, so the
    Jacobian is constant + sparse and the robust loss is the only nonlinearity.
    """
    G, P = len(gate_init), len(pose_init)
    W = 1.0 / np.asarray(cfg.sigma_ned, dtype=np.float64)
    n_par = 3 * G + 3 * max(P - 1, 0)
    anchor = np.asarray(pose_init[0], dtype=np.float64)

    def pose_col(i):                      # parameter column of pose i (i >= 1)
        return 3 * G + 3 * (i - 1)

    def build(keep_mask=None):
        """(J, c) for r = J x - c; rows = kept sighting blocks + smoothness blocks."""
        data, ri, ci, c = [], [], [], []
        row = 0
        kept_obs_rows = 0
        for k, (pi, gi, L) in enumerate(obs):
            if keep_mask is not None and not keep_mask[k]:
                continue
            for a in range(3):
                ri.append(row + a); ci.append(3 * gi + a); data.append(W[a])
                if pi > 0:
                    ri.append(row + a); ci.append(pose_col(pi) + a); data.append(-W[a])
            const = np.asarray(L, dtype=np.float64) + (anchor if pi == 0 else 0.0)
            c.extend((W * const).tolist())
            row += 3
            kept_obs_rows += 1
        if cfg.accel_prior_sigma is not None and P >= 3:
            for i in range(1, P - 1):
                dt0 = max(frame_t[i] - frame_t[i - 1], 1e-3)
                dt1 = max(frame_t[i + 1] - frame_t[i], 1e-3)
                s = 1.0 / (cfg.accel_prior_sigma * 0.5 * (dt0 + dt1))
                coef = [(i + 1, s / dt1), (i, -s * (1.0 / dt1 + 1.0 / dt0)), (i - 1, s / dt0)]
                for a in range(3):
                    const = 0.0
                    for (pj, w) in coef:
                        if pj == 0:
                            const += w * anchor[a]   # r = J x + const  =>  c = -const
                        else:
                            ri.append(row + a); ci.append(pose_col(pj) + a); data.append(w)
                    c.append(-const)
                row += 3
        J = sp.csr_matrix((np.asarray(data), (np.asarray(ri), np.asarray(ci))),
                          shape=(row, n_par))
        return J, np.asarray(c), kept_obs_rows

    def solve(J, c, x_start=None):
        """Two-stage: exact LINEAR pass first, robust loss from that solution.

        The robust loss saturates on residuals >> f_scale, so starting it from a far-off
        init (a chain-init seam can be metres) leaves whole components stranded at their
        init (vanishing gradient -> premature ftol stop; measured). The linear pass is a
        convex quadratic solved globally by lsmr — it drags every component through even a
        weak smoothness bridge; the robust pass then only has to re-weight outliers
        locally, which is exactly the regime it converges in.
        """
        if x_start is None:
            x_start = np.concatenate([np.asarray(gate_init).ravel(),
                                      np.asarray(pose_init[1:]).ravel() if P > 1 else np.zeros(0)])
        lin = least_squares(lambda x: J @ x - c, x_start, jac=lambda x: J,
                            method="trf", tr_solver="lsmr", loss="linear", max_nfev=50)
        if cfg.robust_loss == "linear":
            return lin
        return least_squares(lambda x: J @ x - c, lin.x, jac=lambda x: J,
                             method="trf", tr_solver="lsmr",
                             loss=cfg.robust_loss, f_scale=cfg.f_scale, max_nfev=200)

    keep = np.ones(len(obs), dtype=bool)
    J, c, n_kept = build()
    res = solve(J, c)
    # hard rejection on the whitened sighting residual norm, then one clean re-solve
    r = J @ res.x - c
    norms = np.linalg.norm(r[: 3 * n_kept].reshape(-1, 3), axis=1)
    bad = norms > cfg.reject_norm_k
    n_rejected = int(bad.sum())
    if n_rejected:
        keep[np.flatnonzero(keep)[bad]] = False
        J, c, n_kept = build(keep_mask=keep)
        res = solve(J, c, x_start=res.x)

    # --- per-gate marginal covariance + conditioning from the (whitened) normal matrix ---
    gate_cov = [np.diag(np.asarray(cfg.sigma_ned)**2)] * G
    jtj_cond = jtj_min_eig = None
    if cfg.compute_covariance and n_par <= 3000:
        JTJ = (J.T @ J).toarray()
        eig = np.linalg.eigvalsh(JTJ)
        jtj_min_eig = float(eig[0])
        jtj_cond = float(eig[-1] / max(eig[0], 1e-300))
        # pinv tolerates the (near-)singular disconnected case; NB in that case the
        # unobservable directions get pseudo-variance ~0 — marginals are only meaningful
        # within the anchor component (the `disconnected` flag marks the rest).
        cov_all = np.linalg.pinv(JTJ, rcond=1e-10, hermitian=True)
        gate_cov = [cov_all[3 * g: 3 * g + 3, 3 * g: 3 * g + 3] for g in range(G)]

    n_used_per_gate = [0] * G
    for k, (pi, gi, _L) in enumerate(obs):
        if keep[k]:
            n_used_per_gate[gi] += 1
    x = res.x
    return {"gate_pos": [x[3 * g: 3 * g + 3].copy() for g in range(G)],
            "gate_cov": gate_cov,
            "pose_pos": [anchor] + [x[3 * G + 3 * i: 3 * G + 3 * i + 3].copy()
                                    for i in range(max(P - 1, 0))],
            "n_rejected": n_rejected, "n_used_per_gate": n_used_per_gate,
            "cost": float(res.cost), "jtj_cond": jtj_cond, "jtj_min_eig": jtj_min_eig,
            "obs_mask": keep}


# ---------------------------------------------------------------------------
# Sightings file I/O (the CLI seam; a real recording's extracted detections land here)
# ---------------------------------------------------------------------------
SIGHTINGS_SCHEMA = "racer.mapper_sightings/v1"


def dump_sightings_json(path: str | Path,
                        sightings: list,
                        note: str = "") -> Path:
    """Write sightings to the ``racer.mapper_sightings/v1`` JSON (mode auto-detected)."""
    rows, mode = [], None
    for s in sightings:
        if isinstance(s, GateSighting):
            mode = mode or "pose_aided"
            rows.append({"gate_world_ned": [float(v) for v in s.gate_world_ned],
                         "gate_id": s.gate_id,
                         "yaw_world": None if s.yaw_world is None else float(s.yaw_world),
                         "t": float(s.t), "frame_id": s.frame_id,
                         "range_m": None if s.range_m is None else float(s.range_m)})
        else:
            mode = mode or "relative"
            rows.append({"frame_id": int(s.frame_id),
                         "rel_position_frd": [float(v) for v in s.rel_position_frd],
                         "quat_wxyz": [float(v) for v in s.quat_wxyz],
                         "gate_id": s.gate_id,
                         "rel_yaw": None if s.rel_yaw is None else float(s.rel_yaw),
                         "t": float(s.t)})
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"schema": SIGHTINGS_SCHEMA, "mode": mode or "pose_aided",
                             "note": note, "n": len(rows), "sightings": rows}, indent=1))
    return p


def load_sightings_json(path: str | Path):
    """Read a ``racer.mapper_sightings/v1`` file -> (mode, list[sightings])."""
    data = json.loads(Path(path).read_text())
    if data.get("schema") != SIGHTINGS_SCHEMA:
        raise ValueError(f"not a {SIGHTINGS_SCHEMA} file: {path}")
    mode = data["mode"]
    out = []
    for r in data["sightings"]:
        if mode == "pose_aided":
            out.append(GateSighting(
                gate_world_ned=np.asarray(r["gate_world_ned"], dtype=np.float64),
                gate_id=r.get("gate_id"), yaw_world=r.get("yaw_world"),
                t=float(r.get("t", 0.0)), frame_id=r.get("frame_id"),
                range_m=r.get("range_m")))
        else:
            out.append(RelativeGateSighting(
                frame_id=int(r["frame_id"]),
                rel_position_frd=np.asarray(r["rel_position_frd"], dtype=np.float64),
                quat_wxyz=np.asarray(r["quat_wxyz"], dtype=np.float64),
                gate_id=r.get("gate_id"), rel_yaw=r.get("rel_yaw"),
                t=float(r.get("t", 0.0))))
    return mode, out
