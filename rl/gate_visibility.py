"""rl/gate_visibility.py -- batched (torch) keypoint-visibility + occlusion model.

Decides, from GROUND-TRUTH drone and gate poses, whether each gate is DETECTABLE by the
monocular camera. Pure geometry: no world-position estimate, no measurement noise, no learning.
Visibility is a *physical geometric fact*; the measurement NOISE that a detector would add lives
elsewhere (rl/inc8_estimator_emul.py's fix surrogate) -- do NOT add noise here.

FRAME / CONVENTION (matched to the existing code -- NOT invented):
  * Inputs are in the DiffAero **Z-up / FLU** frame, exactly as ``rl.peregrine_racing`` stores them
    (``gate_pos`` Z-up opening centres, ``gate_yaw`` about +Z; drone pose an XYZW body->world quat
    or a 3x3 body->world matrix). This is the frame a downstream obs builder already holds.
  * Internally we flip to **world NED** with the involutory ``_FLIP = [1,-1,-1]`` (== rl.peregrine_course
    ._FLIP / rl.diffaero_dynamics._FLIP), because the verified camera projection lives in NED:
    ``R_camera_from_body`` (a +20 deg body-Y pitch-up mount composed with the FRD->optical axis swap)
    and the pinhole ``K`` are the NED constants in rl.inc8_estimator_emul. Reusing that exact
    projection (via ``batched_geometry``'s math) keeps this model frame-faithful to the estimator the
    policy is scored against, rather than introducing a second convention.
  * Body->world in NED is obtained from the Z-up body->world by the SAME flip on both world and body
    axes (FRD<->FLU is the same axis flip): ``R_wb_ned = diag(F) @ R_wb_zup @ diag(F)`` with
    F=[1,-1,-1]. This mirrors the NED<->Z-up boundary used by fly_rl / the estimator emul.
  * Per-gate NED gate frame R_world_gate = ``ned_gate_frame(yaw)`` (columns [right, down, downrange]),
    identical to rl.estimator_emul.ned_gate_frame -- so the 8 keypoints are laid out in the SAME
    gate frame the fix surrogate uses.

KEYPOINTS (spec 3.7): the gate is a square frame, OUTER 2.7 m, INNER opening 1.5 m, depth 0.26 m.
The 8 keypoints are the 4 inner + 4 outer corners of the FRONT face (the face nearer the incoming
drone, offset -depth/2 along the gate downrange axis from the opening centre). In the NED gate frame
[right, down, downrange] the corners are at (+-half, +-half, -depth/2).

DETECTABILITY RULE (Fengyou-approved):
  * Project each of the 8 corners into the image. IN-FRAME iff u in [0,W), v in [0,H) AND in front of
    the camera (camera-frame depth tz > 0).
  * OCCLUSION via an IMAGE-SPACE PROJECTED-ANNULUS test (cheap batched stand-in for ray-vs-frame-
    material raycasting): a corner is occluded iff it projects inside a NEARER gate's projected frame
    annulus -- inside that gate's projected OUTER quad but OUTSIDE its projected INNER quad -- AND that
    gate is closer to the camera along the view ray. Per gate-pair, batched over envs.
  * A gate is DETECTABLE iff >= MIN_VISIBLE_CORNERS (=4) of its 8 corners are IN-FRAME AND UNOCCLUDED,
    AND the gate centre is within ``far_cap_m`` (default 30 m). Per-gate independent.

PUBLIC API:
    corners_gate_frame() -> (8,3) canonical NED-gate-frame keypoints
    gate_corners_world_ned(gate_pos_ned, R_world_gate) -> (..., 8, 3)
    project_points_camera(pts_world_ned, drone_pos_ned, R_wb_ned) -> (u, v, tz, in_image)
    gate_detectable(drone_pos, drone_quat_or_R, gate_pos, gate_yaw, *, far_cap_m=30.0,
                    is_quat=None) -> (detectable BoolTensor[N,G], n_visible_corners IntTensor[N,G])
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    from torch import Tensor
except Exception:                       # pragma: no cover - torch absent in some tooling contexts
    torch = None
    Tensor = "Tensor"                   # type: ignore

# Reuse the VERIFIED camera constants + helpers (single source of truth for the projection).
from inc8_estimator_emul import (
    CAMERA_INTRINSICS_K_NP,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    R_CAMERA_FROM_BODY_NP,
    FLIP_NP,
    ned_gate_frame_torch,
    quat_xyzw_to_matrix_torch,
)

# ---- spec 3.7 gate geometry -------------------------------------------------------------------
GATE_OUTER_M = 2.7
GATE_INNER_M = 1.5
GATE_DEPTH_M = 0.26
_HALF_OUTER = GATE_OUTER_M / 2.0        # 1.35 m
_HALF_INNER = GATE_INNER_M / 2.0        # 0.75 m
_FRONT_FACE_DOWNRANGE = -GATE_DEPTH_M / 2.0   # front face is offset toward the incoming drone

MIN_VISIBLE_CORNERS = 4
FAR_CAP_M_DEFAULT = 30.0


# ================================================================================================
# Keypoint generation (NED gate frame [right, down, downrange]).
# ================================================================================================
def corners_gate_frame(device=None, dtype=None) -> Tensor:
    """The 8 front-face keypoints in the NED gate frame, ordered inner(4) then outer(4).

    Order within each quad: (--), (+-), (++), (-+) in (right, down) -- i.e. TL, TR, BR, BL going
    around the ring, so the 4 points form a convex quad in projection order.
    Returns (8, 3): columns [right, down, downrange]; downrange = -depth/2 (front face)."""
    assert torch is not None, "corners_gate_frame requires torch"
    dr = _FRONT_FACE_DOWNRANGE

    def ring(h):
        return [(-h, -h, dr), (h, -h, dr), (h, h, dr), (-h, h, dr)]

    pts = ring(_HALF_INNER) + ring(_HALF_OUTER)
    return torch.tensor(pts, device=device, dtype=dtype)


def gate_corners_world_ned(gate_pos_ned: Tensor, R_world_gate: Tensor) -> Tensor:
    """World-NED positions of the 8 keypoints per gate.

    gate_pos_ned (..., 3); R_world_gate (..., 3, 3) gate->world NED. Returns (..., 8, 3).
    world = gate_centre + R_world_gate @ corner_gate."""
    cg = corners_gate_frame(device=gate_pos_ned.device, dtype=gate_pos_ned.dtype)   # (8,3)
    # (...,3,3) @ (8,3)^T -> (...,3,8) -> (...,8,3)
    world_rel = torch.einsum("...ij,kj->...ki", R_world_gate, cg)                   # (...,8,3)
    return gate_pos_ned.unsqueeze(-2) + world_rel


# ================================================================================================
# Projection (reuses the exact R_camera_from_body + K path from batched_geometry).
# ================================================================================================
def _camera_consts(device, dtype):
    R_cb = torch.as_tensor(R_CAMERA_FROM_BODY_NP, device=device, dtype=dtype)
    K = torch.as_tensor(CAMERA_INTRINSICS_K_NP, device=device, dtype=dtype)
    return R_cb, K


def project_points_camera(pts_world_ned: Tensor, drone_pos_ned: Tensor, R_wb_ned: Tensor):
    """Project world-NED points into the image for a per-env drone pose.

    pts_world_ned (N, P, 3); drone_pos_ned (N, 3); R_wb_ned (N, 3, 3) body->world NED.
    Returns (u (N,P), v (N,P), tz (N,P), in_image (N,P) bool). Mirrors
    inc8_estimator_emul.batched_geometry's projection exactly:
        t_cam = R_camera_from_body @ (R_wb^T @ (p - drone));  u,v = K-project(t_cam); front = tz>0."""
    device, dtype = pts_world_ned.device, pts_world_ned.dtype
    R_cb, K = _camera_consts(device, dtype)
    lever = pts_world_ned - drone_pos_ned.unsqueeze(-2)                 # (N,P,3) world NED
    body_vec = torch.einsum("nji,npj->npi", R_wb_ned, lever)           # R_wb^T @ lever
    t_cam = torch.einsum("ij,npj->npi", R_cb, body_vec)               # (N,P,3)
    tx, ty, tz = t_cam[..., 0], t_cam[..., 1], t_cam[..., 2]
    front = tz > 0.0
    safe_tz = torch.where(front, tz, torch.ones_like(tz))
    u = (K[0, 0] * tx + K[0, 2] * tz) / safe_tz
    v = (K[1, 1] * ty + K[1, 2] * tz) / safe_tz
    in_image = front & (u >= 0.0) & (u < IMAGE_WIDTH) & (v >= 0.0) & (v < IMAGE_HEIGHT)
    return u, v, tz, in_image


# ================================================================================================
# Image-space annulus occlusion.
# ================================================================================================
def _point_in_quad(px: Tensor, py: Tensor, quad: Tensor) -> Tensor:
    """Is (px, py) inside the convex quad? px/py (...,); quad (..., 4, 2) in CONSISTENT winding.

    Uses the sign of the 2D cross product for each edge; inside iff all cross products share a sign
    (>=0 or <=0, robust to CW/CCW winding). Broadcasts over the leading dims of px/py vs quad."""
    inside_pos = torch.ones_like(px, dtype=torch.bool)
    inside_neg = torch.ones_like(px, dtype=torch.bool)
    for i in range(4):
        ax, ay = quad[..., i, 0], quad[..., i, 1]
        bx, by = quad[..., (i + 1) % 4, 0], quad[..., (i + 1) % 4, 1]
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        inside_pos = inside_pos & (cross >= 0.0)
        inside_neg = inside_neg & (cross <= 0.0)
    return inside_pos | inside_neg


def _occluder_projected_quads(gate_pos_ned: Tensor, R_world_gate: Tensor,
                              drone_pos_ned: Tensor, R_wb_ned: Tensor):
    """For each (env, gate) build the projected INNER and OUTER image quads + a validity mask.

    Returns:
      inner_quad (N,G,4,2), outer_quad (N,G,4,2): image (u,v) of the 4 inner / 4 outer corners.
      valid (N,G) bool: all 8 corners are in FRONT of the camera (tz>0) so the projected quad is a
        sensible screen polygon (a corner behind the camera makes the projection meaningless -> we
        do not let such a gate occlude anything).
      z_ref (N,G): representative camera depth of the occluder (its centre's tz) for the near test.
    """
    N, G = gate_pos_ned.shape[0], gate_pos_ned.shape[1]
    corners = gate_corners_world_ned(gate_pos_ned, R_world_gate)        # (N,G,8,3)
    u, v, tz, _ = project_points_camera(corners.reshape(N, G * 8, 3),
                                        drone_pos_ned, R_wb_ned)
    u = u.reshape(N, G, 8); v = v.reshape(N, G, 8); tz = tz.reshape(N, G, 8)
    uv = torch.stack([u, v], dim=-1)                                    # (N,G,8,2)
    inner_quad = uv[:, :, 0:4, :]                                       # (N,G,4,2)
    outer_quad = uv[:, :, 4:8, :]
    valid = (tz > 0.0).all(dim=-1)                                      # (N,G)
    # centre depth of the occluder for the near comparison
    _, _, tz_c, _ = project_points_camera(gate_pos_ned, drone_pos_ned, R_wb_ned)  # (N,G)
    return inner_quad, outer_quad, valid, tz_c


def _annulus_occludes(u_pt: Tensor, v_pt: Tensor, tz_pt: Tensor,
                      inner_quad: Tensor, outer_quad: Tensor,
                      occ_valid: Tensor, occ_z: Tensor) -> Tensor:
    """Does a single projected point (u_pt,v_pt at camera depth tz_pt) fall inside a NEARER gate's
    projected annulus? All target tensors (N,); occluder tensors (N,) for ONE occluder gate.

    Occluded iff: occluder valid AND occluder strictly nearer (occ_z < tz_pt) AND point inside the
    OUTER quad AND point OUTSIDE the INNER quad (the frame material band)."""
    in_outer = _point_in_quad(u_pt, v_pt, outer_quad)
    in_inner = _point_in_quad(u_pt, v_pt, inner_quad)
    nearer = occ_valid & (occ_z < tz_pt)
    return nearer & in_outer & (~in_inner)


# ================================================================================================
# Main entry point.
# ================================================================================================
def _to_R_wb_ned(drone_quat_or_R: Tensor, is_quat: bool | None) -> Tensor:
    """Return body->world NED rotation (N,3,3) from either an XYZW Z-up quat (N,4) or a Z-up
    body->world matrix (N,3,3). NED conversion: R_wb_ned = diag(F) @ R_wb_zup @ diag(F), F=[1,-1,-1]
    (the involutory FLIP applied on both world and body axes)."""
    if is_quat is None:
        is_quat = drone_quat_or_R.shape[-1] == 4 and drone_quat_or_R.dim() == 2
    if is_quat:
        R_wb_zup = quat_xyzw_to_matrix_torch(drone_quat_or_R)          # (N,3,3)
    else:
        R_wb_zup = drone_quat_or_R
    F = torch.as_tensor(FLIP_NP, device=R_wb_zup.device, dtype=R_wb_zup.dtype)
    D = torch.diag(F)
    return D @ R_wb_zup @ D


def gate_detectable(drone_pos: Tensor, drone_quat_or_R: Tensor, gate_pos: Tensor,
                    gate_yaw: Tensor, *, far_cap_m: float = FAR_CAP_M_DEFAULT,
                    is_quat: bool | None = None):
    """Batched keypoint-visibility + occlusion detectability, over N envs and G gates.

    Inputs (DiffAero Z-up / FLU frame, matching rl.peregrine_racing):
      drone_pos        (N, 3)      drone position, Z-up
      drone_quat_or_R  (N, 4) XYZW quat OR (N, 3, 3) body->world matrix, Z-up (auto-detected;
                                   override with is_quat=)
      gate_pos         (N, G, 3)   gate opening centres, Z-up
      gate_yaw         (N, G)      gate through-yaw about +Z (rad)
      far_cap_m                    detectability hard range cap on the gate CENTRE (default 30 m)

    Returns:
      detectable        BoolTensor (N, G)   >= 4 corners in-frame & unoccluded AND centre within cap
      n_visible_corners IntTensor  (N, G)   count of in-frame & unoccluded corners per gate

    Pure geometry from truth: no estimate, no noise, no learning.
    """
    assert torch is not None, "gate_detectable requires torch"
    device = drone_pos.device
    dtype = drone_pos.dtype
    N, G = gate_pos.shape[0], gate_pos.shape[1]

    F = torch.as_tensor(FLIP_NP, device=device, dtype=dtype)
    drone_pos_ned = drone_pos * F                                     # (N,3)
    gate_pos_ned = gate_pos * F                                       # (N,G,3)
    R_wb_ned = _to_R_wb_ned(drone_quat_or_R, is_quat)                # (N,3,3)

    # per (env,gate) NED gate frame (columns [right,down,downrange]); ned_gate_frame_torch takes (K,)
    R_world_gate = ned_gate_frame_torch(gate_yaw.reshape(-1)).reshape(N, G, 3, 3)   # (N,G,3,3)

    # ---- 8 corners per gate, projected; in-frame test ----
    corners = gate_corners_world_ned(gate_pos_ned, R_world_gate)      # (N,G,8,3)
    u, v, tz, in_image = project_points_camera(
        corners.reshape(N, G * 8, 3), drone_pos_ned, R_wb_ned)
    u = u.reshape(N, G, 8); v = v.reshape(N, G, 8)
    tz = tz.reshape(N, G, 8); in_image = in_image.reshape(N, G, 8)    # (N,G,8)

    # ---- occlusion: each corner vs every OTHER gate's projected annulus ----
    inner_q, outer_q, occ_valid, occ_z = _occluder_projected_quads(
        gate_pos_ned, R_world_gate, drone_pos_ned, R_wb_ned)          # (N,G,4,2),(N,G,4,2),(N,G),(N,G)

    # occluded[n, g, c] = any other gate h != g occludes corner c of gate g.
    # Build (N, G_target, 8, G_occ) by broadcasting the target corners against every occluder gate.
    up_e = u[:, :, :, None].expand(N, G, 8, G)    # (N,G,8,G_occ)
    vp_e = v[:, :, :, None].expand(N, G, 8, G)
    tzp_e = tz[:, :, :, None].expand(N, G, 8, G)
    inner_e = inner_q[:, None, None, :, :, :].expand(N, G, 8, G, 4, 2)
    outer_e = outer_q[:, None, None, :, :, :].expand(N, G, 8, G, 4, 2)
    occ_valid_e = occ_valid[:, None, None, :].expand(N, G, 8, G)
    occ_z_e = occ_z[:, None, None, :].expand(N, G, 8, G)

    occ_by = _annulus_occludes(up_e, vp_e, tzp_e, inner_e, outer_e, occ_valid_e, occ_z_e)  # (N,G,8,G)
    # a gate never occludes itself
    self_mask = torch.eye(G, device=device, dtype=torch.bool)[None, :, None, :].expand(N, G, 8, G)
    occ_by = occ_by & (~self_mask)
    occluded = occ_by.any(dim=-1)                                    # (N,G,8)

    visible_corner = in_image & (~occluded)                         # (N,G,8)
    n_visible = visible_corner.sum(dim=-1).to(torch.int32)          # (N,G)

    # ---- far cap on the gate centre range ----
    centre_lever = gate_pos_ned - drone_pos_ned.unsqueeze(-2)        # (N,G,3)
    centre_range = torch.linalg.norm(centre_lever, dim=-1)          # (N,G)
    within_cap = centre_range <= far_cap_m

    detectable = (n_visible >= MIN_VISIBLE_CORNERS) & within_cap
    return detectable, n_visible


# ================================================================================================
# Apparent (projected) inner-opening area -- the vision-faithful "how square-on" cue (Fengyou 2026-07-07).
# ================================================================================================
def gate_apparent_area(drone_pos: Tensor, drone_quat_or_R: Tensor, gate_pos: Tensor,
                       gate_yaw: Tensor, *, is_quat: bool | None = None) -> Tensor:
    """Normalized APPARENT area of each gate's INNER opening as the camera actually sees it, in [0,1]
    with a perfectly SQUARE-ON view == 1.

    This RECALIBRATES the old |cos(view_ray, gate_normal)| foreshortening proxy (Fengyou 2026-07-07: the
    real vision system reports the DETECTED opening area, not a cosine -- and |cos| diverges from the true
    projected area close-in where perspective matters). It is exactly what a corner-detecting detector
    outputs: the image-space (pinhole-projected) area of the 4 INNER corners, divided by the area a
    frontal (square-on) opening of the SAME physical size would project at the SAME range. The range
    normalization cancels the inverse-square shrink AND the intrinsics, so the result is a pure,
    range-invariant "squareness" ratio: ~1 head-on (a big square), -> 0 edge-on (a foreshortened sliver),
    capturing the true perspective the |cos| model misses.

    Uses the SAME verified projection as ``gate_detectable`` (the +20 deg mount + K), so the reward's
    area (privileged GT, noiseless) and the estimator's visible_area obs (this + noise) are the SAME
    quantity. Pure GT geometry -- masking/staleness/noise live in the estimator.

    drone_pos (N,3) Z-up; drone_quat_or_R (N,4) XYZW OR (N,3,3) body->world Z-up; gate_pos (N,G,3) Z-up;
    gate_yaw (N,G). Returns (N,G) in [0,1]. A gate with any inner corner BEHIND the camera -> 0 (its
    projected quad is meaningless; it is not being cleanly seen)."""
    assert torch is not None, "gate_apparent_area requires torch"
    device, dtype = drone_pos.device, drone_pos.dtype
    N, G = gate_pos.shape[0], gate_pos.shape[1]
    F = torch.as_tensor(FLIP_NP, device=device, dtype=dtype)
    drone_pos_ned = drone_pos * F
    gate_pos_ned = gate_pos * F
    R_wb_ned = _to_R_wb_ned(drone_quat_or_R, is_quat)
    R_world_gate = ned_gate_frame_torch(gate_yaw.reshape(-1)).reshape(N, G, 3, 3)
    corners = gate_corners_world_ned(gate_pos_ned, R_world_gate)         # (N,G,8,3)
    inner = corners[:, :, 0:4, :]                                        # (N,G,4,3) the OPENING quad
    u, v, tz, _ = project_points_camera(inner.reshape(N, G * 4, 3), drone_pos_ned, R_wb_ned)
    u = u.reshape(N, G, 4); v = v.reshape(N, G, 4); tz = tz.reshape(N, G, 4)
    # shoelace area of the projected inner quad (corner ring TL,TR,BR,BL -> convex), abs -> unsigned.
    area_px = 0.5 * torch.abs(
        u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
        + u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1]
        + u[..., 2] * v[..., 3] - u[..., 3] * v[..., 2]
        + u[..., 3] * v[..., 0] - u[..., 0] * v[..., 3])                 # (N,G) pixels^2
    # square-on reference at the same range r: a frontal L x L opening at camera-depth r projects to a
    # (fx*L/r) x (fy*L/r) rectangle -> area = fx*fy*L^2/r^2. Dividing cancels r^2 + fx*fy -> square-on==1.
    _R_cb, K = _camera_consts(device, dtype)
    fx, fy = K[0, 0], K[1, 1]
    rng = torch.linalg.norm(gate_pos_ned - drone_pos_ned.unsqueeze(-2), dim=-1).clamp(min=1e-3)  # (N,G)
    ref = (fx * fy * (GATE_INNER_M ** 2)) / (rng * rng)
    all_front = (tz > 0.0).all(dim=-1)                                   # (N,G) opening fully in front
    ratio = torch.where(all_front, area_px / ref.clamp(min=1e-9), torch.zeros_like(area_px))
    return ratio.clamp(0.0, 1.0)


def gate_center_view_cos(drone_pos: Tensor, drone_quat_or_R: Tensor, gate_pos: Tensor,
                         gate_yaw: Tensor, *, is_quat: bool | None = None) -> Tensor:
    """Cosine of the angle between the camera OPTICAL AXIS (+Z_cam) and the drone->gate-CENTRE vector,
    per (env, gate) -> (N,G). ``cos = tz / range``, where tz is the gate-centre camera-depth from the
    SAME verified +20 deg-mount projection as ``gate_detectable`` / ``gate_apparent_area`` and range =
    ‖gate_centre − drone‖. +1 == the gate centre lies exactly on the optical axis (perfectly centred in
    view); it decays as the gate drifts toward the frame edge; a gate BEHIND the camera (tz<0) -> < 0.

    This is the geometric ingredient of the Swift/Geles PERCEPTION reward ``r_perc = λ·exp(−δ_cam⁴)``
    (δ_cam = arccos of this): the field-proven lever that keeps the gate centred in the FOV every step,
    improving the estimate on approach (gate-passing error ~0.5 m -> ~0.15 m in Geles). Same Z-up inputs
    as ``gate_apparent_area``; pass the EMULATED (flipped) camera R so it matches the detector's FOV."""
    assert torch is not None, "gate_center_view_cos requires torch"
    device, dtype = drone_pos.device, drone_pos.dtype
    F = torch.as_tensor(FLIP_NP, device=device, dtype=dtype)
    drone_pos_ned = drone_pos * F
    gate_pos_ned = gate_pos * F
    R_wb_ned = _to_R_wb_ned(drone_quat_or_R, is_quat)
    _, _, tz, _ = project_points_camera(gate_pos_ned, drone_pos_ned, R_wb_ned)   # (N,G) centre depth
    rng = torch.linalg.norm(gate_pos_ned - drone_pos_ned.unsqueeze(-2), dim=-1).clamp(min=1e-6)  # (N,G)
    return (tz / rng).clamp(-1.0, 1.0)
