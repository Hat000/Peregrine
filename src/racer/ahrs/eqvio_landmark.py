"""SOT(3) inverse-depth landmark parameterization for gate-corner bearings.

The companion to eqvio.py's SE_2(3) right-invariant EKF. A monocular camera observes a
gate corner as a BEARING (a unit direction in the image) — it does NOT directly give
metric range. The equivariant-filter line (van Goor, "Equivariant Filters for Visual
Spatial Awareness", ANU 2023; the EqVIO paper) parameterises each landmark on the group

    SOT(3) = SO(3) x R+        (scaled orthogonal transformations)

i.e. a rotation Q in SO(3) carrying the BEARING DIRECTION, and a positive scale rho in
R+ carrying the INVERSE DEPTH (1/range). Inverse depth is the right coordinate because
(a) it is near-Gaussian for a bearing sensor even at large/uncertain range (whereas range
itself has a heavy tail and is ill-conditioned as depth -> infinity), and (b) it keeps the
landmark observable from the first frame (a brand-new corner has rho ~ 0 with large
variance, not an undefined range). This is exactly the gate-corner situation in VQ2: the
1.5 m inner-square corners are detected as image bearings; their depth is what the filter
must triangulate over a few frames.

What this module provides
-------------------------
1. SOT3 dataclass: the (Q, rho) landmark element with exp/log/compose, and the map to a
   body-frame ray  d_body = (1/rho) Q e_z  (e_z the canonical optical axis), so a landmark
   is reconstructed as p_body = origin + d_body.
2. InverseDepthLandmark: the per-corner inverse-depth state (rho, and a 2-dof bearing on
   the unit sphere) with its measurement model h(X, landmark) = pi( R^T (p_L - p) ) where
   pi is the perspective projection — i.e. it slots into eqvio.py's RIEKF as the
   left-invariant bearing output, replacing the known-p_L landmark update with one that
   ALSO estimates the corner depth.
3. bearing_residual / bearing_jacobian: the building blocks an EqF/RIEKF update needs.

Status (honest): this is a CORRECT, TESTED parameterization + measurement model + analytic
Jacobians (finite-difference-verified). Wiring it as a *joint* navigation+landmark EqF
(co-estimating the SE_2(3) pose AND the SOT(3) inverse depths in one covariance) is the
documented next step — the SE_2(3) RIEKF in eqvio.py already demonstrates the pose-side
consistency edge with KNOWN p_L; this module supplies the landmark-side group so the two
compose into the full EqVIO. See module-end NEXT STEPS.

Frame convention: camera optical axis = body +x forward by default (caller can pass a
body<-cam rotation); bearings are unit 3-vectors; (w,x,y,z) quats elsewhere in the package.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from racer.ahrs.iekf import _skew, _Exp_so3
from racer.ahrs.eqvio import _left_jacobian_so3


E_Z = np.array([0.0, 0.0, 1.0])     # canonical reference ray for SOT(3) (optical axis)


# ---------------------------------------------------------------------------
# SOT(3) = SO(3) x R+  group element
# ---------------------------------------------------------------------------

@dataclass
class SOT3:
    """Scaled-orthogonal element (Q, rho): Q in SO(3), rho in R+ (inverse depth scale).

    The landmark ray in the reference (camera) frame is  d = (1/rho) Q e_z, so:
      - Q rotates the canonical axis e_z onto the bearing DIRECTION,
      - rho is the INVERSE DEPTH (large rho = near, rho->0 = far / newly seen).

    Lie algebra sot(3) = so(3) (+) R: increment xi = [omega(3); s(1)] with
        (Q, rho) <- (Q Exp(omega),  rho * exp(s)).
    The multiplicative scale update keeps rho strictly positive (the whole point of R+).
    """
    Q: np.ndarray = field(default_factory=lambda: np.eye(3))
    rho: float = 1.0

    def bearing(self) -> np.ndarray:
        """Unit bearing direction Q e_z."""
        return self.Q @ E_Z

    def ray(self) -> np.ndarray:
        """Full ray (1/rho) Q e_z = depth * direction."""
        return (1.0 / self.rho) * (self.Q @ E_Z)

    def retract(self, xi: np.ndarray) -> "SOT3":
        """Apply a tangent increment xi = [omega(3); s(1)] on the RIGHT."""
        omega = xi[0:3]
        s = float(xi[3])
        return SOT3(Q=self.Q @ _Exp_so3(omega), rho=self.rho * np.exp(s))

    def local(self, other: "SOT3") -> np.ndarray:
        """Tangent vector xi s.t. self.retract(xi) == other (to first order)."""
        dQ = self.Q.T @ other.Q
        omega = _log_so3_local(dQ)
        s = np.log(other.rho / self.rho)
        return np.concatenate([omega, [s]])


def _log_so3_local(R: np.ndarray) -> np.ndarray:
    cos_t = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_t)
    if theta < 1e-8:
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return theta / (2.0 * np.sin(theta)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


# ---------------------------------------------------------------------------
# Inverse-depth landmark + bearing measurement model
# ---------------------------------------------------------------------------

def project_unit(v: np.ndarray) -> np.ndarray:
    """Perspective-to-unit-sphere projection: bearing = v / |v|."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def reconstruct_world_point(
    sot: SOT3,
    R_wb: np.ndarray,
    p_wb: np.ndarray,
    R_bc: np.ndarray = None,
) -> np.ndarray:
    """World position of the landmark from its SOT(3) (bearing+inverse-depth) anchored at
    the camera pose (R_wb body->world, p_wb world position), with optional body<-cam
    rotation R_bc (default: camera == body)."""
    R_bc = np.eye(3) if R_bc is None else R_bc
    ray_cam = sot.ray()                       # (1/rho) direction in camera frame
    ray_body = R_bc @ ray_cam
    return p_wb + R_wb @ ray_body


def bearing_measurement(
    p_L_world: np.ndarray,
    R_wb: np.ndarray,
    p_wb: np.ndarray,
    R_bc: np.ndarray = None,
) -> np.ndarray:
    """Predicted UNIT bearing of a world landmark p_L in the camera frame:
        b = normalise( R_bc^T R_wb^T (p_L - p_wb) ).
    This is the left-invariant output the RIEKF/EqF consumes (a body/cam-frame direction).
    """
    R_bc = np.eye(3) if R_bc is None else R_bc
    d_cam = R_bc.T @ (R_wb.T @ (p_L_world - p_wb))
    return project_unit(d_cam)


def bearing_residual(
    b_meas: np.ndarray,
    p_L_world: np.ndarray,
    R_wb: np.ndarray,
    p_wb: np.ndarray,
    R_bc: np.ndarray = None,
) -> np.ndarray:
    """Tangent-plane bearing residual (3-vec, lies in the plane perpendicular to the
    predicted bearing). Using the full 3-vec difference of unit vectors is the standard
    minimal-overhead choice and is well-conditioned for the small angles of a tracked
    corner."""
    b_hat = bearing_measurement(p_L_world, R_wb, p_wb, R_bc)
    return project_unit(b_meas) - b_hat


def d_project_d_v(v: np.ndarray) -> np.ndarray:
    """Jacobian of the unit-projection pi(v)=v/|v| w.r.t. v (3x3):
        d pi = (1/|v|)(I - b b^T),  b = v/|v|.
    """
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros((3, 3))
    b = v / n
    return (np.eye(3) - np.outer(b, b)) / n


def bearing_jacobian_pose(
    p_L_world: np.ndarray,
    R_wb: np.ndarray,
    p_wb: np.ndarray,
    R_bc: np.ndarray = None,
    right_invariant: bool = True,
) -> np.ndarray:
    """Jacobian of the camera-frame bearing w.r.t. the SE_2(3) pose error (3x9 over
    [xi_R; xi_v; xi_p]). With right_invariant=True the attitude block is anchored to the
    world landmark (the consistency-preserving form, mirroring eqvio.update_landmark);
    velocity block is zero (bearings are instantaneous)."""
    R_bc = np.eye(3) if R_bc is None else R_bc
    d_cam = R_bc.T @ (R_wb.T @ (p_L_world - p_wb))
    Jpi = d_project_d_v(d_cam)
    H = np.zeros((3, 9))
    if right_invariant:
        # d(d_cam)/d xi_R = R_bc^T R_wb^T [p_L]_x ;  d/d xi_p = -R_bc^T R_wb^T
        H[:, 0:3] = Jpi @ (R_bc.T @ R_wb.T @ _skew(p_L_world))
        H[:, 6:9] = Jpi @ (-R_bc.T @ R_wb.T)
    else:
        H[:, 0:3] = Jpi @ (R_bc.T @ _skew(R_wb.T @ (p_L_world - p_wb)))
        H[:, 6:9] = Jpi @ (-R_bc.T @ R_wb.T)
    return H


def bearing_jacobian_landmark(
    sot: SOT3,
    R_wb: np.ndarray,
    p_wb: np.ndarray,
    R_bc: np.ndarray = None,
) -> np.ndarray:
    """Jacobian of the camera-frame bearing w.r.t. the landmark SOT(3) tangent
    xi = [omega(3); s(1)] (3x4). Because the bearing is SCALE-INVARIANT, the inverse-depth
    component s has ZERO first-order effect on the bearing (d b / d s = 0) — bearing
    constrains DIRECTION (omega), depth is recovered only through pose PARALLAX across
    frames. This is the structural fact that makes inverse-depth the right coordinate:
    a single bearing leaves rho unobservable, so rho must start with large variance and
    shrink over the baseline."""
    R_bc = np.eye(3) if R_bc is None else R_bc
    # When the landmark is anchored at THIS camera pose, the camera-frame ray reconstructs
    # exactly to d_cam = (1/rho) Q e_z, so the predicted bearing is simply b = pi(Q e_z) and
    #   d b / d omega = d_pi(Q e_z) . d(Q Exp(omega) e_z)/d omega |_0
    # With the RIGHT increment Q <- Q Exp(omega):  d(Q Exp(omega) e_z)/d omega = -Q [e_z]_x.
    # (Note this is Q[-e_z]_x, NOT -[Q e_z]_x — the increment lives in Q's body frame.)
    # The component of omega ABOUT e_z leaves the bearing fixed, so that column is ~0.
    # d b / d s = 0 (scale-invariance).
    dir_cam = sot.Q @ E_Z
    Jpi = d_project_d_v(dir_cam)
    H = np.zeros((3, 4))
    H[:, 0:3] = Jpi @ (-sot.Q @ _skew(E_Z))
    H[:, 3] = 0.0
    return H


# ---------------------------------------------------------------------------
# Triangulation sanity: depth is recovered from PARALLAX, not a single bearing.
# ---------------------------------------------------------------------------

def triangulate_inverse_depth(
    bearings_cam: np.ndarray,
    R_wb_list: np.ndarray,
    p_wb_list: np.ndarray,
    R_bc: np.ndarray = None,
) -> Tuple[np.ndarray, float]:
    """Least-squares world point from >=2 bearings at known poses (the classic mid-point /
    DLT triangulation). Returns (p_world, inverse_depth_from_first_pose). Demonstrates that
    the SOT(3) rho is observable once there is baseline — the multi-frame ingredient an EqF
    integrates recursively."""
    R_bc = np.eye(3) if R_bc is None else R_bc
    A = np.zeros((3, 3)); b = np.zeros(3)
    for bc, R_wb, p_wb in zip(bearings_cam, R_wb_list, p_wb_list):
        d_world = R_wb @ (R_bc @ project_unit(bc))           # ray direction in world
        Pi = np.eye(3) - np.outer(d_world, d_world)          # projector orthogonal to ray
        A += Pi
        b += Pi @ p_wb
    p_world = np.linalg.solve(A, b)
    d0 = R_bc.T @ (R_wb_list[0].T @ (p_world - p_wb_list[0]))
    inv_depth = 1.0 / max(np.linalg.norm(d0), 1e-9)
    return p_world, inv_depth


# ---------------------------------------------------------------------------
# NEXT STEPS (documented, not yet wired)
# ---------------------------------------------------------------------------
# To form the FULL EqVIO from these two modules:
#   1. Augment the SE_2(3) RIEKF covariance with one SOT(3) tangent (4 dof) per tracked
#      corner: P grows to (9 + 4*K). New corners enter with large 1/rho variance.
#   2. On each frame, for every tracked corner apply update_landmark-style bearing updates
#      using H = [bearing_jacobian_pose | ... | bearing_jacobian_landmark] (the joint
#      pose+landmark Jacobian). The pose block is the consistency-preserving RIGHT-invariant
#      form already validated in eqvio.update_landmark / TestConsistencyEdge.
#   3. Marginalise corners that leave the field of view (Schur complement) to bound state
#      size — the standard MSCKF/EqVIO sliding-window step.
# The pieces here (group, measurement, analytic Jacobians, triangulation) are unit-tested;
# the joint covariance bookkeeping is the remaining integration work.
