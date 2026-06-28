"""AHRS (Attitude and Heading Reference System) filters for VQ2 self-attitude estimation.

VQ2's competitive wire BLOCKS the ATTITUDE MAVLink message (VADR-TS-003 §9.3).
The LinearKF in state_estimator.py trusts given attitude and CANNOT run on the VQ2
wire — so this package builds the missing AHRS layer: estimate orientation from
HIGHRES_IMU (raw accel/gyro; §4.3 confirmed available on the scored wire).

Sub-modules
-----------
imu_gen      -- synthetic high-g IMU trajectory generator (GT-quaternion seeded).
eskf         -- Error-State Kalman Filter AHRS (gyro propagation + accel tilt update
                with two-gate high-g robustness + optional magnetometer yaw update).
iekf         -- Left-Invariant EKF AHRS on SO(3) (Barrau-Bonnabel / van Goor EqF line);
                geometrically-consistent contender; matches the ESKF on attitude-only.
classical    -- Madgwick and Mahony complementary filters (benchmark comparators).
metrics      -- geodesic attitude error (angle-axis), per-filter scoring helpers.

eqvio        -- SE_2(3) RIGHT-invariant EKF (attitude+velocity+position) + a standard-EKF
                foil + a 6-DoF synthetic bench. This is where the invariant filter's
                CONSISTENCY edge materialises (the SO(3) iekf deliberately ties the ESKF):
                the RIEKF's covariance transport is group-affine EXACT (NEES constant under
                propagation) where the standard EKF drifts/mis-calibrates. See traj6dof.
eqvio_landmark-- SOT(3)=SO(3)xR+ inverse-depth gate-corner bearing parameterization (van Goor
                EqVIO line): group + bearing measurement + analytic Jacobians + triangulation.
                The landmark-side group that composes with eqvio's SE_2(3) pose into a full
                EqVIO (joint-covariance wiring is the documented next step).
traj6dof     -- full 6-DoF synthetic trajectory generator (extends imu_gen to v/p GT) +
                NEES/consistency evaluation harness for the SE_2(3) filters.

Extension stubs (NOT implemented — Fengyou's decision point):
  - RIANN / GRU learned gyro-denoising (van Goor ANU thesis / Brossard et al. 2020).
  - FULL EqVIO: augment eqvio's SE_2(3) RIEKF covariance with one eqvio_landmark.SOT3 per
    tracked corner (joint pose+landmark filter). Pose-side consistency + landmark group +
    Jacobians are done & tested; the joint covariance/marginalisation bookkeeping remains.
  These are the natural next-tier after this bench is validated on real twin IMU data
  (plug in via imu_gen.IMUSequence).

Frame convention (matches frames.py throughout):
  - Body frame: FRD (Forward-Right-Down), like the rest of the stack.
  - World frame: NED (North-East-Down).
  - Quaternion layout: (w, x, y, z) scalar-FIRST, matching MAVLink / frames.py convention.
  - Gravity in NED: [0, 0, +9.80665] m/s^2 (down = +Z).
  - Specific force at rest: [0, 0, -g] in body FRD (IMU reports gravity up, sensor down).
"""
from racer.ahrs.imu_gen import IMUSequence, generate_imu_sequence, Scenario
from racer.ahrs.eskf import ESKFAHRS
from racer.ahrs.iekf import LeftInvariantEKF
from racer.ahrs.classical import MadgwickAHRS, MahonyAHRS
from racer.ahrs.metrics import geodesic_error_rad, score_filter
from racer.ahrs.eqvio import SE23RightInvariantEKF, SE23StandardEKF
from racer.ahrs.eqvio_landmark import SOT3
from racer.ahrs.traj6dof import Traj6DoF, generate_traj6dof, run_se23_filter, Traj

__all__ = [
    "IMUSequence",
    "generate_imu_sequence",
    "Scenario",
    "ESKFAHRS",
    "LeftInvariantEKF",
    "MadgwickAHRS",
    "MahonyAHRS",
    "geodesic_error_rad",
    "score_filter",
    "SE23RightInvariantEKF",
    "SE23StandardEKF",
    "SOT3",
    "Traj6DoF",
    "generate_traj6dof",
    "run_se23_filter",
    "Traj",
]
