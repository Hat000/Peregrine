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

Extension stubs (NOT implemented — Fengyou's decision point):
  - RIANN / GRU learned gyro-denoising (van Goor ANU thesis / Brossard et al. 2020).
  - SE_2(3) invariant EKF / EqVIO (the IEKF's consistency advantage materialises on the
    COUPLED attitude+velocity+position problem; the SO(3) iekf here is the attitude core).
  These are the natural next-tier after this T1 bench is validated on real twin IMU data
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
]
