"""A32 byte-identity replay scenarios — shared between the reference-fixture generator (run
against the PRE-A32 tree, f96962d) and the permanent off-path pin in
tests/test_a32_robust_estimation.py (run against the post-A32 tree).

Every scenario drives ONLY pre-A32 API surface with every A32 flag at its default (off), so the
recorded streams must be BIT-IDENTICAL before and after the A32 edits:
  * scenario_eskf        — the shared ESKF AHRS (VQ1 bench path + the A8 motion-reject path)
  * scenario_seeker      — detect_gate_lever track-continuity streams (VQ1 default + the frozen
                           A31 vq2 bundle with the IMU-consistency bearing gate ON)
  * scenario_vert        — VerticalEstimator predict/latch streams (legacy A25 + A28 zoff filter)

Imports of racer.* are LAZY (inside the functions) so the caller controls which tree resolves.
"""
from __future__ import annotations

import numpy as np

_NS = 1_000_000_000


def scenario_eskf() -> dict:
    """Deterministic IMU stream through the ESKF: quiet -> maneuvering -> high-g -> free-fall ->
    quiet, exercising the magnitude band, the gate weight, the chi2 gate and (for the reject
    variant) the A8 motion inflation + R_ref anchoring. Records the quaternion + bias each step."""
    from racer.ahrs.eskf import ESKFAHRS, GRAVITY

    out = {}
    for name, kwargs in (("bare", {}),
                         ("reject", {"use_accel_motion_reject": True})):
        f = ESKFAHRS(gyro_noise_std=0.01, accel_gate_alpha=10.0, **kwargs)
        rng = np.random.default_rng(1234)
        qs, bs = [], []
        dt = 0.055
        for k in range(700):
            t = k * dt
            gyro = np.array([0.3 * np.sin(0.7 * t), 0.2 * np.cos(1.1 * t),
                             0.15 * np.sin(0.4 * t)]) + 0.01 * rng.standard_normal(3)
            accel = np.array([0.0, 0.0, -GRAVITY]) + 0.05 * rng.standard_normal(3)
            if 150 <= k < 250:                       # sustained linear accel, |a| near g
                accel += np.array([2.5, 0.5, 0.0])
            elif 300 <= k < 320:                     # high-g burst
                accel *= 3.2
            elif 400 <= k < 415:                     # near free-fall
                accel *= 0.05
            elif 500 <= k < 560:                     # direction-liar with |a| ~ g
                accel = GRAVITY * np.array([0.6, 0.0, -0.8]) + 0.05 * rng.standard_normal(3)
            f.step(gyro, accel, dt)
            qs.append(f.q_wxyz)
            bs.append(f.gyro_bias)
        out[f"eskf_{name}_q"] = np.asarray(qs, dtype=np.float64)
        out[f"eskf_{name}_b"] = np.asarray(bs, dtype=np.float64)
    return out


def _pose_cam(az, el, r, t_ns, fid):
    from racer.contracts import GatePose
    d = np.array([np.tan(float(az)), np.tan(float(el)), 1.0])
    t_cam = float(r) * d / np.linalg.norm(d)
    return GatePose(frame_id=int(fid), sim_time_ns=int(t_ns),
                    R_cam_gate=np.eye(3), t_cam_gate=t_cam, reproj_error_px=0.0)


def scenario_seeker() -> dict:
    """detect_gate_lever continuity streams: a smoothly-sweeping tracked gate with periodic hop
    candidates + occasional dropouts, run through (a) the VQ1 default config and (b) the frozen
    A31 vq2 bundle (IMU-consistency bearing gate ON, attitude history fed). Records per-frame
    [chosen(0/1), az, el, range, track_range, coast_ticks]."""
    from racer.gate_seeker import GateSeeker, GateSeekerConfig

    frozen_a31 = {"egress_freeze_attitude": True, "hold_last_demand_s": 0.6,
                  "true_attitude_from_ahrs": True, "use_image_servo_lateral": True,
                  "total_accel_cap_mps2": 2.0, "pursuit_yaw_slew_rps": 1.5,
                  "pass_wire_coast_s": 0.25, "pass_coast_accel_mps2": 0.0,
                  "reramp_forward_after_pass": True, "forward_accel_mps2": 0.8,
                  "orbit_guard_rad": 1.75, "orbit_break_s": 1.0, "orbit_yaw_clamp_rad": 2.4,
                  "use_imu_bearing_gate": True, "image_lat_slew_mps3": 6.0,
                  "hold_thrust_lo_frac": 0.90, "hold_thrust_hi_frac": 1.12}
    out = {}
    for name, overrides in (("default", {}), ("vq2_a31", frozen_a31)):
        cfg = GateSeekerConfig(**overrides)
        s = GateSeeker(config=cfg)
        s.detector = object()
        rows = []
        dt_ns = _NS // 20
        for k in range(160):
            t_ns = k * dt_ns
            s._append_att_hist(t_ns, (0.0, 0.0, 0.002 * k))
            az = 0.05 + 0.02 * np.sin(0.15 * k)
            el = 0.02 * np.cos(0.1 * k)
            r = max(12.0 - 0.05 * k, 3.0)
            poses = [_pose_cam(az, el, r, t_ns, fid=k)]
            if k % 11 == 5:                               # a hop candidate rides along
                poses.append(_pose_cam(az + 0.45, el - 0.2, r + 4.0, t_ns, fid=k))
            if k % 17 == 9:                               # the hop is the ONLY candidate
                poses = [_pose_cam(az - 0.5, el + 0.25, r, t_ns, fid=k)]
            if k % 23 == 20:                              # dropout
                poses = []
            s._valid_poses = lambda frame, _p=poses: _p
            from racer.contracts import Frame
            frame = Frame(frame_id=k, sim_time_ns=t_ns,
                          image_bgr=np.ones((4, 4, 3), dtype=np.uint8))
            chosen = s.detect_gate_lever(frame)
            if chosen is None:
                rows.append([0.0, np.nan, np.nan, np.nan,
                             s._track_range_m if s._track_range_m is not None else np.nan,
                             float(s._track_coast_ticks)])
            else:
                b = GateSeeker._pose_bearing(chosen)
                rows.append([1.0, float(b[0]), float(b[1]), float(chosen.range_m),
                             s._track_range_m if s._track_range_m is not None else np.nan,
                             float(s._track_coast_ticks)])
        out[f"seeker_{name}"] = np.asarray(rows, dtype=np.float64)
    return out


def scenario_vert() -> dict:
    """VerticalEstimator predict/latch streams: bias capture, contact spikes, honest latches,
    gross (>2 m) innovation jumps driving the reject/reseed machinery. Records
    [vz, z_off, miss] per tick for the legacy A25 path and the A28 zoff-filter path."""
    from racer.vertical_estimator import VerticalEstimator

    out = {}
    for name, kwargs in (("legacy", {}),
                         ("zoff", {"use_zoff_filter": True, "export_clip_mps": 2.5})):
        v = VerticalEstimator(**kwargs)
        v.seed()
        rng = np.random.default_rng(99)
        rows = []
        dt = 0.055
        for k in range(400):
            a_up = 0.3 * np.sin(0.2 * k) + 0.1 * rng.standard_normal()
            if 200 <= k < 203:
                a_up = 25.0                               # contact spike
            v.predict(float(a_up), dt)
            if k >= 30 and k % 5 == 0:
                off = 1.0 + 0.4 * np.sin(0.05 * k) + 0.05 * rng.standard_normal()
                if 250 <= k < 275:
                    off += 8.0                            # gate-track jump (reject -> reseed)
                v.latch_offset(float(off), 0.1)
            z = v.z_off
            rows.append([v.vz, z if np.isfinite(z) else np.nan, float(v.zoff_miss)])
        out[f"vert_{name}"] = np.asarray(rows, dtype=np.float64)
    return out


def record_all() -> dict:
    d = {}
    d.update(scenario_eskf())
    d.update(scenario_seeker())
    d.update(scenario_vert())
    return d
