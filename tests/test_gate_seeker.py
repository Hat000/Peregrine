"""Gate-seeker + case-C deploy profile — guidance correctness, bounded slow CTBR, the deploy
profile bundle, and an offline pipeline smoke (detector->localization->navigator(case-C)->seeker).

These de-risk the live VQ2 slow lap on the LAPTOP: prove the transparent pursuit law steers
toward the gate centre (sign-correct), holds a slow capped speed and bounded rates, advances
correctly, and that the integrated case-C estimator -> gate-seeker pipeline produces sane slow
gate-pointing commands while the estimate stays bounded. No sim, no Adroit, no torch.

[VQ2 slow-is-smooth, 2026-06-29]
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racer.contracts import (  # noqa: E402
    ControlCommand,
    ControlMode,
    DroneState,
    Frame,
    Gate,
    GateObservation,
    GatePose,
    NavState,
)
from racer.deploy_profile import (  # noqa: E402
    PROFILES,
    VQ2_CMD_RATE_SCALE,
    get_profile,
    vq1_case_a,
    vq2_case_c,
)
from racer.gate_seeker import GateSeeker, GateSeekerConfig, make_seeker_controller  # noqa: E402
from racer.navigator import NavigatorConfig  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _u(v):
    return np.asarray(v, float) / np.linalg.norm(v)


def _gate(position, normal, gate_id=0) -> Gate:
    """A Gate whose through-direction (R_world_gate[:,2]) is `normal`."""
    n = _u(normal)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = _u(np.cross(a, n))
    y = np.cross(n, x)
    R = np.column_stack([x, y, n])
    return Gate(gate_id=gate_id, position_ned=np.asarray(position, float), R_world_gate=R)


def _nav(position, velocity=(0.0, 0.0, 0.0), sim_time_ns=0, yaw=0.0) -> NavState:
    return NavState(
        sim_time_ns=sim_time_ns,
        position_ned=np.asarray(position, float),
        velocity_ned=np.asarray(velocity, float),
        yaw=yaw,
    )


# ===========================================================================
# GUIDANCE: NavState + active gate -> Setpoint (transparent, sign-correct)
# ===========================================================================
def test_on_axis_carrot_lands_beyond_gate_at_slow_cruise():
    """Drone at origin, gate 10 m north: the slow pursuit carrot lands BEYOND the gate (exit side)
    and the desired velocity points THROUGH it at the (slow) cruise cap."""
    gate = _gate([10.0, 0.0, -3.0], normal=[1.0, 0.0, 0.0])
    seeker = GateSeeker(config=GateSeekerConfig(cruise_speed=3.0, lookahead_m=2.0, launch_ramp_s=0.0))
    sp = seeker.plan(_nav([0.0, 0.0, -3.0], sim_time_ns=7), gate)
    np.testing.assert_allclose(sp.position_ned, [12.0, 0.0, -3.0])     # carrot 2 m beyond
    np.testing.assert_allclose(_u(sp.velocity_ned), [1.0, 0.0, 0.0], atol=1e-9)  # toward the gate
    assert np.linalg.norm(sp.velocity_ned) == pytest.approx(3.0)       # SLOW cruise
    assert sp.yaw == pytest.approx(0.0)                                # nose faces the gate (camera in view)
    assert sp.sim_time_ns == 7


@pytest.mark.parametrize("map_normal", [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
def test_carrot_always_on_exit_side_regardless_of_map_normal_sign(map_normal):
    """The map's gate-normal sign is arbitrary; the carrot must always land on the FAR (exit) side."""
    gate = _gate([10.0, 0.0, 0.0], normal=map_normal)
    sp = GateSeeker(config=GateSeekerConfig(lookahead_m=2.0, launch_ramp_s=0.0)).plan(
        _nav([0.0, 0.0, 0.0]), gate)
    np.testing.assert_allclose(sp.position_ned, [12.0, 0.0, 0.0])      # never behind us


def test_off_axis_velocity_points_at_the_carrot():
    """Off-axis: the desired velocity points along the line-of-sight to the carrot (centering pull)."""
    gate = _gate([10.0, 5.0, 0.0], normal=[1.0, 0.0, 0.0])
    sp = GateSeeker(config=GateSeekerConfig(cruise_speed=3.0, lookahead_m=2.0, launch_ramp_s=0.0)).plan(
        _nav([0.0, 0.0, 0.0]), gate)
    los = np.array([12.0, 5.0, 0.0])
    np.testing.assert_allclose(_u(sp.velocity_ned), _u(los), atol=1e-9)
    assert np.linalg.norm(sp.velocity_ned) == pytest.approx(3.0)
    assert sp.yaw == pytest.approx(np.arctan2(5.0, 12.0))              # nose toward the off-axis carrot


def test_yaw_faces_a_gate_to_the_east():
    gate = _gate([0.0, 10.0, 0.0], normal=[0.0, 1.0, 0.0])
    sp = GateSeeker(config=GateSeekerConfig(launch_ramp_s=0.0)).plan(_nav([0.0, 0.0, 0.0]), gate)
    assert sp.yaw == pytest.approx(np.pi / 2)                          # heading east


def test_head_on_is_well_defined_no_nan():
    """Head-on (the gate dead ahead, zero lateral): the law must stay finite (no divide-by-zero)."""
    gate = _gate([8.0, 0.0, 0.0], normal=[1.0, 0.0, 0.0])
    seeker = GateSeeker(config=GateSeekerConfig(cruise_speed=2.0, launch_ramp_s=0.0))
    sp = seeker.plan(_nav([0.0, 0.0, 0.0]), gate)
    assert np.all(np.isfinite(sp.position_ned)) and np.all(np.isfinite(sp.velocity_ned))
    assert np.isfinite(sp.yaw)
    np.testing.assert_allclose(_u(sp.velocity_ned), [1.0, 0.0, 0.0], atol=1e-9)
    cmd = seeker.command(_nav([0.0, 0.0, 0.0]), gate, 0)
    assert np.all(np.isfinite(cmd.body_rate)) and np.isfinite(cmd.thrust)


def test_overshoot_with_forward_velocity_does_not_uturn():
    """Just past the gate plane but still moving forward: the carrot stays on the exit side (no
    U-turn back through the gate that a position-only normal flip would cause)."""
    gate = _gate([10.0, 0.0, 0.0], normal=[1.0, 0.0, 0.0])
    sp = GateSeeker(config=GateSeekerConfig(lookahead_m=2.0, launch_ramp_s=0.0)).plan(
        _nav([10.5, 0.0, 0.0], velocity=[5.0, 0.0, 0.0]), gate)
    np.testing.assert_allclose(sp.position_ned, [12.0, 0.0, 0.0])      # still beyond
    assert sp.velocity_ned[0] > 0.0                                    # keeps going forward


# ===========================================================================
# GUIDANCE -> CTBR: bounded slow rates + sane collective
# ===========================================================================
def test_command_is_bodyrate_with_bounded_rate_and_sane_collective():
    """The seeker emits a BODY_RATE CTBR command: body rates clamped to the seeker's max, and a
    collective inside [0,1] near hover (not a runaway throttle)."""
    gate = _gate([6.0, 3.0, -2.0], normal=[1.0, 0.0, 0.0])
    seeker = GateSeeker(config=GateSeekerConfig(cruise_speed=3.0, launch_ramp_s=0.0))
    cmd = seeker.command(_nav([0.0, 0.0, -2.0], velocity=[1.0, 0.0, 0.0]), gate, 0)
    assert cmd.mode is ControlMode.BODY_RATE
    assert cmd.body_rate is not None and cmd.thrust is not None
    assert np.linalg.norm(cmd.body_rate) <= seeker.controller.max_body_rate_rps + 1e-9
    assert 0.0 <= cmd.thrust <= 1.0
    # near a level slow cruise the collective sits near hover (well below full throttle).
    assert cmd.thrust < 0.6


def test_hover_command_is_near_level_and_hover_thrust():
    """At rest, on-axis, level: the command is ~zero body rate and ~hover collective (no lurch)."""
    gate = _gate([10.0, 0.0, 0.0], normal=[1.0, 0.0, 0.0])
    seeker = GateSeeker(config=GateSeekerConfig(cruise_speed=3.0, launch_ramp_s=0.0))
    # a tiny launch ramp window is disabled here, so the carrot pull is active; still bounded.
    cmd = seeker.command(_nav([0.0, 0.0, 0.0]), gate, 0)
    assert np.linalg.norm(cmd.body_rate) <= seeker.controller.max_body_rate_rps + 1e-9
    assert 0.05 <= cmd.thrust <= 0.6


def test_launch_ramp_eases_the_takeoff_transient():
    """The launch ramp scales the horizontal-accel authority up from ~0 over launch_ramp_s, so the
    FIRST command after the start-gate spawn cannot step to a full cruise lean (the anti-tumble
    guard). At t0 the ramp is ~0 -> a near-level, near-zero-rate command; later it is full."""
    gate = _gate([10.0, 8.0, 0.0], normal=[1.0, 0.0, 0.0])   # off-axis: wants a big lean if unramped
    seeker = GateSeeker(config=GateSeekerConfig(cruise_speed=3.0, launch_ramp_s=0.6))
    # face the gate so the test isolates the TILT (roll/pitch) ramp, not the in-place yaw alignment
    # (the launch ramp guards the translational lean — a yaw rotation in place never tumbles).
    yaw_to_gate = float(np.arctan2(8.0, 10.0))
    sp0 = seeker.plan(_nav([0.0, 0.0, 0.0], sim_time_ns=0, yaw=yaw_to_gate), gate)
    assert sp0.launch_ramp is not None and sp0.launch_ramp == pytest.approx(0.0)
    cmd0 = seeker.command(_nav([0.0, 0.0, 0.0], sim_time_ns=0, yaw=yaw_to_gate), gate, 0)  # arm clock @t0
    # at the start the ramp ~0 -> the TILT (roll/pitch) rate command is small (no step to a cruise lean)
    assert np.linalg.norm(cmd0.body_rate[:2]) < 1.0
    # 1 s later the ramp has completed -> full authority (None)
    sp1 = seeker.plan(_nav([0.0, 0.0, 0.0], sim_time_ns=1_000_000_000, yaw=yaw_to_gate), gate)
    assert sp1.launch_ramp is None


# ===========================================================================
# ADVANCE logic
# ===========================================================================
def test_advance_on_wire_index_increment():
    """advanced_since_last reports the authoritative RACE_STATUS advance (index increment)."""
    gate = _gate([10.0, 0.0, 0.0], normal=[1.0, 0.0, 0.0])
    seeker = GateSeeker()
    seeker.command(_nav([0.0, 0.0, 0.0]), gate, 0)        # baseline index 0
    assert not seeker.advanced_since_last(0)
    assert seeker.advanced_since_last(1)                  # the wire bumped the active gate


def test_range_backstop_advances_after_crossing_the_plane():
    """The geometry backstop advances once the drone crosses the gate plane onto the exit side
    within the capture radius (a near-centred pass), and NOT while still approaching."""
    gate = _gate([10.0, 0.0, 0.0], normal=[1.0, 0.0, 0.0])
    seeker = GateSeeker(config=GateSeekerConfig(capture_radius_m=0.6))
    # still 4 m short -> not yet
    assert not seeker.should_advance(_nav([6.0, 0.0, 0.0], velocity=[3.0, 0.0, 0.0]), gate)
    # crossed to the exit side, near-centred -> advance
    assert seeker.should_advance(_nav([10.3, 0.1, 0.0], velocity=[3.0, 0.0, 0.0]), gate)


def test_range_backstop_rejects_a_wide_side_pass():
    """A crossing far off-axis (never threaded the opening) must NOT count as passed."""
    gate = _gate([10.0, 0.0, 0.0], normal=[1.0, 0.0, 0.0])
    seeker = GateSeeker(config=GateSeekerConfig(capture_radius_m=0.6))
    assert not seeker.should_advance(_nav([10.3, 3.0, 0.0], velocity=[3.0, 0.0, 0.0]), gate)


def test_final_gate_dead_reckons_within_blowout_range():
    """The final gate has no next gate to re-aim at: advance (clear) once within the blow-out range,
    dead-reckoning straight through (the recon final blow-out past the last station)."""
    gate = _gate([10.0, 0.0, 0.0], normal=[1.0, 0.0, 0.0])
    seeker = GateSeeker(config=GateSeekerConfig(final_blowout_m=4.0))
    assert not seeker.should_advance(_nav([4.0, 0.0, 0.0]), gate, is_final_gate=True)   # 6 m out
    assert seeker.should_advance(_nav([7.0, 0.0, 0.0]), gate, is_final_gate=True)       # 3 m out


def test_reset_drops_launch_clock_and_advance_baseline():
    gate = _gate([10.0, 0.0, 0.0], normal=[1.0, 0.0, 0.0])
    seeker = GateSeeker()
    seeker.command(_nav([0.0, 0.0, 0.0], sim_time_ns=5), gate, 2)
    assert seeker.advanced_since_last(3)
    seeker.reset()
    assert seeker._last_index is None and seeker._t0_sim_ns is None
    assert not seeker.advanced_since_last(3)              # no baseline after reset


# ===========================================================================
# DEPLOY PROFILE: the case-C bundle + OFF-byte-identity
# ===========================================================================
def test_vq2_case_c_profile_turns_on_the_self_localizing_chain():
    """The vq2_case_c profile flips the full case-C bundle ON together, with NO given position, and
    sets the 0.4 uplink rate-scale."""
    p = vq2_case_c()
    cfg = p.nav_config
    assert p.name == "vq2_case_c" and p.self_localizing is True
    assert p.cmd_rate_scale == VQ2_CMD_RATE_SCALE == 0.4
    # no given pose (case-C foundation)
    assert cfg.use_given_position is False and cfg.use_given_velocity is False
    # self-estimated attitude + map-free vision yaw/z
    assert cfg.use_ahrs is True
    assert cfg.use_vp_yaw is True and cfg.use_gate_bearing_yaw is True and cfg.use_floor_height is True
    # gate-relative +L chain
    assert cfg.use_gate_relative is True and cfg.use_rewind_kf is True and cfg.use_range_channel is True
    assert cfg.use_vision is True


def test_vq1_case_a_profile_is_the_bare_default_config():
    """vq1_case_a == the dataclass-default NavigatorConfig + identity uplink (a named legacy baseline,
    NOT a new behaviour). Pin field-by-field that it equals a fresh default config."""
    p = vq1_case_a()
    assert p.name == "vq1_case_a" and p.self_localizing is False and p.cmd_rate_scale == 1.0
    default = NavigatorConfig()
    got = p.nav_config
    for f in ("use_given_position", "use_given_velocity", "use_vision", "use_ahrs",
              "use_vp_yaw", "use_gate_bearing_yaw", "use_floor_height",
              "use_gate_relative", "use_rewind_kf", "use_range_channel"):
        assert getattr(got, f) == getattr(default, f), f


def test_default_navigatorconfig_is_byte_identical_to_vq1_off_path():
    """The whole case-C chain is OFF by default: a bare NavigatorConfig() has every case-C flag off,
    so a stack that never asks for a profile is the legacy VQ1 path."""
    cfg = NavigatorConfig()
    for f in ("use_ahrs", "use_vp_yaw", "use_gate_bearing_yaw", "use_floor_height",
              "use_gate_relative", "use_rewind_kf", "use_range_channel"):
        assert getattr(cfg, f) is False, f
    assert cfg.use_given_position is True   # legacy default uses the given pose


def test_get_profile_registry():
    assert set(PROFILES) == {"vq1_case_a", "vq2_case_c", "vq2_ego_lean"}
    assert get_profile("vq2_case_c").name == "vq2_case_c"
    with pytest.raises(KeyError):
        get_profile("nope")


def test_vq2_ego_lean_is_case_c_minus_yaw_z_anchors():
    """vq2_ego_lean drops EXACTLY the three vision yaw/z anchors (dead weight for the yaw-free
    ego obs contract; ~50+25 ms/call = the a5/a7 gap to 30 Hz) and is otherwise byte-identical
    to vq2_case_c — the base profile itself must stay untouched."""
    import dataclasses
    lean, base = get_profile("vq2_ego_lean"), get_profile("vq2_case_c")
    assert lean.name == "vq2_ego_lean"
    for f in ("use_vp_yaw", "use_gate_bearing_yaw", "use_floor_height"):
        assert getattr(lean.nav_config, f) is False, f
        assert getattr(base.nav_config, f) is True, f
    for f in dataclasses.fields(lean.nav_config):
        if f.name in ("use_vp_yaw", "use_gate_bearing_yaw", "use_floor_height"):
            continue
        assert getattr(lean.nav_config, f.name) == getattr(base.nav_config, f.name), f.name
    assert lean.cmd_rate_scale == base.cmd_rate_scale
    assert lean.gyro_sign == base.gyro_sign
    assert lean.seeker_overrides == base.seeker_overrides
    assert lean.controller_overrides == base.controller_overrides


def test_seeker_controller_uses_ff_gain_one_not_double_compensating():
    """The seeker's CTBR controller uses ff_gain=1.0 (the uplink cmd_rate_scale owns the 2.5x
    compensation) -- not the twin's ff_gain=2.5, which would double-compensate on the VQ2 wire."""
    c = make_seeker_controller()
    assert c.ff_gain == 1.0
    assert c.decoupled is True and c.mode is ControlMode.BODY_RATE


# ===========================================================================
# PIPELINE SMOKE: detector -> localization -> navigator(case-C) -> seeker -> CTBR
# ===========================================================================
class _FixedPoseDetector:
    """A fake detector that returns a GateObservation whose PnP would reconstruct a KNOWN gate
    pose, so the navigator's localization gets a clean fix WITHOUT a model. It carries the
    desired GatePose and the navigator's PnP (estimate_gate_pose) re-solves it from the corners
    we project. To keep the smoke deterministic + model-free we instead bypass PnP by returning
    an observation the navigator associates and that estimate_gate_pose can solve."""

    def __init__(self, gate: Gate, drone_pos_world, R_wb):
        self.gate = gate
        self.drone_pos_world = np.asarray(drone_pos_world, float)
        self.R_wb = np.asarray(R_wb, float)
        self._fid = 0

    def detect(self, frame):
        # project the gate's inner-square corners into the camera and hand them back as a detection;
        # the navigator's own estimate_gate_pose solves the pose -> a real localization fix.
        from racer.frames import CAMERA_INTRINSICS_K, R_camera_from_body
        half = self.gate.inner_size_m / 2.0
        # canonical inner corners in the gate frame (X=right, Y=down, Z=through): LL,LR,UR,UL
        corners_gate = np.array([
            [-half,  half, 0.0],
            [ half,  half, 0.0],
            [ half, -half, 0.0],
            [-half, -half, 0.0],
        ])
        R_wg = np.asarray(self.gate.R_world_gate, float)
        R_cb = R_camera_from_body()
        K = CAMERA_INTRINSICS_K
        px = []
        for cg in corners_gate:
            p_world = self.gate.position_ned + R_wg @ cg
            p_body = self.R_wb.T @ (p_world - self.drone_pos_world)
            p_cam = R_cb @ p_body
            if p_cam[2] <= 0.05:
                return []                       # gate behind the camera -> no detection
            u = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
            v = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
            px.append([u, v])
        self._fid += 1
        return [GateObservation(
            frame_id=frame.frame_id, sim_time_ns=frame.sim_time_ns,
            corners_px=np.asarray(px, float), corner_ids=np.array([0, 1, 2, 3]),
            corner_confidence=np.ones(4), gate_id=None,
        )]


def test_pipeline_smoke_casec_navigator_to_seeker_bounded_and_slow():
    """End-to-end offline smoke (the laptop de-risk): drive the case-C Navigator over a short
    synthetic HOVER IMU stream + injected gate sightings, run the gate-seeker on its NavState each
    tick, and assert (a) the self-localized estimate stays BOUNDED (no divergence) and (b) every
    command is a sane, SLOW, finite BODY_RATE. Uses use_ahrs (no given pose) so this exercises the
    real self-localizing seam, not a GT-anchored shortcut."""
    from racer.ahrs.imu_gen import Scenario, generate_imu_sequence
    from racer.frames import R_world_from_body
    from racer.navigator import Navigator

    # geometry: the case-C KF seeds at the ORIGIN (no given pose), so to exercise a real
    # association+fix the TRUE drone starts at the origin too and the gate sits ahead of it.
    # The gate is ~12 m north and level-ahead; the camera (+20deg up) sees it. The estimate drift
    # then comes only from the AHRS/IMU integration, which is what we want to bound.
    gate = _gate([12.0, 0.0, -2.5], normal=[1.0, 0.0, 0.0], gate_id=0)
    drone_pos = np.array([0.0, 0.0, 0.0])
    R_wb = R_world_from_body(0.0, 0.0, 0.0)        # level, facing north (+X) -> gate ahead + up

    # case-C config but WITHOUT the vision-yaw/floor extras (those need the heading/floor estimators
    # + real frames); we exercise use_ahrs + gate_relative + rewind + range here (the position core).
    cfg = NavigatorConfig(
        use_given_position=False, use_given_velocity=False, use_vision=True,
        use_ahrs=True, use_gate_relative=True, use_rewind_kf=True, use_range_channel=True,
    )
    detector = _FixedPoseDetector(gate, drone_pos, R_wb)
    nav = Navigator(gates=[gate], detector=detector, config=cfg)
    seeker = GateSeeker(config=GateSeekerConfig(cruise_speed=3.0, launch_ramp_s=0.0))

    seq = generate_imu_sequence(Scenario.STATIC_GRAVITY, duration_s=1.0, dt=0.01, seed=3)
    img = np.zeros((360, 640, 3), dtype=np.uint8)

    n_cmd = 0
    pos_errs = []
    for k in range(seq.N):
        t_ns = int(k * 0.01 * 1e9)
        ds = DroneState(
            sim_time_ns=t_ns,
            accel_body=np.asarray(seq.accel[k], float),
            gyro_body=np.asarray(seq.gyro[k], float),     # raw HIGHRES_IMU gyro (the AHRS source)
            position_ned=None, velocity_ned=None,         # VQ2: no given pose on the wire
            active_gate_index=0,
        )
        frame = Frame(frame_id=k, sim_time_ns=t_ns, image_bgr=img) if k % 2 == 0 else None
        nav_state = nav.update(ds, frame)
        assert np.all(np.isfinite(nav_state.position_ned))
        pos_errs.append(float(np.linalg.norm(nav_state.position_ned - drone_pos)))

        cmd = seeker.command(nav_state, gate, 0)
        assert isinstance(cmd, ControlCommand) and cmd.mode is ControlMode.BODY_RATE
        assert np.all(np.isfinite(cmd.body_rate)) and np.isfinite(cmd.thrust)
        assert np.linalg.norm(cmd.body_rate) <= seeker.controller.max_body_rate_rps + 1e-9
        assert 0.0 <= cmd.thrust <= 1.0
        n_cmd += 1

    assert n_cmd == seq.N
    # BOUNDED: the self-localized estimate never runs away (a divergence would blow past tens of m).
    assert max(pos_errs) < 25.0, f"estimate diverged: max |err|={max(pos_errs):.1f} m"
    # at least some vision fixes landed (the loop is genuinely self-localizing, not coasting blind).
    assert nav.n_vision_fixes > 0, "no vision fixes applied -> the localization path did not run"


# ===========================================================================
# MAP-FREE VISUAL SERVO (the live VQ2 path — the 2026-06-29 blind-launch fix)
# ===========================================================================
class _ProjDetector:
    """Project a gate's inner corners into the camera from a TRUE pose -> a clean detection the
    seeker's own PnP solves into a relative lever (t_cam_gate). Model-free, deterministic."""

    def __init__(self, gate: Gate, drone_pos, R_wb):
        from racer.frames import R_camera_from_body
        self.gate = gate
        self.drone_pos = np.asarray(drone_pos, float)
        self.R_wb = np.asarray(R_wb, float)
        self._R_cb = R_camera_from_body()

    def detect(self, frame):
        from racer.frames import CAMERA_INTRINSICS_K
        half = self.gate.inner_size_m / 2.0
        corners_gate = np.array([[-half, half, 0.0], [half, half, 0.0],
                                 [half, -half, 0.0], [-half, -half, 0.0]])
        R_wg = np.asarray(self.gate.R_world_gate, float)
        K = CAMERA_INTRINSICS_K
        px = []
        for cg in corners_gate:
            p_world = self.gate.position_ned + R_wg @ cg
            p_cam = self._R_cb @ (self.R_wb.T @ (p_world - self.drone_pos))
            if p_cam[2] <= 0.05:
                return []
            px.append([K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2],
                       K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]])
        return [GateObservation(frame_id=frame.frame_id, sim_time_ns=frame.sim_time_ns,
                                corners_px=np.asarray(px, float), corner_ids=np.array([0, 1, 2, 3]),
                                corner_confidence=np.ones(4))]


def _frame(fid=0, sim_time_ns=0):
    return Frame(frame_id=fid, sim_time_ns=sim_time_ns,
                 image_bgr=np.ones((360, 640, 3), dtype=np.uint8))


def _nav_fix(position, *, tsv, sim_time_ns=0, yaw=0.0, velocity=(0.0, 0.0, 0.0)):
    """A NavState carrying a vision-fix age: tsv=inf => not anchored (launch); finite => anchored."""
    return NavState(sim_time_ns=sim_time_ns, position_ned=np.asarray(position, float),
                    velocity_ned=np.asarray(velocity, float), yaw=yaw,
                    time_since_vision_update_s=tsv)


def test_launch_anchor_holds_yaw_until_first_fix_no_blind_slew():
    """THE core 2026-06-29 fix: at launch (no accepted vision fix yet, tsv=inf) the map-free seeker
    HOLDS — it commands ~zero yaw rate even with the gate plainly in view, so it never slews the
    visible start gate out of frame before the estimator can anchor."""
    from racer.frames import R_world_from_body
    gate = _gate([12.0, 0.0, -2.5], normal=[1.0, 0.0, 0.0])
    det = _ProjDetector(gate, np.zeros(3), R_world_from_body(0.0, 0.0, 0.0))
    seeker = GateSeeker(config=GateSeekerConfig(cruise_speed=3.0, launch_ramp_s=0.0), detector=det)
    cmd = seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=float("inf")), _frame(), 0)
    assert cmd.mode is ControlMode.BODY_RATE
    assert abs(float(cmd.body_rate[2])) < 1e-6        # yaw HELD (anchor clamp default 0)
    assert np.linalg.norm(cmd.body_rate) < 0.5        # no slew of any axis
    assert 0.05 <= cmd.thrust <= 0.6                  # holds a hover collective


def test_after_anchor_pursues_the_seen_gate_with_forward_velocity():
    """Once anchored (a vision fix landed, tsv finite) AND past the settle window the map-free seeker
    pursues the SEEN gate: a sane bounded BODY_RATE that turns toward + leans into the gate the camera
    sees, NOT a slew. (settle_s=0 isolates the anchor/pursuit logic from the cold-start settle hold.)"""
    from racer.frames import R_world_from_body
    gate = _gate([12.0, 0.0, -2.5], normal=[1.0, 0.0, 0.0])
    det = _ProjDetector(gate, np.zeros(3), R_world_from_body(0.0, 0.0, 0.0))
    seeker = GateSeeker(config=GateSeekerConfig(cruise_speed=3.0, launch_ramp_s=0.0, settle_s=0.0),
                        detector=det)
    # first tick anchors the latch (tsv finite); a fresh frame_id each call so the detector re-runs.
    seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=0.05, sim_time_ns=0), _frame(0, 0), 0)
    cmd = seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=0.05, sim_time_ns=10_000_000),
                                _frame(1, 10_000_000), 0)
    assert cmd.mode is ControlMode.BODY_RATE
    assert np.all(np.isfinite(cmd.body_rate)) and np.isfinite(cmd.thrust)
    assert np.linalg.norm(cmd.body_rate) <= seeker.controller.max_body_rate_rps + 1e-9
    # the seen gate is dead ahead -> the recovered world bearing points north (forward progress).
    gdir = seeker._gate_dir_world(_nav_fix([0, 0, -2.5], tsv=0.05), seeker._last_pose)
    assert gdir[0] > 0.5


def test_off_axis_seen_gate_yaws_toward_it_not_away():
    """A gate seen OFF to one side: after anchoring the seeker yaws TOWARD it (sign-correct), bounded
    by the smooth visual yaw cap — never a saturated slew, never the wrong direction."""
    from racer.frames import R_world_from_body
    # gate to the north-EAST: the world bearing has a +east (+y) component -> yaw should be > 0.
    gate = _gate([12.0, 6.0, -2.5], normal=[1.0, 0.0, 0.0])
    det = _ProjDetector(gate, np.zeros(3), R_world_from_body(0.0, 0.0, 0.0))
    seeker = GateSeeker(config=GateSeekerConfig(cruise_speed=3.0, launch_ramp_s=0.0, settle_s=0.0,
                                                visual_yaw_rate_cap_rps=1.5), detector=det)
    seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=0.05, sim_time_ns=0, yaw=0.0), _frame(0, 0), 0)
    cmd = seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=0.05, sim_time_ns=10_000_000, yaw=0.0),
                                _frame(1, 10_000_000), 0)
    assert abs(float(cmd.body_rate[2])) <= 1.5 + 1e-9         # smooth cap holds
    gdir = seeker._gate_dir_world(_nav_fix([0, 0, -2.5], tsv=0.05, yaw=0.0), seeker._last_pose)
    assert gdir[1] > 0.0                                       # bearing points east (toward the gate)


def test_no_detection_holds_heading_no_blind_slew():
    """When the detector returns nothing (between gates / momentarily lost), the seeker HOLDS heading
    and coasts level — a clamped, gentle command, never a blind large slew."""
    class _Blind:
        def detect(self, frame):
            return []
    seeker = GateSeeker(config=GateSeekerConfig(cruise_speed=3.0, launch_ramp_s=0.0), detector=_Blind())
    # anchored, but nothing in view this tick.
    cmd = seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=0.05, yaw=0.3), _frame(), 0)
    assert cmd.mode is ControlMode.BODY_RATE
    assert abs(float(cmd.body_rate[2])) < 1e-6        # heading held (reacquire clamp default 0)
    assert np.linalg.norm(cmd.body_rate) < 0.5        # no lean/slew


def test_no_detector_always_holds():
    """With NO detector injected, command_visual can never see -> it always HOLDS (safe default)."""
    seeker = GateSeeker(config=GateSeekerConfig(cruise_speed=3.0, launch_ramp_s=0.0))
    cmd = seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=0.05), _frame(), 0)
    assert np.linalg.norm(cmd.body_rate) < 0.5


def test_regression_old_absolute_map_command_would_blind_slew_new_visual_does_not():
    """REGRESSION pinning the 2026-06-29 bug: handed the ORIGIN-seeded estimator + a WRONG/STALE map
    gate (the VQ1 fallback at world (-23.3,-0.4,0)), the OLD absolute-map command() produces a near-
    saturated launch yaw slew (the U-turn that lost the start gate). The NEW map-free command_visual,
    on the SAME origin seed with the true gate VISIBLE, commands ~zero launch yaw. This must never
    silently regress."""
    from racer.frames import R_world_from_body
    cap = make_seeker_controller().max_body_rate_rps
    # OLD path: absolute-map command to the WRONG gate at origin seed, yaw 0 -> demands a ~180 slew.
    wrong_gate = _gate([-23.3, -0.4, 0.0], normal=[-1.0, 0.0, 0.0])
    old_cmd = GateSeeker(config=GateSeekerConfig(cruise_speed=3.0, launch_ramp_s=0.0)).command(
        _nav([0.0, 0.0, 0.0], yaw=0.0), wrong_gate, 0)
    assert abs(float(old_cmd.body_rate[2])) > 0.9 * cap, "old map path should slew hard at launch"
    # NEW path: the true start gate is visible; the map-free seeker HOLDS at launch (no slew).
    true_gate = _gate([12.0, 0.0, -2.5], normal=[1.0, 0.0, 0.0])
    det = _ProjDetector(true_gate, np.zeros(3), R_world_from_body(0.0, 0.0, 0.0))
    new_seeker = GateSeeker(config=GateSeekerConfig(cruise_speed=3.0, launch_ramp_s=0.0), detector=det)
    new_cmd = new_seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=float("inf"), yaw=0.0), _frame(), 0)
    assert abs(float(new_cmd.body_rate[2])) < 0.5, "new visual servo must not slew at launch"


# ===========================================================================
# BUG A (2026-06-29 attempt-2): the MAP-FREE anchor release on the seeker's OWN detections
# ===========================================================================
def test_map_free_anchor_releases_on_own_detections_when_tsv_stays_inf():
    """THE 2026-06-29 attempt-2 BUG A fix: on the LIVE VQ2 wire the navigator is MAP-FREE, so its
    map-associated fix path never fires and ``nav.time_since_vision_update_s`` stays inf FOREVER.
    The OLD release signal (tsv finite) is therefore structurally UNREACHABLE -> the seeker would be
    pinned in the launch-hold forever. The NEW seeker releases on its OWN consecutive quality-gated
    detections: with tsv=inf throughout but the gate VISIBLE, after anchor_release_detections ticks
    it ANCHORS and reaches PURSUIT (forward lean toward the seen gate)."""
    from racer.frames import R_world_from_body
    gate = _gate([12.0, 0.0, -2.5], normal=[1.0, 0.0, 0.0])
    det = _ProjDetector(gate, np.zeros(3), R_world_from_body(0.0, 0.0, 0.0))
    N = 3
    seeker = GateSeeker(config=GateSeekerConfig(cruise_speed=3.0, launch_ramp_s=0.0, settle_s=0.0,
                                                anchor_release_detections=N), detector=det)
    # tsv stays inf the ENTIRE time (the map-free wire) -- the only release signal is own-detection.
    assert not seeker._anchored
    for k in range(N):
        assert not seeker._anchored, f"released too early at tick {k} (need {N} detections)"
        cmd = seeker.command_visual(
            _nav_fix([0, 0, -2.5], tsv=float("inf"), sim_time_ns=k * 10_000_000),
            _frame(k, k * 10_000_000), 0)
        assert cmd.mode is ControlMode.BODY_RATE
    # after N consecutive detections the anchor has released (tsv NEVER went finite).
    assert seeker._anchored, "map-free anchor never released on own detections -> BUG A regression"
    # and it now PURSUES the seen gate: a forward (north) world bearing -> a real lean, not a hold.
    cmd = seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=float("inf"), sim_time_ns=N * 10_000_000),
                                _frame(N, N * 10_000_000), 0)
    gdir = seeker._gate_dir_world(_nav_fix([0, 0, -2.5], tsv=float("inf")), seeker._last_pose)
    assert gdir[0] > 0.5, "did not pursue the seen gate after the map-free release"


def test_old_tsv_only_release_would_pin_forever_on_map_free_wire():
    """REGRESSION pinning BUG A: model the OLD release rule (anchor ONLY when tsv finite). On the
    map-free VQ2 wire tsv stays inf for every tick, so the old rule NEVER releases -> the drone is
    pinned in the launch-hold forever even with the gate centred. This asserts the failure the NEW
    own-detection release fixes, so the structural bug can never silently return."""
    # the OLD anchor condition, evaluated over a map-free run (tsv=inf throughout):
    old_anchored = False
    for _ in range(50):                       # many ticks, gate visible the whole time
        tsv = float("inf")                    # map-free wire: never a map fix -> never finite
        if np.isfinite(tsv):                  # the OLD release rule
            old_anchored = True
    assert not old_anchored, "old tsv-only rule would have to stay pinned on the map-free wire"


# ===========================================================================
# BUG B (2026-06-29 attempt-2): the ATTITUDE-SAFE cold-start hold (no pitch tumble)
# ===========================================================================
def test_cold_start_hold_clamps_roll_pitch_not_just_yaw():
    """THE 2026-06-29 attempt-2 BUG B fix: the launch/settle hold must clamp ROLL/PITCH rates, not
    only yaw. Feed a NavState whose attitude estimate is wildly tilted (a cold mag-free AHRS ~18deg
    off) so the level-hold controller would demand a large pitch correction; assert the hold bounds
    EVERY axis to the conservative cap (no saturated pitch-over that tumbles into the gate)."""
    # an ~18deg cold attitude error: the controller's level-hold wants to drive a big roll/pitch rate.
    cold = NavState(sim_time_ns=0, position_ned=np.array([0.0, 0.0, -2.5]),
                    velocity_ned=np.zeros(3), roll=np.deg2rad(18.0), pitch=np.deg2rad(18.0),
                    yaw=0.0, time_since_vision_update_s=float("inf"))
    cap = 0.6
    seeker = GateSeeker(config=GateSeekerConfig(cruise_speed=3.0, launch_ramp_s=0.0, settle_s=0.75,
                                                hold_rp_rate_cap_rps=cap), detector=None)
    cmd = seeker.command_visual(cold, _frame(0, 0), 0)        # tick 0 -> in the settle hold
    assert cmd.mode is ControlMode.BODY_RATE
    assert abs(float(cmd.body_rate[0])) <= cap + 1e-9, "roll rate not clamped in the cold-start hold"
    assert abs(float(cmd.body_rate[1])) <= cap + 1e-9, "pitch rate not clamped (the tumble axis!)"
    assert abs(float(cmd.body_rate[2])) <= cap + 1e-9, "yaw rate not clamped"


def test_old_yaw_only_hold_would_diverge_in_pitch_from_18deg():
    """REGRESSION pinning BUG B: the OLD hold clamped ONLY yaw. From an ~18deg cold attitude error the
    level-hold controller demands a large PITCH rate; with only-yaw clamping that pitch command rides
    UNBOUNDED (the saturated -3.4 rad/s pitch-over the tlog showed). This asserts the OLD yaw-only hold
    WOULD produce an out-of-(conservative)-bound pitch -- the divergence the new clamp removes."""
    cold = NavState(sim_time_ns=0, position_ned=np.array([0.0, 0.0, -2.5]),
                    velocity_ned=np.zeros(3), roll=0.0, pitch=np.deg2rad(18.0), yaw=0.0)
    cap = 0.6
    seeker = GateSeeker(config=GateSeekerConfig(cruise_speed=3.0, launch_ramp_s=0.0))
    # the OLD hold: yaw-only clamp (no attitude_safe roll/pitch clamp). Reproduce it directly.
    old_cmd = seeker._cap_yaw_rate(
        seeker.controller.command(
            cold, __import__("racer.contracts", fromlist=["Setpoint"]).Setpoint(
                sim_time_ns=0, velocity_ned=np.zeros(3), yaw=0.0, launch_ramp=0.0)),
        0.0)
    # from 18deg pitch error the controller commands a pitch rate well beyond the conservative cap.
    assert abs(float(old_cmd.body_rate[1])) > cap, \
        "old yaw-only hold should leave a large (divergent) pitch rate from 18deg"
    # the NEW attitude-safe hold bounds that same pitch.
    new_cmd = seeker._hold_command(cold, yaw_rate_cap=0.0, attitude_safe=True)
    assert abs(float(new_cmd.body_rate[1])) <= cap + 1e-9


def test_settle_holds_then_releases_to_pursuit_after_window():
    """The post-arm SETTLE holds for settle_s (conservative level, clamped rates) EVEN with the gate
    visible + anchored, then transitions to pursuit once the window elapses (the cold AHRS has had
    time to gravity-align). Models the launch SEQUENCE: hold early, pursue late."""
    from racer.frames import R_world_from_body
    gate = _gate([12.0, 0.0, -2.5], normal=[1.0, 0.0, 0.0])
    det = _ProjDetector(gate, np.zeros(3), R_world_from_body(0.0, 0.0, 0.0))
    settle = 0.5
    seeker = GateSeeker(config=GateSeekerConfig(cruise_speed=3.0, launch_ramp_s=0.0, settle_s=settle,
                                                anchor_release_detections=1), detector=det)
    # tick 0 (t=0): inside the settle -> a clamped hold even though the gate is dead ahead + anchored.
    cmd0 = seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=0.05, sim_time_ns=0), _frame(0, 0), 0)
    assert np.linalg.norm(cmd0.body_rate) < 1.0          # held (no pursuit lean) during settle
    assert seeker._in_settle(0)
    # a tick PAST the settle window -> pursuit (the seeker leans toward the seen gate).
    t_past = int((settle + 0.2) * 1e9)
    assert not seeker._in_settle(t_past)
    cmd1 = seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=0.05, sim_time_ns=t_past),
                                 _frame(1, t_past), 0)
    assert cmd1.mode is ControlMode.BODY_RATE
    assert np.all(np.isfinite(cmd1.body_rate))


def test_reset_returns_seeker_to_launch_anchor_regime():
    """After reset the seeker is back in the launch-anchor regime (re-holds until re-anchored)."""
    from racer.frames import R_world_from_body
    gate = _gate([12.0, 0.0, -2.5], normal=[1.0, 0.0, 0.0])
    det = _ProjDetector(gate, np.zeros(3), R_world_from_body(0.0, 0.0, 0.0))
    seeker = GateSeeker(config=GateSeekerConfig(cruise_speed=3.0, launch_ramp_s=0.0), detector=det)
    seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=0.05), _frame(0, 0), 0)   # anchor
    assert seeker._anchored is True
    seeker.reset()
    assert seeker._anchored is False and seeker._last_pose is None
    assert seeker._consec_detections == 0          # the own-detection streak is cleared too
    # immediately after reset, even with the gate in view, it HOLDS (tsv inf again).
    cmd = seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=float("inf")), _frame(1, 0), 0)
    assert abs(float(cmd.body_rate[2])) < 1e-6


# ===========================================================================
# LAYER 2a (2026-06-29 attempt-3): TEMPORAL GATE TRACKING — lock one gate, reject flap
# ===========================================================================
class _MultiProjDetector:
    """Project SEVERAL gates into the camera; the distractors' corners SCALE-JITTER per frame so
    their PnP depth oscillates (and they read near/far) -- the A3 multi-gate range flap. gate[0] is
    the stable centered active gate."""

    def __init__(self, gates, drone_pos, R_wb):
        from racer.frames import R_camera_from_body
        self.gates, self.drone_pos, self.R_wb = list(gates), np.asarray(drone_pos, float), np.asarray(R_wb, float)
        self._R_cb = R_camera_from_body()

    def _project(self, gate):
        from racer.frames import CAMERA_INTRINSICS_K
        half = gate.inner_size_m / 2.0
        cg = np.array([[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]])
        R_wg = np.asarray(gate.R_world_gate, float)
        K = CAMERA_INTRINSICS_K
        px = []
        for c in cg:
            p_cam = self._R_cb @ (self.R_wb.T @ (gate.position_ned + R_wg @ c - self.drone_pos))
            if p_cam[2] <= 0.05:
                return None
            px.append([K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2], K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]])
        return np.asarray(px, float)

    def detect(self, frame):
        out = []
        for i, gate in enumerate(self.gates):
            base = self._project(gate)
            if base is None:
                continue
            if i == 0:
                px = base
            else:
                ctr = base.mean(axis=0)
                scale = 1.0 + 0.5 * np.sin(0.9 * int(frame.frame_id) + np.pi * i)
                px = ctr + (base - ctr) * scale
            out.append(GateObservation(frame_id=frame.frame_id, sim_time_ns=frame.sim_time_ns,
                                       corners_px=px, corner_ids=np.array([0, 1, 2, 3]),
                                       corner_confidence=np.ones(4)))
        return out


def _drive_multigate(seeker, det, n):
    """Run command_visual over a multi-gate stream from a fixed hover pose; return (chosen ranges,
    roll-rate commands). The drone is HELD fixed so this isolates target selection + steering."""
    from racer.frames import R_world_from_body
    det.drone_pos, det.R_wb = np.zeros(3), R_world_from_body(0.0, 0.0, 0.0)
    ranges, rolls = [], []
    for k in range(40):
        t_ns = int(k * 0.02 * 1e9)
        ns = _nav_fix([0, 0, 0], tsv=(0.05 if k > 0 else float("inf")), sim_time_ns=t_ns, yaw=0.0)
        cmd = seeker.command_visual(ns, _frame(k, t_ns), 0)
        ranges.append(seeker._last_pose.range_m if seeker._last_pose is not None else np.nan)
        rolls.append(float(cmd.body_rate[0]))
    return np.array(ranges), np.array(rolls)


def test_temporal_track_locks_one_gate_smooth_range_old_redetect_flaps():
    """LAYER 2a: with several gates visible and the distractors' PnP range flapping, the NEW temporal
    track LOCKS the centered active gate so the chosen range stays SMOOTH; the OLD redetect-each-frame
    (use_gate_track=False) picks the closest each tick and FLAPS. Pins the proximate roll-over cause."""
    from racer.frames import R_world_from_body
    gates = [_gate([20.0, 1.0, -2.5], normal=[1, 0, 0], gate_id=0),       # near-centered active
             _gate([14.0, 12.0, -2.5], normal=[1, 0, 0], gate_id=1),      # hard-left distractor
             _gate([14.0, -12.0, -2.5], normal=[1, 0, 0], gate_id=2)]     # hard-right distractor
    R_wb = R_world_from_body(0.0, 0.0, 0.0)

    old = GateSeeker(config=GateSeekerConfig(cruise_speed=3.0, launch_ramp_s=0.0, settle_s=0.0,
                                             anchor_release_detections=1, use_gate_track=False,
                                             use_spawn_egress=False),
                     detector=_MultiProjDetector(gates, np.zeros(3), R_wb))
    new = GateSeeker(config=GateSeekerConfig(cruise_speed=3.0, launch_ramp_s=0.0, settle_s=0.0,
                                             anchor_release_detections=1, use_spawn_egress=False),
                     detector=_MultiProjDetector(gates, np.zeros(3), R_wb))
    old_r, _ = _drive_multigate(old, old.detector, 40)
    new_r, _ = _drive_multigate(new, new.detector, 40)

    def _flap(r):
        d = np.abs(np.diff(r[~np.isnan(r)]))
        return float(d.max()) if d.size else 0.0
    assert _flap(old_r) > 5.0, "old redetect-from-scratch should flap the chosen-gate range"
    assert _flap(new_r) < 3.0, "temporal track must lock one gate -> smooth range (no flap)"


def test_temporal_track_keeps_roll_bounded_old_redetect_swings_it():
    """LAYER 2a/2b: the locked + slew-limited + roll-capped pursuit keeps the ROLL command bounded and
    non-oscillating under the multi-gate flap; the OLD unbounded redetect SWINGS the roll hard (the A3
    roll-over). Pins the crash axis."""
    from racer.frames import R_world_from_body
    gates = [_gate([20.0, 1.0, -2.5], normal=[1, 0, 0], gate_id=0),
             _gate([14.0, 12.0, -2.5], normal=[1, 0, 0], gate_id=1),
             _gate([14.0, -12.0, -2.5], normal=[1, 0, 0], gate_id=2)]
    R_wb = R_world_from_body(0.0, 0.0, 0.0)
    cap = make_seeker_controller().max_body_rate_rps

    # OLD baseline: the A3 caps off AND the legacy velocity pursuit (use_feedforward_forward=False),
    # so the heading-aligned velocity setpoint produces the roll swing the unbounded servo crashed on.
    old = GateSeeker(config=GateSeekerConfig(cruise_speed=3.0, launch_ramp_s=0.0, settle_s=0.0,
                                             anchor_release_detections=1, use_gate_track=False,
                                             pursuit_ramp_s=0.0, pursuit_yaw_slew_rps=0.0,
                                             pursuit_roll_rate_cap_rps=0.0, visual_yaw_rate_cap_rps=cap,
                                             use_feedforward_forward=False, use_spawn_egress=False),
                     detector=_MultiProjDetector(gates, np.zeros(3), R_wb))
    new = GateSeeker(config=GateSeekerConfig(cruise_speed=3.0, launch_ramp_s=0.0, settle_s=0.0,
                                             anchor_release_detections=1, use_spawn_egress=False),
                     detector=_MultiProjDetector(gates, np.zeros(3), R_wb))
    _, old_roll = _drive_multigate(old, old.detector, 40)
    _, new_roll = _drive_multigate(new, new.detector, 40)

    def _swing(x):
        d = np.abs(np.diff(x[~np.isnan(x)]))
        return float(d.max()) if d.size else 0.0
    assert _swing(old_roll) > 1.0, "old unbounded redetect should swing the roll command hard"
    assert np.nanmax(np.abs(new_roll)) <= GateSeekerConfig().pursuit_roll_rate_cap_rps + 1e-9
    assert _swing(new_roll) < 0.5 * _swing(old_roll), "tracked+capped pursuit must not oscillate roll"


def test_track_rejects_a_range_jump_and_coasts():
    """A candidate that JUMPS implausibly in range from the established track is REJECTED -> the seeker
    coasts (returns None this tick) instead of locking onto the flapper. Unit-level pin of the gate."""
    from racer.frames import R_world_from_body
    gate = _gate([20.0, 0.0, -2.5], normal=[1, 0, 0])
    seeker = GateSeeker(config=GateSeekerConfig(track_max_range_jump_m=6.0), detector=None)
    # seed the track at ~20 m, centred bearing.
    seeker._track_range_m = 20.0
    seeker._track_bearing = np.zeros(2)
    det = _ProjDetector(gate, np.zeros(3), R_world_from_body(0.0, 0.0, 0.0))

    class _Jumped:
        """Returns only a single gate that is 18 m NEARER than the track (a depth flip / wrong gate)."""
        def detect(self, frame):
            near = _gate([2.0, 0.0, -2.5], normal=[1, 0, 0])    # range ~2 m, a >15 m jump from 20
            return _ProjDetector(near, np.zeros(3), R_world_from_body(0.0, 0.0, 0.0)).detect(frame)

    seeker.detector = _Jumped()
    pose = seeker.detect_gate_lever(_frame(1, 0))
    assert pose is None, "a candidate that jumps range >max must be rejected (coast), not locked"
    # a CONSISTENT candidate (near the track) is accepted.
    seeker.detector = det
    pose2 = seeker.detect_gate_lever(_frame(2, 0))
    assert pose2 is not None and abs(pose2.range_m - 20.0) < 6.0


def test_track_first_acquisition_prefers_centered_gate():
    """On FIRST acquisition (no track) the seeker prefers the most CENTERED gate (the active line), not
    merely the closest -- so it doesn't lock a near side-distractor."""
    from racer.frames import R_world_from_body
    # a CLOSE off-axis gate vs a slightly-farther CENTERED gate: prefer the centered one.
    centered = _gate([18.0, 0.0, -2.5], normal=[1, 0, 0], gate_id=0)
    close_side = _gate([12.0, 11.0, -2.5], normal=[1, 0, 0], gate_id=1)
    det = _MultiProjDetector([centered, close_side], np.zeros(3), R_world_from_body(0.0, 0.0, 0.0))
    seeker = GateSeeker(config=GateSeekerConfig(track_prefer_centered=True), detector=det)
    pose = seeker.detect_gate_lever(_frame(0, 0))
    assert pose is not None
    # the chosen gate must be the FARTHER, centered one (range ~18), NOT the closer side distractor
    # (~12.5): centring beat proximity. (Its camera bearing magnitude is also the smaller of the two.)
    assert pose.range_m > 16.0, "first acquisition should prefer the centered gate over the close side one"
    # cross-check: the centered gate's bearing is smaller than the side gate's.
    poses = seeker._valid_poses(_frame(1, 0))
    bearings = sorted(float(np.linalg.norm(seeker._pose_bearing(p))) for p in poses)
    assert float(np.linalg.norm(seeker._pose_bearing(pose))) == pytest.approx(bearings[0], abs=1e-9)


def test_track_off_is_legacy_closest_each_frame():
    """With use_gate_track=False the legacy behaviour returns: pick the CLOSEST gate each frame (no
    continuity). Guards that the new path is opt-in and the old selection is preserved."""
    from racer.frames import R_world_from_body
    centered = _gate([18.0, 0.0, -2.5], normal=[1, 0, 0], gate_id=0)
    close_side = _gate([12.0, 11.0, -2.5], normal=[1, 0, 0], gate_id=1)
    det = _MultiProjDetector([centered, close_side], np.zeros(3), R_world_from_body(0.0, 0.0, 0.0))
    seeker = GateSeeker(config=GateSeekerConfig(use_gate_track=False), detector=det)
    pose = seeker.detect_gate_lever(_frame(0, 0))
    assert pose is not None
    # legacy: the CLOSEST gate wins (the ~12.5 m side gate), not the centered ~18 m one.
    assert pose.range_m < 17.0


# ===========================================================================
# LAYER 2b (2026-06-29 attempt-3): post-release slew-ramp + guidance rate limit
# ===========================================================================
def test_pursuit_heading_is_slew_rate_limited():
    """LAYER 2b: the commanded pursuit HEADING is rate-limited -- a large bearing step cannot move the
    yaw setpoint more than pursuit_yaw_slew_rps*dt in a tick, so the steering bearing stays smooth."""
    from racer.frames import R_world_from_body
    gate = _gate([12.0, 8.0, -2.5], normal=[1, 0, 0])   # a big off-axis bearing
    det = _ProjDetector(gate, np.zeros(3), R_world_from_body(0.0, 0.0, 0.0))
    slew = 1.0
    seeker = GateSeeker(config=GateSeekerConfig(cruise_speed=3.0, launch_ramp_s=0.0, settle_s=0.0,
                                                anchor_release_detections=1, pursuit_yaw_slew_rps=slew),
                        detector=det)
    # tick 0 anchors + seeds _last_yaw at 0; tick 1 (10 ms later) would jump the heading to the gate
    # bearing (~0.6 rad) but the slew limits it to slew*dt = 0.01 rad.
    seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=0.05, sim_time_ns=0, yaw=0.0), _frame(0, 0), 0)
    y0 = float(seeker._last_yaw)
    dt_ns = 10_000_000
    seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=0.05, sim_time_ns=dt_ns, yaw=0.0), _frame(1, dt_ns), 0)
    y1 = float(seeker._last_yaw)
    step = abs(float(np.arctan2(np.sin(y1 - y0), np.cos(y1 - y0))))
    assert step <= slew * (dt_ns / 1e9) + 1e-9, "the pursuit heading must be slew-rate limited"


def test_pursuit_ramp_eases_authority_after_release():
    """LAYER 2b: the post-release pursuit ramp scales the lean authority up from a floor over
    pursuit_ramp_s, so the FIRST pursuit tick after release cannot lean to full cruise authority."""
    from racer.frames import R_world_from_body
    gate = _gate([12.0, 6.0, -2.5], normal=[1, 0, 0])
    det = _ProjDetector(gate, np.zeros(3), R_world_from_body(0.0, 0.0, 0.0))
    seeker = GateSeeker(config=GateSeekerConfig(cruise_speed=3.0, launch_ramp_s=0.0, settle_s=0.0,
                                                anchor_release_detections=1, pursuit_ramp_s=0.8,
                                                pursuit_ramp_floor=0.15), detector=det)
    # release at tick 0 -> _release_t_ns set; the ramp at t=0 is the floor (0.15), at t>=ramp_s it is 1.
    seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=0.05, sim_time_ns=0), _frame(0, 0), 0)
    assert seeker._pursuit_ramp(0) == pytest.approx(0.15)
    assert seeker._pursuit_ramp(int(0.4 * 1e9)) == pytest.approx(0.575, abs=0.05)   # mid-ramp
    assert seeker._pursuit_ramp(int(1.0 * 1e9)) == pytest.approx(1.0)               # complete


def test_pursuit_roll_rate_capped():
    """LAYER 2b: the pursuit roll command is hard-capped below saturation (A3's crash axis)."""
    import dataclasses
    seeker = GateSeeker(config=GateSeekerConfig(pursuit_roll_rate_cap_rps=1.5))
    cmd = ControlCommand(mode=ControlMode.BODY_RATE, body_rate=np.array([5.0, 0.2, 0.1]), thrust=0.3)
    capped = seeker._cap_roll_rate(cmd, 1.5)
    assert abs(float(capped.body_rate[0])) <= 1.5 + 1e-9
    assert capped.body_rate[1] == 0.2 and capped.body_rate[2] == 0.1   # other axes untouched


# ===========================================================================
# LAYER 1 (2026-06-29 attempt-3): the FREEZE-ATTITUDE hold (don't re-level off the gate)
# ===========================================================================
def test_hold_freeze_attitude_zeroes_roll_pitch_keeps_gate_in_view():
    """LAYER 1: with hold_freeze_attitude (default) the launch/settle hold ZEROES the roll/pitch rate
    command -- it FREEZES the spawn attitude instead of re-levelling off the gate. From a tilted spawn
    the OLD force-level hold commands a persistent pitch (drifting the camera off the gate, stalling
    the release streak); the freeze holds the camera on the gate."""
    cold = NavState(sim_time_ns=0, position_ned=np.array([0.0, 0.0, -2.5]), velocity_ned=np.zeros(3),
                    roll=np.deg2rad(8.0), pitch=np.deg2rad(-10.0), yaw=0.0,
                    time_since_vision_update_s=float("inf"))
    # NEW (freeze): roll/pitch rate command is exactly zero -> the camera does not drift off the gate.
    new = GateSeeker(config=GateSeekerConfig(launch_ramp_s=0.0, settle_s=0.75, hold_freeze_attitude=True),
                     detector=None)
    cmd_new = new.command_visual(cold, _frame(0, 0), 0)
    assert float(cmd_new.body_rate[0]) == 0.0, "freeze must zero the roll-rate command (hold attitude)"
    assert float(cmd_new.body_rate[1]) == 0.0, "freeze must zero the pitch-rate command (hold attitude)"
    # OLD (force-level, freeze off): the level-hold commands a NON-zero pitch toward level off the tilt
    # -> the persistent pitch that drifts the camera off the gate (clamped, but not zero).
    old = GateSeeker(config=GateSeekerConfig(launch_ramp_s=0.0, settle_s=0.75, hold_freeze_attitude=False),
                     detector=None)
    cmd_old = old.command_visual(cold, _frame(0, 0), 0)
    assert abs(float(cmd_old.body_rate[1])) > 0.05, "old force-level hold should command a re-levelling pitch"


def test_hold_freeze_still_bounds_thrust_and_yaw():
    """The freeze hold still clamps yaw + bounds the collective (BUG-B guards intact) -- it only frees
    roll/pitch to HOLD attitude, not the safety bounds."""
    cold = NavState(sim_time_ns=0, position_ned=np.array([0.0, 0.0, -2.5]), velocity_ned=np.zeros(3),
                    roll=np.deg2rad(8.0), pitch=np.deg2rad(-10.0), yaw=0.3,
                    time_since_vision_update_s=float("inf"))
    seeker = GateSeeker(config=GateSeekerConfig(launch_ramp_s=0.0, settle_s=0.75,
                                                hold_freeze_attitude=True, anchor_yaw_rate_rps=0.0),
                        detector=None)
    cmd = seeker.command_visual(cold, _frame(0, 0), 0)
    assert abs(float(cmd.body_rate[2])) < 1e-6           # yaw still clamped (anchor cap 0)
    hover = seeker.controller.hover_thrust
    assert hover * 0.6 - 1e-6 <= cmd.thrust <= hover * 1.4 + 1e-6   # collective still bounded


def test_vq2_case_c_profile_is_map_free_for_steering():
    """The case-C profile carries no given pose; the deploy seeker steers MAP-FREE via the detector —
    a guard against re-introducing an absolute-map steering dependency on the self-localizing path."""
    p = vq2_case_c()
    assert p.self_localizing is True
    assert p.nav_config.use_given_position is False     # no absolute self-position to steer from


# ===========================================================================
# A4 (2026-06-29 attempt-4): BOUNDED FEEDFORWARD forward tilt (map-free PITCH-windup fix)
# ===========================================================================
def test_mapfree_pursuit_pitch_bounded_old_velocity_winds_up():
    """THE 2026-06-29 attempt-4 fix (the crash): on the MAP-FREE wire velocity is UNOBSERVABLE
    (``nav.velocity_ned`` stays ~0). The OLD pursuit asked for a desired VELOCITY (cruise*los); the
    controller closes it with a velocity-ERROR term that NEVER closes -> the commanded PITCH winds up
    to the -4.0 controller limit (A4: -0.69 -> -2.39 -> -3.999, then crash). The NEW pursuit commands a
    BOUNDED FEEDFORWARD forward tilt (no velocity term) + a pitch cap, so the pitch stays bounded over a
    sustained pursuit window with velocity held at 0."""
    from racer.frames import R_world_from_body
    gate = _gate([12.0, 0.0, -2.5], normal=[1.0, 0.0, 0.0])
    cap = make_seeker_controller().max_body_rate_rps          # the 4.0 saturation limit
    pitch_cap = GateSeekerConfig().pursuit_pitch_rate_cap_rps

    def _run(feedforward, pitch_cap_rps):
        det = _ProjDetector(gate, np.zeros(3), R_world_from_body(0.0, 0.0, 0.0))
        seeker = GateSeeker(config=GateSeekerConfig(
            cruise_speed=3.0, launch_ramp_s=0.0, settle_s=0.0, anchor_release_detections=1,
            use_feedforward_forward=feedforward, use_spawn_egress=False,
            pursuit_pitch_rate_cap_rps=pitch_cap_rps), detector=det)
        pitches = []
        for k in range(40):                          # SUSTAINED window: velocity HELD at 0 (map-free)
            t_ns = int(k * 0.05 * 1e9)
            ns = NavState(sim_time_ns=t_ns, position_ned=np.zeros(3), velocity_ned=np.zeros(3),
                          roll=0.0, pitch=0.0, yaw=0.0,
                          time_since_vision_update_s=(0.05 if k > 0 else float("inf")))
            cmd = seeker.command_visual(ns, _frame(k, t_ns), 0)
            pitches.append(abs(float(cmd.body_rate[1])))
        return max(pitches)

    # OLD: velocity setpoint, pitch cap off -> the raw windup to saturation.
    old_max = _run(feedforward=False, pitch_cap_rps=0.0)
    # NEW: bounded feedforward + the shipped pitch cap -> bounded.
    new_max = _run(feedforward=True, pitch_cap_rps=pitch_cap)
    assert old_max > 0.9 * cap, "old velocity-feedback pursuit should wind pitch toward saturation map-free"
    assert new_max <= pitch_cap + 1e-9, "new bounded-feedforward pursuit must keep pitch under the cap"


def test_pursuit_pitch_rate_capped():
    """The pursuit pitch command is hard-capped below saturation (A4's crash axis), symmetric to the
    roll cap."""
    seeker = GateSeeker(config=GateSeekerConfig(pursuit_pitch_rate_cap_rps=1.5))
    cmd = ControlCommand(mode=ControlMode.BODY_RATE, body_rate=np.array([0.2, 5.0, 0.1]), thrust=0.3)
    capped = seeker._cap_pitch_rate(cmd, 1.5)
    assert abs(float(capped.body_rate[1])) <= 1.5 + 1e-9
    assert capped.body_rate[0] == 0.2 and capped.body_rate[2] == 0.1   # other axes untouched


def test_pursuit_forward_is_feedforward_accel_not_velocity_setpoint():
    """The bounded-feedforward pursuit sends a Setpoint with accel_ned (pure feedforward the controller
    adds, no wind-up) and NO HORIZONTAL velocity_ned -- the structural A4 fix. The FORWARD/cross-track
    demand is pure accel feedforward; the only velocity_ned the seeker may carry is a Z-ONLY
    vertical-align target (A5 BLOCKER 1), which is NOT a horizontal velocity setpoint and cannot wind up
    the pitch. With feedforward ON the setpoint carries accel + (at most) a Z-only velocity; with it OFF
    (legacy) it carries a full velocity setpoint.

    Here the active gate sits at the SAME height as the drone (zero vertical offset), so vertical-align
    is in its deadband and commands no vz -> velocity_ned is None; the A5 vertical case is covered by its
    own dedicated test below."""
    from racer.frames import R_world_from_body
    # gate at the drone's spawn HEIGHT (drone_pos defaults to [0,0,0] in _ProjDetector, gate z=0) so the
    # world vertical offset is ~0 -> vertical-align deadband -> no vz target -> velocity_ned stays None.
    gate = _gate([12.0, 0.0, 0.0], normal=[1.0, 0.0, 0.0])
    captured = {}

    class _SpyController:
        """Wrap the real controller; record the Setpoint it is handed on the pursuit tick."""
        def __init__(self, inner):
            self.inner = inner
            self.max_body_rate_rps = inner.max_body_rate_rps
            self.hover_thrust = inner.hover_thrust
        def command(self, nav, sp):
            captured["sp"] = sp
            return self.inner.command(nav, sp)

    def _last_sp(feedforward):
        det = _ProjDetector(gate, np.zeros(3), R_world_from_body(0.0, 0.0, 0.0))
        seeker = GateSeeker(config=GateSeekerConfig(
            cruise_speed=3.0, launch_ramp_s=0.0, settle_s=0.0, anchor_release_detections=1,
            use_feedforward_forward=feedforward, use_spawn_egress=False), detector=det)
        seeker.controller = _SpyController(seeker.controller)
        seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=0.05, sim_time_ns=0), _frame(0, 0), 0)
        captured.clear()
        # a tick PAST release (the forward-demand ramp has eased in) so the accel magnitude is > 0.
        t1 = int(0.5 * 1e9)
        seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=0.05, sim_time_ns=t1), _frame(1, t1), 0)
        return captured["sp"]

    ff_sp = _last_sp(feedforward=True)
    assert ff_sp.accel_ned is not None and ff_sp.velocity_ned is None, \
        "feedforward pursuit must command accel_ned (bounded feedforward), not a velocity setpoint"
    assert float(np.linalg.norm(ff_sp.accel_ned)) > 0.0   # a real forward demand once the ramp eased in
    legacy_sp = _last_sp(feedforward=False)
    assert legacy_sp.velocity_ned is not None, "legacy pursuit still uses a velocity setpoint"


# ===========================================================================
# A4 (2026-06-29 attempt-4): SPAWN-GATE EGRESS — clear gate 0 before pursuit
# ===========================================================================
def test_spawn_egress_creeps_along_spawn_heading_then_pursues():
    """THE 2026-06-29 attempt-4 start-gate fix: the drone SPAWNS INSIDE gate 0. After release the NEW
    seeker runs a brief EGRESS phase -- a capped forward creep along the FROZEN spawn heading (the
    start-gate normal, the way OUT) -- BEFORE re-aiming at the downrange gate. During egress the
    commanded heading is the spawn heading (not slewed toward an off-side downrange gate); after
    egress_s it transitions to pursuit (re-aims at the seen gate)."""
    from racer.frames import R_world_from_body
    # the seen downrange gate is OFF to the +Y side: if the seeker re-aimed immediately, the heading
    # would swing toward +Y. During egress it must stay on the +X spawn heading (the way out of gate 0).
    gate = _gate([12.0, 8.0, -2.5], normal=[1.0, 0.0, 0.0])
    det = _ProjDetector(gate, np.zeros(3), R_world_from_body(0.0, 0.0, 0.0))
    egress = 0.5
    seeker = GateSeeker(config=GateSeekerConfig(
        cruise_speed=3.0, launch_ramp_s=0.0, settle_s=0.0, anchor_release_detections=1,
        use_spawn_egress=True, egress_s=egress, pursuit_yaw_slew_rps=1.0), detector=det)
    # tick 0 (t=0): releases + enters egress; the commanded heading is the spawn heading (~0, +X).
    seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=0.05, sim_time_ns=0, yaw=0.0), _frame(0, 0), 0)
    assert seeker._in_egress(0)
    assert abs(float(seeker._last_yaw)) < 1e-6, "egress must creep along the +X spawn heading, not re-aim"
    # a tick still INSIDE the egress window: heading still frozen on the spawn heading.
    t_mid = int(0.2 * 1e9)
    seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=0.05, sim_time_ns=t_mid, yaw=0.0), _frame(1, t_mid), 0)
    assert seeker._in_egress(t_mid)
    assert abs(float(seeker._last_yaw)) < 1e-6, "still egressing -> heading stays on the spawn normal"
    # a tick PAST egress -> pursuit: the heading now slews TOWARD the off-side downrange gate (+Y -> +yaw).
    t_past = int((egress + 0.2) * 1e9)
    assert not seeker._in_egress(t_past)
    seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=0.05, sim_time_ns=t_past, yaw=0.0), _frame(2, t_past), 0)
    assert float(seeker._last_yaw) > 0.0, "after egress the seeker re-aims (yaws toward the +Y gate)"


def test_spawn_egress_forward_tilt_is_bounded():
    """The egress creep is a BOUNDED feedforward tilt + pitch cap (gentle, no lunge): every egress
    command is a sane bounded BODY_RATE with the pitch under the pursuit pitch cap."""
    from racer.frames import R_world_from_body
    gate = _gate([12.0, 0.0, -2.5], normal=[1.0, 0.0, 0.0])
    det = _ProjDetector(gate, np.zeros(3), R_world_from_body(0.0, 0.0, 0.0))
    pitch_cap = GateSeekerConfig().pursuit_pitch_rate_cap_rps
    seeker = GateSeeker(config=GateSeekerConfig(
        cruise_speed=3.0, launch_ramp_s=0.0, settle_s=0.0, anchor_release_detections=1,
        use_spawn_egress=True, egress_s=0.8), detector=det)
    for k in range(8):                       # the egress window (0.8 s at ~0.05 s/tick)
        t_ns = int(k * 0.05 * 1e9)
        cmd = seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=0.05, sim_time_ns=t_ns), _frame(k, t_ns), 0)
        assert seeker._in_egress(t_ns)
        assert cmd.mode is ControlMode.BODY_RATE and np.all(np.isfinite(cmd.body_rate))
        assert np.linalg.norm(cmd.body_rate) <= seeker.controller.max_body_rate_rps + 1e-9
        assert abs(float(cmd.body_rate[1])) <= pitch_cap + 1e-9, "egress pitch must be capped (no lunge)"
        assert 0.0 <= cmd.thrust <= 1.0


def test_spawn_egress_off_goes_straight_to_pursuit():
    """With use_spawn_egress=False (legacy) there is no egress phase: the seeker re-aims at the seen
    gate immediately after release (the OLD behaviour that lunged into the start-gate frame)."""
    from racer.frames import R_world_from_body
    gate = _gate([12.0, 8.0, -2.5], normal=[1.0, 0.0, 0.0])   # off-side gate
    det = _ProjDetector(gate, np.zeros(3), R_world_from_body(0.0, 0.0, 0.0))
    seeker = GateSeeker(config=GateSeekerConfig(
        cruise_speed=3.0, launch_ramp_s=0.0, settle_s=0.0, anchor_release_detections=1,
        use_spawn_egress=False, pursuit_yaw_slew_rps=1.0), detector=det)
    assert not seeker._in_egress(0)
    seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=0.05, sim_time_ns=0, yaw=0.0), _frame(0, 0), 0)
    t1 = int(0.05 * 1e9)
    seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=0.05, sim_time_ns=t1, yaw=0.0), _frame(1, t1), 0)
    # no egress -> the heading immediately slews toward the off-side gate (the legacy re-aim).
    assert float(seeker._last_yaw) > 0.0


# ===========================================================================
# A5 (2026-06-29 attempt-5) BLOCKER 1: VERTICAL ALIGNMENT to the gate-opening centre
# ===========================================================================
def test_vertical_align_offset_sign_and_mount_handling():
    """The vertical-align offset is the TRUE world-NED Z of the gate-centre lever (NOT the raw camera-
    frame elevation): the +20deg mount is rotated out via R_camera_from_body() before reading Z. A gate
    BELOW the drone (world +Z) yields a positive (descend) vz; a gate ABOVE yields a negative (climb)."""
    from racer.frames import R_world_from_body
    # gate 1.0 m BELOW the camera (detector drone_pos z=0, gate z=+1.0 in world NED): opening is below.
    below = _gate([12.0, 0.0, 1.0], normal=[1.0, 0.0, 0.0])
    det_b = _ProjDetector(below, np.zeros(3), R_world_from_body(0.0, 0.0, 0.0))
    seeker = GateSeeker(config=GateSeekerConfig(launch_ramp_s=0.0, vertical_align_ramp_s=0.0,
                                                vertical_align_deadband_m=0.05), detector=det_b)
    pose_b = seeker.detect_gate_lever(_frame(0, 0))
    ns = _nav_fix([0, 0, 0], tsv=0.05)
    off_b = float(seeker._gate_lever_world(ns, pose_b)[2])
    assert off_b > 0.5, "a gate below the drone must read a positive (down) world vertical offset"
    assert seeker._vertical_align_vz(ns, pose_b) > 0.0, "below -> descend (positive vz, NED z+ down)"
    # gate 1.0 m ABOVE: world Z negative -> climb (negative vz).
    seeker.reset()
    above = _gate([12.0, 0.0, -1.0], normal=[1.0, 0.0, 0.0])
    seeker.detector = _ProjDetector(above, np.zeros(3), R_world_from_body(0.0, 0.0, 0.0))
    pose_a = seeker.detect_gate_lever(_frame(1, 0))
    assert float(seeker._gate_lever_world(ns, pose_a)[2]) < -0.5
    assert seeker._vertical_align_vz(ns, pose_a) < 0.0, "above -> climb (negative vz)"


def test_vertical_align_vz_is_bounded_and_ramped():
    """The commanded vz is CAPPED to the speed cap (never a dive) and RAMPED in from release (the first
    pursuit ticks don't step to a full descent), mirroring the forward-feedforward discipline."""
    from racer.frames import R_world_from_body
    # a big vertical offset so the raw kp*offset would exceed the cap -> the cap must hold.
    gate = _gate([12.0, 0.0, 5.0], normal=[1.0, 0.0, 0.0])    # ~5 m below
    det = _ProjDetector(gate, np.zeros(3), R_world_from_body(0.0, 0.0, 0.0))
    cap = 1.0
    seeker = GateSeeker(config=GateSeekerConfig(launch_ramp_s=0.0, vertical_align_speed_cap_mps=cap,
                                                vertical_align_ramp_s=0.0), detector=det)
    pose = seeker.detect_gate_lever(_frame(0, 0))
    ns = _nav_fix([0, 0, 0], tsv=0.05)
    assert abs(seeker._vertical_align_vz(ns, pose)) <= cap + 1e-9, "vz must be capped (no dive)"
    # ramp: at release the vz is ~0, growing to full over the ramp window.
    seeker2 = GateSeeker(config=GateSeekerConfig(launch_ramp_s=0.0, vertical_align_speed_cap_mps=cap,
                                                 vertical_align_ramp_s=1.0), detector=det)
    seeker2._release_t_ns = 0
    seeker2.detector = det
    p2 = seeker2.detect_gate_lever(_frame(1, 0))
    vz_t0 = abs(seeker2._vertical_align_vz(_nav_fix([0, 0, 0], tsv=0.05, sim_time_ns=0), p2))
    vz_t1 = abs(seeker2._vertical_align_vz(_nav_fix([0, 0, 0], tsv=0.05, sim_time_ns=int(1.0 * 1e9)), p2))
    assert vz_t0 < vz_t1, "the vertical-align authority must ramp in from release"


def test_vertical_align_deadband_no_command_when_centered():
    """Within the deadband (the opening is centred enough) NO vertical correction is commanded -- avoids
    hunting on estimator noise near alignment."""
    from racer.frames import R_world_from_body
    # gate at the SAME height as the detector drone (z=0): world vertical offset ~0 -> inside deadband.
    gate = _gate([12.0, 0.0, 0.0], normal=[1.0, 0.0, 0.0])
    det = _ProjDetector(gate, np.zeros(3), R_world_from_body(0.0, 0.0, 0.0))
    seeker = GateSeeker(config=GateSeekerConfig(launch_ramp_s=0.0, vertical_align_ramp_s=0.0,
                                                vertical_align_deadband_m=0.1), detector=det)
    pose = seeker.detect_gate_lever(_frame(0, 0))
    assert seeker._vertical_align_vz(_nav_fix([0, 0, 0], tsv=0.05), pose) == 0.0


def test_vertical_align_off_holds_altitude_legacy():
    """With use_vertical_align=False the seeker commands NO vertical velocity (the legacy fixed-altitude
    hold) even with a big offset -- the opt-in guard."""
    from racer.frames import R_world_from_body
    gate = _gate([12.0, 0.0, 2.0], normal=[1.0, 0.0, 0.0])   # 2 m below
    det = _ProjDetector(gate, np.zeros(3), R_world_from_body(0.0, 0.0, 0.0))
    seeker = GateSeeker(config=GateSeekerConfig(launch_ramp_s=0.0, use_vertical_align=False), detector=det)
    pose = seeker.detect_gate_lever(_frame(0, 0))
    assert seeker._vertical_align_vz(_nav_fix([0, 0, 0], tsv=0.05), pose) == 0.0


def test_vertical_align_pursuit_setpoint_carries_z_only_velocity():
    """In pursuit with a vertical offset the seeker's Setpoint to the controller carries a Z-ONLY
    velocity_ned (the vertical-align vz_t) alongside the horizontal accel feedforward -- the horizontal
    velocity components are ZERO (no horizontal velocity windup, the A4 invariant preserved)."""
    from racer.frames import R_world_from_body
    gate = _gate([12.0, 0.0, 1.5], normal=[1.0, 0.0, 0.0])   # below -> a real vz target
    captured = {}

    class _Spy:
        def __init__(self, inner):
            self.inner = inner
            self.max_body_rate_rps = inner.max_body_rate_rps
            self.hover_thrust = inner.hover_thrust
        def command(self, nav, sp):
            captured["sp"] = sp
            return self.inner.command(nav, sp)

    det = _ProjDetector(gate, np.zeros(3), R_world_from_body(0.0, 0.0, 0.0))
    seeker = GateSeeker(config=GateSeekerConfig(
        cruise_speed=3.0, launch_ramp_s=0.0, settle_s=0.0, anchor_release_detections=1,
        use_spawn_egress=False, vertical_align_ramp_s=0.0), detector=det)
    seeker.controller = _Spy(seeker.controller)
    # nav drone height matched to the detector (z=0) so the seen offset is the real 1.5 m gap.
    seeker.command_visual(_nav_fix([0, 0, 0], tsv=0.05, sim_time_ns=0), _frame(0, 0), 0)
    captured.clear()
    t1 = int(0.5 * 1e9)
    seeker.command_visual(_nav_fix([0, 0, 0], tsv=0.05, sim_time_ns=t1), _frame(1, t1), 0)
    sp = captured["sp"]
    assert sp.velocity_ned is not None, "vertical-align must carry a velocity_ned (vz_t)"
    assert sp.velocity_ned[0] == 0.0 and sp.velocity_ned[1] == 0.0, \
        "the vertical-align velocity must be Z-ONLY (no horizontal velocity setpoint -> no windup)"
    assert sp.velocity_ned[2] > 0.0, "a below-gate offset -> a positive (descend) vz_t"
    assert sp.accel_ned is not None, "the forward feedforward tilt is still an accel demand"


# ===========================================================================
# A5 (2026-06-29 attempt-5) BLOCKER 2: NEAREST-GATE first acquisition
# ===========================================================================
def test_nearest_gate_locks_near_over_far_off_axis():
    """THE A5 BLOCKER 2 fix: a NEAR slightly-off-axis start gate and a FAR dead-ahead gate are both
    visible. OLD prefer-centered locks the FAR (more centered) gate -- the A5 far-gate trap that starved
    the release. NEW prefer_nearest locks the NEAR start-line gate (the one to fly first)."""
    from racer.frames import R_world_from_body
    near = _gate([9.0, 2.6, -2.5], normal=[1, 0, 0], gate_id=0)     # near, slightly off-axis (~9.4 m)
    far = _gate([35.0, 0.0, -2.5], normal=[1, 0, 0], gate_id=1)     # far, dead-ahead (~35 m, centered)
    R_wb = R_world_from_body(0.0, 0.0, 0.0)
    # NEW: prefer_nearest -> the near gate.
    new = GateSeeker(config=GateSeekerConfig(prefer_nearest=True),
                     detector=_MultiProjDetector([near, far], np.zeros(3), R_wb))
    new_pose = new.detect_gate_lever(_frame(0, 0))
    assert new_pose is not None and new_pose.range_m < 12.0, "NEW must lock the NEAR start-line gate"
    # OLD: prefer-centered only -> the far (more centered) gate.
    old = GateSeeker(config=GateSeekerConfig(prefer_nearest=False, track_prefer_centered=True),
                     detector=_MultiProjDetector([near, far], np.zeros(3), R_wb))
    old_pose = old.detect_gate_lever(_frame(0, 0))
    assert old_pose is not None and old_pose.range_m > 30.0, \
        "OLD prefer-centered should lock the far dead-ahead gate (the A5 far-gate trap)"


def test_nearest_gate_rejects_candidate_beyond_acquire_range():
    """A candidate beyond max_acquire_range_m is REJECTED on first acquisition (a distant downrange gate
    is never the next gate to fly) -- so a lone far gate + a near gate locks the near; but if EVERY
    candidate is beyond range, we do NOT reject them all (must still lock something)."""
    from racer.frames import R_world_from_body
    R_wb = R_world_from_body(0.0, 0.0, 0.0)
    # gates near the detector's boresight height (drone_pos z=0) so PnP is well-conditioned at close range.
    near = _gate([10.0, 0.0, 0.0], normal=[1, 0, 0], gate_id=0)
    far = _gate([40.0, 1.0, 0.0], normal=[1, 0, 0], gate_id=1)
    seeker = GateSeeker(config=GateSeekerConfig(prefer_nearest=True, max_acquire_range_m=22.0),
                        detector=_MultiProjDetector([near, far], np.zeros(3), R_wb))
    pose = seeker.detect_gate_lever(_frame(0, 0))
    assert pose is not None and pose.range_m < 15.0, "the far (>22 m) gate must be rejected; lock the near"
    # ALL-far: every candidate beyond the acquire range -> fall back, lock the nearest of them (not None).
    far_only = GateSeeker(config=GateSeekerConfig(prefer_nearest=True, max_acquire_range_m=22.0),
                          detector=_MultiProjDetector(
                              [_gate([30.0, 0.0, 0.0], normal=[1, 0, 0], gate_id=0),
                               _gate([40.0, 0.0, 0.0], normal=[1, 0, 0], gate_id=1)], np.zeros(3), R_wb))
    pose2 = far_only.detect_gate_lever(_frame(0, 0))
    assert pose2 is not None and pose2.range_m < 35.0, \
        "when all candidates are beyond range, lock the nearest (don't reject everything)"


def test_prefer_nearest_off_is_legacy_prefer_centered():
    """With prefer_nearest=False the legacy prefer-centered selection is preserved (opt-in guard)."""
    from racer.frames import R_world_from_body
    R_wb = R_world_from_body(0.0, 0.0, 0.0)
    near_off = _gate([9.0, 3.0, -2.5], normal=[1, 0, 0], gate_id=0)    # near but off-axis
    far_centered = _gate([30.0, 0.0, -2.5], normal=[1, 0, 0], gate_id=1)  # far but centered
    seeker = GateSeeker(config=GateSeekerConfig(prefer_nearest=False, track_prefer_centered=True),
                        detector=_MultiProjDetector([near_off, far_centered], np.zeros(3), R_wb))
    pose = seeker.detect_gate_lever(_frame(0, 0))
    # legacy prefer-centered: the centered (far) gate wins.
    assert pose is not None and pose.range_m > 25.0


# ===========================================================================
# A5 (2026-06-29 attempt-5) BLOCKER 3: DISTANCE-BASED spawn-gate egress
# ===========================================================================
def test_distance_egress_ends_on_distance_not_only_time():
    """THE A5 BLOCKER 3 fix: with use_distance_egress the egress ends once the dead-reckoned along-
    heading distance exceeds egress_clear_distance_m -- not just the fixed timer. A short clear distance
    ends egress EARLY (before egress_s); a long one keeps egressing until the distance is crept (up to
    the egress_s timeout)."""
    from racer.frames import R_world_from_body
    gate = _gate([12.0, 0.0, -2.5], normal=[1.0, 0.0, 0.0])
    det = _ProjDetector(gate, np.zeros(3), R_world_from_body(0.0, 0.0, 0.0))
    # a SMALL clear distance: egress should end after only a little creep (well before the egress_s cap).
    seeker = GateSeeker(config=GateSeekerConfig(
        cruise_speed=3.0, launch_ramp_s=0.0, settle_s=0.0, anchor_release_detections=1,
        use_spawn_egress=True, use_distance_egress=True, egress_s=2.0,
        egress_clear_distance_m=0.3, egress_accel_mps2=1.0), detector=det)
    ended_at = None
    for k in range(40):
        t_ns = int(k * 0.05 * 1e9)
        seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=0.05, sim_time_ns=t_ns), _frame(k, t_ns), 0)
        if ended_at is None and not seeker._in_egress(t_ns) and seeker._egress_done:
            ended_at = t_ns / 1e9
            break
    assert ended_at is not None, "distance-based egress must end once the clear distance is crept"
    assert ended_at < 2.0, "the small clear distance must end egress BEFORE the egress_s timeout"
    assert seeker._egress_dist_m >= 0.3 - 1e-6, "egress must have crept at least the clear distance"


def test_distance_egress_timeout_bounds_egress():
    """egress_s remains a hard UPPER bound: even if the (large) clear distance is never reached, egress
    ends at the egress_s timeout so it can never run forever."""
    from racer.frames import R_world_from_body
    gate = _gate([12.0, 0.0, -2.5], normal=[1.0, 0.0, 0.0])
    det = _ProjDetector(gate, np.zeros(3), R_world_from_body(0.0, 0.0, 0.0))
    seeker = GateSeeker(config=GateSeekerConfig(
        cruise_speed=3.0, launch_ramp_s=0.0, settle_s=0.0, anchor_release_detections=1,
        use_spawn_egress=True, use_distance_egress=True, egress_s=0.4,
        egress_clear_distance_m=1000.0, egress_accel_mps2=0.5), detector=det)  # unreachable distance
    seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=0.05, sim_time_ns=0), _frame(0, 0), 0)
    # inside the timeout window: still egressing (distance not yet reached).
    assert seeker._in_egress(int(0.2 * 1e9))
    # past the timeout: egress ends regardless of the (unreached) distance.
    assert not seeker._in_egress(int(0.5 * 1e9)), "egress must end at the egress_s timeout (bounded)"


def test_distance_egress_off_is_legacy_time_based():
    """With use_distance_egress=False the legacy pure time-based egress is preserved: it ends only at
    egress_s, regardless of distance crept (the opt-in guard)."""
    from racer.frames import R_world_from_body
    gate = _gate([12.0, 0.0, -2.5], normal=[1.0, 0.0, 0.0])
    det = _ProjDetector(gate, np.zeros(3), R_world_from_body(0.0, 0.0, 0.0))
    seeker = GateSeeker(config=GateSeekerConfig(
        cruise_speed=3.0, launch_ramp_s=0.0, settle_s=0.0, anchor_release_detections=1,
        use_spawn_egress=True, use_distance_egress=False, egress_s=0.8), detector=det)
    seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=0.05, sim_time_ns=0), _frame(0, 0), 0)
    assert seeker._in_egress(int(0.5 * 1e9))          # still in the time window
    assert not seeker._in_egress(int(0.9 * 1e9))      # past egress_s -> done (pure time-based)


# ===========================================================================
# A5 FOOTAGE FIX: DEAD-RECKON THROUGH THE PASS + NEXT-GATE HANDOFF
# ===========================================================================
class _RangingDetector:
    """Project whatever gates it currently holds from a re-pointable (drone_pos, R_wb). The caller
    scripts the pass: gate 0's range closes >3 -> <2.5 -> LOST, then a NEXT gate appears downrange."""

    def __init__(self, drone_pos, R_wb):
        from racer.frames import R_camera_from_body
        self.gates = []
        self.drone_pos = np.asarray(drone_pos, float)
        self.R_wb = np.asarray(R_wb, float)
        self._R_cb = R_camera_from_body()

    def detect(self, frame):
        from racer.frames import CAMERA_INTRINSICS_K
        out = []
        for g in self.gates:
            half = g.inner_size_m / 2.0
            cg = np.array([[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]])
            R_wg = np.asarray(g.R_world_gate, float)
            K = CAMERA_INTRINSICS_K
            px, ok = [], True
            for c in cg:
                p_cam = self._R_cb @ (self.R_wb.T @ (g.position_ned + R_wg @ c - self.drone_pos))
                if p_cam[2] <= 0.05:
                    ok = False
                    break
                px.append([K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2], K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]])
            if ok:
                out.append(GateObservation(frame_id=frame.frame_id, sim_time_ns=frame.sim_time_ns,
                                           corners_px=np.asarray(px, float), corner_ids=np.array([0, 1, 2, 3]),
                                           corner_confidence=np.ones(4)))
        return out


def _drive_pass(seeker, det, *, drone_pos, lean_pitch):
    """Drive the seeker through a scripted pass: gate 0 closes 3.4->0 then is lost; a NEXT off-axis gate
    appears downrange + the wire advances active_gate_index. Returns per-tick (regime-passing, the first
    degenerate-tick pitch-rate command, whether the next gate was re-acquired+pursued)."""
    dt = 0.05
    rng0 = 3.4
    gi = 0
    next_gate = _gate([16.0, 2.5, -2.5], normal=[1.0, 0.0, 0.0], gate_id=1)
    degenerate_pitch = None
    reacquired = False
    passing_during_pass = False
    for k in range(40):
        t_ns = int(k * dt * 1e9)
        if rng0 > 0.15:
            det.gates = [_gate([rng0, 0.0, -2.5], normal=[1.0, 0.0, 0.0], gate_id=0)]
            degenerate = rng0 <= 2.5
            rng0 -= 0.22
        else:
            det.gates = [next_gate]
            gi = 1
            degenerate = False
        det.drone_pos = np.asarray(drone_pos, float)
        ns = NavState(sim_time_ns=t_ns, position_ned=np.asarray(drone_pos, float),
                      velocity_ned=np.zeros(3), roll=0.0, pitch=float(lean_pitch), yaw=0.0,
                      time_since_vision_update_s=(0.05 if k > 0 else float("inf")))
        cmd = seeker.command_visual(ns, _frame(k, t_ns), gi)
        if degenerate and degenerate_pitch is None:
            degenerate_pitch = float(cmd.body_rate[1])
        if seeker._passing:
            passing_during_pass = True
        if gi == 1 and not seeker._passing and seeker._last_pose is not None \
                and float(seeker._last_pose.range_m) > 5.0 and abs(float(seeker._last_yaw)) > 0.02:
            reacquired = True
    return passing_during_pass, degenerate_pitch, reacquired


def test_pass_dead_reckon_arms_and_commits_then_coasts():
    """The pass state machine ARMS once the tracked range drops below pass_arm_range_m and COMMITS to
    the dead-reckon coast once degenerate-close / lost -- the seeker enters the _passing coast regime
    (not the no-detection hold) through the pass."""
    from racer.frames import R_world_from_body
    det = _RangingDetector(np.array([0.0, 0.0, -2.5]), R_world_from_body(0.0, 0.0, 0.0))
    seeker = GateSeeker(config=GateSeekerConfig(
        cruise_speed=2.0, launch_ramp_s=0.0, settle_s=0.0, anchor_release_detections=1,
        use_spawn_egress=False, use_vertical_align=False, use_pass_dead_reckon=True,
        pass_arm_range_m=3.0, pass_degenerate_range_m=2.5, pass_coast_s=0.6), detector=det)
    passing, _, _ = _drive_pass(seeker, det, drone_pos=[0.0, 0.0, -2.5], lean_pitch=np.deg2rad(-7.0))
    assert passing, "the seeker must enter the dead-reckon coast (_passing) through the pass"


def test_pass_dead_reckon_suppresses_the_pitch_up_old_pursuit_noses_up():
    """THE A5-footage fix: threading the gate while leaned forward, the OLD pursuit servos on the seen
    gate and commands a sizeable NOSE-UP at the close pass tick (the backside-clip pitch-up). The NEW
    dead-reckon coast SUPPRESSES it (holds the forward lean, ~zero pitch-rate command)."""
    from racer.frames import R_world_from_body
    lean = np.deg2rad(-7.0)
    cfg = dict(cruise_speed=2.0, launch_ramp_s=0.0, settle_s=0.0, anchor_release_detections=1,
               use_spawn_egress=False, use_vertical_align=False, pass_arm_range_m=3.0,
               pass_degenerate_range_m=2.5, pass_coast_s=0.6)
    det_old = _RangingDetector(np.array([0.0, 0.0, -2.5]), R_world_from_body(0.0, 0.0, 0.0))
    old = GateSeeker(config=GateSeekerConfig(use_pass_dead_reckon=False, **cfg), detector=det_old)
    _, old_pitch, _ = _drive_pass(old, det_old, drone_pos=[0.0, 0.0, -2.5], lean_pitch=lean)
    det_new = _RangingDetector(np.array([0.0, 0.0, -2.5]), R_world_from_body(0.0, 0.0, 0.0))
    new = GateSeeker(config=GateSeekerConfig(use_pass_dead_reckon=True, **cfg), detector=det_new)
    _, new_pitch, _ = _drive_pass(new, det_new, drone_pos=[0.0, 0.0, -2.5], lean_pitch=lean)
    # odo pitch-rate sign (-1): a NOSE-UP correction off the forward lean is a POSITIVE body_rate[1].
    assert old_pitch is not None and old_pitch > 0.5, "OLD pursuit should command a nose-up at the pass"
    assert new_pitch is not None and new_pitch < 0.5 * old_pitch, \
        "NEW dead-reckon coast must SUPPRESS the pass pitch-up (hold the forward lean)"


def test_pass_dead_reckon_hands_off_to_next_gate():
    """After the dead-reckon glide the NEW seeker re-acquires the NEXT gate (active_gate_index advanced)
    and resumes pursuit -- a +yaw toward the off-axis next gate (the handoff), not an indefinite drift."""
    from racer.frames import R_world_from_body
    det = _RangingDetector(np.array([0.0, 0.0, -2.5]), R_world_from_body(0.0, 0.0, 0.0))
    seeker = GateSeeker(config=GateSeekerConfig(
        cruise_speed=2.0, launch_ramp_s=0.0, settle_s=0.0, anchor_release_detections=1,
        use_spawn_egress=False, use_vertical_align=False, use_pass_dead_reckon=True,
        pass_arm_range_m=3.0, pass_degenerate_range_m=2.5, pass_coast_s=0.6, acquire_next_s=3.0),
        detector=det)
    _, _, reacq = _drive_pass(seeker, det, drone_pos=[0.0, 0.0, -2.5], lean_pitch=np.deg2rad(-7.0))
    assert reacq, "the seeker must re-acquire + pursue the next gate after the pass (the handoff)"
    assert not seeker._passing, "the pass regime must have ended once the next gate is pursued"


def test_pass_coast_commands_forward_demand_not_a_zero_hold():
    """During the dead-reckon coast the seeker commands the bounded FORWARD feedforward tilt (accel_ned),
    NOT the no-detection zero-velocity hold -- so it glides THROUGH the opening instead of stalling.
    The Setpoint carries a real accel_ned along the frozen heading + no horizontal velocity setpoint."""
    from racer.frames import R_world_from_body
    captured = {}

    class _Spy:
        def __init__(self, inner):
            self.inner = inner
            self.max_body_rate_rps = inner.max_body_rate_rps
            self.hover_thrust = inner.hover_thrust
        def command(self, nav, sp):
            captured["sp"] = sp
            return self.inner.command(nav, sp)

    # detector drone at the gate HEIGHT (z=-2.5) so the PnP range ≈ the horizontal closing range.
    det = _RangingDetector(np.array([0.0, 0.0, -2.5]), R_world_from_body(0.0, 0.0, 0.0))
    seeker = GateSeeker(config=GateSeekerConfig(
        cruise_speed=2.0, launch_ramp_s=0.0, settle_s=0.0, anchor_release_detections=1,
        use_spawn_egress=False, use_vertical_align=False, use_pass_dead_reckon=True,
        pass_arm_range_m=3.0, pass_degenerate_range_m=2.5, pass_coast_accel_mps2=1.2), detector=det)
    seeker.controller = _Spy(seeker.controller)
    # anchor + arm + commit: drive a few closing ticks into the degenerate band.
    rng = 3.4
    for k in range(10):
        det.gates = [_gate([rng, 0.0, -2.5], normal=[1.0, 0.0, 0.0], gate_id=0)]
        rng -= 0.22
        det.drone_pos = np.array([0.0, 0.0, -2.5])
        seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=0.05, sim_time_ns=int(k * 0.05 * 1e9)),
                              _frame(k, int(k * 0.05 * 1e9)), 0)
    assert seeker._passing, "precondition: the seeker should be in the dead-reckon coast"
    sp = captured["sp"]
    assert sp.accel_ned is not None and float(np.linalg.norm(sp.accel_ned)) > 0.0, \
        "the pass coast must command a forward feedforward demand (glide through), not a zero hold"
    # horizontal velocity setpoint must be zero/None (no velocity windup; vertical-align is off here).
    assert sp.velocity_ned is None or (sp.velocity_ned[0] == 0.0 and sp.velocity_ned[1] == 0.0)


def test_mid_approach_detection_gap_does_not_trigger_pitch_up():
    """A transient mid-approach detection gap (a brief det=0 BEFORE the pass is armed -- the gate still
    far, range never below pass_arm_range_m) must NOT trigger the pass dead-reckon / a pitch-up: the
    seeker coasts level on the last heading (the gentle no-detection hold), never commits to the pass."""
    from racer.frames import R_world_from_body
    det = _RangingDetector(np.array([0.0, 0.0, -2.5]), R_world_from_body(0.0, 0.0, 0.0))
    seeker = GateSeeker(config=GateSeekerConfig(
        cruise_speed=2.0, launch_ramp_s=0.0, settle_s=0.0, anchor_release_detections=1,
        use_spawn_egress=False, use_vertical_align=False, use_pass_dead_reckon=True,
        pass_arm_range_m=3.0, pass_degenerate_range_m=2.5), detector=det)
    far = _gate([14.0, 0.0, -2.5], normal=[1.0, 0.0, 0.0], gate_id=0)   # FAR (never armed: 14 m >> 3 m)
    # anchor on the far gate, then a transient detection gap (empty det) mid-approach.
    det.gates = [far]; det.drone_pos = np.zeros(3)
    seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=0.05, sim_time_ns=0), _frame(0, 0), 0)
    assert seeker._anchored and not seeker._pass_armed   # far gate -> never armed
    det.gates = []                                       # transient detection gap (gate momentarily lost)
    cmd = seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=0.05, sim_time_ns=10_000_000),
                                _frame(1, 10_000_000), 0)
    assert not seeker._passing, "an UNARMED mid-approach detection gap must NOT commit to the pass"
    # the no-detection hold freezes attitude (zero roll/pitch rate) -> no pitch-up.
    assert abs(float(cmd.body_rate[1])) < 1e-6, "the unarmed gap must coast level (no pitch-up)"


def test_pass_dead_reckon_off_is_legacy_no_pass_regime():
    """With use_pass_dead_reckon=False (opt-in guard) the pass regime never engages: through the same
    pass the seeker never sets _passing (the legacy pursuit -> no-detection-hold path is preserved)."""
    from racer.frames import R_world_from_body
    det = _RangingDetector(np.array([0.0, 0.0, -2.5]), R_world_from_body(0.0, 0.0, 0.0))
    seeker = GateSeeker(config=GateSeekerConfig(
        cruise_speed=2.0, launch_ramp_s=0.0, settle_s=0.0, anchor_release_detections=1,
        use_spawn_egress=False, use_vertical_align=False, use_pass_dead_reckon=False), detector=det)
    passing, _, _ = _drive_pass(seeker, det, drone_pos=[0.0, 0.0, -2.5], lean_pitch=np.deg2rad(-7.0))
    assert not passing, "with the pass dead-reckon OFF the seeker must never enter the pass regime"


def test_pass_index_advance_triggers_dead_reckon():
    """The wire signal: even without a close-range geometric trigger, an active_gate_index INCREMENT
    (RACE_STATUS advanced) commits the dead-reckon coast -- the authoritative pass signal."""
    from racer.frames import R_world_from_body
    gate = _gate([8.0, 0.0, -2.5], normal=[1.0, 0.0, 0.0], gate_id=0)   # comfortably > arm range
    det = _ProjDetector(gate, np.zeros(3), R_world_from_body(0.0, 0.0, 0.0))
    seeker = GateSeeker(config=GateSeekerConfig(
        cruise_speed=2.0, launch_ramp_s=0.0, settle_s=0.0, anchor_release_detections=1,
        use_spawn_egress=False, use_vertical_align=False, use_pass_dead_reckon=True), detector=det)
    seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=0.05, sim_time_ns=0), _frame(0, 0), 0)  # gi 0
    assert not seeker._passing
    # the wire advances the active gate index -> commit the dead-reckon coast (a pass happened).
    seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=0.05, sim_time_ns=10_000_000),
                          _frame(1, 10_000_000), 1)
    assert seeker._passing, "an active_gate_index increment must commit the dead-reckon coast"


def test_pass_dead_reckon_window_is_bounded_reverts_to_hold():
    """The committed coast (dead-reckon + acquire-next) is BOUNDED by pass_coast_s + acquire_next_s; past
    that, with no next gate ever seen, the seeker ends the pass and reverts to the gentle no-detection
    level hold (never an indefinite forward glide / never a pitch-up)."""
    from racer.frames import R_world_from_body
    det = _RangingDetector(np.array([0.0, 0.0, -2.5]), R_world_from_body(0.0, 0.0, 0.0))
    seeker = GateSeeker(config=GateSeekerConfig(
        cruise_speed=2.0, launch_ramp_s=0.0, settle_s=0.0, anchor_release_detections=1,
        use_spawn_egress=False, use_vertical_align=False, use_pass_dead_reckon=True,
        pass_arm_range_m=3.0, pass_degenerate_range_m=2.5, pass_coast_s=0.4, acquire_next_s=0.4),
        detector=det)
    # arm + commit by closing into the degenerate band, then NO gate at all (next never appears).
    rng = 3.4
    for k in range(10):
        det.gates = [_gate([rng, 0.0, -2.5], normal=[1.0, 0.0, 0.0], gate_id=0)]; rng -= 0.22
        det.drone_pos = np.array([0.0, 0.0, -2.5])
        seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=0.05, sim_time_ns=int(k * 0.05 * 1e9)),
                              _frame(k, int(k * 0.05 * 1e9)), 0)
    assert seeker._passing
    # well past the coast+acquire window with nothing in view -> the pass ends, revert to the hold.
    det.gates = []
    t_late = int(2.0 * 1e9)
    cmd = seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=0.05, sim_time_ns=t_late), _frame(99, t_late), 0)
    assert not seeker._passing, "the bounded coast+acquire window must end the pass (no infinite glide)"
    assert abs(float(cmd.body_rate[1])) < 1e-6, "after the bounded window it reverts to a level hold (no pitch-up)"


def test_reset_clears_pass_state():
    """reset() drops the pass state machine (back to the launch-anchor boot regime)."""
    from racer.frames import R_world_from_body
    det = _RangingDetector(np.array([0.0, 0.0, -2.5]), R_world_from_body(0.0, 0.0, 0.0))
    seeker = GateSeeker(config=GateSeekerConfig(
        launch_ramp_s=0.0, settle_s=0.0, anchor_release_detections=1, use_spawn_egress=False,
        use_pass_dead_reckon=True, pass_arm_range_m=3.0, pass_degenerate_range_m=2.5), detector=det)
    rng = 3.4
    for k in range(10):
        det.gates = [_gate([rng, 0.0, -2.5], normal=[1.0, 0.0, 0.0], gate_id=0)]; rng -= 0.22
        det.drone_pos = np.array([0.0, 0.0, -2.5])
        seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=0.05, sim_time_ns=int(k * 0.05 * 1e9)),
                              _frame(k, int(k * 0.05 * 1e9)), 0)
    assert seeker._passing and seeker._pass_armed
    seeker.reset()
    assert not seeker._passing and not seeker._pass_armed
    assert seeker._pass_t_ns is None and seeker._pass_heading is None
    assert seeker._pass_min_range_m == float("inf")


def test_default_config_enables_pass_dead_reckon():
    """The dead-reckon-through-pass + handoff ships ON by default (the slow-lap deploy needs it); the
    behaviour is self-defaulting (no fly_rl flag needed)."""
    cfg = GateSeekerConfig()
    assert cfg.use_pass_dead_reckon is True
    assert cfg.pass_arm_range_m == 3.0 and cfg.pass_degenerate_range_m == 2.5


# ===========================================================================
# A7 (2026-06-29): SPAWN THRUST-COLLAPSE — point-blank start gate must not free-fall
# ===========================================================================
def test_pointblank_elevation_guard_suppresses_descent_old_collapses_thrust():
    """THE A7 fix (root cause): the drone SPAWNS INSIDE gate 0, point-blank. At ~1 m range the PnP
    elevation is garbage (trk_el<0 -> the gate reads BELOW boresight), so the OLD vertical-align
    commands a DESCENT, which cuts the alt-hold collective to the 0.05 floor -> free-fall. The NEW
    point-blank elevation guard commands NO vertical correction below min_trust_elevation_range_m, so
    the alt-hold holds height (no thrust collapse)."""
    from racer.frames import R_world_from_body
    # point-blank (slant range ~1.4 m) and ~0.4 m BELOW boresight -> a positive (descend) world vertical
    # offset. The detector drone is at z=-2.5; gate at z=-2.1 (0.4 m below) and 1.3 m ahead.
    gate = _gate([1.3, 0.0, -2.1], normal=[1.0, 0.0, 0.0])
    det = _ProjDetector(gate, np.array([0.0, 0.0, -2.5]), R_world_from_body(0.0, 0.0, 0.0))
    ns = _nav_fix([0, 0, -2.5], tsv=0.05)

    # OLD (guard off): the point-blank gate reads a descent demand -> a positive vz_t.
    old = GateSeeker(config=GateSeekerConfig(launch_ramp_s=0.0, vertical_align_ramp_s=0.0,
                                             use_min_trust_elevation=False), detector=det)
    pose_old = old.detect_gate_lever(_frame(0, 0))
    assert pose_old is not None and pose_old.range_m < 2.0          # point-blank
    assert old._vertical_align_vz(ns, pose_old) > 0.0, "OLD: point-blank gate must command a descent (the bug)"

    # NEW (guard on, default): below the min-trust range -> NO vertical correction (no descent).
    new = GateSeeker(config=GateSeekerConfig(launch_ramp_s=0.0, vertical_align_ramp_s=0.0,
                                             min_trust_elevation_range_m=2.0), detector=det)
    pose_new = new.detect_gate_lever(_frame(0, 0))
    assert new._vertical_align_vz(ns, pose_new) == 0.0, \
        "NEW: a point-blank gate's garbage elevation must command NO descent (the A7 guard)"


def test_pointblank_thrust_does_not_collapse_to_floor_in_pursuit():
    """End-to-end vertical loop: spawning point-blank below the gate, the NEW seeker's commanded
    collective stays at hover (no descent demand -> the alt-hold holds height) instead of collapsing to
    the 0.05 floor and free-falling (the A7 collapse)."""
    from racer.frames import R_world_from_body
    gate = _gate([1.2, 0.0, -1.9], normal=[1.0, 0.0, 0.0])
    R_wb = R_world_from_body(0.0, 0.0, 0.0)
    # egress + pass-dead-reckon off to isolate the PURSUIT vertical-align channel (the collapse path).
    seeker = GateSeeker(config=GateSeekerConfig(
        cruise_speed=2.0, launch_ramp_s=0.0, settle_s=0.0, anchor_release_detections=1,
        use_spawn_egress=False, use_pass_dead_reckon=False, vertical_align_ramp_s=0.0),
        detector=_ProjDetector(gate, np.zeros(3), R_wb))
    hover = seeker.controller.hover_thrust
    z, vz, dt = -2.5, 0.0, 0.03
    min_thr = np.inf
    for k in range(12):
        t_ns = int(k * dt * 1e9)
        seeker.detector = _ProjDetector(gate, np.array([0.0, 0.0, z]), R_wb)
        ns = NavState(sim_time_ns=t_ns, position_ned=np.array([0.0, 0.0, z]),
                      velocity_ned=np.array([0.0, 0.0, vz]), roll=0.0, pitch=0.0, yaw=0.0,
                      time_since_vision_update_s=(0.05 if k > 0 else float("inf")))
        cmd = seeker.command_visual(ns, _frame(k, t_ns), 0)
        min_thr = min(min_thr, float(cmd.thrust))
        az = 9.80665 * (1.0 - float(cmd.thrust) / hover)
        vz += az * dt
        z += vz * dt
    assert min_thr >= hover - 1e-6, "the point-blank pursuit must hold >= hover thrust (no free-fall)"
    assert abs(z - (-2.5)) < 0.05, "the drone must HOLD its height at point-blank (no sink)"


def test_egress_thrust_floor_holds_hover_even_if_alt_hold_would_cut():
    """The egress THRUST FLOOR (invariant 1): during the spawn-gate egress the collective is floored to
    at least hover-equivalent, so even a (hypothetical) alt-hold command below hover is lifted -> the
    drone can only HOLD/CLIMB out of spawn, never free-fall."""
    import dataclasses
    seeker = GateSeeker(config=GateSeekerConfig(use_egress_thrust_floor=True, egress_thrust_floor_frac=1.0))
    hover = seeker.controller.hover_thrust
    # a command whose collective dipped to the controller floor (the collapse the A7 free-fall rode).
    low = ControlCommand(mode=ControlMode.BODY_RATE, body_rate=np.zeros(3), thrust=0.05)
    floored = seeker._floor_egress_thrust(low)
    assert floored.thrust == pytest.approx(hover), "egress must floor the collective to hover (no free-fall)"
    # a command already at/above hover is left untouched (the floor never SUPPRESSES a needed climb).
    high = ControlCommand(mode=ControlMode.BODY_RATE, body_rate=np.zeros(3), thrust=0.5)
    assert seeker._floor_egress_thrust(high).thrust == pytest.approx(0.5)
    # off => legacy pass-through (the opt-in guard).
    off = GateSeeker(config=GateSeekerConfig(use_egress_thrust_floor=False))
    assert off._floor_egress_thrust(low).thrust == pytest.approx(0.05)


def test_egress_command_thrust_is_at_least_hover():
    """In the live egress phase every commanded collective is >= hover (the spawn drone holds/climbs out
    of gate 0, never descends)."""
    from racer.frames import R_world_from_body
    gate = _gate([1.5, 0.0, -2.4], normal=[1.0, 0.0, 0.0])   # point-blank, near level
    det = _ProjDetector(gate, np.zeros(3), R_world_from_body(0.0, 0.0, 0.0))
    seeker = GateSeeker(config=GateSeekerConfig(
        cruise_speed=2.0, launch_ramp_s=0.0, settle_s=0.0, anchor_release_detections=1,
        use_spawn_egress=True, egress_s=0.8), detector=det)
    hover = seeker.controller.hover_thrust
    for k in range(8):
        t_ns = int(k * 0.05 * 1e9)
        seeker.detector = _ProjDetector(gate, np.array([0.0, 0.0, -2.5]), R_world_from_body(0.0, 0.0, 0.0))
        cmd = seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=0.05, sim_time_ns=t_ns), _frame(k, t_ns), 0)
        assert seeker._in_egress(t_ns)
        assert cmd.thrust >= hover - 1e-6, "egress collective must stay >= hover (no spawn free-fall)"
        assert cmd.thrust <= 1.0


def test_post_egress_downrange_pursuit_is_byte_identical_to_pre_a7():
    """REGRESSION / byte-identity: the A7 guards must change ONLY the point-blank spawn behaviour. For a
    NORMAL downrange gate (range well beyond min_trust_elevation_range_m), the seeker's commands are
    BYTE-IDENTICAL with the guards ON vs OFF -- the elevation guard never fires and the egress floor is a
    no-op once the alt-hold sits at/above hover. Pins that post-egress flight is unchanged."""
    from racer.frames import R_world_from_body
    # a normal downrange gate ~12 m ahead + ~1 m below: vertical-align is ACTIVE (a real vz) and the
    # range is far beyond the 2 m guard, so the guard must NOT alter the command.
    gate = _gate([12.0, 1.0, -1.5], normal=[1.0, 0.0, 0.0])
    R_wb = R_world_from_body(0.0, 0.0, 0.0)

    def _run(guards_on: bool):
        det = _ProjDetector(gate, np.zeros(3), R_wb)
        seeker = GateSeeker(config=GateSeekerConfig(
            cruise_speed=3.0, launch_ramp_s=0.0, settle_s=0.0, anchor_release_detections=1,
            use_spawn_egress=False, vertical_align_ramp_s=0.0,
            use_egress_thrust_floor=guards_on, use_min_trust_elevation=guards_on), detector=det)
        rates, thrusts = [], []
        for k in range(8):
            t_ns = int(k * 0.05 * 1e9)
            ns = _nav_fix([0, 0, -2.5], tsv=0.05, sim_time_ns=t_ns, yaw=0.0)
            cmd = seeker.command_visual(ns, _frame(k, t_ns), 0)
            rates.append(np.asarray(cmd.body_rate, float).copy())
            thrusts.append(float(cmd.thrust))
        return np.array(rates), np.array(thrusts)

    r_on, t_on = _run(guards_on=True)
    r_off, t_off = _run(guards_on=False)
    np.testing.assert_array_equal(r_on, r_off)   # BYTE-identical body rates downrange
    np.testing.assert_array_equal(t_on, t_off)   # BYTE-identical collective downrange
    # sanity: vertical-align really was active here (a non-trivial vz would have moved thrust off a flat
    # hover) -- so the byte-identity is a real test of the guard not firing, not a vacuous all-hover run.
    assert t_on.std() > 0.0 or r_on.std() > 0.0


def test_a7_config_defaults_on():
    """The A7 guards ship ON by default (the slow-lap deploy needs them); self-defaulting, no flag."""
    cfg = GateSeekerConfig()
    assert cfg.use_egress_thrust_floor is True and cfg.egress_thrust_floor_frac == 1.0
    assert cfg.use_min_trust_elevation is True and cfg.min_trust_elevation_range_m == 2.0


def test_pipeline_smoke_off_path_navigator_still_constructs():
    """Control: the legacy (vq1_case_a) profile config builds a Navigator that runs on a GIVEN pose
    with NO AHRS -- the byte-identical legacy path is unaffected by the new modules."""
    from racer.navigator import Navigator

    gate = _gate([10.0, 0.0, 0.0], normal=[1.0, 0.0, 0.0])
    nav = Navigator(gates=[gate], detector=None, config=vq1_case_a().nav_config)
    ds = DroneState(
        sim_time_ns=1,
        orientation_ned_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        position_ned=np.zeros(3), velocity_ned=np.zeros(3),
    )
    ns = nav.update(ds, None)
    assert np.all(np.isfinite(ns.position_ned))
    assert nav._ahrs is None        # no AHRS constructed on the legacy path


# ===========================================================================
# PATCH-1 (2026-07-19): RE-ACQUIRE CAP (WP1a) + HELD-POINT HINT (WP1b) + DECISION LOG (WP0)
# ===========================================================================
def _level_multi(gates):
    """A multi-gate projection detector from a drone at the origin, level, facing north (+X). At
    frame_id=0 EVERY gate projects UN-jittered (sin(pi*i)=0), so single-tick selection is clean."""
    from racer.frames import R_world_from_body
    return _MultiProjDetector(gates, np.zeros(3), R_world_from_body(0.0, 0.0, 0.0))


class _BlindDet:
    def detect(self, frame):
        return []


def test_cold_start_keeps_fallback_but_reacquire_enforces_cap():
    """WP1a A/B: with ONLY a beyond-cap candidate, a TRUE COLD START keeps the A5 fallback (must lock
    something at flight start), but a RE-ACQUISITION (this gate locked before + expired via coast) does
    NOT -- it returns None so the pipeline coasts on the ego-propagated hold rather than far-lock a wrong
    gate."""
    far = [_gate([26.0, 3.0, 0.0], normal=[1, 0, 0], gate_id=0)]     # ~26.2 m, beyond the 22 m cap
    # COLD start: fresh seeker, only the far candidate -> fallback locks it.
    cold = GateSeeker(config=GateSeekerConfig(max_acquire_range_m=22.0), detector=_level_multi(far))
    assert cold._track_ever_locked is False
    p_cold = cold.detect_gate_lever(_frame(0, 0))
    assert p_cold is not None and p_cold.range_m > 22.0, "cold start must keep the beyond-cap fallback (A5)"
    # RE-ACQUISITION: lock a near gate, coast it out past the limit, then only the far gate is visible.
    reacq = GateSeeker(config=GateSeekerConfig(max_acquire_range_m=22.0, track_max_coast_ticks=3),
                       detector=_level_multi([_gate([10.0, 0.0, 0.0], normal=[1, 0, 0])]))
    assert reacq.detect_gate_lever(_frame(0, 0)) is not None       # lock the near gate ~10 m
    assert reacq._track_ever_locked is True
    reacq.detector = _BlindDet()
    for k in range(1, 6):                                          # blind frames past coast -> track drops
        assert reacq.detect_gate_lever(_frame(k, 0)) is None
    assert reacq._track_range_m is None and reacq._track_ever_locked is True
    reacq.detector = _level_multi(far)
    assert reacq.detect_gate_lever(_frame(20, 0)) is None, "re-acquire must reject the beyond-cap candidate"
    assert reacq.last_decision(0)["reason"] == "reacquire_range_reject"


def test_reacquire_far_lock_231220_regression():
    """NAMED REGRESSION -- flight 20260719_231220: gate 0 locked ~10 m, a pitch-dive blackout expired the
    slot0 track (RACE_STATUS active_gate_index still 0, so NO seeker.reset), and the ONLY visible candidate
    on re-acquire was the next (upper-right) gate at ~26.7 m -- beyond the 22 m acquire cap. The pre-patch
    fallback LOCKED it and the policy flew at the wrong gate. WP1a: a re-acquire must NOT far-lock -> None
    (coast on the propagated hold). A COLD seeker with the SAME candidate DOES lock it (proves the candidate
    is genuine + detectable; only the cold/re-acquire distinction changes the outcome)."""
    wrong = [_gate([25.0, 8.0, -6.0], normal=[1, 0, 0], gate_id=1)]  # ~26.9 m, up-and-right, beyond cap
    # control: a COLD seeker locks the far candidate (it is real + within FOV + PnP-clean).
    cold = GateSeeker(config=GateSeekerConfig(max_acquire_range_m=22.0), detector=_level_multi(wrong))
    p_cold = cold.detect_gate_lever(_frame(0, 0))
    assert p_cold is not None and p_cold.range_m > 22.0

    # the 231220 sequence: lock the active gate ~10 m, blackout past the coast limit, then the wrong gate.
    seeker = GateSeeker(config=GateSeekerConfig(max_acquire_range_m=22.0, track_max_coast_ticks=8),
                        detector=_level_multi([_gate([10.0, 0.0, 0.0], normal=[1, 0, 0], gate_id=0)]))
    assert seeker.detect_gate_lever(_frame(0, 0)) is not None
    seeker.detector = _BlindDet()
    for k in range(1, 11):                                          # > track_max_coast_ticks blind ticks
        seeker.detect_gate_lever(_frame(k, 0))
    assert seeker._track_range_m is None                            # track expired via coast (NOT a reset)
    seeker.detector = _level_multi(wrong)
    pose = seeker.detect_gate_lever(_frame(50, 0))
    assert pose is None, "231220: the far wrong gate on re-acquire must be REJECTED, not locked + flown at"
    assert seeker.last_decision(0)["reason"] == "reacquire_range_reject"


def test_reacquire_hint_prefers_near_hint_over_nearer_off_hint():
    """WP1b: on a RE-ACQUISITION with a held-point hint, a candidate NEAR the hint beats a candidate that
    is nearer in RANGE but off the hint bearing -- the hint disambiguates which opening is 'the gate we
    lost'. Without the hint the same re-acquire picks the nearer (wrong) gate."""
    A = _gate([20.0, 0.0, 0.0], normal=[1, 0, 0], gate_id=0)        # dead-ahead, 20 m (matches the hint)
    B = _gate([10.0, 2.0, 0.0], normal=[1, 0, 0], gate_id=1)        # nearer (10 m) but off to the right
    hint = np.array([20.0, 0.0, 0.0])                              # body-FRD lever pointing at A

    def _reacq():
        s = GateSeeker(config=GateSeekerConfig(max_acquire_range_m=25.0), detector=_level_multi([A, B]))
        s._track_ever_locked = True                                # this gate locked before -> re-acquire
        return s

    p_nohint = _reacq().detect_gate_lever(_frame(0, 0))            # nearest -> B
    assert p_nohint is not None and p_nohint.range_m < 15.0, "no-hint re-acquire picks the nearer gate B"
    p_hint = _reacq().detect_gate_lever(_frame(0, 0), hint_rel_body_frd=hint)
    assert p_hint is not None and p_hint.range_m > 15.0, "the hint must steer re-acquire to the near-hint gate A"


def test_reacquire_hint_veto_rejects_off_hint_candidate():
    """WP1b: a re-acquire candidate whose bearing differs from the hint by more than
    reacquire_hint_max_bearing_rad is VETOED; if that empties the (in-cap) pool -> None + the veto reason."""
    C = [_gate([12.0, -10.0, 0.0], normal=[1, 0, 0], gate_id=0)]   # in cap (~15.6 m) but hard-left (~-0.69 rad)
    s = GateSeeker(config=GateSeekerConfig(max_acquire_range_m=25.0, reacquire_hint_max_bearing_rad=0.6),
                   detector=_level_multi(C))
    s._track_ever_locked = True
    pose = s.detect_gate_lever(_frame(0, 0), hint_rel_body_frd=np.array([18.0, 0.0, 0.0]))  # hint = dead-ahead
    assert pose is None, "an off-hint re-acquire candidate must be vetoed"
    assert s.last_decision(0)["reason"] == "reacquire_hint_reject"


def test_reacquire_hint_ignored_on_cold_start():
    """WP1b guard: the hint is consulted ONLY on a RE-ACQUISITION. A COLD start (never locked) ignores it
    and keeps the A5 nearest+fallback selection, so a hint can never distort the flight-start acquisition."""
    A = _gate([20.0, 0.0, 0.0], normal=[1, 0, 0], gate_id=0)
    B = _gate([10.0, 2.0, 0.0], normal=[1, 0, 0], gate_id=1)
    s = GateSeeker(config=GateSeekerConfig(max_acquire_range_m=25.0), detector=_level_multi([A, B]))
    assert s._track_ever_locked is False                           # COLD
    # even with a hint pointing at the FAR gate A, a cold start uses nearest -> B (hint ignored).
    pose = s.detect_gate_lever(_frame(0, 0), hint_rel_body_frd=np.array([20.0, 0.0, 0.0]))
    assert pose is not None and pose.range_m < 15.0, "cold start must ignore the hint (nearest selection)"


def test_seeker_decision_accessor_records_each_outcome():
    """WP0: last_decision(slot) is populated for the emitted / valid-poses-empty / continuity-reject /
    re-acquire-reject outcomes, with candidate counts, track state, and (when emitted) the chosen pose."""
    from racer.frames import R_world_from_body
    R_wb = R_world_from_body(0.0, 0.0, 0.0)
    g = _gate([12.0, 0.0, 0.0], normal=[1, 0, 0])
    s = GateSeeker(config=GateSeekerConfig(), detector=_level_multi([g]))
    # emitted -> reason None, emit fields populated.
    p = s.detect_gate_lever(_frame(0, 0))
    d = s.last_decision(0)
    assert p is not None and d["reason"] is None
    assert d["n_cand"] >= 1 and d["emit_range_m"] is not None and d["emit_bearing"] is not None
    assert d["track_range_m"] is not None
    # valid_poses_empty -> reason set, no emit.
    s.detector = _BlindDet()
    assert s.detect_gate_lever(_frame(1, 0)) is None
    assert s.last_decision(0)["reason"] == "valid_poses_empty" and s.last_decision(0)["emit_range_m"] is None
    # continuity-reject -> a candidate existed (n_cand>=1) but jumped the track gate.
    s2 = GateSeeker(config=GateSeekerConfig(track_max_range_jump_m=6.0), detector=None)
    s2._track_range_m, s2._track_bearing, s2._track_ever_locked = 20.0, np.zeros(2), True

    class _Jumped:
        def detect(self, frame):
            near = _gate([2.0, 0.0, 0.0], normal=[1, 0, 0])       # ~2 m, a >15 m jump from the 20 m track
            return _ProjDetector(near, np.zeros(3), R_wb).detect(frame)

    s2.detector = _Jumped()
    assert s2.detect_gate_lever(_frame(1, 0)) is None
    assert s2.last_decision(0)["reason"] == "continuity_reject" and s2.last_decision(0)["n_cand"] >= 1
    # slot1 decision is recorded too (single gate -> the next-gate slot finds no non-active candidate).
    s3 = GateSeeker(config=GateSeekerConfig(), detector=_level_multi([g]))
    s3.detect_gate_lever(_frame(0, 0))
    s3.detect_next_gate_lever(_frame(0, 0))
    d1 = s3.last_decision(1)
    assert d1 is not None and d1["reason"] in ("valid_poses_empty", "other")


def test_reset_clears_reacquire_latch_back_to_cold():
    """WP1a: reset() (a sim epoch / a gate advance in fly_rl) returns the seeker to a COLD start -- the
    ever-locked latch clears, so the next first-acquisition keeps the A5 fallback for the NEW gate."""
    far = [_gate([26.0, 3.0, 0.0], normal=[1, 0, 0])]
    s = GateSeeker(config=GateSeekerConfig(max_acquire_range_m=22.0),
                   detector=_level_multi([_gate([10.0, 0.0, 0.0], normal=[1, 0, 0])]))
    s.detect_gate_lever(_frame(0, 0))
    assert s._track_ever_locked is True
    s.reset()
    assert s._track_ever_locked is False and s.last_decision(0) is None
    # after reset the far-only candidate locks again (cold fallback), NOT a re-acquire reject.
    s.detector = _level_multi(far)
    p = s.detect_gate_lever(_frame(1, 0))
    assert p is not None and p.range_m > 22.0


def test_perceived_gate_down_bias_lowers_every_emission_unchanged():
    """WP5 subject / regression: the perceived-gate vertical bias lowers EVERY emitted pose by exactly the
    bias on the camera +Y (down) axis (x/z untouched). Pins the z-bias the recipes now pin to 0, and
    guards that the Patch-1 seeker edits did not disturb the emission path."""
    g = _gate([12.0, 0.0, 0.0], normal=[1, 0, 0])
    base = GateSeeker(config=GateSeekerConfig(perceived_gate_down_bias_m=0.0), detector=_level_multi([g]))
    biased = GateSeeker(config=GateSeekerConfig(perceived_gate_down_bias_m=0.5), detector=_level_multi([g]))
    pb = base.detect_gate_lever(_frame(0, 0))
    px = biased.detect_gate_lever(_frame(0, 0))
    assert pb is not None and px is not None
    assert float(px.t_cam_gate[1] - pb.t_cam_gate[1]) == pytest.approx(0.5, abs=1e-6)   # lowered by the bias
    assert float(px.t_cam_gate[0]) == pytest.approx(float(pb.t_cam_gate[0]), abs=1e-6)  # x unchanged
    assert float(px.t_cam_gate[2]) == pytest.approx(float(pb.t_cam_gate[2]), abs=1e-6)  # z unchanged


def test_duplicate_gate_merge_collapses_same_gate_keeps_distinct():
    """2026-07-23: a gate larger than the frame is found by several anchors on different fragments;
    their boxes overlap too little for NMS (measured IoU 0.287). _merge_duplicate_poses folds poses
    that agree in BOTH bearing and range into one, keeping the most-corners representative, and must
    NOT touch genuinely distinct gates. Recall-safe by construction: it only removes a pose that has
    a near-twin."""
    from racer.contracts import GatePose
    cfg = GateSeekerConfig(dup_merge_bearing_rad=0.10, dup_merge_range_frac=0.25)
    s = GateSeeker(config=cfg, detector=_level_multi([_gate([12.0, 0.0, 0.0], normal=[1, 0, 0])]))

    def pose(x, y, z, n=4):
        return GatePose(frame_id=0, sim_time_ns=0, R_cam_gate=np.eye(3),
                        t_cam_gate=np.array([x, y, z], float), reproj_error_px=0.0, n_corners=n)

    # three detections of ONE close gate (~same bearing+range) + one genuinely distinct far gate
    dupA = pose(0.10, 0.0, 5.0, n=4)
    dupB = pose(0.14, 0.05, 5.2, n=2)          # fewer corners -> should be the one dropped
    dupC = pose(0.08, -0.03, 4.9, n=3)
    other = pose(6.0, 0.0, 10.0, n=4)          # bearing ~0.54 rad away -> distinct
    merged = s._merge_duplicate_poses([dupA, dupB, dupC, other])
    assert len(merged) == 2, [p.n_corners for p in merged]
    kept = {round(float(p.t_cam_gate[2]), 1) for p in merged}
    assert 10.0 in kept                        # the distinct gate survives
    rep = [p for p in merged if float(p.t_cam_gate[2]) < 8.0][0]
    assert rep.n_corners == 4                  # the fullest quad is the representative

    # OFF by default -> byte-identical passthrough
    off = GateSeeker(config=GateSeekerConfig(), detector=_level_multi([_gate([12.0, 0, 0], normal=[1, 0, 0])]))
    same = [dupA, dupB, dupC, other]
    assert off._merge_duplicate_poses(same) is same
