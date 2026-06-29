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
    assert set(PROFILES) == {"vq1_case_a", "vq2_case_c"}
    assert get_profile("vq2_case_c").name == "vq2_case_c"
    with pytest.raises(KeyError):
        get_profile("nope")


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
    adds, no wind-up) and NO velocity_ned -- the structural fix. With feedforward ON the seeker's
    setpoint to the controller carries accel and not velocity; with it OFF (legacy) it carries velocity."""
    from racer.frames import R_world_from_body
    gate = _gate([12.0, 0.0, -2.5], normal=[1.0, 0.0, 0.0])
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
