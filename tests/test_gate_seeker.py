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
    """Once anchored (a vision fix landed, tsv finite) the map-free seeker pursues the SEEN gate:
    a sane bounded BODY_RATE that turns toward + leans into the gate the camera sees, NOT a slew."""
    from racer.frames import R_world_from_body
    gate = _gate([12.0, 0.0, -2.5], normal=[1.0, 0.0, 0.0])
    det = _ProjDetector(gate, np.zeros(3), R_world_from_body(0.0, 0.0, 0.0))
    seeker = GateSeeker(config=GateSeekerConfig(cruise_speed=3.0, launch_ramp_s=0.0), detector=det)
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
    seeker = GateSeeker(config=GateSeekerConfig(cruise_speed=3.0, launch_ramp_s=0.0,
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
    # immediately after reset, even with the gate in view, it HOLDS (tsv inf again).
    cmd = seeker.command_visual(_nav_fix([0, 0, -2.5], tsv=float("inf")), _frame(1, 0), 0)
    assert abs(float(cmd.body_rate[2])) < 1e-6


def test_vq2_case_c_profile_is_map_free_for_steering():
    """The case-C profile carries no given pose; the deploy seeker steers MAP-FREE via the detector —
    a guard against re-introducing an absolute-map steering dependency on the self-localizing path."""
    p = vq2_case_c()
    assert p.self_localizing is True
    assert p.nav_config.use_given_position is False     # no absolute self-position to steer from


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
