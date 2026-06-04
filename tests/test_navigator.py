import json

import numpy as np

from racer.contracts import (
    ControlMode,
    DroneState,
    Frame,
    Gate,
    GateObservation,
)
from racer.controller import Controller
from racer.frames import R_camera_from_body, R_world_from_body
from racer.mission import Mission, MissionConfig, MissionState
from racer.navigator import (
    Navigator,
    NavigatorConfig,
    gates_from_track_records,
    load_track_map,
)
from racer.planner import ReactivePlanner
from racer.vision.gate_pose import project_gate_corners

_HOVER_ACCEL = np.array([0.0, 0.0, -9.80665])   # FRD specific force at level hover
_LEVEL_Q = np.array([1.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _records(positions, gate_id0=0):
    return [
        {"gate_id": gate_id0 + i, "position_ned": list(p),
         "orientation_ned_wxyz": [0.7071, 0.0, 0.0, 0.7071], "width_m": 2.72, "height_m": 2.72}
        for i, p in enumerate(positions)
    ]


def _ds(sim_time_ns, position=None, velocity=None, *, reset_counter=0,
        roll=0.0, pitch=0.0, yaw=0.0):
    return DroneState(
        sim_time_ns=int(sim_time_ns),
        orientation_ned_wxyz=_LEVEL_Q.copy(),
        roll=roll, pitch=pitch, yaw=yaw,
        accel_body=_HOVER_ACCEL.copy(),
        position_ned=None if position is None else np.asarray(position, float),
        velocity_ned=None if velocity is None else np.asarray(velocity, float),
        reset_counter=reset_counter,
    )


def _gate_facing_north(position, gate_id=0, inner=1.5) -> Gate:
    """Upright gate whose through-direction is +X (north): X=East, Y=Down, Z=North."""
    R = np.column_stack([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    return Gate(gate_id=gate_id, position_ned=np.asarray(position, float), R_world_gate=R, inner_size_m=inner)


def _project_gate(gate: Gate, drone_pos, R_wb, inner=1.5) -> np.ndarray:
    """The 4 inner corners the camera would see for this gate at this drone pose (ground truth)."""
    R_camera_world = (R_wb @ R_camera_from_body().T).T
    t_cam_gate = R_camera_world @ (gate.position_ned - np.asarray(drone_pos, float))
    R_cam_gate = R_camera_world @ gate.R_world_gate
    return project_gate_corners(R_cam_gate, t_cam_gate, inner_size_m=inner)


class _FakeDetector:
    """Returns a fixed set of (gate, true-drone-pos) projections as GateObservations."""

    def __init__(self, gate: Gate, drone_pos, inner=1.5):
        self.gate, self.drone_pos, self.inner = gate, np.asarray(drone_pos, float), inner
        self.calls = 0

    def detect(self, frame: Frame):
        self.calls += 1
        corners = _project_gate(self.gate, self.drone_pos, np.eye(3), self.inner)
        return [GateObservation(
            frame_id=frame.frame_id, sim_time_ns=frame.sim_time_ns,
            corners_px=corners, corner_confidence=np.ones(4), score=0.9,
        )]


def _frame(frame_id, sim_time_ns):
    return Frame(frame_id=frame_id, sim_time_ns=int(sim_time_ns),
                 image_bgr=np.zeros((360, 640, 3), np.uint8))


# ---------------------------------------------------------------------------
# map loading
# ---------------------------------------------------------------------------
def test_loads_saved_track_map_as_six_gates_with_inner_size():
    gates = load_track_map("handoff/shadowpc-firstcontact-2026-06-02/track_map.json")
    assert len(gates) == 6
    assert all(g.inner_size_m == 1.5 for g in gates)          # inner (PnP), NOT the 2.72 outer
    np.testing.assert_allclose(gates[0].position_ned, [-23.2979679, -0.39990234, -0.031958], atol=1e-4)


def test_through_directions_point_down_course():
    # The course runs in -X; each gate's normal must point toward the NEXT gate so the planner
    # places its carrot on the exit side (never an approach-side U-turn).
    gates = load_track_map("handoff/shadowpc-firstcontact-2026-06-02/track_map.json")
    for i in range(len(gates) - 1):
        to_next = gates[i + 1].position_ned - gates[i].position_ned
        assert float(gates[i].normal_ned @ to_next) > 0.0
    # R_world_gate is a proper rotation (orthonormal, det +1).
    for g in gates:
        np.testing.assert_allclose(g.R_world_gate.T @ g.R_world_gate, np.eye(3), atol=1e-9)
        assert float(np.linalg.det(g.R_world_gate)) > 0.99


def test_gates_from_records_geometry_matches_input():
    gates = gates_from_track_records(_records([[5, 0, -1.5], [10, 0, -1.5], [15, 0, -1.5]]))
    assert [g.gate_id for g in gates] == [0, 1, 2]
    for g in gates:                                            # all through-directions +X
        np.testing.assert_allclose(g.normal_ned, [1.0, 0.0, 0.0], atol=1e-9)


def test_corner_to_center_lifts_vertically_only():
    # The map position is the gate's BOTTOM-CENTRE: centred in width (keep y), at the base in height.
    # corner_to_center lifts it by half the gate height along the height axis (col2 of the true
    # quaternion) and makes NO lateral shift. _records quat is Rz(90): col2=+z, so a 2.72 m gate
    # lifts z by -1.36 and leaves y untouched.
    recs = _records([[-5, 0, 0], [-10, 0, 0], [-15, 0, 0]])   # course runs -X like the real track
    corner = gates_from_track_records(recs, corner_to_center=False)
    center = gates_from_track_records(recs, corner_to_center=True)
    np.testing.assert_allclose(corner[0].position_ned, [-5, 0, 0], atol=1e-9)
    np.testing.assert_allclose(center[0].position_ned, [-5.0, 0.0, -1.36], atol=1e-6)  # y unchanged, z up
    for c, k in zip(corner, center):                          # frame + through-dir unchanged
        np.testing.assert_allclose(c.R_world_gate, k.R_world_gate, atol=1e-9)


# ---------------------------------------------------------------------------
# given-state estimation (no detector)
# ---------------------------------------------------------------------------
def test_tracks_given_position_and_velocity():
    nav = Navigator(gates=[], detector=None)
    nav.update(_ds(0, position=[1.0, 2.0, -3.0], velocity=[0.5, 0.0, 0.0]))   # init
    ns = None
    for k in range(1, 20):
        ns = nav.update(_ds(k * 10_000_000, position=[1.0 + 0.5 * k * 0.01, 2.0, -3.0],
                            velocity=[0.5, 0.0, 0.0]))
    assert ns is not None
    np.testing.assert_allclose(ns.position_ned, [1.0 + 0.5 * 19 * 0.01, 2.0, -3.0], atol=0.1)
    np.testing.assert_allclose(ns.velocity_ned, [0.5, 0.0, 0.0], atol=0.1)
    assert ns.time_since_vision_update_s == float("inf")      # no vision ran


def test_zero_dt_tick_returns_cached_state_without_crash():
    nav = Navigator(gates=[], detector=None)
    nav.update(_ds(1000, position=[0.0, 0.0, 0.0]))
    a = nav.update(_ds(2000, position=[1.0, 0.0, 0.0]))
    b = nav.update(_ds(2000, position=[9.0, 9.0, 9.0]))       # same sim_time -> no new IMU
    np.testing.assert_allclose(a.position_ned, b.position_ned)  # cached, given pos NOT re-applied


def test_reset_counter_change_reinitializes():
    nav = Navigator(gates=[], detector=None)
    nav.update(_ds(0, position=[0.0, 0.0, 0.0]))
    nav.update(_ds(10_000_000, position=[0.1, 0.0, 0.0]))
    ns = nav.update(_ds(20_000_000, position=[50.0, -10.0, -5.0], reset_counter=1))
    np.testing.assert_allclose(ns.position_ned, [50.0, -10.0, -5.0], atol=1e-6)  # snapped, not integrated


def test_uninitialized_navstate_is_finite():
    nav = Navigator(gates=[], detector=None)
    ns = nav.update(_ds(0))                                    # no position ever -> seeds at origin
    assert np.all(np.isfinite(ns.position_ned))


# ---------------------------------------------------------------------------
# vision -> KF (synthetic projector + fake detector; no model needed)
# ---------------------------------------------------------------------------
def test_vision_only_converges_to_true_position():
    # Given position OFF: the KF starts at the origin and must be pulled to the true drone
    # position purely by gate fixes -- exercises detector -> PnP -> assoc -> localization -> KF
    # and catches any frame/sign error end-to-end.
    true_pos = np.array([1.0, -0.5, -2.0])
    gate = _gate_facing_north([9.0, 0.0, -2.5], gate_id=0)
    nav = Navigator(gates=[gate], detector=_FakeDetector(gate, true_pos),
                    config=NavigatorConfig(use_given_position=False, use_given_velocity=False))
    nav.update(_ds(0), _frame(0, 0))                          # init at origin (no given pos)
    ns = None
    for k in range(1, 60):
        ns = nav.update(_ds(k * 10_000_000), _frame(k, k * 10_000_000))
    assert nav.n_vision_fixes > 0
    np.testing.assert_allclose(ns.position_ned, true_pos, atol=0.25)
    assert ns.time_since_vision_update_s < 0.05               # fresh vision


def test_detector_runs_once_per_frame_id():
    gate = _gate_facing_north([9.0, 0.0, -2.5])
    det = _FakeDetector(gate, [1.0, -0.5, -2.0])
    nav = Navigator(gates=[gate], detector=det,
                    config=NavigatorConfig(use_given_position=False))
    nav.update(_ds(0), _frame(0, 0))                          # init tick: no vision yet
    nav.update(_ds(10_000_000), _frame(1, 10_000_000))        # detect -> call 1
    nav.update(_ds(20_000_000), _frame(1, 20_000_000))        # SAME frame_id -> not re-detected
    nav.update(_ds(30_000_000), _frame(2, 30_000_000))        # new frame_id -> call 2
    assert det.calls == 2


def test_innovation_gate_rejects_inconsistent_fix():
    # Given position ON + tight at the origin; a detection that PnP-resolves to a gate ~9 m
    # ahead but is associated to a map gate placed far away implies a drone position far from
    # the origin -> the Mahalanobis gate rejects it; the estimate stays anchored to truth.
    far_gate = _gate_facing_north([200.0, 0.0, -2.5], gate_id=0)   # map says gate is 200 m north
    seen_gate = _gate_facing_north([9.0, 0.0, -2.5], gate_id=0)    # but we actually see one 9 m ahead
    nav = Navigator(gates=[far_gate], detector=_FakeDetector(seen_gate, [0.0, 0.0, 0.0]),
                    config=NavigatorConfig(use_given_position=True, given_pos_std=0.05))
    nav.update(_ds(0, position=[0.0, 0.0, 0.0]), _frame(0, 0))
    ns = None
    for k in range(1, 20):
        ns = nav.update(_ds(k * 10_000_000, position=[0.0, 0.0, 0.0]), _frame(k, k * 10_000_000))
    assert nav.n_vision_rejected > 0
    np.testing.assert_allclose(ns.position_ned, [0.0, 0.0, 0.0], atol=0.2)   # anchored, not yanked


def test_no_detections_no_vision_update():
    class _Empty:
        def detect(self, frame):
            return []

    nav = Navigator(gates=[_gate_facing_north([9, 0, -2.5])], detector=_Empty())
    nav.update(_ds(0, position=[0, 0, 0]), _frame(0, 0))
    ns = nav.update(_ds(10_000_000, position=[0, 0, 0]), _frame(1, 10_000_000))
    assert ns.time_since_vision_update_s == float("inf")
    assert nav.n_vision_fixes == 0


# ---------------------------------------------------------------------------
# wiring: Navigator -> Mission.run drives a course to FINISHED
# ---------------------------------------------------------------------------
def test_navigator_drives_mission_to_finished():
    gates = gates_from_track_records(_records([[5, 0, -1.5], [10, 0, -1.5], [15, 0, -1.5]]))
    nav = Navigator(gates=gates, detector=None)
    m = Mission(gates=gates, planner=ReactivePlanner(cruise_speed=4.0),
                controller=Controller(mode=ControlMode.POSITION),
                config=MissionConfig(takeoff_altitude_m=1.5, takeoff_tol_m=0.3, gate_pass_radius_m=1.0))

    plant = {"pos": np.zeros(3), "vel": np.zeros(3), "k": 0}
    dt, max_speed = 0.02, 4.0

    def navigator():
        plant["k"] += 1
        return nav.update(_ds(plant["k"] * 20_000_000, position=plant["pos"].copy(),
                              velocity=plant["vel"].copy()))

    class _Transport:
        def send_command(self, cmd):
            target = cmd.position_ned if cmd.position_ned is not None else plant["pos"]
            to = np.asarray(target, float) - plant["pos"]
            d = float(np.linalg.norm(to))
            step = (to / d) * min(d, max_speed * dt) if d > 1e-9 else np.zeros(3)
            plant["pos"] = plant["pos"] + step
            plant["vel"] = step / dt

    final = m.run(navigator, _Transport(), max_steps=8000)
    assert final is MissionState.FINISHED
    assert m.gate_index == 3
    np.testing.assert_allclose(plant["pos"], [15.0, 0.0, -1.5], atol=1.0)


def test_saved_map_json_is_well_formed():
    data = json.loads(open("handoff/shadowpc-firstcontact-2026-06-02/track_map.json").read())
    assert data["num_gates"] == len(data["gates"]) == 6
