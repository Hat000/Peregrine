"""Tests for the inc8 reference line Gamma (``rl/reference_line_inc8.json``), rebuilt on the
MEASURED corrected-aero plant by ``rl/build_reference_line.py`` (P2 INC8-RL, handoff
p2-inc8-rl-refline-2026-06-14).

The five contracts the rebuild must hold (replacing the drag-infeasible linear-plant
``reference_line_vq1.json``):
  1. ROUND-TRIP: loads through the real ``ReferenceLine`` loader; ``.progress()`` is STRICTLY
     monotonic along the emitted samples (arc-length increases, no fold-backs).
  2. CONTACT-SAFE: every gate crossing's in-plane miss vs the CANONICAL gate centre
     (fly_rl._GATE_POS_ZUP * _FLIP) is inside the contact-true pass band at r=0.38
     (< HALF_OPEN - 0.38 = 0.37 m); dead-centre target < 0.05 m.
  3. ORIENTATION: velocity faces DOWN-COURSE (positive projection on the local path tangent;
     X strictly decreasing; yaw in the -X hemisphere) -- the ~170 deg vq1 inversion is gone.
  4. DRAG-FEASIBLE on the MEASURED envelope: peak collective <= 1.0 (|f_thrust| <= the convex
     full-stick ceiling 78.28 m/s^2), per-axis body rate <= ~11 rad/s, speed < ~39 m/s wall.
  5. REPRODUCIBLE + CANONICAL: the committed JSON regenerates bit-for-bit from the generator,
     which sources gate centres from the canonical fly_rl constants (not the json).

NOTE: the honest UPRIGHT lap is ~8.5 s, NOT the brief's ~4.6-4.7 s -- that bound is the
full-attitude (TOGT/proto) regime, which requires sustained INVERTED thrust on the descent and
yaw rate ~13 > 11 rad/s (rate-infeasible). See the handoff REPORT.md for the full data.
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_ROOT / "rl") not in sys.path:
    sys.path.insert(0, str(_ROOT / "rl"))

from racer.reference_line import ReferenceLine  # noqa: E402

REF_PATH = _ROOT / "rl" / "reference_line_inc8.json"

# spec / validity constants (proto_envelope_topp + d5 contact band)
HALF_OPEN = 0.75
RADIUS_PRIMARY = 0.38
PASS_BAND = HALF_OPEN - RADIUS_PRIMARY            # 0.37 m contact-true band at r=0.38
DEAD_CENTRE = 0.05                                # dead-centre target
A_UP_MAX = 78.282838504684648                     # rl_plant COLL_MAP_ACCEL_MEASURED[-1] (full stick)
V_DRAG_WALL = 39.0                                # ~39 m/s v^2-drag wall
RATE_ENV = 11.0                                   # measured super-rate envelope (per axis)


def _canonical_gates_ned() -> np.ndarray:
    """Independently AST-source the 6 gate centres = _GATE_POS_ZUP * _FLIP from rl/fly_rl.py.
    Deliberately NOT importing fly_rl (it pulls torch) and NOT using the generator's helper --
    this is an INDEPENDENT check that the emitted line used the canonical centres."""
    tree = ast.parse((_ROOT / "rl" / "fly_rl.py").read_text())
    vals: dict[str, np.ndarray] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id in ("_GATE_POS_ZUP", "_FLIP"):
                    if isinstance(node.value, ast.Call) and node.value.args:
                        vals[tgt.id] = np.array(ast.literal_eval(node.value.args[0]), dtype=float)
    return vals["_GATE_POS_ZUP"] * vals["_FLIP"]


@pytest.fixture(scope="module")
def ref() -> ReferenceLine:
    return ReferenceLine.load(REF_PATH)


@pytest.fixture(scope="module")
def raw() -> dict:
    return json.loads(REF_PATH.read_text())


# --------------------------------------------------------------------------- 1. round-trip
def test_roundtrip_load_and_progress_strictly_monotonic(ref):
    assert ref.raw["schema"] == "peregrine.reference_line.v1"
    assert ref.lap_time_s > 0.0
    assert ref.total_duration_s >= ref.lap_time_s
    # cumulative arc length strictly increasing (no duplicate / fold-back samples)
    assert np.all(np.diff(ref.arc) > 0.0), "stored polyline arc length not strictly increasing"
    # .progress() of each emitted sample is strictly increasing (the RL reward primitive)
    prog = np.array([ref.progress(p) for p in ref.pos])
    d = np.diff(prog)
    assert np.all(d > 0.0), f".progress() not strictly monotonic; min step {d.min():.3e} m"
    # progress spans (approximately) the full arc length
    assert prog[-1] == pytest.approx(ref.arc[-1], abs=1e-6)


# --------------------------------------------------------------------------- 2. contact-safe
def test_gate_crossings_contact_safe_and_dead_centre(ref):
    gates = _canonical_gates_ned()
    assert len(ref.gate_crossings) == 6
    misses = []
    for c in ref.gate_crossings:
        gc = gates[int(c["gate_id"])]
        p = np.asarray(c["pos_ned"], dtype=float)
        # gates face yaw=pi (level) -> plane normal +-X -> in-plane (aperture) miss = Y-Z offset
        miss = float(np.hypot(p[1] - gc[1], p[2] - gc[2]))
        misses.append(miss)
        # the crossing is on the gate plane (X == gate_X) and dead-centre vs the canonical centre
        assert abs(p[0] - gc[0]) < 1e-3, f"gate {c['gate_id']} crossing not on the gate plane"
        assert miss < PASS_BAND, f"gate {c['gate_id']} miss {miss:.4f} >= contact band {PASS_BAND}"
        # the stored miss_m must be the honest in-plane miss vs the canonical centre
        assert c["miss_m"] == pytest.approx(miss, abs=1e-6)
    print("per-gate in-plane miss (m):", [round(m, 6) for m in misses])
    assert max(misses) < DEAD_CENTRE, f"max gate miss {max(misses):.4f} exceeds dead-centre {DEAD_CENTRE}"


# --------------------------------------------------------------------------- 3. orientation
def test_orientation_velocity_faces_down_course(ref):
    pos, vel = ref.pos, ref.vel
    seg = np.diff(pos, axis=0)
    seg /= np.maximum(np.linalg.norm(seg, axis=1, keepdims=True), 1e-9)
    vproj = np.einsum("ij,ij->i", vel[:-1], seg)
    assert np.all(vproj > 0.0), f"velocity not aligned with travel; min projection {vproj.min():.3f}"
    # course runs -X: the North/X velocity component is negative everywhere (no 170/180 flip)
    assert np.all(vel[:, 0] < 0.0), "velocity_ned X must be negative (down-course) at every sample"
    # yaw faces travel: heading in the -X hemisphere AND equal to atan2(vE, vN)
    yaw = np.asarray(ref.raw["yaw"], dtype=float)
    assert np.all(np.cos(yaw) < 0.0), "yaw not in the down-course (-X) hemisphere -- inversion?"
    dyaw = np.angle(np.exp(1j * (np.arctan2(vel[:, 1], vel[:, 0]) - yaw)))
    assert np.max(np.abs(dyaw)) < np.radians(1.0), "yaw inconsistent with velocity heading"
    # gate-approach heading faces travel
    for c in ref.gate_crossings:
        assert ref.sample(c["t"]).velocity_ned[0] < 0.0


# --------------------------------------------------------------------------- 4. drag-feasible
def test_drag_feasible_collective_rate_speed(ref, raw):
    speed = np.linalg.norm(ref.vel, axis=1)
    omega = np.asarray(raw["omega_frd"], dtype=float)
    peak_speed = float(speed.max())
    peak_axis = np.abs(omega).max(axis=0)
    peak_coll = float(ref.thrust_norm.max())
    peak_fmag = float(raw["source"]["feasibility"]["peak_thrust_a_up_mps2"])
    print(f"peak speed {peak_speed:.2f} m/s | per-axis rate r/p/y "
          f"{peak_axis.round(2)} rad/s | peak collective {peak_coll:.3f} | peak |f| {peak_fmag:.2f}")
    # speed below the v^2 drag wall
    assert peak_speed < V_DRAG_WALL, f"peak speed {peak_speed:.2f} >= drag wall {V_DRAG_WALL}"
    # collective never saturates past full stick; required specific force within the convex ceiling
    assert peak_coll <= 1.0 + 1e-6, f"collective {peak_coll:.4f} exceeds full stick"
    assert peak_fmag <= A_UP_MAX + 1e-3, f"required |f| {peak_fmag:.3f} exceeds ceiling {A_UP_MAX:.3f}"
    # per-axis body rate within the measured super-rate envelope
    assert np.all(peak_axis <= RATE_ENV + 1e-6), f"per-axis rate {peak_axis} exceeds {RATE_ENV}"
    # honest upright line: no inverted thrust
    assert raw["source"]["feasibility"]["frac_inverted_tilt_gt90"] == 0.0


# --------------------------------------------------------------------------- 5. reproducible + canonical
def test_generator_reproduces_committed_file_from_canonical_gates():
    import build_reference_line as B

    # generator sources the canonical centres (independent AST check matches the generator)
    assert np.allclose(B.load_canonical_gates_ned(), _canonical_gates_ned(), atol=1e-9)

    payload, *_ = B.generate()                       # default = cone, tilt 75 (the emitted config)
    committed = json.loads(REF_PATH.read_text())
    assert payload["schema"] == committed["schema"]
    assert payload["lap_time_s"] == pytest.approx(committed["lap_time_s"], abs=1e-6)
    assert len(payload["pos_ned"]) == len(committed["pos_ned"])
    assert np.allclose(payload["pos_ned"], committed["pos_ned"], atol=1e-9), \
        "committed reference_line_inc8.json is stale -- re-run rl/build_reference_line.py --emit"
    # provenance records the MEASURED plant, not the linear-drag fiction
    plant = committed["source"]["plant"]
    assert plant["thrust_to_weight"] > 7.0          # ~7.98, NOT the linear plant's 3.765
    assert "supersedes" in committed["source"]


def test_supersedes_vq1_and_is_not_the_linear_plant_fiction(raw):
    # the rebuilt line must NOT re-import the vq1 4.55 s linear-plant infeasibility
    assert raw["lap_time_s"] > 4.7, "lap claims <=4.7 s -- re-importing the linear-plant fiction"
    assert raw["source"]["plant"]["quad_drag_c2_pooled"] == pytest.approx(0.052)
