"""Generate TOGT-Planner case directories for the VQ1 time-optimal bound.

Each case = a self-contained TOGT parameter dir (peregrine_setups.yaml + quad + init/refine
planning + lbfgs + track.yaml + meta.json) consumed by the patched `planners` CLI
(scripts/togt/traj_planner_peregrine.cpp) and then by the multiple-shooting refine
(scripts/togt/refine/refine_peregrine.py). See scripts/togt/README.md for the pipeline.

PLANT -> TOGT MAPPING (measured plant: handoff/shadowpc-characterize-sweep-2026-06-10 §1/§3,
twin defaults src/racer/twin.py; all per-unit-mass, mass=1 kg):
  - Collective ceiling: live normalized thrust 1.0 -> a_up = g/hover = 9.80665/0.2656
    = 36.92 m/s^2 = 3.765 g  =>  per-rotor thrust_max = 3.765*9.8066/4 = 9.2305 N.
    (TOGT's G is 9.8066; the 5e-5 g difference is far below measurement error.)
    Per-rotor floor 0.1 N (0.04 g total): the true floor is ~0, but minThr 0.05 makes the
    TOGT L-BFGS fail outright (flatness singularity near zero thrust); 0.1 N solves and
    biases the bound conservative by an immeasurable amount.
  - Body-rate authority: cmd clamp +-3.14 rad/s through the measured static super-rate map
    g(|c|)=G0/(1-s*|c|/pi), G0=[2.501,2.504,2.231], s~=0.30 -> SUSTAINED reachable rate
    ~= 11.0-11.2 rad/s roll/pitch (measured 11.05-11.12 holds) => omega_xy 11.0.
    Yaw: level-attitude plateau ~2.35*3.14 = 7.4 rad/s (maneuver-dependent; racing yaw
    commands are small) => omega_z 7.0, slightly conservative.
  - Linear drag 0.21/s (twin linear_drag), world-frame isotropic. The C++ TOGT phase has
    no drag model (init/warm-start only); the refine includes it exactly (PEREGRINE patch).
  - Inertia [0.001,0.001,0.0017] + arm 0.15/beta 45/torCoeff 0.05 (cpc values): TINY on
    purpose -- the rotor-differential torque envelope then never binds, reducing the
    per-rotor model to collective-only + body-rate bounds = our CTBR plant class. The
    inner-loop lag (tau~=0.02 s) and slew (~260 rad/s^2 r/p) are NOT modeled here; the twin
    replay (scripts/twin_track_reference.py) is the feasibility check for those.
  - NOT modeled anywhere: aero at racing airspeed beyond linear drag (sweep caveat).

COURSE (rl/peregrine_course_diffaero.json, diffaero z-up frame = NED*[1,-1,-1]):
  - 6 gates, inner opening 1.5 m; validity = in-plane miss < 0.75 m half-opening.
    Gate normal along -X (map yaw pi) -> RectanglePrisma rpy [0,-90,0]; marginW/H shrink
    the traversal region: half-opening = (1.5 - margin)/2.
  - Start: live spawn, z-up [0,0,-0.02], at rest (the race clock starts with the drone
    parked on the pad; the ~17.8 deg pad tilt and yaw-pi facing are irrelevant to the
    polynomial boundary condition, which pins PVAJ only -- noted in the WRITEUP).
  - Finish: time is measured at the GATE-5 PLANE CROSSING (analyze.py); the trajectory
    itself continues to a virtual endpoint 25 m past gate 5 (vel 0) so the forced
    terminal state cannot slow the finish crossing (braking from ~30 m/s needs ~13 m).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _write_lf(path: Path, text: str) -> None:
    """Write with LF line endings regardless of platform: TOGT's hand-rolled C++ YAML
    parser chokes on CRLF (values parse as '0.05\\r' -> garbage params -> segfault;
    found the hard way, 2026-06-11)."""
    with open(path, "w", newline="\n") as f:
        f.write(text)

G = 9.8066                      # TOGT's gravity convention
TW_NOMINAL = 3.765              # live collective ceiling, g-equivalent
OMEGA_NOMINAL = (11.0, 11.0, 7.0)
DRAG = 0.21                     # 1/s, twin linear_drag
RUNOUT_M = 25.0                 # virtual endpoint distance beyond gate 5 along its normal

CASES = [
    # name,              TW,          omega_xyz,            margin, tol,  drag, shape
    # -- rectangle gates: the full 1.5x1.5 square is the region => the optimum clips gate
    #    CORNERS (in-plane |h|,|v| <= 0.75 each, Euclidean miss up to 1.06). Valid only if
    #    the race rule is the square opening (unverified live -- VQ1 passes were all inside
    #    the inscribed circle).
    ("bound_nominal",    TW_NOMINAL,  OMEGA_NOMINAL,        0.0,    0.01, DRAG, "rect"),
    ("bound_free",       TW_NOMINAL,  OMEGA_NOMINAL,        0.6,    0.30, DRAG, "rect"),
    ("ref_margin",       TW_NOMINAL,  OMEGA_NOMINAL,        0.7,    0.10, DRAG, "rect"),
    ("sens_omega785",    TW_NOMINAL,  (7.85, 7.85, 7.0),    0.0,    0.01, DRAG, "rect"),
    ("sens_thr50",       TW_NOMINAL * 0.50, OMEGA_NOMINAL,  0.0,    0.01, DRAG, "rect"),
    ("sens_thr75",       TW_NOMINAL * 0.75, OMEGA_NOMINAL,  0.0,    0.01, DRAG, "rect"),
    ("sens_omega_unbounded", TW_NOMINAL, (25.0, 25.0, 25.0), 0.0,   0.01, DRAG, "rect"),
    ("bound_nodrag",     TW_NOMINAL,  OMEGA_NOMINAL,        0.0,    0.01, 0.0,  "rect"),
    # -- ball gates: crossing constrained to the INSCRIBED circle (Euclidean miss < 0.75 =
    #    the validity rule exactly as we measure it) => the defensible bound + the
    #    committed reference line.
    ("bound_circle",     TW_NOMINAL,  OMEGA_NOMINAL,        0.0,    0.01, DRAG, "ball"),
    ("ref_circle",       TW_NOMINAL,  OMEGA_NOMINAL,        0.7,    0.10, DRAG, "ball"),
]

# EXPLORATORY (twin-falsify 2026-06-11, banked mid-session): linear drag + the 3.765 g
# linear collective ceiling were FALSIFIED -- real aero is quadratic body-frame drag
# c2~0.052/m (measured to 7.6 m/s, EXTRAPOLATED above) and the convex thrust curve
# reaches ~8 g at full stick. This case brackets where the corrected bound likely lands
# (isotropic c2, box thrust to 8 g); the authoritative re-run is queued post-S16.
CASES.append(
    ("expl_corrected_aero", 8.0, OMEGA_NOMINAL, 0.0, 0.01, 0.0, "ball", 0.052))


def quad_yaml(tw: float, omega: tuple, drag: float, quad_drag: float = 0.0) -> str:
    thr_max = tw * G / 4.0
    return f"""# Peregrine plant mapped to the TOGT quad model -- see gen_cases.py docstring.
mass:               1.00
gravity:            {G}
inertia:            [0.001, 0.001, 0.0017]
armLength:          0.15
beta:               45.0
torCoeff:           0.05

# refine (13-state multiple shooting): per-rotor thrust + body-rate state bounds
thrust_min: 0.0
thrust_max: {thr_max:.4f}   # = {tw:.4f} g total on 1 kg
omega_max:  [{omega[0]}, {omega[1]}, {omega[2]}]         # [rad/s]
linear_drag: {drag}         # 1/s, PEREGRINE refine-only (isotropic, world frame)
quad_drag: {quad_drag}      # 1/m, PEREGRINE refine-only (isotropic |v|*v)
"""


def planning_yaml(pieces: int, omega: tuple, thr_max: float) -> str:
    return f"""piecesPerSegment:   {pieces}
speedGuess:         1.0
maxVelNorm:         60.0
maxOmgXY:           {omega[0]}
maxOmgZ:            {omega[2]}
maxTiltedAngle:     6.28
maxThr:             {thr_max:.4f}
minThr:             0.1

weightTime:         1.0
weightEnergy:       0.0
weightPos:          1.0
weightVel:          1.0
weightOmg:          1.0
weightRot:          1.0
weightThr:          1.0
smoothingEps:       1.0e-2
numConstPena:       16

dynamicConstCheck:  true
minNumCheck:        8
maxNumCheck:        32
checkTimeSec:       0.05

boundX:             [-195, 10]
boundY:             [-20, 25]
boundZ:             [-32, 8]
"""


LBFGS_YAML = """memorySize:         256
past:               3
minStep:            1.0e-32
maxLinesearch:      64
maxIterations:      0
relCostTolerance:   1.0e-5
relGradTolerance:   0.0
"""

SETUPS_YAML = """planningInit: "init/peregrine_planning.yaml"
lbfgsInit:     "init/peregrine_lbfgs.yaml"

planningRefine: "refine/peregrine_planning.yaml"
lbfgsRefine:     "refine/peregrine_lbfgs.yaml"

quadrotor: "peregrine_quad.yaml"
"""


def state_yaml(key: str, pos, vel) -> str:
    return f"""{key}:
  pos: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]
  vel: [{vel[0]:.1f}, {vel[1]:.1f}, {vel[2]:.1f}]
  acc: [0.0, 0.0, 0.0]
  jer: [0.0, 0.0, 0.0]
  rot: [1.0, 0.0, 0.0, 0.0]
  cthrustmass: {G}
  euler: [0.0, 0.0, 0.0]
"""


def track_yaml(course: dict, margin: float, shape: str = "rect") -> str:
    gates = course["gates"]
    out = [state_yaml("initState", (0.0, 0.0, -0.02), (0, 0, 0))]
    # Virtual endpoint past the last gate along its through-direction (-X for yaw pi).
    g5 = gates[-1]["pos_zup"]
    yaw = gates[-1]["yaw"]
    n = (math.cos(yaw), math.sin(yaw), 0.0)               # gate normal in z-up
    end = (g5[0] + RUNOUT_M * n[0], g5[1] + RUNOUT_M * n[1], g5[2])
    out.append(state_yaml("endState", end, (0, 0, 0)))
    names = [f"Gate{i + 1}" for i in range(len(gates))]
    out.append("orders: [" + ", ".join(f"'{n_}'" for n_ in names) + "]\n")
    for name, g in zip(names, gates):
        p = g["pos_zup"]
        if shape == "ball":
            # 3D ball region, effective radius = radius - margin/2 (ball.cpp) = the
            # INSCRIBED circle of the opening when radius = inner/2: Euclidean miss bound.
            out.append(f"""{name}:
  type: 'SingleBall'
  name: 'vq1_gate'
  position: [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]
  radius: {course["inner_opening_m"] / 2}
  margin: {margin}
  stationary: true
""")
            continue
        # rpy [0,-90,yaw_deg]: prisma axis (local Z, the through axis) -> horizontal.
        # Map yaw pi == normal [-1,0,0] == rpy yaw 0 (see WRITEUP frame notes).
        yaw_deg = math.degrees(g["yaw"]) - 180.0
        out.append(f"""{name}:
  type: 'RectanglePrisma'
  name: 'vq1_gate'
  position: [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]
  rpy: [0.0, -90.0, {yaw_deg:.4f}]
  width: {course["inner_opening_m"]}
  height: {course["inner_opening_m"]}
  marginW: {margin}
  marginH: {margin}
  length: 0.0
  midpoints: 0
  stationary: true
""")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--course", default="rl/peregrine_course_diffaero.json")
    ap.add_argument("--out", default="handoff/laptop-togt-bound-2026-06-10/cases")
    ap.add_argument("--only", nargs="*", default=None,
                    help="generate only these case names (default: all)")
    args = ap.parse_args()

    course = json.loads(Path(args.course).read_text())
    out_root = Path(args.out)
    for name, tw, omega, margin, tol, drag, shape, *rest in CASES:
        if args.only is not None and name not in args.only:
            continue
        quad_drag = rest[0] if rest else 0.0
        d = out_root / name
        (d / "init").mkdir(parents=True, exist_ok=True)
        (d / "refine").mkdir(parents=True, exist_ok=True)
        thr_max = tw * G / 4.0
        _write_lf(d / "peregrine_setups.yaml", SETUPS_YAML)
        _write_lf(d / "peregrine_quad.yaml", quad_yaml(tw, omega, drag, quad_drag))
        _write_lf(d / "init" / "peregrine_planning.yaml", planning_yaml(1, omega, thr_max))
        _write_lf(d / "init" / "peregrine_lbfgs.yaml", LBFGS_YAML)
        _write_lf(d / "refine" / "peregrine_planning.yaml", planning_yaml(5, omega, thr_max))
        _write_lf(d / "refine" / "peregrine_lbfgs.yaml", LBFGS_YAML)
        _write_lf(d / "track.yaml", track_yaml(course, margin, shape))
        (d / "meta.json").write_text(json.dumps({
            "case": name, "thrust_to_weight": tw, "omega_max": list(omega),
            "gate_margin_m": margin, "refine_tol_m": tol, "linear_drag": drag,
            "quad_drag": quad_drag, "gate_shape": shape,
            "half_opening_effective_m": (course["inner_opening_m"] - margin) / 2,
            "max_center_miss_m": (course["inner_opening_m"] - margin) / 2 + tol,
            "refine_pieces_per_segment": 5,
        }, indent=1))
        print(f"{name}: T/W {tw:.3f}, omega {omega}, margin {margin}, tol {tol}, "
              f"drag {drag}, quad_drag {quad_drag}, shape {shape}")
    print(f"-> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
