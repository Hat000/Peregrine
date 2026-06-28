"""Self-contained CasADi minimum-time trajectory optimizer for the Peregrine gate track.

Computes the TIME-OPTIMAL racing line + the lap-time LOWER BOUND for our measured plant,
with NO external warm-start (no C++ TOGT phase): a straight-line / constant-speed initial
guess is enough for IPOPT to converge. This is the tractable direct-multiple-shooting
minimum-time formulation the mission specifies as the CPC fallback; the gap to full CPC is
documented in scripts/planning/README.md and in MEMORY-DELTA.

FORMULATION (direct multiple shooting, free per-segment node time):
    * State x = [p(3), v(3), q(4)]  (point-mass-with-attitude, the CTBR plant class).
    * Control u = [a_c, wx, wy, wz]: body-up specific accel a_c in [0, A_MAX] (the measured
      convex-map ceiling, ~8 g) and body rates bounded by OMEGA_MAX.
    * Dynamics (continuous), NED world frame (z down, gravity +z):
          p_dot = v
          v_dot = R(q) @ [0,0,-a_c] + [0,0,+g] + a_drag_world(q, v)
          q_dot = 0.5 * q (x) [0, w]
      a_drag = measured quadratic body-frame drag -c2|v_b|v_b (rotated to world), optional.
    * Each gate-to-gate segment has its own node count Ns[i] and a single free duration
      DT[i] (so dt is uniform within a segment but free across segments -- the standard
      TOGT/CPC time discretization). Objective = sum_i Ns[i]*DT[i] = total time.
    * Gate constraint: the LAST node of segment i lands within GATE_BALL_RADIUS (Euclidean)
      of gate i  ->  inscribed-circle crossing = our exact validity rule. Terminal node hits
      a virtual endpoint past the last gate (so the forced stop can't slow the finish).
    * RK4 collocation per node; quaternion renormalised each step.

Run (CasADi venv):
    PYTHONPATH=src .venv/Scripts/python.exe scripts/planning/trajopt.py \
        --course rl/peregrine_course_diffaero.json --out OUTDIR [--no-drag] [--nodes 12]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import casadi as ca

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plant_params as PP  # noqa: E402


# ----------------------------------------------------------------------------- quaternion algebra
def quat_mult(q1, q2):
    """Hamilton product (wxyz), CasADi symbolic."""
    return ca.vertcat(
        q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
        q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
        q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
        q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0],
    )


def rotate_vec_by_quat(q, v):
    """Rotate body vector v into world by q (wxyz) = R(q) @ v."""
    qc = ca.vertcat(q[0], -q[1], -q[2], -q[3])
    r = quat_mult(quat_mult(q, ca.vertcat(0, v)), qc)
    return ca.vertcat(r[1], r[2], r[3])


def rotate_vec_by_quat_inv(q, v):
    """Rotate world vector v into body by q (wxyz) = R(q)^T @ v."""
    qc = ca.vertcat(q[0], -q[1], -q[2], -q[3])
    r = quat_mult(quat_mult(qc, ca.vertcat(0, v)), q)
    return ca.vertcat(r[1], r[2], r[3])


# ----------------------------------------------------------------------------- dynamics
def make_dynamics(use_drag: bool):
    """Continuous dynamics f(x, u) -> x_dot in NED (z down, gravity +z, thrust -z body)."""
    p = ca.MX.sym("p", 3)
    v = ca.MX.sym("v", 3)
    q = ca.MX.sym("q", 4)
    a_c = ca.MX.sym("a_c")              # body-up specific accel magnitude (>=0)
    w = ca.MX.sym("w", 3)              # body rates
    x = ca.vertcat(p, v, q)
    u = ca.vertcat(a_c, w)

    g_vec = ca.DM([0, 0, PP.G])                          # NED: gravity is +z
    thrust_world = rotate_vec_by_quat(q, ca.vertcat(0, 0, -a_c))  # body -z (up) -> world
    v_dot = thrust_world + g_vec
    if use_drag:
        v_body = rotate_vec_by_quat_inv(q, v)
        spd_b = ca.sqrt(v_body.T @ v_body + 1e-9)
        a_drag_body = -PP.QUAD_DRAG_C2 * spd_b * v_body
        v_dot = v_dot + rotate_vec_by_quat(q, a_drag_body)
    x_dot = ca.vertcat(v, v_dot, 0.5 * quat_mult(q, ca.vertcat(0, w)))
    return ca.Function("f", [x, u], [x_dot], ["x", "u"], ["x_dot"])


def make_rk4(f, M: int = 1):
    """One discrete step x_{k+1} = RK4(f, x, u, dt) with quaternion renorm."""
    x0 = ca.MX.sym("x0", 10)
    u = ca.MX.sym("u", 4)
    dt = ca.MX.sym("dt")
    h = dt / M
    x = x0
    for _ in range(M):
        k1 = f(x, u)
        k2 = f(x + 0.5 * h * k1, u)
        k3 = f(x + 0.5 * h * k2, u)
        k4 = f(x + h * k3, u)
        x = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    qn = ca.sqrt(x[6:10].T @ x[6:10])
    x = ca.vertcat(x[0:6], x[6:10] / qn)
    return ca.Function("F", [x0, u, dt], [x], ["x0", "u", "dt"], ["x1"])


# ----------------------------------------------------------------------------- course IO
def load_course(course_json: Path):
    """Load gates (NED) + build the start/end boundary. The course json is in diffaero z-up
    (= NED * [1,-1,-1]); we convert gate centres to NED for planning."""
    c = json.loads(Path(course_json).read_text())
    gates_zup = [g["pos_zup"] for g in c["gates"]]
    yaw = c["gates"][0]["yaw"]
    # z-up -> NED: y,z flip (NED = zup * [1,-1,-1]).
    gates_ned = [[p[0], -p[1], -p[2]] for p in gates_zup]
    start_ned = [0.0, 0.0, 0.02]                       # live spawn pad, NED (z down small)
    # virtual endpoint: 25 m past the last gate along its through-direction (-X for yaw pi)
    last = gates_ned[-1]
    n = np.array([np.cos(yaw), np.sin(yaw), 0.0])      # gate normal (z-up xy == NED xy)
    end_ned = [last[0] + 25.0 * n[0], last[1] + 25.0 * n[1], last[2]]
    return np.array(gates_ned), np.array(start_ned), np.array(end_ned)


# ----------------------------------------------------------------------------- the NLP
class TrajOpt:
    def __init__(self, gates_ned, start_ned, end_ned, *, use_drag=True,
                 nodes_per_seg=12, gate_radius=PP.GATE_BALL_RADIUS, dt_max=0.5):
        self.gates = gates_ned
        self.start = start_ned
        self.end = end_ned
        self.use_drag = use_drag
        self.gate_radius = gate_radius
        self.f = make_dynamics(use_drag)
        self.F = make_rk4(self.f, M=1)

        # waypoints in order = gates + virtual endpoint; one segment per waypoint, from the
        # PREVIOUS waypoint (start for seg 0). seg_num = n_gates + 1 (last seg -> endpoint).
        self.wps = np.vstack([gates_ned, end_ned])     # (n_gates+1, 3) targets
        self.seg_num = len(self.wps)
        self.Ns = [nodes_per_seg] * self.seg_num
        self.horizon = sum(self.Ns)
        self.base = [0]
        for n in self.Ns:
            self.base.append(self.base[-1] + n)

        self.Xdim, self.Udim = 10, 4
        self.dt_max = dt_max
        self._build()

    def _state_bounds(self):
        # p free; v bounded generously; quaternion in [-1,1].
        x_lb = [-ca.inf] * 3 + [-120.0] * 3 + [-1.0] * 4
        x_ub = [ca.inf] * 3 + [120.0] * 3 + [1.0] * 4
        return x_lb, x_ub

    def _ctrl_bounds(self):
        om = PP.OMEGA_MAX
        u_lb = [0.0, -om[0], -om[1], -om[2]]
        u_ub = [PP.A_MAX, om[0], om[1], om[2]]
        return u_lb, u_ub

    def _build(self):
        H, Xd, Ud = self.horizon, self.Xdim, self.Udim
        Xs = ca.MX.sym("Xs", Xd, H)
        Us = ca.MX.sym("Us", Ud, H)
        DTs = ca.MX.sym("DTs", self.seg_num)
        x_lb, x_ub = self._state_bounds()
        u_lb, u_ub = self._ctrl_bounds()

        # boundary states: start at rest identity attitude; end at rest.
        x_init = np.concatenate([self.start, [0, 0, 0], [1, 0, 0, 0]])
        x_end_pos = self.end

        # Decision vector layout: [vec(Xs) (col-major), vec(Us), DTs]. Build bounds in the
        # SAME order so they line up with ca.veccat of the symbol matrices (which keeps the
        # vector "purely symbolic" as IPOPT requires).
        lbx, ubx = [], []
        # state bounds, column-major (node 0 all dims, node 1 ...), matching ca.vec(Xs)
        for k in range(H):
            lbx += x_lb; ubx += x_ub
        for k in range(H):
            lbx += u_lb; ubx += u_ub
        for i in range(self.seg_num):
            lbx += [1e-3]; ubx += [self.dt_max]

        g, lbg, ubg = [], [], []
        obj = 0.0
        for i in range(self.seg_num):
            b = self.base[i]
            for j in range(self.Ns[i]):
                k = b + j
                x_prev = x_init if k == 0 else Xs[:, k - 1]
                g.append(Xs[:, k] - self.F(x_prev, Us[:, k], DTs[i]))
                lbg += [0.0] * Xd; ubg += [0.0] * Xd
            obj += DTs[i] * self.Ns[i]
            k_last = self.base[i + 1] - 1
            if i < self.seg_num - 1:
                d = Xs[0:3, k_last] - self.wps[i]
                g.append(d.T @ d)
                lbg += [0.0]; ubg += [self.gate_radius ** 2]   # within inscribed circle
            else:
                d = Xs[0:3, k_last] - x_end_pos
                g.append(d.T @ d)
                lbg += [0.0]; ubg += [1.0 ** 2]                # 1 m terminal ball
                g.append(Xs[3:6, k_last])                       # v_end == 0
                lbg += [0.0] * 3; ubg += [0.0] * 3

        self._nlp_x = ca.veccat(Xs, Us, DTs)
        self._g = ca.vertcat(*g)
        self._lbx, self._ubx = lbx, ubx
        self._lbg, self._ubg = lbg, ubg
        self._x_init = x_init
        opts = {"ipopt.max_iter": 3000, "ipopt.print_level": 0, "print_time": False,
                "ipopt.tol": 1e-4, "ipopt.acceptable_tol": 1e-3,
                "ipopt.acceptable_iter": 10}
        self._solver = ca.nlpsol("trajopt", "ipopt",
                                 {"f": obj, "x": self._nlp_x, "g": self._g}, opts)

    def initial_guess(self):
        """Straight-line, constant-speed warm start through the waypoints. Cheap + robust."""
        H, Xd, Ud = self.horizon, self.Xdim, self.Udim
        v_guess = 20.0                                     # m/s nominal cruise guess
        X0 = np.zeros((Xd, H)); U0 = np.zeros((Ud, H))
        DT0 = np.zeros(self.seg_num)
        prev = self.start
        for i in range(self.seg_num):
            tgt = self.wps[i]
            seg = tgt - prev
            L = np.linalg.norm(seg) + 1e-9
            DT0[i] = max((L / v_guess) / self.Ns[i], 5e-3)
            dirn = seg / L
            for j in range(self.Ns[i]):
                k = self.base[i] + j
                frac = (j + 1) / self.Ns[i]
                X0[0:3, k] = prev + frac * seg
                X0[3:6, k] = dirn * v_guess
                X0[6, k] = 1.0                              # identity attitude guess
                U0[0, k] = PP.G                             # hover-ish thrust guess
            prev = tgt
        # taper end velocity to zero over the last segment
        for j in range(self.Ns[-1]):
            k = self.base[self.seg_num - 1] + j
            frac = 1.0 - (j + 1) / self.Ns[-1]
            X0[3:6, k] *= frac
        # match ca.veccat(Xs, Us, DTs): column-major flatten of X0, then U0, then DT0.
        return np.concatenate([X0.flatten(order="F"), U0.flatten(order="F"), DT0])

    def solve(self):
        z0 = self.initial_guess()
        res = self._solver(x0=z0, lbx=self._lbx, ubx=self._ubx,
                           lbg=self._lbg, ubg=self._ubg)
        stats = self._solver.stats()
        z = res["x"].full().flatten()
        return self._unpack(z), stats

    def _unpack(self, z):
        H, Xd, Ud = self.horizon, self.Xdim, self.Udim
        # inverse of ca.veccat(Xs, Us, DTs): X block (col-major), U block, DT.
        nX, nU = H * Xd, H * Ud
        X = z[:nX].reshape((Xd, H), order="F").T
        U = z[nX:nX + nU].reshape((Ud, H), order="F").T
        DT = z[nX + nU: nX + nU + self.seg_num]
        return {"X": X, "U": U, "DT": DT}


# ----------------------------------------------------------------------------- validation + IO
def validate(sol, opt: TrajOpt):
    """Re-check actuator limits, gate misses, lap time at the last-gate crossing."""
    X, U, DT = sol["X"], sol["U"], sol["DT"]
    # per-node absolute time
    t = np.zeros(opt.horizon)
    tt = 0.0
    idx = 0
    for i in range(opt.seg_num):
        for j in range(opt.Ns[i]):
            tt += DT[i]
            t[idx] = tt
            idx += 1
    # gate misses (Euclidean, at the segment's last node)
    misses = []
    for i in range(len(opt.gates)):
        k_last = opt.base[i + 1] - 1
        misses.append(float(np.linalg.norm(X[k_last, 0:3] - opt.gates[i])))
    # lap time = crossing time of the LAST gate (segment len(gates)-1's last node)
    k_lastgate = opt.base[len(opt.gates)] - 1
    lap = float(t[k_lastgate])
    # actuator usage
    a_c = U[:, 0]
    speed = np.linalg.norm(X[:, 3:6], axis=1)
    om = np.abs(U[:, 1:4])
    return {
        "lap_time_s": lap,
        "total_time_to_endpoint_s": float(t[-1]),
        "max_gate_miss_m": float(np.max(misses)),
        "gate_misses_m": [round(m, 4) for m in misses],
        "gate_radius_m": opt.gate_radius,
        "all_gates_passed": bool(np.max(misses) <= opt.gate_radius + 1e-3),
        "a_c_max_mps2": float(a_c.max()),
        "a_c_max_g": float(a_c.max() / PP.G),
        "a_c_ceiling_g": PP.A_MAX_G,
        "thrust_sat_frac": float((a_c > 0.98 * PP.A_MAX).mean()),
        "omega_max_used_rps": [float(om[:, 0].max()), float(om[:, 1].max()),
                               float(om[:, 2].max())],
        "omega_limits_rps": PP.OMEGA_MAX.tolist(),
        "max_speed_mps": float(speed.max()),
        "mean_speed_mps": float(speed.mean()),
        "use_drag": opt.use_drag,
        "_t": t, "_speed": speed, "_a_c": a_c,
    }


def save_csv(sol, opt: TrajOpt, path: Path):
    X, U, DT = sol["X"], sol["U"], sol["DT"]
    t = 0.0
    rows = []
    idx = 0
    for i in range(opt.seg_num):
        for j in range(opt.Ns[i]):
            t += DT[i]
            x = X[idx]; u = U[idx]
            rows.append([t, *x[0:3], *x[3:6], *x[6:10], *u])
            idx += 1
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "p_x", "p_y", "p_z", "v_x", "v_y", "v_z",
                    "q_w", "q_x", "q_y", "q_z", "a_c", "w_x", "w_y", "w_z"])
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--course", default="rl/peregrine_course_diffaero.json")
    ap.add_argument("--out", default="scripts/planning/out")
    ap.add_argument("--nodes", type=int, default=12, help="nodes per gate-to-gate segment")
    ap.add_argument("--no-drag", action="store_true", help="drop quadratic drag (abs bound)")
    ap.add_argument("--radius", type=float, default=PP.GATE_BALL_RADIUS,
                    help="gate inscribed-circle radius (m)")
    args = ap.parse_args()

    PP.verify_against_rl_plant()
    gates, start, end = load_course(Path(args.course))
    opt = TrajOpt(gates, start, end, use_drag=not args.no_drag,
                  nodes_per_seg=args.nodes, gate_radius=args.radius)
    print(f"course: {len(gates)} gates, horizon={opt.horizon} nodes, "
          f"drag={'OFF' if args.no_drag else 'ON'}, A_MAX={PP.A_MAX_G:.2f} g")
    sol, stats = opt.solve()
    val = validate(sol, opt)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_csv(sol, opt, out / "trajectory.csv")
    summary = {k: v for k, v in val.items() if not k.startswith("_")}
    summary["ipopt_status"] = stats.get("return_status", "?")
    summary["ipopt_success"] = bool(stats.get("success", False))
    summary["iter_count"] = int(stats.get("iter_count", -1))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"-> {out}/trajectory.csv  +  summary.json")
    return 0 if summary["ipopt_success"] and summary["all_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
