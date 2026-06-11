"""Dissect the case-C 54 m failure: init vs final, residuals, solver status."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import numpy as np
import scipy.sparse as sp
from scipy.optimize import least_squares

from racer.gate_mapper import MapperConfig, _chain_init, _covis_components, _solve_no_pose
from racer.gate_mapper_synth import MEASURED_NOISE, generate_relative_sightings, \
    simulate_exploration_path, true_gates

centres, yaws = true_gates()
poses = simulate_exploration_path(centres, scan_pitch_down_deg=25.0)
sc = generate_relative_sightings(poses, rng=np.random.default_rng(100),
                                 noise=MEASURED_NOISE.scaled(leak_rate=0.0, assoc_error_rate=0.0))
print(f"{len(sc)} sightings (leak/assoc OFF for clean diagnosis)")

cfg = MapperConfig(accel_prior_sigma=1.0)

by_frame = {}
for s in sc:
    by_frame.setdefault(int(s.frame_id), []).append(s)
frame_ids = sorted(by_frame, key=lambda f: (min(s.t for s in by_frame[f]), f))
frames = [by_frame[f] for f in frame_ids]
frame_t = np.array([float(np.median([s.t for s in fr])) for fr in frames])
gaps = np.diff(frame_t)
print(f"frames={len(frames)}  gaps>0.5s: {np.sum(gaps > 0.5)}  max gap {gaps.max():.2f} s "
      f"at t={frame_t[np.argmax(gaps)]:.1f}")

levers = [[s.lever_world() for s in fr] for fr in frames]
obs4, gate_init, pose_init, n_rej, first_seen = _chain_init(frames, levers, cfg)
obs = [(f, g, L) for (f, g, L, _s) in obs4]
G, P = len(gate_init), len(pose_init)
print(f"chain: G={G} P={P} rejected={n_rej}")
print("gate_init vs truth (raw):")
for g in range(G):
    d = np.linalg.norm(gate_init[g] - centres[g]) if g < len(centres) else float("nan")
    print(f"  g{g}: init={np.round(gate_init[g], 2)} err {d:.2f}")

# pose-init error along the chain
pose_err = [np.linalg.norm(pose_init[i] - np.asarray(poses[frame_ids[i]][1])) for i in range(P)]
print("pose-init err quantiles:", np.round(np.percentile(pose_err, [50, 90, 99, 100]), 2))

comp, n_comp = _covis_components(obs, G, P)
print(f"components: {n_comp}  comp_of_gate={comp}")

sol = _solve_no_pose(obs, gate_init, pose_init, frame_t, cfg)
print("\nafter BA (raw errors):")
for g in range(G):
    d = np.linalg.norm(sol["gate_pos"][g] - centres[g])
    di = np.linalg.norm(sol["gate_pos"][g] - gate_init[g])
    print(f"  g{g}: err {d:.2f}  moved-from-init {di:.2f}")
print(f"cost={sol['cost']:.1f} resid_rej={sol['n_rejected']} cond={sol['jtj_cond']:.2e}")

# raw least_squares introspection: rebuild and look at convergence
W = 1.0 / np.asarray(cfg.sigma_ned)
print("\nsolver introspection with jac COPY each call:")
n_par = 3 * G + 3 * (P - 1)
# quick rebuild via _solve_no_pose internals is private; emulate with linear obs rows only
data, ri, ci, c = [], [], [], []
row = 0
anchor = pose_init[0]
for (pi, gi, L) in obs:
    for a in range(3):
        ri.append(row + a); ci.append(3 * gi + a); data.append(W[a])
        if pi > 0:
            ri.append(row + a); ci.append(3 * G + 3 * (pi - 1) + a); data.append(-W[a])
    const = L + (anchor if pi == 0 else 0.0)
    c.extend((W * const).tolist())
    row += 3
J = sp.csr_matrix((data, (ri, ci)), shape=(row, n_par))
cvec = np.asarray(c)
x0 = np.concatenate([np.asarray(gate_init).ravel(), np.asarray(pose_init[1:]).ravel()])
for jac_mode in ("shared", "copy"):
    jac = (lambda x: J) if jac_mode == "shared" else (lambda x: J.copy())
    res = least_squares(lambda x: J @ x - cvec, x0, jac=jac, method="trf",
                        tr_solver="lsmr", loss="soft_l1", f_scale=2.0, max_nfev=200)
    errs = [float(np.linalg.norm(res.x[3 * g: 3 * g + 3] - centres[g])) for g in range(G)]
    print(f"  jac={jac_mode}: status={res.status} nfev={res.nfev} cost={res.cost:.1f} "
          f"gate errs {np.round(errs, 2)}")
