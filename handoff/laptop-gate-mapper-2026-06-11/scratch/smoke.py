"""Smoke + magnitude calibration for the gate mapper (not a test; numbers feed thresholds)."""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import numpy as np

from racer.gate_mapper import MapperConfig, MEASURED_FIX_BIAS_NED, estimate_map_no_pose, \
    estimate_map_pose_aided
from racer.gate_mapper_synth import MEASURED_NOISE, evaluate_map, \
    generate_pose_aided_sightings, generate_relative_sightings, simulate_exploration_path, \
    true_gates

centres, yaws = true_gates()
print("true centres:")
for i, (c, y) in enumerate(zip(centres, yaws)):
    print(f"  g{i}: {np.round(c, 2)}  yaw {np.degrees(y):.1f} deg")
seg = np.linalg.norm(np.diff(centres, axis=0), axis=1)
print("inter-gate spacing:", np.round(seg, 1))

poses = simulate_exploration_path(centres)
print(f"\npath: {len(poses)} poses, {poses[-1][0]:.1f} s")

# visibility census
from racer.gate_mapper_synth import visible_gate_indices
vis_count = np.zeros(len(centres), int)
multi = 0
for (t, p, R) in poses:
    v = visible_gate_indices(p, R, centres, MEASURED_NOISE)
    for g in v:
        vis_count[g] += 1
    multi += len(v) >= 2
print("visible-frames per gate:", vis_count, f" multi-gate frames: {multi}/{len(poses)}")

for seed in range(3):
    rng = np.random.default_rng(seed)
    sa = generate_pose_aided_sightings(poses, rng=rng)
    est = estimate_map_pose_aided(sa)
    ev = evaluate_map(est)
    print(f"\n[A labeled seed {seed}] n={len(sa)} matched={ev['n_matched']} "
          f"err mean/max {ev['pos_err_mean_m']:.3f}/{ev['pos_err_max_m']:.3f} "
          f"inplane max {ev['inplane_err_max_m']:.3f} yaw max {ev['yaw_err_max_deg']:.2f}")
    cfgb = MapperConfig(bias_correction_ned=MEASURED_FIX_BIAS_NED)
    evb = evaluate_map(estimate_map_pose_aided(sa, config=cfgb))
    print(f"  bias-corrected: err mean/max {evb['pos_err_mean_m']:.3f}/{evb['pos_err_max_m']:.3f}")
    # cluster mode
    sau = generate_pose_aided_sightings(poses, rng=np.random.default_rng(seed), labeled=False)
    evc = evaluate_map(estimate_map_pose_aided(sau))
    print(f"  cluster mode: matched={evc['n_matched']} err max {evc['pos_err_max_m']:.3f}")

poses_scan = simulate_exploration_path(centres, scan_pitch_down_deg=25.0)
vis2 = 0
for (t, p, R) in poses_scan:
    vis2 += len(visible_gate_indices(p, R, centres, MEASURED_NOISE)) >= 2
print(f"\nwith scan: multi-gate frames {vis2}/{len(poses_scan)}")

for scan, pp in (("scan", poses_scan), ("noscan", poses)):
    for sig_a in (6.0, 1.0, 0.5):
        errs = []
        for seed in range(3):
            sc = generate_relative_sightings(pp, rng=np.random.default_rng(100 + seed))
            cfg = MapperConfig(accel_prior_sigma=sig_a)
            t0 = time.perf_counter()
            est = estimate_map_no_pose(sc, config=cfg, n_gates_hint=6)
            dt = time.perf_counter() - t0
            ev = evaluate_map(est, align_translation=True)
            evr = evaluate_map(est, align_translation=False)
            d = est.diagnostics
            errs.append(ev["pos_err_max_m"])
            if seed == 0:
                print(f"\n[C {scan} sig_a={sig_a}] comps={d['covis_components']} "
                      f"virt={d['n_virtual_poses']} ({dt:.1f} s) "
                      f"resid_rej={d['n_residual_rejected']}")
                print(f"  aligned mean/max {ev['pos_err_mean_m']:.2f}/{ev['pos_err_max_m']:.2f} "
                      f"raw max {evr['pos_err_max_m']:.2f} "
                      f"inplane max {ev['inplane_err_max_m']:.2f} yaw max {ev['yaw_err_max_deg']:.1f}")
                print("  per-gate aligned:",
                      " ".join(f"g{pg['true_gate']}:{pg['err_m']:.2f}" for pg in ev["per_gate"]))
        print(f"  [{scan} sig_a={sig_a}] aligned max over 3 seeds: "
              f"{np.min(errs):.2f}..{np.max(errs):.2f}")
