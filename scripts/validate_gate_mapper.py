"""scripts/validate_gate_mapper.py — the mapper's proof: sweep the MEASURED noise model.

Runs the offline gate mapper (cases A, B, C) over synthetic exploration laps generated
from the measured perception-noise model (handoff/perception-char-2026-06-08) on the real
VQ1 geometry, sweeping noise scale / catastrophic-leak rate / association-error rate /
sighting coverage / case-C visibility+bridge knobs, multi-seed. Emits the validation
tables (markdown) to stdout and ``--out``; ``--dump-sightings DIR`` also writes sample
``racer.mapper_sightings/v1`` files (the run_mapper_offline.py input schema).

Usage:  python scripts/validate_gate_mapper.py [--seeds 5] [--c-seeds 3] [--quick]
            [--out handoff/.../validation_table.md] [--dump-sightings DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from racer.gate_mapper import (
    MEASURED_FIX_BIAS_NED,
    MapperConfig,
    dump_sightings_json,
    estimate_map_no_pose,
    estimate_map_pose_aided,
)
from racer.gate_mapper_synth import (
    MEASURED_NOISE,
    VQ1_TRACK_RECORDS,
    evaluate_map,
    generate_pose_aided_sightings,
    generate_relative_sightings,
    simulate_exploration_path,
    true_gates,
)

CENTRES, _YAWS = true_gates()


def _coverage_poses(coverage: float, scan: bool = False):
    passes = max(int(np.ceil(coverage)), 1)
    poses = simulate_exploration_path(CENTRES, passes=passes,
                                      scan_pitch_down_deg=25.0 if scan else 0.0)
    if coverage < 1.0:
        poses = poses[: int(len(poses) * coverage)]
    return poses


def _agg(vals):
    return f"{np.mean(vals):.2f} ({np.max(vals):.2f})"


def run_case_a(label, seeds, coverage=1.0, noise_scale=1.0, leak=None, assoc=None,
               labeled=True, bias_correct=False):
    noise = MEASURED_NOISE.scaled(noise_scale)
    if leak is not None:
        noise = noise.scaled(leak_rate=leak)
    if assoc is not None:
        noise = noise.scaled(assoc_error_rate=assoc)
    cfg = MapperConfig(bias_correction_ned=MEASURED_FIX_BIAS_NED if bias_correct else None)
    rows = {"matched": [], "max": [], "mean": [], "inpl": [], "yaw": [], "n": [], "rej": []}
    for seed in range(seeds):
        poses = _coverage_poses(coverage)
        s = generate_pose_aided_sightings(poses, noise=noise,
                                          rng=np.random.default_rng(seed), labeled=labeled)
        est = estimate_map_pose_aided(s, config=cfg)
        ev = evaluate_map(est)
        rows["matched"].append(ev["n_matched"])
        rows["max"].append(ev["pos_err_max_m"])
        rows["mean"].append(ev["pos_err_mean_m"])
        rows["inpl"].append(ev["inplane_err_max_m"])
        rows["yaw"].append(ev["yaw_err_max_deg"])
        rows["n"].append(len(s) / max(ev["n_matched"], 1))
        rows["rej"].append(est.diagnostics["n_rejected"])
    return (f"| {label} | {np.mean(rows['n']):.0f} | {min(rows['matched'])}/6 "
            f"| {_agg(rows['mean'])} | {_agg(rows['max'])} | {_agg(rows['inpl'])} "
            f"| {_agg(rows['yaw'])} | {np.mean(rows['rej']):.0f} |")


def _rough_prior(rng, offset_m):
    recs = []
    for r in VQ1_TRACK_RECORDS:
        d = rng.normal(size=3)
        d = d / np.linalg.norm(d) * offset_m
        recs.append({**r, "position_ned": list(np.asarray(r["position_ned"]) + d)})
    return recs


def run_case_b(label, seeds, offset_m=2.0, coverage=1.0, prior_sigma=2.0):
    fused_max, prior_max = [], []
    for seed in range(seeds):
        rng = np.random.default_rng(1000 + seed)
        prior = _rough_prior(rng, offset_m)
        prior_max.append(max(
            np.linalg.norm(np.asarray(p["position_ned"]) - np.asarray(t["position_ned"]))
            for p, t in zip(prior, VQ1_TRACK_RECORDS)))
        s = generate_pose_aided_sightings(_coverage_poses(coverage),
                                          rng=np.random.default_rng(seed))
        cfg = MapperConfig(prior_sigma_pos=prior_sigma)
        ev = evaluate_map(estimate_map_pose_aided(s, prior_records=prior, config=cfg))
        fused_max.append(ev["pos_err_max_m"])
    return (f"| {label} | {_agg(prior_max)} | {_agg(fused_max)} "
            f"| {np.mean(prior_max) / max(np.mean(fused_max), 1e-9):.1f}x |")


def run_case_c(label, seeds, scan=True, max_range=28.0, accel_sigma=1.0, leak=None,
               labeled=False, coverage=1.0):
    noise = MEASURED_NOISE.scaled(max_range_m=max_range)
    if leak is not None:
        noise = noise.scaled(leak_rate=leak)
    cfg = MapperConfig(accel_prior_sigma=accel_sigma)
    rows = {"matched": [], "amax": [], "amean": [], "rmax": [], "inpl": [], "yaw": [],
            "comp": [], "disc": [], "extra": [], "t": []}
    for seed in range(seeds):
        s = generate_relative_sightings(_coverage_poses(coverage, scan=scan), noise=noise,
                                        rng=np.random.default_rng(100 + seed),
                                        labeled=labeled)
        t0 = time.perf_counter()
        est = estimate_map_no_pose(s, config=cfg, n_gates_hint=6)
        rows["t"].append(time.perf_counter() - t0)
        ev = evaluate_map(est, align_translation=True)
        evr = evaluate_map(est, align_translation=False)
        d = est.diagnostics
        rows["matched"].append(ev["n_matched"])
        rows["amax"].append(ev["pos_err_max_m"])
        rows["amean"].append(ev["pos_err_mean_m"])
        rows["rmax"].append(evr["pos_err_max_m"])
        rows["inpl"].append(ev["inplane_err_max_m"])
        rows["yaw"].append(ev["yaw_err_max_deg"])
        rows["comp"].append(d["covis_components"])
        rows["disc"].append(sum("disconnected" in g.flags for g in est.gates))
        rows["extra"].append(ev["n_extra_est"])
    return (f"| {label} | {min(rows['matched'])}/6 (+{max(rows['extra'])}) "
            f"| {max(rows['comp'])} | {max(rows['disc'])} | {_agg(rows['amean'])} "
            f"| {_agg(rows['amax'])} | {_agg(rows['rmax'])} | {_agg(rows['inpl'])} "
            f"| {_agg(rows['yaw'])} | {np.mean(rows['t']):.0f}s |")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5, help="seeds for cases A/B")
    ap.add_argument("--c-seeds", type=int, default=3, help="seeds for case C (slower)")
    ap.add_argument("--quick", action="store_true", help="2 seeds everywhere")
    ap.add_argument("--out", default=None, help="also write the markdown here")
    ap.add_argument("--dump-sightings", default=None,
                    help="write sample mapper_sightings/v1 files into this dir")
    args = ap.parse_args()
    sa, sc = (2, 2) if args.quick else (args.seeds, args.c_seeds)

    lines = []
    say = lines.append
    say("# Gate-mapper validation — measured noise model, real VQ1 geometry")
    say("")
    say(f"Noise (perception-char 2026-06-08): sigma {MEASURED_NOISE.sigma_ned.tolist()} m "
        f"(N,E,D), bias {MEASURED_NOISE.bias_ned.tolist()} m, leak "
        f"{MEASURED_NOISE.leak_rate:.1%} @ 3-8 m, assoc errors "
        f"{MEASURED_NOISE.assoc_error_rate:.0%}, detect {MEASURED_NOISE.detect_prob:.0%}, "
        f"range {MEASURED_NOISE.min_range_m}-{MEASURED_NOISE.max_range_m} m. "
        f"Validity yardstick: 0.75 m half-opening (in-plane).")
    say("")
    say(f"## Case A — pose-aided ({sa} seeds; cells: mean (worst) over seeds)")
    say("")
    say("| config | n/gate | gates | err mean m | err max m | in-plane max m "
        "| yaw max deg | rej |")
    say("|---|---|---|---|---|---|---|---|")
    say(run_case_a("baseline (labeled, 1 lap)", sa))
    say(run_case_a("+ measured-bias correction", sa, bias_correct=True))
    say(run_case_a("unlabeled (clustering)", sa, labeled=False))
    say(run_case_a("noise x0.5", sa, noise_scale=0.5))
    say(run_case_a("noise x2", sa, noise_scale=2.0))
    say(run_case_a("leak 0%", sa, leak=0.0))
    say(run_case_a("leak 5%", sa, leak=0.05))
    say(run_case_a("assoc errors 5%", sa, assoc=0.05))
    say(run_case_a("assoc errors 15%", sa, assoc=0.15))
    say(run_case_a("0.25 lap", sa, coverage=0.25))
    say(run_case_a("0.5 lap", sa, coverage=0.5))
    say(run_case_a("2 laps", sa, coverage=2.0))
    print("\n".join(lines[-15:]))

    n0 = len(lines)
    say("")
    say(f"## Case B — rough prior + refinement ({sa} seeds)")
    say("")
    say("| config | prior err max m | fused err max m | gain |")
    say("|---|---|---|---|")
    say(run_case_b("offset 2 m, 1 lap, prior_sigma 2", sa))
    say(run_case_b("offset 2 m, 0.25 lap", sa, coverage=0.25))
    say(run_case_b("offset 3 m, 1 lap", sa, offset_m=3.0))
    say(run_case_b("offset 1 m, 1 lap, prior_sigma 1", sa, offset_m=1.0, prior_sigma=1.0))
    print("\n".join(lines[n0:]))

    n0 = len(lines)
    say("")
    say(f"## Case C — no pose ({sc} seeds; 'aligned' = translation gauge removed)")
    say("")
    say("| config | gates (+extra) | comps | disc | aligned mean m | aligned max m "
        "| raw max m | in-plane max m | yaw max deg | t |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    say(run_case_c("baseline (scan, range 28, sig_a 1.0)", sc))
    say(run_case_c("no scan nod", sc, scan=False))
    say(run_case_c("range 24 (measured flat-to)", sc, max_range=24.0))
    say(run_case_c("range 32 (live cap)", sc, max_range=32.0))
    say(run_case_c("sig_a 0.5 (smoother lap)", sc, accel_sigma=0.5))
    say(run_case_c("sig_a 6.0 (sporty lap)", sc, accel_sigma=6.0))
    say(run_case_c("NO smoothness bridge", sc, accel_sigma=None))
    say(run_case_c("leak 5%", sc, leak=0.05))
    say(run_case_c("labels given (naming)", sc, labeled=True))
    say(run_case_c("2 passes", sc, coverage=2.0))
    print("\n".join(lines[n0:]))

    if args.dump_sightings:
        d = Path(args.dump_sightings)
        poses = _coverage_poses(1.0, scan=True)
        pa = dump_sightings_json(
            d / "sample_pose_aided.json",
            generate_pose_aided_sightings(poses, rng=np.random.default_rng(0)),
            note="synthetic, measured noise, VQ1 geometry, 1 lap")
        pc = dump_sightings_json(
            d / "sample_relative.json",
            generate_relative_sightings(poses, rng=np.random.default_rng(0)),
            note="synthetic, measured noise, VQ1 geometry, 1 lap + scan nod")
        say("")
        say(f"Sample sightings dumped: {pa}, {pc}")
        print(f"dumped {pa} and {pc}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
