"""scripts/run_mapper_offline.py — sightings in, track map out (the between-attempts step).

Offline gate mapping for VQ2 cases B (pose + rough map) and C (no pose): feed the gate
sightings extracted from an exploration-lap recording, get back a ``track_map_*.json`` the
navigator loads exactly like the given map (``navigator.load_track_map``). Offline
processing between attempts is legal; this is the one-command version of it.

Input: a ``racer.mapper_sightings/v1`` JSON (``racer.gate_mapper.dump_sightings_json``).
  * mode "pose_aided": rows {gate_world_ned, gate_id?, yaw_world?, t, frame_id?, range_m?}
    — gate_world_ned = drone_position_ned + R_world_camera @ t_cam_gate, i.e. the trusted
    pose plus the PnP lever (the same two quantities the live chain logs per fix).
  * mode "relative":   rows {frame_id, rel_position_frd, quat_wxyz, gate_id?, rel_yaw?, t}
    — the PnP lever in body FRD + the GIVEN attitude; no drone position anywhere.
The extractor that turns a ``data/runs/<session>`` recording into this file follows the
``export_course_bundle.py`` alignment recipe (dedup frames by frame_id, nearest ODOMETRY
quat by recv clock, detector -> PnP); until VQ2 tells us which telemetry exists, the
synthetic generator (``racer.gate_mapper_synth`` / scripts/validate_gate_mapper.py
--dump-sightings) writes the same schema, so this CLI is exercised end-to-end today.

Usage:
  python scripts/run_mapper_offline.py SIGHTINGS.json --out refined_map.json
      [--prior data/runs/track_map_*.json]     # case B: rough prior to refine
      [--prior-sigma-pos 2.0] [--prior-sigma-yaw-deg 15]
      [--bias-correct]                          # opt-in measured fix-bias removal
      [--no-smoothness] [--accel-sigma 1.0]     # case C bridge knobs
      [--n-gates 6] [--report]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from racer.gate_mapper import (
    MEASURED_FIX_BIAS_NED,
    MapperConfig,
    estimate_map_no_pose,
    estimate_map_pose_aided,
    load_sightings_json,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sightings", help="racer.mapper_sightings/v1 JSON")
    ap.add_argument("--out", required=True, help="output track map JSON path")
    ap.add_argument("--prior", default=None,
                    help="rough prior map JSON (capture_track_map schema) -> case B fusion")
    ap.add_argument("--prior-sigma-pos", type=float, default=2.0,
                    help="prior position trust, 1-sigma metres (case B)")
    ap.add_argument("--prior-sigma-yaw-deg", type=float, default=15.0)
    ap.add_argument("--bias-correct", action="store_true",
                    help="apply the MEASURED fix-bias correction (perception-char "
                         "2026-06-08; course/direction-specific — opt-in)")
    ap.add_argument("--no-smoothness", action="store_true",
                    help="case C: disable the accel smoothness prior (pure observations)")
    ap.add_argument("--accel-sigma", type=float, default=1.0,
                    help="case C smoothness prior, m/s^2 1-sigma (exploration-lap honesty)")
    ap.add_argument("--n-gates", type=int, default=None,
                    help="expected gate count (diagnostic warning when mismatched)")
    ap.add_argument("--report", action="store_true", help="print the per-gate table")
    args = ap.parse_args()

    mode, sightings = load_sightings_json(args.sightings)
    cfg = MapperConfig(
        bias_correction_ned=MEASURED_FIX_BIAS_NED if args.bias_correct else None,
        prior_sigma_pos=args.prior_sigma_pos,
        prior_sigma_yaw_rad=float(np.deg2rad(args.prior_sigma_yaw_deg)),
        accel_prior_sigma=None if args.no_smoothness else args.accel_sigma,
    )
    prior_records = None
    if args.prior:
        prior_records = json.loads(Path(args.prior).read_text())["gates"]

    if mode == "pose_aided":
        est = estimate_map_pose_aided(sightings, prior_records=prior_records, config=cfg)
    else:
        if prior_records is not None:
            print("NB: --prior is ignored in relative (no-pose) mode — the no-pose map is "
                  "gauge-anchored, not world-registered; register it to the prior afterwards.",
                  file=sys.stderr)
        est = estimate_map_no_pose(sightings, config=cfg, n_gates_hint=args.n_gates)

    out = est.save_track_map_json(
        args.out, note=f"run_mapper_offline mode={mode} sightings={args.sightings} "
                       f"bias_correct={args.bias_correct}")
    d = est.diagnostics
    print(f"mode={mode}  sightings={d.get('n_sightings')}  gates={len(est.gates)}  -> {out}")
    for k in ("association", "labels", "n_rejected", "n_assoc_rejected",
              "n_residual_rejected", "covis_components", "n_merged_gates",
              "warning_connectivity", "warning_gate_count"):
        if k in d:
            print(f"  {k}: {d[k]}")
    if args.report:
        print(f"  {'gate':>4} {'position_ned (opening centre)':>34} {'yaw_deg':>8} "
              f"{'std_ned':>20} {'n':>4} flags")
        for g in est.gates:
            std = np.sqrt(np.diag(g.pos_cov))
            yaw = "-" if g.yaw is None else f"{np.degrees(g.yaw):+.1f}"
            print(f"  {g.gate_id:>4} {np.array2string(g.position_ned, precision=2):>34} "
                  f"{yaw:>8} {np.array2string(std, precision=2):>20} {g.n_used:>4} "
                  f"{','.join(g.flags) or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
