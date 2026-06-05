"""Build sysid/manifest.json from the per-run extracts (reproducible summary for the laptop)."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SY = HERE / "sysid"

ORDER = ["rate1", "rate2", "hsweep1", "hsweep2", "hsweep3", "gate0_course1"]
PURPOSE = {
    "rate1": "open-loop body-rate doublets (pitch,roll,yaw @0.25 thrust); aborted +8m",
    "rate2": "open-loop body-rate doublets, RICHEST (11s); best for rate_gain/sign/tau",
    "hsweep1": "thrust held 0.48 only (climb-away, aborted +8m) -> one vertical-accel point",
    "hsweep2": "thrust held 0.25 only (0.6s anti-damping tumble) -> low value, included for completeness",
    "hsweep3": "thrust 0.25 then 0.30 -> two vertical-accel points",
    "gate0_course1": "CLOSED-LOOP validation: clean dense FINISHED origin->gate0 flight (~6.6s)",
}

runs = []
for name in ORDER:
    p = SY / f"{name}.json"
    if not p.exists():
        continue
    d = json.loads(p.read_text(encoding="utf-8"))
    v = d["validation"]
    runs.append({
        "file": p.name,
        "run": d["run"],
        "purpose": PURPOSE.get(name, ""),
        "command_schema": d["command_schema"],
        "n_samples": v["n_samples"],
        "rate_hz": d["rate_hz"],
        "meta": d["meta"],
        "validation": v,
        "size_kb": p.stat().st_size // 1024,
    })

manifest = {
    "schema": "racer.sysid_extract_manifest/v1",
    "created_for": "laptop CTBR twin fit (rate_gain, rate_sign, rate_tau, thrust hover/slope) + validation",
    "source_commit": runs[0]["validation"] and json.loads((SY / f"{ORDER[0]}.json").read_text())["source_commit"],
    "sim_host": "ShadowPC",
    "extractor": "scripts/extract_sysid.py",
    "per_sample_fields": [
        "t_ns (sim clock = time_usec*1000; ODOMETRY/ACTUATOR/commands share it)",
        "t_s (relative seconds)",
        "odo_q_wxyz (body->world quat, scalar-first)",
        "odo_angular_rate (BODY FRD rad/s, RAW; sign-inverted on pitch -- prefer quaternion finite-diff)",
        "odo_vel_body (BODY FRD m/s, raw twist)",
        "odo_vel_world (WORLD NED m/s = c3b5a8e fix rotation -- the CORRECTED velocity)",
        "odo_pos (WORLD NED m)",
        "lpn_pos / lpn_vel (WORLD NED, LOCAL_POSITION_NED, independent; lpn_vel cross-checks odo_vel_world)",
        "actuators[4] (ACTUATOR_OUTPUT_STATUS motor outputs 0..1)",
        "cmd_body_rate (OUR commanded body rate rad/s, ZOH), cmd_thrust (OUR collective 0..1, ZOH)",
        "cmd_age_ms (staleness of the held command), ctx (phase/kind/axis or mission state)",
    ],
    "notes": [
        "Velocity is the FIXED (rotated) world velocity in odo_vel_world; odo_vel_body is the raw "
        "body twist; lpn_vel is the independent world reference. validation.rot_odo_vs_lpn_vel_rms_mps "
        "confirms the fix per run (tiny on clean runs; inflated on hsweeps by 75/97Hz stagger during "
        "fast vertical accel).",
        "All commands WERE logged (commands.jsonl) -- no controller replay needed; actuators included anyway.",
        "Rate-gain SIGNS recovered from these extracts: roll -, pitch +, yaw - (memo-consistent). "
        "Use a settled-tail fit (analyze_sysid.py method) for the ~2.4-2.7 steady magnitude.",
        "hsweeps are NOT clean multi-level sweeps (aborted early); POOL the discrete thrust levels "
        "{0.25,0.30,0.48} with the rate runs (0.25) and course1 (~0.26-0.27) to fit hover/slope.",
        "t_ns is absolute sim ns and joins natively with each run's commands.jsonl sim_time_ns/sim_t.",
    ],
    "runs": runs,
}
(SY / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"wrote {SY / 'manifest.json'} with {len(runs)} runs")
for r in runs:
    print(f"  {r['run']:>34}  n={r['n_samples']:<4} {r['size_kb']:>4}KB  "
          f"rot-vs-lpn RMS={r['validation']['rot_odo_vs_lpn_vel_rms_mps']}")
