"""rl/vq2_curriculum.py -- the VQ2 slow-is-smooth curriculum stages (spec T3.2).

The curriculum is expressed as a LADDER of config-override dicts (no new env machinery -- each stage is
a set of `+env.`/`+dynamics.` hydra overrides layered on the base VQ2 recipe). This mirrors the
DIFFICULTY_PRESETS pattern: pure DATA + a tiny helper that renders the hydra override strings the sbatch
consumes. Laptop-testable; imports nothing heavy.

The four stages (spec T3.2 -- SLOW throughout; the high-speed rungs are GATED on T1's above-hover
thrust/drag validation and are NOT part of this slow ladder):

  1. HOVER / STATION-KEEP -- the safest ground (only the T1-validated channels: CTBR action map + AHRS
     + hover point). A degenerate 1-gate course whose single gate sits just ahead so the episode is
     essentially hold-position; blackout OFF; NO look-at needed (nothing to chase yet).
  2. SINGLE GATE, HEAD-ON, SLOW -- one gate, start BEYOND the ~4.5 m blackout and stop before the blind
     zone (blackout OFF; the estimator-emulated obs, NEVER perfect pose). Through-centering carries it.
  3. SINGLE GATE + TERMINAL BLACKOUT -- drive THROUGH the <4.5 m blind zone (blackout ON at the measured
     ~4.3 m). This is where the IMU-coast + belief-state earn their keep (the GRU seam, if built).
  4. MULTI-GATE LAP -- the full random vq2_like course (6 gates) incl. the HIGH-climb gate-2 case + a
     turn between gates. Blackout ON. Lap completion on estimator-emulated obs.

Each stage dict overrides the BASE recipe (defined in the sbatch COMMON). Only the stage-SPECIFIC knobs
are listed; anything not named inherits the base. n_gates is a course-sampler override (the stager
narrows the course toward the full 6-gate lap); the slow-ladder keeps forward speed low throughout via
a generous cone + the low-drop presets -- there is no explicit forward-speed knob to set here (speed
emerges from the reward + the drop/turn ranges, which stages 1-3 keep gentle).
"""
from __future__ import annotations

# The ordered stage ladder. Values are hydra-override key->value (rendered as `+env.<k>=<v>` etc. by
# render_overrides). course_mode/track_difficulty select the layout; blackout + seg-len narrowing stage
# the difficulty. All are config values the env already reads (peregrine_racing_inc8 / peregrine_course).
STAGES: dict[str, dict] = {
    # 1. HOVER: a single nearby gate (course_mode=random narrowed to n_gates=1, short spawn), blackout
    #    OFF, look-at OFF (nothing to chase). Confirms the action map + AHRS + hover point transfer.
    "hover": {
        "course_mode": "random",
        "track_difficulty": "vq2_like",
        "course_n_gates": 1,
        "course_seg_len_lo": 23.7, "course_seg_len_hi": 26.0,   # short, so it is near-station-keep
        "emul_blackout_range_m": 0.0,
        "lookat_g_yaw": 0.0, "lookat_g_pitch": 0.0,             # no pointing primitive yet
        "rw_through_centering": 10.0,
    },
    # 2. SINGLE GATE, HEAD-ON, SLOW: one gate, start beyond ~4.5 m, blackout OFF (stop before the blind
    #    zone). Estimator-emulated obs (never perfect pose). Look-at ON (validated signs).
    "single_gate": {
        "course_mode": "random",
        "track_difficulty": "vq2_like",
        "course_n_gates": 1,
        "emul_blackout_range_m": 0.0,
        "lookat_g_yaw": -3.0, "lookat_g_pitch": 3.0,
        "lookat_warmup_updates": 200,
        "rw_through_centering": 10.0,
    },
    # 3. SINGLE GATE + TERMINAL BLACKOUT: drive through the <4.5 m blind zone (blackout ON at ~4.3 m).
    #    The IMU-coast + belief state (GRU seam) earn their keep here. Vision LATENCY (content lag) ON at
    #    the measured bimodal regime -- a fix carries the t-Delta geometry, age carries the same Delta.
    "blackout_pass": {
        "course_mode": "random",
        "track_difficulty": "vq2_like",
        "course_n_gates": 1,
        "emul_blackout_range_m": 4.3,
        "emul_lat_max_s": 1.0, "emul_lat_healthy_frac": 0.5,             # bimodal content-lag ON
        "emul_lat_healthy_lo": 0.07, "emul_lat_healthy_hi": 0.12,        # healthy fed mode ~70-120 ms
        "emul_lat_cont_lo": 0.15, "emul_lat_cont_hi": 0.55,             # GPU-contention p50~0.25/p90~0.55
        "emul_tau_stale": 0.5,                                           # discriminate the fed regime
        "lookat_g_yaw": -3.0, "lookat_g_pitch": 3.0,
        "lookat_warmup_updates": 200,
        "rw_through_centering": 10.0,
    },
    # 4. MULTI-GATE LAP: full 6-gate vq2_like course incl. the HIGH-climb gate-2 + a turn. Blackout ON.
    #    Vision latency (content lag) ON + an intermittent GPU-contention stall (the 1.0 s clamp tail).
    "multi_gate": {
        "course_mode": "random",
        "track_difficulty": "vq2_like",
        "course_n_gates": 6,
        "emul_blackout_range_m": 4.3,
        "emul_lat_max_s": 1.0, "emul_lat_healthy_frac": 0.5,
        "emul_lat_healthy_lo": 0.07, "emul_lat_healthy_hi": 0.12,
        "emul_lat_cont_lo": 0.15, "emul_lat_cont_hi": 0.55,
        "emul_pose_age_stall_p": 0.01,                                   # intermittent stall -> 1.0 s clamp
        "emul_tau_stale": 0.5,
        "lookat_g_yaw": -3.0, "lookat_g_pitch": 3.0,
        "lookat_warmup_updates": 200,
        "rw_through_centering": 10.0,
    },
}

STAGE_ORDER = ("hover", "single_gate", "blackout_pass", "multi_gate")

# Keys that are COURSE-SAMPLER overrides (forwarded to sample_courses via the env, NOT +env.<k>). The
# env maps course_n_gates -> the sampler's n_gates and course_seg_len_{lo,hi} -> seg_len_m; these live
# under the same +env. namespace but are documented here so the sbatch/renderer and any future wiring
# agree on the contract. (They are additive env cfg keys; unset => the vq2_like preset defaults.)
COURSE_SAMPLER_KEYS = ("course_n_gates", "course_seg_len_lo", "course_seg_len_hi")


def render_overrides(stage: str, prefix: str = "+env.") -> list[str]:
    """Render a stage's overrides as hydra CLI tokens (e.g. '+env.emul_blackout_range_m=4.3'). The
    ``prefix`` is applied to every key (all VQ2 stage knobs live under env.*). Raises on an unknown
    stage so a typo in the sbatch fails loudly."""
    if stage not in STAGES:
        raise ValueError(f"unknown curriculum stage {stage!r}; valid: {sorted(STAGES)}")
    toks = []
    for k, v in STAGES[stage].items():
        toks.append(f"{prefix}{k}={v}")
    return toks


if __name__ == "__main__":
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "multi_gate"
    print(" ".join(render_overrides(stage)))
