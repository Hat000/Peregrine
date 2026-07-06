"""rl/vq2_curriculum.py -- the VQ2 slow-is-smooth curriculum stages (spec T3.2).

The curriculum is expressed as a LADDER of config-override dicts (no new env machinery -- each stage is
a set of `+env.`/`+dynamics.` hydra overrides layered on the base VQ2 recipe). This mirrors the
DIFFICULTY_PRESETS pattern: pure DATA + a tiny helper that renders the hydra override strings the sbatch
consumes. Laptop-testable; imports nothing heavy.

2026-07-05 REBUILD (training-audit A0-A4; the 4-stage ladder's multi_gate collapse + the camera-mount
inversion). What changed and why:
  * emul_camera_flip=True EVERYWHERE: training flies TAIL-FIRST (spawn yaw = gate_yaw + pi) but the
    legacy emulated camera pointed at the NOSE = backward -- the gate sat BEHIND the camera the whole
    approach, so vision/pointing/fix economy were structurally inert in EVERY prior stage (audit A0).
    With the mount fixed the look-at gains return to the ANALYTIC signs (+3/+3): the old -3 yaw was
    an empirical compensation for the mirrored frame (see inc8_reward.lookat_correction docstring +
    the band_az "analytical sign was WRONG" note it left behind).
  * rw_gate_progress=10 EVERYWHERE: random courses have NO dense along-track drive (R1' is the exact
    zero without a global line) -- the 24-38.5 m inter-gate stretch was a reward desert where quitting
    beat racing (audit A2). Signed target-gate approach delta fills it.
  * rw_estimerr_clamp=0.5 EVERYWHERE: the unclamped GT anchor (-2*err_ip/step) made early termination
    reward-OPTIMAL on a lost estimator (audit A2).
  * emul_tau_stale=0.5 EVERYWHERE: the old 0.1->0.5 mid-ladder switch silently changed obs[19]'s
    semantics between stages (audit: curriculum discontinuity).
  * NEW dual_gate rung: obs[13:17] (next-gate block) is identically ZERO on every 1-gate stage, so its
    first-layer weights sat at random init until multi_gate slammed 24-38 m vectors through them (audit
    A1's biggest single shock). dual_gate activates the channel + the first gate HANDOFF + the first
    turn/climb at a narrowed segment length BEFORE the full course.
  * multi_gate: max_time 40->100 s (a genuine standing-start 6-gate lap was arithmetically UNFINISHABLE
    in 40 s at slow-lap speeds -- the finish bonus was dead, audit A4) + gamma 0.995 (30 Hz: 0.99 gives
    a ~2.3 s credit half-life vs 5-12 s between gates) + clip_value_loss off (a raw-unit 0.2 clip is
    meaningless at multi-gate return scales and throttles critic recovery).
  * hover REMOVED from the default ladder (kept as an optional stage): its only distinguishing override
    was dead at G=1, it duplicated single_gate, and with the logstd reset at every warm-start (see
    inc8_warmstart.reset_actor_logstd) a fresh single_gate start costs nothing.

The default ladder (each stage warm-starts from the previous stage's checkpoint; the sbatch passes
+warmstart_reset_logstd=true +critic_warmup_updates=100 so exploration and the critic survive every
boundary):

  1. SINGLE GATE, HEAD-ON, SLOW -- one gate, start beyond the blind zone, blackout OFF, camera FIXED,
     look-at ON at the analytic signs. Estimator-emulated obs (never perfect pose).
  2. SINGLE GATE + TERMINAL BLACKOUT -- drive THROUGH the <4.5 m blind zone (blackout ON at the
     measured ~4.3 m) + the full bimodal content-lag latency.
  3. DUAL GATE (NEW) -- 2 gates, narrowed segment (23.7-28 m): activates obs[13:17], the first gate
     handoff, the first turn/climb class. Blackout + latency ON, no stall yet.
  4. MULTI-GATE LAP -- the full random vq2_like course (6 gates), blackout + latency + contention
     stall, 100 s clock, gamma 0.995. Lap completion on estimator-emulated obs.

  B2 2026-07-06 (handoff diagnosis): handoff_drill (drop [-2,+4] m) inserted before dual_gate_full
  (= the flown dual_gate, full drop band). The flown dual_gate dict is FROZEN for reproducibility.
  B2b (2026-07-06): M3 noise_anneal + M4 fix economy REVERTED after the curr_b probe verdict (the
  0.35 ceiling starved discovery-phase exploration; economy misapplied outside the post-handoff
  regime) -- see the _COMMON note. STRUCTURAL wins kept: spawn fix, honest metrics, FIX-B, this
  ladder. noise-anneal redesign (looser ceiling) + economy are deferred to measured increments.
"""
from __future__ import annotations

# Knobs shared by EVERY stage (the audit-A0/A2 fixes -- keeping them in one place so a stage cannot
# silently drop one). Merged into each stage dict below; a stage may override.
_COMMON = {
    "course_mode": "random",
    "track_difficulty": "vq2_like",
    "emul_camera_flip": True,        # camera faces the tail-first flight direction (audit A0)
    "emul_tau_stale": 0.5,           # ONE age-channel semantics across the whole ladder
    "rw_gate_progress": 10.0,        # dense signed along-track drive on random courses (audit A2)
    "rw_estimerr_clamp": 0.5,        # GT anchor capped -- quitting can never beat racing (audit A2)
    "rw_through_centering": 10.0,    # the proven lateral restoring pull
    "lookat_g_yaw": 3.0,             # ANALYTIC signs (frame fixed; -3 was the mirror compensation)
    "lookat_g_pitch": 3.0,
    "lookat_warmup_updates": 200,
}

# B2b (2026-07-06): M3 noise_anneal + M4 fix economy REVERTED after the curr_b probe verdict.
# curr_b stopped at single_gate (0.82->0.035; even the 1 m near-spawn dash 4.3%). Isolation probes:
#   sgpA (anneal OFF, econ ON)  -> raw 0.79 / standing 0.39  (FLEW, ~= curr_a restored)
#   sgpB (anneal ON,  econ OFF) -> raw 0.20 @ upd 1120 climbing (vs curr_b ~0.001) -- econ also hurt
# => the 0.35 std ceiling starved the discovery-phase exploration single_gate needs (curr_a hit 0.64
# by upd 400 on wider noise), and M4's fix_bonus/anchor rebalance is misapplied outside the >0.5 m
# post-handoff regime it was designed for. Both are DEFERRED to their own measured increments on top
# of a completing baseline. STRUCTURAL B2 wins (M1 spawn fix, M2 honest metrics, M5 FIX-B, M6
# handoff_drill ladder) are KEPT. _NOISE_ANNEAL_RAW retained (unused) for the redesign: a LOOSER
# ceiling (~1.0-1.5, above healthy discovery noise, below the sigma~4.9 runaway) is the next attempt.
_NOISE_ANNEAL_RAW = {          # DEFERRED (unused): 0.35 too tight; redesign with a looser ceiling
    "+algo.noise_anneal": True,
    "+algo.noise_std_hold": 0.35,
    "+algo.noise_std_floor": 0.10,
}

# The ordered stage ladder. Values are hydra-override key->value rendered as `+env.<k>=<v>`, EXCEPT the
# special "_raw" sub-dict whose entries are emitted VERBATIM (for existing non-env keys: algo.gamma,
# env.max_time, algo.clip_value_loss -- these exist in the base config so they take no `+`).
STAGES: dict[str, dict] = {
    # OPTIONAL (not in STAGE_ORDER): the old hover rung, kept for one-off diagnostics only.
    "hover": {
        **_COMMON,
        "course_n_gates": 1,
        "course_seg_len_lo": 23.7, "course_seg_len_hi": 26.0,
        "emul_blackout_range_m": 0.0,
        "lookat_g_yaw": 0.0, "lookat_g_pitch": 0.0,
    },
    # 1. SINGLE GATE, HEAD-ON, SLOW: one gate, blackout OFF (stop before the blind zone), camera FIXED,
    #    look-at ON (analytic signs). The fresh-start stage.
    "single_gate": {
        **_COMMON,
        "course_n_gates": 1,
        "emul_blackout_range_m": 0.0,
    },
    # 2. SINGLE GATE + TERMINAL BLACKOUT: through the <4.5 m blind zone (~4.3 m measured) + the full
    #    bimodal content-lag latency (fix carries t-Delta geometry; age carries the same Delta).
    "blackout_pass": {
        **_COMMON,
        "course_n_gates": 1,
        "emul_blackout_range_m": 4.3,
        "emul_lat_max_s": 1.0, "emul_lat_healthy_frac": 0.5,
        "emul_lat_healthy_lo": 0.07, "emul_lat_healthy_hi": 0.12,
        "emul_lat_cont_lo": 0.15, "emul_lat_cont_hi": 0.55,
    },
    # 3-as-flown. FROZEN reproducibility pin (job 3295856 curr_a, B1 HEAD f63b4d9): the dict that
    # actually flew. LITERAL (not **_COMMON) so _COMMON drift (e.g. the B2 M4 economy change) can
    # never alter this render; NOT in STAGE_ORDER. Byte-identity pinned by tests/test_vq2_b2_ladder.py.
    "dual_gate": {
        "course_mode": "random",
        "track_difficulty": "vq2_like",
        "emul_camera_flip": True,
        "emul_tau_stale": 0.5,
        "rw_gate_progress": 10.0,
        "rw_estimerr_clamp": 0.5,
        "rw_through_centering": 10.0,
        "lookat_g_yaw": 3.0,
        "lookat_g_pitch": 3.0,
        "lookat_warmup_updates": 200,
        "course_n_gates": 2,
        "course_seg_len_lo": 23.7, "course_seg_len_hi": 28.0,
        "emul_blackout_range_m": 4.3,
        "emul_lat_max_s": 1.0, "emul_lat_healthy_frac": 0.5,
        "emul_lat_healthy_lo": 0.07, "emul_lat_healthy_hi": 0.12,
        "emul_lat_cont_lo": 0.15, "emul_lat_cont_hi": 0.55,
        "_raw": {"env.max_time": 60, "algo.gamma": 0.995, "algo.clip_value_loss": False},
    },
    # 3. HANDOFF_DRILL (B2, 2026-07-06 diagnosis M6): the first gate handoff with the DROP CLAMPED to
    #    [-2, +4] m (+down), i.e. inside the vertical-FOV visibility band -- the full vq2_like drop
    #    (-6..+12) makes ~42% of segments put gate 2 permanently outside a level drone's FOV (camera
    #    +20 deg up, half-FOV 29.36 deg), so the flown dual_gate mixed "learn the handoff" with
    #    "gate 2 is unseeable". Drill the handoff FIRST where the gate is always acquirable.
    "handoff_drill": {
        **_COMMON,
        "course_n_gates": 2,
        "course_seg_len_lo": 23.7, "course_seg_len_hi": 28.0,
        "course_drop_lo": -2.0, "course_drop_hi": 4.0,
        "emul_blackout_range_m": 4.3,
        "emul_lat_max_s": 1.0, "emul_lat_healthy_frac": 0.5,
        "emul_lat_healthy_lo": 0.07, "emul_lat_healthy_hi": 0.12,
        "emul_lat_cont_lo": 0.15, "emul_lat_cont_hi": 0.55,
        "_raw": {"env.max_time": 60, "algo.gamma": 0.995, "algo.clip_value_loss": False},
    },
    # 4. DUAL_GATE_FULL (B2): the flown dual_gate with the drop UNSET (full vq2_like -6..+12 band
    #    returns). Same narrowed segment band; blackout + latency ON.
    "dual_gate_full": {
        **_COMMON,
        "course_n_gates": 2,
        "course_seg_len_lo": 23.7, "course_seg_len_hi": 28.0,
        "emul_blackout_range_m": 4.3,
        "emul_lat_max_s": 1.0, "emul_lat_healthy_frac": 0.5,
        "emul_lat_healthy_lo": 0.07, "emul_lat_healthy_hi": 0.12,
        "emul_lat_cont_lo": 0.15, "emul_lat_cont_hi": 0.55,
        "_raw": {"env.max_time": 60, "algo.gamma": 0.995, "algo.clip_value_loss": False},
    },
    # 4. MULTI-GATE LAP: full 6-gate vq2_like course incl. the HIGH-climb gate + turns. Blackout +
    #    latency + contention stall. 100 s clock (a 40 s standing-start lap was unfinishable), gamma
    #    0.995 (credit across 5-12 s inter-gate spans at 30 Hz), value clip off.
    "multi_gate": {
        **_COMMON,
        "course_n_gates": 6,
        "emul_blackout_range_m": 4.3,
        "emul_lat_max_s": 1.0, "emul_lat_healthy_frac": 0.5,
        "emul_lat_healthy_lo": 0.07, "emul_lat_healthy_hi": 0.12,
        "emul_lat_cont_lo": 0.15, "emul_lat_cont_hi": 0.55,
        "emul_pose_age_stall_p": 0.01,
        "_raw": {"env.max_time": 100, "algo.gamma": 0.995, "algo.clip_value_loss": False},
    },
}

# B2 ladder (2026-07-06): handoff_drill (drop-clamped first handoff) BEFORE dual_gate_full (full
# vq2_like drop). The as-flown "dual_gate" stays in STAGES (frozen) but is NOT flown.
STAGE_ORDER = ("single_gate", "blackout_pass", "handoff_drill", "dual_gate_full", "multi_gate")

# Keys that are COURSE-SAMPLER overrides (forwarded to sample_courses via the env, NOT +env.<k>). The
# env maps course_n_gates -> the sampler's n_gates and course_seg_len_{lo,hi} -> seg_len_m; these live
# under the same +env. namespace but are documented here so the sbatch/renderer and any future wiring
# agree on the contract. (They are additive env cfg keys; unset => the vq2_like preset defaults.)
COURSE_SAMPLER_KEYS = ("course_n_gates", "course_seg_len_lo", "course_seg_len_hi",
                       "course_drop_lo", "course_drop_hi")


def render_overrides(stage: str, prefix: str = "+env.") -> list[str]:
    """Render a stage's overrides as hydra CLI tokens (e.g. '+env.emul_blackout_range_m=4.3'). The
    ``prefix`` is applied to every key EXCEPT the "_raw" sub-dict, whose entries are emitted verbatim
    (existing non-env keys like algo.gamma / env.max_time take no '+'). Raises on an unknown stage so
    a typo in the sbatch fails loudly."""
    if stage not in STAGES:
        raise ValueError(f"unknown curriculum stage {stage!r}; valid: {sorted(STAGES)}")
    toks = []
    for k, v in STAGES[stage].items():
        if k == "_raw":
            for rk, rv in v.items():
                toks.append(f"{rk}={rv}")
        else:
            toks.append(f"{prefix}{k}={v}")
    return toks


if __name__ == "__main__":
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "multi_gate"
    print(" ".join(render_overrides(stage)))
