# VQ2 gate-handoff diagnosis — 2026-07-06

**Trigger (Fengyou):** dual_gate read as "~90% pass gate 1, only 32% complete gate 2" — demanded a
root-cause dive before the next training round.
**Method:** 33-agent workflow (8 finders → adversarial verification, 2 refuters per critical →
synthesis), run twice (first pass truncated by session limits; resume re-verified the starved
findings). Full structured findings + verifier notes: `findings.json` (16 confirmed).
Source run: job 3295856 `curr_a_s0c` (single 0.82 / blackout 0.60 / dual 0.32 / multi 0.115 — all
now known to be contaminated numbers, see RC1).

## Headline corrections (both invalidate earlier readings)

1. **SPAWN LOTTERY (RC1).** 70% of resets spawn 1 m before a uniformly-random gate INCLUDING the
   last (`peregrine_racing.py:814-827`, `standing_start_frac=0.3`). For G=2: 35% of episodes score
   "success" via a perception-free 1 m dash at a truth-cold-initialized KF. Genuine standing-start
   2-gate completion ≈ **5–15%** (verifier-corrected band; not the reported 32%). The
   "~90% pass gate 1" reading is arithmetically impossible under the spawn mix (n_passed 0.625 is
   the RAW mean). The flat 1000–2000 plateau = constant trivial-success floor, near-zero learnable
   signal on real handoffs. All stage verdicts carry this inflation.
2. **ENTROPY SIGN INVERSION.** `entropy_loss` logs **−H**. The banked "logstd consumed →
   deterministic" story was backwards: −12 ⇒ H=+12 ⇒ pre-tanh σ≈4.9 — **noise RUNAWAY** toward the
   e² clamp (bang-bang actions; explains mean_speed 9–11.5 m/s "crash speeds" in a slow-lap stage,
   degraded pointing, sampled-vs-mean policy gap). Triple-pinned arithmetically (warmup const
   +1.18344 = −H(0.18) exact). The earlier "exploration exhausted σ=0.012" finding is REFUTED;
   fix = the dormant `noise_anneal` **ceiling** (`inc8_noise_anneal.py`, wired at
   `peregrine_train_inc8.py:179`), NOT an entropy floor/sweep-up.

## Root causes (synthesis attribution)

| # | Cause | Share |
|---|-------|-------|
| 1 | Spawn lottery (metric corruption + gradient dilution) | ~100% of the misread; 10–15% of learning failure |
| 2 | Vertical-FOV structural blindness: camera +20° up, half-FOV 29.36°, drops to +12 m ⇒ ~42% of segments make gate 2 permanently invisible to a level drone (p_accept out-of-image 0.0002) | 30–40% of genuine failures |
| 3 | Entropy runaway (see above) — corrupts every stage verdict | 20–30% |
| 4 | Dead post-handoff reward economy: GT-anchor clamp 0.5 m < post-handoff err 0.68–0.75 m ⇒ saturated −1.0/step, zero reacquisition gradient; first recovered fix pays +0.0000; `rw_fix_bonus` OFF; net seg-2 continuation ~0 | 20–30% (owns plateau persistence) |
| 5 | Option-1 latency forward-fuse poisons KF in-plane on turns/climbs (fix from t−Δ fused as current, no v·Δ covariance) — offline-proven; grows with speed | 10–15% |
| 6 | Look-at pitch primitive injects 1.5–2.3 rad/s (48–73% of authority) in descent class, post-PPO-ratio | 5–10% |

**Exonerated / refuted:** the target-gate switch itself injects zero KF error; gate-frame velocity
rotation is decodable (not a flip); rhi 28.14 vs seg-max 28.0 is not a razor edge (post-handoff fix
drought <0.2 s); fix-cadence oversupply (2–4.3× vs 7–15 Hz real) is real but did not bind tonight
(policy realized 2.6 Hz — pointing-starved, not cadence-limited).
**Also confirmed:** `inc8_centering`≡0 is a metric-name collision (logs the OFF `rw_centering`, not
the configured `rw_through_centering`); terminal-step-only sampling in metrics/fix_rate+estim_err;
FLIGHTCHECK dual 0.64 was the step-0 restore artifact (sustained max 0.343); **deploy:**
`fly_rl.py:1035-1039` still switches on `active_gate_index` (~9 m early) vs the plane-crossing
convention — relay to vision commander before any RL-policy flight (fix exists:
`mission.py:161-187`); clean-map training is the top transfer risk once handoffs complete (defer,
ship WITH next-gate pre-acquisition).

## Next round (B2) — synthesis plan

**MUST-DO:** (1) spawn fix — near-spawns exclude the last gate when n_gates≥2
(`peregrine_racing.py:816` + pin test); (2) measurement stack — per-spawn-class success metrics,
FLIGHTCHECK skips step-0, deterministic stage-end eval gates advancement; (3) `+algo.noise_anneal=true`
per stage (ceiling ~0.35→~0.10 back-half; entropy_weight 0.01→0 back-half; REJECT noise floors);
(4) fix economy — `rw_estimerr` 2.0→1.0 (KEEP clamp 0.5) + `rw_fix_bonus=0.75`; (5) FIX-B latency
covariance inflation (`inc8_estimator_emul.py:722-724`, cov += outer(v̂Δ)+εI; offline-proven
never-worse; update parity pins); (6) **handoff_drill rung** — new `course_drop_lo/hi` sampler keys,
drop clamped [−2,+4] m (below the 3.9–4.6 m FOV exclusion cliff) inserted before full-drop dual.
Ladder: single → blackout → handoff_drill → dual_gate_full → multi.

**ARM (portfolio):** `lookat_max_rate=1.0` cap (default 0 = byte-id); `rw_centering=1.0`
warm-started (+ rename colliding scalar → `inc8_near_centering`); γ 0.995→0.9975 for dual/multi.
**DEFERRED (with triggers):** 7 Hz frame-cadence mask (trigger: genuine dual completion >0.5);
per-gate map error ~N(0,0.2 m) shipped together with next-gate pre-acquisition; rhi→30-31 +
in_image edge-margin softening (needs real pointed-data recal); accel-bias term (with map-error pair).

**Open questions / asks:** s_trivial direct measurement (first 50 updates post-instrumentation);
confirm TB n_passed normalization at the diffaero trainer; VISION asks — YOLO detection fraction on
center-point-excluded partial gates; pointed-data fix_surrogate recal at 24–32 m + sustained usable
Hz under race load; plane-crossing switch decision (fly_rl) before first RL flight.
