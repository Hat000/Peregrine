# VQ2 RL Training Audit — 2026-07-05

36-agent multi-angle audit of the VQ2 second-controller training stack (reward wiring, optimization
dynamics, curriculum, sim-to-deploy fidelity, DiffAero internals, literature), every critical/high
finding independently adversarially verified. Verdicts: 13 CONFIRMED, 16 PARTIAL (confirmed with
corrections), 51 plausible-unverified.

HEADLINE (A0, since PROVEN OFFLINE by tests/test_vq2_audit_fixes.py): the emulated camera pointed
BACKWARD relative to the tail-first training flight convention — every pre-fix curriculum stage
trained perception-blind. The B1 fix package landed in commit 35bb820.

Post-audit corrections from the stack author (2026-07-05): logstd reset retuned to intermediate
std~0.18 (not full fresh 0.223); training IMU noise to be verified against real HIGHRES_IMU logs
(suspicion: emulated IMU too clean — why truth-init dead-reckoning coasted so well); map-noise DR
stays provisional pending GT validation of this session'''s reconstruction.

---

# VQ2 RACER TRAINING — SYNTHESIS ACTION PLAN
(Synthesized from adversarially-verified audit findings; every item cites its source finding. Verdict tags: [C]=CONFIRMED, [P]=PARTIAL-corrected, [PL]=PLAUSIBLE.)

---

## A. CAUSAL DIAGNOSIS OF THE multi_gate COLLAPSE

One coherent mechanism, five layers, ranked by causal role:

**A0. Pre-existing invalidation (deepest, ranked first because it re-scopes everything): the curriculum never trained the target skill.** Vision never engaged in ANY stage — the KF is truth-initialized (pos+vel) each episode with weak white noise, so dead-reckoning ≈ truth over 1-gate courses, AND the emulated camera is mounted on training-body +x while tail-first spawns/deploy convention put flight along −x, so the gate sits BEHIND the emulated camera (lockband_pointing 0.00000 every row; 99% success at 0.3% pointing_rate is impossible with a flight-direction camera) [P-vision-dead, corrected — mount inversion effectively certain]. Consequence: R5', fix_bonus, look-at, and the confidence triple were structurally inert all ladder long; the 99.3/99.5/99.8% stage certificates certify blind dead-reckoning plus spawn-mix inflation, not perception-based racing. This also explains why the analytical look-at yaw sign was "empirically wrong."

**A1. Trigger: compound stage-boundary shock → critic explosion (steps 0–100).** multi_gate simultaneously introduced ≥6 novelties: 6 gates, first-ever turns (±35°), first-ever climbs, contention stall, return-scale jump, and obs[13:17] flipping from identically-zero (6000 updates of exactly-zero gradient → frozen random-init first-layer columns) to raw 24–38 m vectors with no obs normalization [C-obs-zero ×2, C-difficulty-cliff, C-curriculum-discontinuity]. The warm-started critic (fresh Adam moments, shared lr 2.6e-3, value_weight=2, critic_grad_norm=null → NEVER clipped, raw-space value clip 0.2 vs O(100–700) errors, self-bootstrapped targets) hit this and diverged: value_loss 617→269,156 at step 100 [C-unbounded-critic]. **Adjudication vs the "clip-licensed drift" and "never heals" claims:** refuted by the trace — value_loss recovered to ~616 by step 200, so the clip is a soft throttle and the spike is OOD extrapolation + regime change, not accumulated drift [P-value-clip, corrected]. With l_rollout=16, advantages were bootstrap-dominated critic garbage; norm_adv preserved the wrong signs; the actor executed ~200 updates of confident noise with no KL guard → success 16.3%→1–2% by step 300 [P-critic-shock, corrected].

**A2. Direction of the failed optimum: the reward landscape ranks quitting above racing.** With progress zeroed and R1' provably inert on random courses [C-no-dense-drive, P-no-along-track ×3], the only along-track dense pay is R5' (≤0.03/step in-band, ~0.003 beyond 28 m) while the UNCLAMPED GT anchor drains −2·err_ip every step (−0.2…−10/step; blind 40 s episode ≈ −9,600 expected) — at γ=0.99 even a WELL-TRACKING env's continuation value (≈−23) loses to an immediate miss (−15) [C-GT-anchor, corrected: terminate-fast dominates racing throughout multi_gate, not just when lost]. through_centering is cross-track only. PPO found exactly the ordering the reward specifies: instant-finish > trivial-pass > deliberate-miss > hover > racing [PL-termination-economics]. **Adjudication:** the reward hole alone does NOT explain the collapse (the observed 1–2% falls below the reward-rational floor of ~12%+ artifact finishes [P-no-along-track corrected]) — it explains why the destroyed policy had no gradient back and why even a healthy policy would cap at ~0.5 gates.

**A3. Irreversibility: exploration budget exhausted before the only hard stage.** actor_logstd is carried through every weights-only warm-start; three near-saturated easy stages (fixed 2000 updates each, no early stop) ground per-dim std 0.223→0.050 (4.4× deficit) before multi_gate, and post-collapse it fell to ~0.008 (entropy pinned −13.676 ≈ logstd −4.84/dim — NEAR, not AT, the −5 floor; all-at-floor would read −14.324) [C-entropy-budget; P-entropy ×4, corrected]. **Adjudication:** the floor is not a mathematically absorbing state (Adam is scale-invariant to the tanh Jacobian; the queued ew=0.03 job restarts from blackout_pass where the squash gradient is 82% of max) — but empirically the channel was impotent at 0.01, and NOTHING can rediscover 24–38 m gate-chaining at std 0.008. Entropy exhaustion is the *why-it-never-recovered*, not the *why-it-collapsed* (collapse completed by step 300 at std 0.03–0.05).

**A4. Concealment: every metric that should have caught this was blind.** 70% of resets spawn at rest 1 m up-course of a uniform-random gate → ≥11.7% of multi_gate episodes are instant finishes worth ~+69 incl. the finish_time bonus (which actively amplifies the artifact in the gradient); the 16.3% "peak" is mostly artifact; a genuine standing-start lap is arithmetically unfinishable in max_time=40 s at curriculum speeds (needs ≥3.5–5.5 m/s avg) — though note the finish bonus is ALSO killed by γ=0.99 discounting beyond ~5 s regardless of the clock [C-metric-contamination; P-max_time, corrected]. FLIGHTCHECK gates on max-over-run with near-vacuous OR thresholds satisfiable by the step-0 inherited value; the ladder warm-starts from the FINAL (most entropy-collapsed, or destroyed) checkpoint, not best — the sbatch "best checkpoint" comment is false [P-FLIGHTCHECK, corrected: blackout_pass's 0.998 headline was itself a length-censored step-0 artifact; the stage improved ZERO in 2000 updates and the gate couldn't see it].

**Causal chain in one sentence:** a curriculum that never engaged perception (A0) and never taught gate handoff walked a near-deterministic (A3) policy into a six-axis distribution shock (A1) whose critic explosion destroyed the transferred behavior, whereupon a reward landscape that pays quitting over racing (A2) defined the new optimum, exploration exhaustion made it permanent (A3), and the metrics reported FLEW (A4).

---

## B. IMMEDIATE PREP (no GPU; queue-blocked window)

Grouped by priority. Files under `C:/Users/Fengy/Downloads/Projects/Anduril/rl/` unless noted.

**B1 — Retrain blockers (must land before ANY retrain):**
1. **Camera-frame fix** [P-vision-dead]: in `peregrine_racing_inc8.py:332-333` (fix-geometry path) and the look-at mount (`inc8_reward.py:180-186`), apply the fly_rl virtual π body-z flip so the emulated camera faces the flight direction under tail-first spawns; add a per-run logged `t_cam_z_pos_frac` KPI. Keep truth-R for fix content generation.
2. **GT anchor clamp** [C-GT-anchor]: `inc8_reward.py:91-95` → `−rw_estimerr·min(err_ip, 0.5)` (caps drain at −1/step; episode integral can no longer exceed terminal penalties). Pending Fengyou sign-off (Q1) — if he wants the anti-damping lever intact, gate it to the fixable band instead.
3. **Dense along-track progress on random courses** [C/P-no-drive]: `peregrine_racing_inc8.py:413/429` — when `self._random_course`, pay `w.progress·(prev_d2g − curr_d2g)` SIGNED (do not reuse R5's clamp(·,0) — retreat must cost [PL-R5-one-sided]); recheck the pass-bonus ≥2.5× sizing invariant (`peregrine_racing.py:103`).
4. **logstd reset at every warm-start** [C-entropy-budget]: `inc8_warmstart.py` `maybe_warmstart`, after `agent.load`: `actor_logstd.data.zero_()` (std 0.22) or solve for std≈0.15. One line; decouples ladder length from entropy budget.
5. **Critic protections** [C-unbounded-critic, P-critic-shock]: hydra `algo.critic_grad_norm=1.0` all stages; critic-only warmup ~100 updates at every warm-started stage entry (zero actor grads inside the existing `GuardedAPPO` optim.step wrapper in `peregrine_train_inc8.py` — no diffaero edit); `algo.clip_value_loss=false` for multi-gate stages (raw-unit 0.2 clip is meaningless at value scale O(50–300)).

**B2 — Curriculum/metrics restructure (`vq2_curriculum.py`, `peregrine_vq2_curriculum.sbatch`, `inc8_tb_trace.py`):**
6. Insert rungs: **R1 = 2 gates, no stall** (activates obs[13:17], one turn, one climb class), **R2 = 6 gates no stall**, **R3 = full**; delete or fix the hover stage (its seg_len override is dead code at G=1 — with the logstd reset, starting fresh at single_gate costs nothing [PL-hover-not-hover]); narrow first multi-gate seg_len to (23.7, 28) then widen [C-handoff-dead-zone].
7. FLIGHTCHECK → END-WINDOW mean (last 200 updates), per-stage thresholds, plus perception KPIs (lockband_pointing > 0.5, in-band fix_rate > 0.3) so blind policies can't climb [P-FLIGHTCHECK, P-vision-dead]; add success/npg SPLIT by spawn class (`standing` mask → stats_raw).
8. Chunked adaptive budgets: 250-update chunks with `+init_from` chaining, advance on plateau (>95% for 200 updates), cap easy stages ~600 [C-entropy-budget, PL-fixed-budgets].
9. Warm-start from a selected snapshot, not final: enable `inc8_snapshots.py` hook (`peregrine_train_inc8.py:205`), pick earliest-within-1%-of-max (higher entropy, same competence) [PL-final-ckpt]; fix best-save modulus bug in the wrapper [PL-trainer-footguns].
10. multi_gate `max_time` → 90–120 s or scale with sampled course length; score/gate on standing-start npg [C-metric-contamination]. Spawn mix: single-gate stages standing_start_frac→0.6–1.0; multi-gate replace 1 m rest-spawns with mid-segment MOVING spawns (U(2,10) m or mid-leg, v 0–2 m/s down-course) [PL-spawn-distribution ×2].
11. `emul_tau_stale=0.5` in ALL stages (kills the mid-ladder obs[19] semantics shift) [PL-tau-stale]; lookat warmup force-off made conditional on source-stage gains matching (sidecar records gains) [PL-lookat ×3].

**B3 — Conditioning + trainer hygiene:**
12. Fixed per-channel obs scale table (pos/next-gate ÷20, vel ÷8, angles/rates ÷π, collective ÷3.765; constants mirrored in fly_rl, cfg-gated, sidecar-tagged) [C-obs-zero, P-obs-norm — severity medium under Adam, but it also defuses the obs[13:17] shock]; zero next-gate block at the last gate (phantom fix) [PL-phantom].
13. GuardedPPO/APPO `bootstrap()` override: cut the GAE λ-chain at truncations with (1−reset) while keeping the (1−terminated) delta mask (~6 lines) [PL-GAE-leak ×2].
14. Entropy-pinned-at-floor abort (>100 updates flat → kill run); log critic input_dim=37 assertion in the sidecar [C-entropy-budget, PL-verify-appo].
15. Fidelity one-liners (all default-off/byte-id): `emul_fix_rate_hz` knob (p_accept × min(1, rate·dt)); KF init offset draw consistent with P; per-episode accel-bias DR (±0.2 m/s²); yaw-bias+drift Rz(δψ) on obs-R and KF-predict-R only; 2-state Markov stall replacing i.i.d. p=0.01 (age GROWS by dt, delete pin-to-clamp); decorrelate lateral/vertical fix bias (vertical up to ~0.25 m to cover ε_vert); per-episode map noise on obs[13:17] shadow copy only [C-fix-rate, C-yaw-truth, PL-KF-truth-init, PL-stall-model, PL-map-noise].
16. Deploy-contract fix: write `tau_stale`/`sigma_ref`/obs-scale-tag/lookat-gains into the actor.json sidecar; `src/racer/estimator_obs.py` consumes sidecar tau; update `tests/test_deploy_obs20.py` to pin sidecar==deploy [PL-deploy-tau — guaranteed field bug otherwise].

**Adjudicated NON-changes:** keep lr=2.6e-3 globally (upstream default, proven on inc5/inc7 at identical scale — the "9× too high" comparison used small-batch baselines [P-lr, corrected]) but add ~50-update lr warmup after each warm-start; keep γ=0.99 for 1–2-gate stages, raise to 0.995 only for 6-gate stages [PL-gamma-fine + PL-discount-horizon adjudication: dense progress restoration is the primary fix, γ the secondary]; do NOT build per-env reference lines or RewindKF now (delta_gate progress + OOSM-covariance-inflation are the stopgaps) [PL-rewind].

---

## C. DIAGNOSTIC LADDER (when GPU frees; ~30 min per 2000-update stage)

Designed so early rungs disambiguate trigger (critic shock) vs recovery-blocker (entropy) vs limiter (reward desert), then validate the perception fix before committing to the full retrain.

**Rung 0 — job 3295803 (ew=0.03), already queued, cost 0.** Treat strictly as a CONTROL. Prediction: little recovery (starting std 0.05 + intact critic shock + intact reward desert). It cannot disambiguate causes [P-critic-shock corrected]. If it unexpectedly recovers → entropy pressure was near-sufficient; still adopt B1 fixes.

**Rung 1 — mechanical fixes only (~15 min, 1000 updates).** multi_gate from the same blackout_pass ckpt; delta: logstd reset + critic_grad_norm=1 + critic-only warmup 100 + lr warmup. NO reward change. Confirms: value spike vanishes (<10× step-0) and success holds ≥ entry level through step 300 → A1/A3 mechanism confirmed. Expected plateau at ~0.5–1 gates (artifact + first-gate) → confirms A2 (reward desert) as the residual limiter. If it STILL collapses with a spike → something else at the boundary (suspect obs[13:17] shock alone; go to Rung 1b = same + obs scale table).

**Rung 2 — + reward economics (~30 min).** Rung 1 + GT clamp + signed delta_gate progress + γ=0.995 + max_time 100 s. Confirms A2 if standing-start npg climbs past 1 and keeps rising; refutes if plateau persists (then suspect the handoff dead-zone/lookat_r_hi — widen to 38.5 and re-run, 15 min).

**Rung 3 — camera-frame validation (~10 min, 400 updates, can run whenever a slot opens; independent).** single_gate config with the mount fix, logging `t_cam_z_pos_frac`, pointing_rate, in-band fix_rate. Confirms mount inversion if pointing_rate jumps 0.003→>0.3 and fixes flow. This is the gate on all perception shaping; if pointing does NOT jump, the inversion hypothesis is wrong and the vision-dead cause is purely the truth-init KF (then Rung 3b: honest KF init/drift arm — blind policies should now FAIL, success drop is the desired signal).

**Rung 4 — full new ladder, production candidate (~2–3 h with early stops).** single_gate(fresh) → blackout(2-gate) → 6-gate-no-stall → full, all B1–B3 fixes, chunked budgets, end-window gates with perception KPIs. Pass = standing-start lap success > 0 with n_passed ≥ 3–4 and entropy > −9 at end of the hard stage.

**Rung 5 — fidelity arms (~1 h).** Two arms of the Rung-4 winner's final stage: `emul_fix_rate_hz=7` (flyable-NOW checkpoint) vs 30 (post-A20 target), + yaw-drift ON. Compares degradation; the 7 Hz arm is the first sim-to-deploy candidate [C-fix-rate].

---

## D. TRAINING-RECIPE CHANGES TO ADOPT REGARDLESS

All from B, restated as standing policy with rationale:
1. **logstd reset at every warm-start** — transfer the mean, never the exploitation schedule; makes ladder length entropy-neutral [C].
2. **critic_grad_norm=1.0 + critic-only warmup + lr warmup at every stage entry** — every warm-start is a distribution shift onto fresh Adam; protect the window every time [C/P].
3. **clip_value_loss=false (or return normalization) for multi-gate-scale stages** — raw-unit 0.2 clip is a silent value-learning throttle at these scales; also nullifies the privileged-critic advantage [P, PL-verify-appo].
4. **Never advance on max-over-run metrics; never warm-start from the final checkpoint** — end-window means, per-stage thresholds incl. perception KPIs, snapshot selection [P-FLIGHTCHECK].
5. **Spawn-class-split metrics always on** — no headline number that mixes 1 m drift-throughs with standing laps again [C-metric-contamination].
6. **Fixed obs scale table + sidecar-versioned obs contract (tau_stale, scale tag, lookat gains)** — deploy determinism, kills silent contract breaks [PL-deploy-tau].
7. **Adaptive stage budgets (plateau early-stop)** — post-convergence updates only buy entropy destruction [C-entropy-budget].
8. **Every stage keeps all channels live** — no identically-zero obs blocks (2-gate minimum or virtual next-gate); no reward term slammed on at full gain mid-ladder (honor warmups when gains change) [C-obs-zero, PL-lookat].
9. **Automatic aborts**: entropy pinned >100 updates; value_loss >100× baseline for >50 updates.
10. **Consider Rudin-style per-env difficulty mixing inside the hard stage** (per-env turn_rad/seg_len buckets promoted on success; n_gates stays fixed at 6 — per-env G unsupported by the sampler) — smooths the return distribution by construction [C-literature; caveat from verification: geometry knobs cheap, G is not].

---

## E. FIDELITY UPGRADES, RANKED (transfer-risk-reduction ÷ effort)

1. **Camera mount/π-flip fix** — currently the ENTIRE perception loop trains inverted; every perception-shaping term is fiction until this lands. Effort: small. [P-vision-dead]
2. **`emul_fix_rate_hz` knob + 7 Hz arm** — 2.5–4× optimistic fix-supply gap, in the unsafe direction; one line. [C-fix-rate]
3. **Honest KF: init-offset draw + accel-bias DR** — removes the truth anchor that lets blind flight pass stages; a few lines. [PL-KF-init, PL-IMU-bias]
4. **Yaw-error model (bias + drift, vision-reset)** — no-mag vehicle; belief-frame shear exactly in the blackout where margin is 0.30 m; ~20 lines. Medium priority at slow lap, mandatory pre-speed-ramp. [C-yaw-truth]
5. **tau_stale sidecar→deploy plumbing** — zero training benefit, prevents a guaranteed silent field bug. Trivial. [PL-deploy-tau]
6. **Bursty 2-state Markov stall (age grows, no pin-to-clamp), coupled to latency mode** — fixes the temporal signature obs[19] exists to convey; ~10 lines. [PL-stall-model]
7. **Map noise on obs[13:17] + decorrelated/vertical-heavy fix bias** — sized from Q5; covers ε_vert≈0.215 and the perfect-map over-trust; one draw each. [PL-map-noise ×2]
8. **Append previous 4-dim action (or last 3 actions) to obs** — makes the 1–3-step latency DR identifiable; Swift does it; deploy-safe. Do at the next obs-contract rev, not mid-fix. [PL-Swift-delta, PL-latency-DR]
9. **Privileged critic enrichment (ω, last action, time-left, DR draws → ~49-d)** — critic-only, deploy-safe; bundle with a stage-reset since dims change. [PL-critic-input]
10. **Parked for the speed ramp**: motion-blur accept factor; OOSM covariance inflation (RewindKF approximation); Swift-style empirical-residual fine-tune stage; VQ2-build transport-latency remeasure (checkbox). [PL ×4]

---

## F. QUESTIONS FOR FENGYOU (deduped, decision-blocking first)

1. **GT anchor (rw_estimerr=2.0)**: deliberately ON for the slow arms, or inherited? Sign-off to clamp at 0.5 m / band-gate — it currently makes quitting optimal [C-GT-anchor].
2. **Race structure & scoring**: gates per lap (track has stations 01–20 — is 6-gate training G right?), number of laps, lap-time limit, partial credit for gates passed vs finish-time-only? Sets max_time, G, and whether finish_time should become per-gate split bonuses.
3. **Target slow-lap speed band** for the first deployable checkpoint (2–3 vs 4–5 m/s)? Pins max_time and the γ=0.995-vs-0.997 choice.
4. **Deploy start condition**: always standing GO ~20–26 m from gate 0, or moving handoff from Policy 1? Decides whether standing_start_frac anneals →1.0 in the final stage.
5. **Policy-1 surveyed-map error stats**: per-gate σ (lateral/vertical/along-track breakdown), common-mode vs independent per gate? Sizes the obs[13:17] map-noise DR (current guess 0.15–0.4 m, vertical worst).
6. **First flown checkpoint cadence**: assume current ~7 Hz or post-A20 25–30 Hz? Picks the emul_fix_rate_hz arm to prioritize.
7. **Deploy obs[0:3] source**: does the current-gate world position come from the surveyed map (map error leaks into gate-relative obs) or pure gate-relative vision?
8. **tau_stale contract intent**: thread sidecar 0.5 into deploy estimator_obs, or retrain final policy at the frozen 0.10? (test_deploy_obs20 must change either way.)
9. **Real detector lower fix edge**: do real fixes extend below accept_rlo=12 m (to ~5–8 m)? If yes the trained blind zone is ~2× too long.
10. **Track spacing**: can real inter-gate spacing exceed 35 m (fix guard) / 30 m (look-at band)? Decides whether the >35 m acquisition dead zone is a skill to train or an artifact to narrow.
11. **Race clock**: is elapsed-time-since-GO available to Policy 2 (RACE_STATUS)? If yes, time-remaining can enter the ACTOR obs, not just the critic.
12. **Job 3295803**: keep as control (recommended if the queue slot is otherwise idle) or kill for Rung 1? Prediction on record: little recovery.
13. **Rate/thrust bounds**: is ±3.14 rad/s the intended deploy clamp on VQ2 (climb-gate pitch-over worth checking)?
14. **Deploy estimator start**: KF/VIO initialized at known pad pose at GO (truth-init faithful at t=0)? And measured VIO drift rate without gate fixes — calibrates the honest-KF DR.
15. **Historical**: earlier inc8 pilots reportedly DID drive pointing_rate high — what config differed? (Would independently confirm when/where the camera-frame inversion entered.)

---

**Single-paragraph bottom line:** the multi_gate run died of a protected-window failure (critic shock at a six-axis compound boundary) made permanent by an exhausted exploration budget and made *rational* by a reward that pays quitting over racing — but the deeper problem is that the ladder never trained the actual task: perception was structurally inert (truth-init KF + inverted camera mount) and the metrics were spawn-contaminated, so even the "successful" stages certified the wrong skill. Land B1 (camera flip, GT clamp, signed progress, logstd reset, critic protections) before any GPU is spent; run Rungs 1–3 to confirm the mechanism cheaply; then rebuild the ladder per B2 and fly the 7 Hz-arm checkpoint from Rung 5.