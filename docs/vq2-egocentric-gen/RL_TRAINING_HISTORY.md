# VQ2 RL Training — Full History & Regression Map (as of 2026-07-07)

*Purpose (Fengyou's ask): map every VQ2 RL training attempt since the start — what was tried, how it
failed, what we identified as the bug, and what we did to fix it — so the "why did training regress"
question can be read from the whole arc, not one snapshot.*

**Reliability note.** Rows marked ✅direct = observed first-hand this session (job IDs, TB metrics).
Rows marked 📖recorded = from the memory bank / audit reports written at the time — and this session
proved several of *those* verdicts were themselves inflated or artifact-driven, so treat pre-session
"success" numbers with suspicion. The single most important lesson below is exactly that: **almost every
"success" or "regression" we logged turned out to be a measurement problem, not a policy fact.**

---

## The one-paragraph answer to "why did we regress?"

There isn't one regression — there are three things people have called "regression," and only one was a
true policy regression:

1. **A real regression (gate-frame):** `curr_a → curr_b`, single_gate 0.82 → 0.035, caused by
   over-applying a hard-stage exploration-noise knob to the easy discovery stage. Fixed by reverting
   (B2b). *A knob-scoping mistake.*
2. **The architectural reset (the big one):** we *pivoted* from the working gate-frame policy (inc7,
   live-confirmed, threads gates) to a from-scratch **egocentric position-free** policy (inc9) that has
   **never threaded a gate.** This is the "regression" that matters — a deliberate reset for
   drift-immunity that has been fighting structural bugs ever since.
3. **An illusory regression (this session):** "the fixed policy now crashes into the floor 100%." That
   was a **broken diagnostic render**, not a real regression — the *training* metrics showed the fix
   *improved* behavior.

Recurring root-cause theme across ALL of it: **we keep fighting observability, not just the policy** —
a backward camera (twice), inflated success metrics (twice), and unreliable renders. Every genuine step
forward came from *building a reliable measurement first*.

---

## ERA 0 — Gate-frame VQ2 "second controller" (2026-07-04 → 07-06)

Env built `3fd9396` (2026-07-04): `vq2_like` curriculum + faithful vision-lag + APPO privileged critic.
Obs = gate-frame, launders through a drifting world-NED KF. Ladder: single_gate → blackout → dual → multi.

| Attempt | Result | Bug identified | Fix |
|---|---|---|---|
| **curr_a_s0c** (job 3295856, ladder v1) 📖 | All 4 stages "flew" (single 0.82) — but **CONTAMINATED** | **A0: emulated camera pointed BACKWARD** — training flies TAIL-FIRST, camera sat on the nose = backward → every stage trained perception-**BLIND** (passed via dead-reckoning + near-spawn drift-through). Plus multi_gate collapse chain: stage-boundary shock, unbounded-critic explosion (value_loss 617→269k), reward paid quitting over racing, exploration spent, metrics blind. | **B1 package** `35bb820`: `camera_flip` + tail-mount + look-at +3/+3; GT-anchor clamp 0.5 + dense signed `gate_progress=10`; warm-start logstd reset→0.18; critic grad-norm 1.0 + 100-update warmup; ladder rebuilt. |
| **B2 diagnosis** (33-agent workflow) 📖 | Exposed curr_a's numbers as fake | **SPAWN LOTTERY**: 70% of resets spawn 1 m before a random gate (incl. the last) → the "0.82" was drift-through inflation; genuine standing-start 2-gate ≈ **5–15%**. **ENTROPY SIGN INVERSION**: `entropy_loss = −H`, so the "deterministic" read was backwards — it was noise **RUNAWAY** (bang-bang, 9–11 m/s crashes). | B2 must-do: spawn fix (exclude last gate), per-spawn-class metrics, noise-anneal ceiling, economy retune, handoff_drill rung. |
| **curr_b** (job 3296751, B2 fixes) 📖 | **FAILED — single_gate REGRESSED 0.82 → 0.035** (even the trivial 1 m dash → 4.3%) | **Over-applied the two HARD-STAGE knobs (noise_anneal std_hold=0.35 + economy) to the EASY stage via `_COMMON`.** Isolation probes: the **std_hold=0.35 ceiling was TOO TIGHT → starved the discovery exploration** single_gate needs (curr_a hit 0.64 by upd 400 on *wider* noise). Anneal = dominant cause, economy secondary. | **B2b** `ca8fdf9`: reverted noise_anneal + economy to curr_a dynamics; KEPT structural wins (spawn fix, metrics, FIX-B, handoff_drill). **LESSON: validate new knobs on the EASY stage before baking into `_COMMON`.** |
| **curr_b2** (job 3296780, B2b) 📖 | Ran; superseded by the pivot before a full verdict | — (the dual-gate handoff *frame-teleport* was the standing weakness) | — |

**Why we left ERA 0:** the gate-frame obl laundered through a drifting world-NED KF and the dual-gate
handoff produced a frame-teleport. inc7 (this generation, VQ1) is **live-confirmed and threads gates**
— it remains the only working policy — but its drift weakness motivated the pivot.

---

## ERA 1 — Egocentric position-free (inc9), the PIVOT (2026-07-07)

Committed `668347f`: 21-dim position-free obs (body velocity + attitude + rates + 2-gate rel_pos slider),
16-dim privileged critic (APPO), "refined-B" reward, keypoint-visibility model. Ladder: single_gate →
handoff_drill → dual_gate_full → multi_gate.

| Attempt | Result | Bug identified | Fix |
|---|---|---|---|
| **ego_a ladder** (job 3297042) 📖 | single_gate trained clean but **LEARNED NOTHING** (success 0.003) | **THREE independent fatal bugs** (5-agent workflow + probes + A/B): **RC1 — emulated camera pointed BACKWARD *again*** (the ego visibility path never got the flip → `gate_detectable`=0% → obs BLIND). **RC2 — noise RUNAWAY** (std=7.39 bang-bang; the noise-ceiling brake wasn't wired). **RC3 — approach was −EV** (finish 20 vs terminal 200 → approach-then-miss nets −base). | `5a252a4`: RC1 `_cam_R_wb` virtual π-flip (0%→100% detectable). RC2 wire the noise ceiling, held std=0.30. RC3 `rw_progress` 1→6, miss/oob 200→30. **⚠️ Chose config-4b (forfeit-on-ALL terminals) over 4a (contact-only)** to avoid an "approach-and-miss farming" optimum. **← THIS CHOICE WAS THE FORFEIT BUG (see ERA 2).** |
| **ego_b** (miss/oob=30) 📖 | **96% OOB** | Both miss and oob cheap → no pressure to stay in the arena | `3566799`: OOB back to 200, miss stays 30 |
| **ego_c** (progress=6, miss=30, oob=200) 📖 | OOB fixed (8%) but **90% MISS** | *Hypothesized:* refined-B dropped inc7's dense centering pull → wide flyby → "aperture centering" is the last blocker | *(left for Fengyou — became the starting point of this session; the "aperture centering" framing later proved to be a **render artifact**, not the real issue)* |

---

## ERA 2 — This session (2026-07-07, the diagnostic breakthrough) ✅direct

Starting from the "aperture centering" hypothesis. All runs single_gate unless noted.

| Attempt | Job | Result (reliable = training metrics) | Read |
|---|---|---|---|
| reward redesign (area-dist coupling, spawn-heading pinned, miss 30/oob 200) | 3297209 | FAILED 0.0024; render 93% wide miss | dive solved, now "wide miss" |
| **+ dense centering penalty 0.15** | 3297232 (ego_ctr) | FAILED 0.0005 | render showed 72% OOB — centering *magnitude penalty* triggered "give-up-and-leave" |
| centering + **non-terminating miss** | 3297250 (ego_ctr_nt) | WORSE — reward never climbs | discard |
| **progress → 3D distance-to-center** (Swift/inc7 homing) | 3297559 (p2c) | FAILED 0.0 | reward flat; render (unreliable) "floor-dive", training = 100% collision |
| p2c + area-off + finish_time cut | 3297560 (p2c_hw) | FAILED, reward less-negative | area-coupling was a headwind |
| **FORFEIT FIX** (contact-only) + area-off + miss 30→8 | 3297564 (p2cfix, 4000 upd) | FAILED 0% thread — **but IMPROVED**: banked 2→6, prog −→+, **crosses gate plane 68%**, collides 28%, oob 87%→5.6% | the fix *worked directionally* |
| **static gate + box-exit metrics** | 3297576 (sgs) | *RUNNING* — the reliable geometry probe | pending |

### The bug this session found — and it had been there since RC3
A **6-lens reward-shaping audit workflow** (adversarially verified) found the real root cause of the
**universal 0%**: `ego_reward.py:terminal_penalty` forfeited the banked progress on **miss/OOB, not just
contact** (`fired=(coll+miss+ob)`). Because `banked` is the *same* dense progress already streamed into
the return, the forfeit **exactly cancels the homing gradient on ~100% of (single_gate=0%) rollouts**,
AND makes "approach then loiter/miss" out-earn committing. **This is the config-4b choice made at RC3** —
adopted deliberately to prevent an "approach-and-miss farming" optimum, but it was itself the fatal bug.
The audit proved the farming worry was overblown and the gradient-cancellation was lethal.
**Fix:** forfeit contact-only (a frame clip is still contact, so anti-sprint-and-clip survives) + area
coupling off + miss base 30→8. Training then improved (p2cfix above).

### Two methodology bugs this session also exposed
- 🚩 **The render tool (`scratchpad/ego_render_rollout.py`) is UNTRUSTWORTHY** — it disagrees with the
  training metrics for BOTH p2c and p2cfix, in *opposite* directions. It runs the deterministic mean
  without the held-0.30 exploration noise the policy trains under, plus tool-faithfulness bugs. **Nearly
  every confusing flip this session (floor-dive, "vertical miss", degenerate-mean, 5 m offsets) came
  from trusting it.** → Diagnose from *training* metrics.
- 🚩 **Held exploration noise (std 0.30, never decays) is a structural cap.** The policy optimizes
  "mean + 0.30 noise", so for a precision task the *deterministic* mean is never pressured to sharpen
  onto a threadable trajectory. The deployable policy **can't become good without an endgame
  noise-anneal.** (This is also why the deterministic render looks like garbage mid-training.)
- **All prior ego "successes" (0.0005–0.0024) were NOISE** (~1 env of 2048 threading once); the ego has
  never had a good policy. inc7 (gate-frame) at ~0.7+ is real — a noise floor can dress up 0.0005, not 0.7.

### Built this session (the reliable testbed)
- In-training **box-exit classifier** (`metrics/exit_{thread,plane_miss,frame,floor,ceiling,side,back,
  front,timeout}` + `cross_offset_m`) — reads the *stochastic* rollouts, pure diagnostic.
- **`single_gate_static`** curriculum stage — fixed 15 m level dead-ahead gate, doesn't move; isolates
  the gross control failure. (job sgs, running.)

### Asymmetry finding (workflow, code-grounded) — banked for the endgame, not the current blocker
Vertical IS genuinely harder than lateral (refutes "the obs is symmetric"): gravity vertical-only + no
gravity compensation; thrust-vectoring makes forward/lateral tilt *steal* altitude (one-way); gate
`rel_pos` vision noise anisotropic (vertical σ 0.28 vs lateral 0.10, 2.7×). BUT all are **sub-meter**,
and the real failure is ~5 m off in *both* axes = a gross, axis-agnostic control failure. Fix the
asymmetries only once we're close.

---

## Patterns across the whole arc (the actual "why")

1. **Observability failures dominate.** Backward camera (ERA 0 A0 *and* ERA 1 RC1), spawn-lottery
   inflation, entropy-sign inversion, noise-floor false success, unreliable renders — over and over the
   *measurement* was wrong before the policy was even the question. Every real advance required building
   a trustworthy probe first (offline camera probe → per-spawn metrics → box-exit classifier).
2. **"Fixes" that were bugs.** RC3's forfeit-on-all (chosen to stop farming) cancelled the homing
   gradient. Knobs meant for hard stages (noise_anneal 0.35) crushed the easy stage. The lesson each
   time: **scope + validate on the easy stage before generalizing.**
3. **Inflated success masked reality.** curr_a's 0.82 (spawn lottery) and the ego's 0.0005 (noise) both
   read as "progress" and both were fake. We have never had a good *ego* policy.
4. **The pivot cost a working system.** inc7 threads gates; the ego reset to 0 and has spent its whole
   life on structural bugs. Whether the pivot pays off is still unproven.

## Where we are today

### FINAL DATA POINT — `single_gate_static` box-exit distribution (job sgs 3297576, 2026-07-07)
The reliable geometric read, on a fixed 15 m level dead-ahead gate:

| updates      | floor | ceiling | plane-miss | thread | x-offset @ gate plane |
|-------------:|------:|--------:|-----------:|-------:|----------------------:|
| 10 (flail)   |   0%  |   87%   |    12%     |   0%   |        9.5 m          |
| 170 → 1930   | **100%** |  0%  |     0%     |   0%   |      **9.70 m**       |

**100% FLOOR-DIVE, converged by update 170, dead-stable to the end** — one failure mode, total
domination. The `x-offset = 9.70 m` is the L-inf offset *at the moment the drone crosses the gate plane
(x=15 m)*: it flies the full 15 m forward (forward control is FINE) but is already ~9.7 m **below center**
at the plane, then floors. It flies a steep committed downward glide: ~15 m forward, ~10 m down.

**Diagnosis: the binding failure is ALTITUDE HOLD during transit, NOT aperture precision.** The drone
never gets close enough to thread — it loses the vertical fight and hits the ground first. This
*overturns* the "held-noise precision-wall" framing (real, but downstream — you cannot anneal noise out
of a floor-dive; altitude is UPSTREAM of precision).

**It is a control-LEARNING failure, not a reward pathology:** the reward *prefers* level flight —
`progress = −‖pos − gate_center‖`, so straight-and-level scores *more* progress than diving (dist 7.5 vs
12.3 for the same forward distance). The policy chooses a *worse-reward* behavior → it simply never
learns gravity compensation: mean action is sub-hover; forward pitch bleeds lift via thrust-vectoring
(`a_up·cos(tilt)`); the floor becomes a basin it falls into by update 170. inc7 flies this **same control
stack** and holds altitude → altitude-hold IS learnable here; the ego reward/obs just isn't teaching it.
(The earlier `p2cfix` "68% plane-cross / 28% collision" was a pre-classifier inference and is superseded
by this direct box-exit read.)

**Recommended lever (Fengyou's MPCC idea, geometry only): a dense CORRIDOR / CONTOURING penalty** on
perpendicular (esp. vertical) deviation from the straight line spawn→gate_center — manufactures the clean
vertical gradient a 3-D distance norm buries. Minimal = altitude-to-corridor term; principled = MPCC
contouring error (sets up turning stages; NO speed term yet). Optional cheap probe: a hover-hold stage.
inc7 remains the only gate-threading policy in the program.
