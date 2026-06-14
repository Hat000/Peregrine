# d5 — INC8 RETRAIN SPEC (Adroit-ready, build-ready)

Agent **d5** (RL/inc8, opus-4.8). COMPONENT 5 of the gate-relative case-C VQ2 pipeline design.
DESIGN + OFFLINE only — no source edits, no SLURM, no live sim. All load-bearing numbers re-derived
from source or reproduced in `.venv` (see §0). This file is the build-ready inc8 spec the commander
hands to the training agent.

> **Scope discipline.** This spec defines the inc8 *training env changes* + the *Adroit launch matrix*
> + the *selection metric*. It CONSUMES component-1's gate-relative obs layout (the new confidence
> channel) and component-3's achieved per-fix σ (the speed-ladder gate). Where component-1/-3 had not
> yet written their artifacts at authoring time, this spec STATES the contract they must satisfy and
> flags it ASSUMED-PENDING-INTERFACE; the numeric DR/eval values are MEASURED and stand regardless.

---

## 0. Re-derived load-bearing numbers (verification log)

| # | claim | value | how verified | class |
|---|---|---|---|---|
| 0.1 | rel-arm gate-4 in-plane RMS (warm prior) | **0.139 m** (p50 0.117, **p90 0.203**) | RAN `c1_gate_relative.py` → reproduced exactly | MEASURED (sim) |
| 0.2 | gate-relative per-fix LATERAL per-axis σ | **0.265 m** (lat_rms 0.375) | `c1_gate_relative_results.json::part_b.per_axis_lat_sigma_m` + RAN | MEASURED |
| 0.3 | rel-arm removes map bias | E_bias **−0.000 m**, D_bias **−0.004 m** | RAN c1; abs arm E_bias +0.176; submap (anti-pattern) +0.174 | MEASURED |
| 0.4 | KF averages ~6–9 effective fixes; filtered ≈ per-fix/√N_eff (~/2.8) | N_eff 6.2 @30Hz/47% | READ `c2_findings.md` §2 (N_eff = (σ_perfix/σ_filt)²) | MEASURED |
| 0.5 | per-fix→max-valid-speed gate (0.155 m margin) | per-fix ≤0.10 m → margin held >55 m/s; 0.20 m → 17 m/s | READ `c5_findings.md` HEADLINE table | MEASURED (sim) |
| 0.6 | per-track world N-bias drops out gate-relative | +0.67 m systematic N-bias, removed exactly (`db` cancels) | READ `c1_findings.md` §(a)/(b); CONTEXT.md | MEASURED |
| 0.7 | latency staleness is 98.4% ALONG-TRACK at gate-4 | in-plane leak 5.2 mm edge / 41 mm CPU | READ `a4_findings.md` §2–3 | MEASURED |
| 0.8 | gate-4 contact-true margin | **0.155 m @ r=0.38**, SIMSTART; gate-5 clean 0.314 m | CONTEXT.md / `contact_true_eval.py` PASS_BAND | MEASURED |
| 0.9 | 17-dim obs layout + gate frame math | exact (see §1) | READ `rl/fly_rl.py::obs_from_zup` + `peregrine_racing.get_observations` | MEASURED (source) |
| 0.10 | reward contract R1–R7/T1–T4 + weights | exact (see §3) | READ `rl/peregrine_racing.py::RewardWeights`, `compute_reward_terms` | MEASURED (source) |

**Repro command (CONFIRMED working):**
`.venv\Scripts\python.exe handoff/ultracode-estimator-racespeed-2026-06-13/c1_gate_relative.py`
→ rel inplane_rms 0.139, per-fix lateral σ 0.265, E/D_bias −0.000/−0.004.

---

## 1. OBS — gate-relative + uncertainty-aware confidence channel

### 1.1 The existing 17-dim obs (FROZEN deployment contract, `peregrine_racing.get_observations` ==
`fly_rl.obs_from_zup`, bit-exact P4-C05):

```
[0:3]   pos_g = R_w2g @ (gate_pos - pos)   gate-relative position   <-- THE channel case-C estimates
[3:6]   vel_g = R_w2g @ vel                velocity, gate frame
[6:9]   rpy_g = ZYX(R_w2g @ R_b2w)         attitude vs gate
[9:12]  body_rates (FLU)
[12]    prev_normed_thrust (rescaled g-units)
[13:16] next_relpos (next-gate lookahead, gate (i-1) frame)
[16]    next_relyaw
```

**The unification crux (CONTEXT.md §P4-C05, confirmed in source).** In case A/B, `pos = p_KF_absolute`
and `gate_pos = gate_map_centre`. In **case C the policy must consume `pos_g` derived from the SEEN
gate** (PnP `−L` lever): the estimator delivers the gate-relative offset DIRECTLY. The training env must
populate `pos_g` (obs[0:3]) — and `vel_g`, `rpy_g` insofar as they ride the same pose — as
**ground-truth gate-relative offset + the MEASURED estimator residual error** (§2), NOT pristine truth.
This is what makes **deploy == train**. The anti-pattern `R_w2g @ (gate_map − p_KF_absolute)`
(re-injects the +0.67 m map bias; c1 submap arm E_bias +0.174 m) is FORBIDDEN — never reference
`gate_map` in the in-plane channel.

### 1.2 The new confidence channel (consumes component-1's obs layout)

Append **3 scalars** → **obs_dim = 20** (17 → 20). Layout (component-1 owns the exact wire; this is the
contract it must satisfy):

```
[17]  c_inplane = clip(sigma_ref / sigma_inplane_hat, 0, 1)   in-plane (E,D gate-frame) confidence
[18]  c_along   = clip(sigma_ref / sigma_along_hat,  0, 1)   along-track (N gate-frame) confidence
[19]  age_norm  = clip(t_since_last_accepted_fix / TAU_STALE, 0, 1)   staleness clock
```

Design rationale (each load-bearing):
- **Two confidence axes, not one.** c5/a4 prove the error is anisotropic at gate-4: in-plane (binding
  miss) and along-track (phase/timing, latency-dominated) decouple. A single scalar conflates the
  margin-critical axis with the benign one. `c_inplane` is the fast-vs-careful signal; `c_along` lets
  the policy hold/extend approach when phase is stale without braking laterally.
- **Confidence = ratio, bounded [0,1], NOT raw σ.** `sigma_ref` is a fixed normalizer (recommend
  `sigma_ref = 0.05 m`, the 1-σ bar) so `c=1` ⇔ "at-or-better-than-bar confident", `c→0` ⇔ uncertain.
  Bounded + sign-stable → stable PPO input; raw σ has no natural scale and explodes the obs-normalizer.
- **`sigma_*_hat` is the CALIBRATED KF covariance** (component-1/RewindKF supply `sqrt(diag(P))` rotated
  into the gate frame), NOT the rejected logstd-as-uncertainty. **Calibration is mandatory:** c2/c5
  measured the raw absolute KF as overconfident (NEES 4–62) because it tracks bias as truth. On the
  gate-relative path the bias is removed (NEES 1.96, c1) so P is honest — but component-1 must DELIVER a
  NEES≈3 (chi²(3)) calibrated P, else the confidence channel lies. **This is the binding interface
  requirement on component-1.** (ASSUMED-PENDING-INTERFACE: exact P→gate-frame rotation + calibration
  factor.)
- **`age_norm` is the staleness clock.** Already half-present in `navigator.py:426`
  (`time_since_vision_update_s`, corrupted by the TIMESYNC P0 bug #2 — must be fixed first). Recommend
  `TAU_STALE = 0.10 s` (≈ 3 ticks @ 30 Hz; covers the CPU-latency band where age matters). Gives the
  policy the "my pose is going stale, the next correction will be large" signal — the in-loop analogue
  of c5's reaction-time-shrinks-1/v finding.

### 1.3 Critic (asymmetric) consumption

`get_state` (asymmetric-critic input) currently carries 3-gate lookahead of clean `pos_g/vel_g/rpy_g`
(33-dim). **The critic gets GROUND-TRUTH gate-relative state + the SAME confidence triple** (so the
value function can credit "it was right to slow under low confidence"). Append the same 3 scalars to
`get_state` → critic dim 33→36. The critic seeing truth while the actor sees noised-obs is the standard
asymmetric setup and is what lets the reward (§3) avoid teaching damping: the critic knows the true
margin, so a policy that brakes needlessly under HIGH confidence is correctly penalized by the advantage.

### 1.4 Cheap ablation to run (CONTEXT.md): all-gate look-ahead vs next-gate-only. Keep next-gate-only
(current contract) as the default; the all-gate variant is a 1-seed A/B, not a portfolio member.

---

## 2. MEASURED-ERROR DOMAIN RANDOMIZATION (inject the case-C error classes onto the obs)

**Doctrine (CONTEXT.md / index-rl-training):** DR on the **MEASURED error classes, NOT arbitrary
noise.** The policy trains to a gate-relative obs carrying the estimator's residual error + confidence,
so deploy == train. Below, every distribution ties to a re-derived number from §0.

The DR is applied **to the obs the policy sees** (a perturbation layer on `pos_g`, `vel_g`, and the
confidence channel), resampled per-fix-event within an episode — NOT a static per-episode offset. It
SIMULATES the case-C estimator: a discrete fix stream + KF smoothing, not Gaussian-on-every-tick.

### 2.1 The estimator-emulation layer (`+env.estim_dr=case_c`)

At each env step, maintain a per-env **emulated gate-relative pose error** `e_g = [e_N, e_E, e_D]`
(gate frame) and a **confidence state** that the policy reads. The error evolves as a fix-driven
Ornstein-Uhlenbeck-style smoothed process matched to the REAL KF behaviour (c1/c2/c5):

**(a) In-plane per-fix LATERAL noise — the residual VARIANCE term.**
- Per-accepted-fix lateral draw: `n_E, n_D ~ N(0, sigma_perfix_lat²)`, **sigma_perfix_lat sampled per
  episode ~ U[0.08, 0.30] m** (the DR range spanning the measured near-band 0.10–0.20 m through the
  0–12 m average 0.265 m to the margin-edge 0.30 m; §0.2, c1 table). The per-episode sample is the
  speed-rung coupling lever (§5).
- The KF SMOOTHS these: emulate filtered in-plane σ ≈ `sigma_perfix_lat / sqrt(N_eff)`, with **N_eff
  drawn ~ U[4, 9]** per approach (c5 cadence: ~5.6–9 effective fixes at 8–37 m/s; §0.4). Implement as a
  1st-order low-pass on the per-fix draws with time-constant set so the realized in-plane σ at the gate
  plane equals `sigma_perfix_lat / sqrt(N_eff)` — this reproduces the c1 rel-arm 0.095 m filtered 1σ at
  sigma_perfix_lat=0.265, N_eff≈7.8. **VALIDATION TARGET: the emulated `e_E,e_D` at the gate-4 plane
  must have 1σ matching the c1 rel table (0.265→0.095, 0.20→0.080, 0.08→0.049).**

**(b) Range-collapsing STRUCTURE (NOT flat noise).** The per-fix σ shrinks as the drone approaches the
gate (c1 band table: ≤9 m lat 0.10–0.20 m vs 9–12 m 0.41 m). Emulate `sigma_perfix_lat(range)` as a
linear collapse: σ at range r ≈ `sigma_perfix_lat * clip(r / 9.0, 0.5, 2.2)` (caps the near-band at
~0.5× and the far-band at ~2.2× the episode sample — matches the measured 0.20 m near / 0.41 m far
ratio). This is WHY the policy can be aggressive late: the obs gets *more* accurate as it closes, and
the confidence channel (§1.2) reflects it. Reward (§3) must let the policy exploit this.

**(c) The +0.67 m systematic N-bias / range-collapsing bias.** On the gate-relative path this MAP
bias DROPS OUT exactly (c1 proof: `db` cancels; E/D_bias −0.000/−0.004). **So DO NOT inject a per-track
in-plane bias** — that would be modelling the anti-pattern we are removing. BUT inject the
**residual one-signed PnP/extrinsic bias** that the gate-relative path does NOT remove (c1 caveat,
bound ≤0.19 m magnitude, sign UNMEASURED): **per-episode in-plane bias `b_E, b_D ~ N(0, 0.05²) m`,
truncated at ±0.10 m** — a small, honest, zero-mean-across-episodes residual (the policy must be robust
to a per-track systematic it cannot observe, but we do NOT know its sign, so it is sampled zero-mean per
episode). Flag: **`b_inplane` magnitude is ASSUMED (0.05 m σ) — the decisive resolver is the ShadowPC
at-speed gate-4 recording (c1/c5 caveat).** Keep it small so it does not dominate the variance term.

**(d) Latency staleness — ALONG-TRACK, not in-plane (a4 §0.7).** Inject staleness ONLY on the N
(along-track) channel of `e_g` and on `age_norm`: per fix, `e_N += v_along * L`, **L sampled per
episode ~ U[0.006, 0.125] s** (edge p50 5.8 ms → CPU p90 125 ms; a4 bands). The in-plane LEAK is tiny
(`v*L*sin(0.5°)` = 5–41 mm) — inject as `e_E,e_D += v_along * L * sin(theta_head)`,
`theta_head ~ N(0, (0.5°)²)`. **RewindKF removes the first-order along-track staleness**, so model the
*residual* after rewind: along-track residual `~ U[0, 0.05] m` (rewind leaves <5 cm; a4 §4) when
`+env.rewind_on=true` (DEFAULT), or the full `v*L` when modelling the no-rewind ablation. This teaches
the policy that phase/timing is the stale axis, not the lateral miss — and the `c_along` channel carries
it.

### 2.2 Confidence channel population (the SIMULATE-variable-confidence requirement)

The training env MUST populate `c_inplane`, `c_along` (§1.2) from the **emulated KF covariance**, NOT
from the true error. Maintain an emulated `P_hat` that tracks the *expected* error magnitude given the
fix history (range, N_eff so far, time-since-fix), then set:
- `sigma_inplane_hat = sqrt(P_hat_E + P_hat_D)/sqrt(2)`, collapsing with range per §2.1(b);
- `sigma_along_hat` grows with `age_norm` between fixes (latency), resets on each accepted fix.
- **Calibration:** the realized `|e_g|/sigma_hat` distribution must have NEES≈3 (chi²(3)) — i.e. the
  confidence the policy reads is HONEST (no overconfidence). This is the calibrated-covariance
  requirement; an overconfident channel would teach the policy to trust a lie. Assert in a unit test:
  `mean(e_g**2 / diag(P_hat)) ∈ [0.8, 1.3]` over a rollout batch.
- **Variable confidence across episodes:** because `sigma_perfix_lat`, `N_eff`, `L`, and the
  range-collapse all vary per episode/range, the policy sees a CONTINUUM of confidence regimes within
  and across episodes — exactly the "fast-when-confident / careful-when-uncertain" training signal.
  Critically, a fraction of episodes (`p_dropout ~ 0.10`) drop a run of fixes (FoV exit, chi² reject
  burst) → `age_norm → 1`, confidence → 0, and the policy must coast on IMU — the case-C tail.

### 2.3 What DR must NOT do (NEVER reward damping under noise)

The DR injects error on the OBS only; the **true dynamics + true contact geometry are unperturbed**
(the world model owns margin, per the inc7 contact-true doctrine; reward never sees the injected error).
A policy that brakes whenever confidence drops is NOT what we want — see §3.4 for how the reward + the
truth-seeing critic prevent it. The DR is honest perception noise, not a penalty.

---

## 3. SPEED-PRESERVING REWARD that exploits the confidence channel

Relate to the R1–R7 / T1–T4 contract (`RewardWeights`, re-derived §0.10). **Three changes; everything
else frozen.**

### 3.1 R1 replacement — hybrid arc-length progress over the REBUILT contact-safe line

Per planning-report §3 step-1 (the highest-value graft) + index-rl-training inc8 sub-task 1:
**Replace R1 (progress-to-gate-CENTER, `rw_progress * (d2g_prev − d2g_curr)`) with progress-ALONG-Γ:**
```
R1'  + rw_progress * (s(curr) - s(prev))      [10 / m of arc-length advanced along Γ]
```
where `s(·) = ReferenceLine.progress(pos_ned)` projects onto the **REBUILT corrected-aero contact-safe
reference line** (planning §3 step-2: `rl/reference_line_vq1.json` is drag-infeasible + 170° inverted →
MUST be rebuilt by component/step-2). **Loader = `src/racer/reference_line.py::ReferenceLine.load()` /
`.progress(position_ned)`** (VERIFIED present: arc-length cumulative `self.arc` + `progress()` returns
projected arc-length in m — exactly this graft; memory's `rl/reference_line.py` is an abbreviation, the
real path is `src/racer/reference_line.py`). This closes failure-mode
M-2 BY CONSTRUCTION: progress-to-center can pay a speed-pushed policy to clip the frame; arc-length
progress over a contact-safe line cannot (you only get rewarded for advancing along a line that already
clears the gate). **Crossings DEAD-CENTER on Γ → full 0.37 m contact-true band free @ r=0.38**
(planning §4 C1). Keep `rw_progress = 10.0`.

> **Reward-damping guard on R1'.** Because Γ is contact-safe and progress is along it, the policy is
> rewarded for SPEED along the safe line regardless of obs confidence — there is no term that pays the
> policy to slow down when uncertain. Confidence enters ONLY through where the policy is ALLOWED to be
> aggressive (§3.3), never as a progress discount.

### 3.2 Keep T4 finish-time pressure (the speed reward)

`(rw_finish + rw_finish_time * t_left_s)` unchanged (20 + 1.0/s). This is the explicit faster-lap
pressure (VQ2 ranks on time). It is the dominant speed-preserving term and is **obs-confidence-blind by
design** — the policy cannot earn time by being timid. Combined with R1' it guarantees the reward
gradient always points "faster along Γ".

### 3.3 NEW confidence-gated tilt headroom (the fast-when-confident lever, NOT a damping term)

The cleanest way to "exploit confidence" WITHOUT rewarding damping: make the **tilt free-cone
(R4)breathe with confidence** rather than adding a speed-discount. R4 is currently
`rw_tilt * relu(cos(tilt_free) − R33)²` with a fixed 60° free cone. Replace the fixed `tilt_free_rad`
with a **confidence-modulated free cone**:
```
R4'  - rw_tilt * relu(cos(tilt_free(c_inplane)) - R33)^2
      tilt_free(c) = tilt_free_lo + (tilt_free_hi - tilt_free_lo) * c_inplane
```
- HIGH `c_inplane` (confident, late in approach where range-collapse tightens σ) → wide cone
  (`tilt_free_hi`, the relaxed envelope §4) → the policy may tilt hard / go fast.
- LOW `c_inplane` (uncertain, far / stale / dropped fix) → narrow cone (`tilt_free_lo`) → aggressive
  tilt is penalized, so the policy naturally flies conservatively WHEN BLIND — but this is a *style/
  envelope* term, not a progress penalty: a confident policy pays ZERO and flies full speed.

**Why this is not damping:** R4' only ever taxes TILT BEYOND the cone; inside the cone it is exactly 0.
A policy that stays inside the (confidence-appropriate) cone pays nothing and is driven full-speed by
R1'+T4. The lever does not reward slowing — it removes the *permission* to take the highest-tilt /
highest-risk line when the obs cannot support it. Under HIGH confidence the cone is wide and the policy
is uncapped. This directly couples to c5: the achievable speed is gated by the achieved σ, and the
policy learns to spend its tilt budget where the estimator is accurate (close range), exactly the
range-collapse structure §2.1(b) provides.

Recommended: `tilt_free_lo = 60° (1.0472 rad)`, `tilt_free_hi` = the ladder target (§4). Keep
`rw_tilt` on the ladder schedule (§4). **Ablation:** also train the fixed-cone variant (R4 unchanged,
just relaxed) as a portfolio member — if the confidence-gated cone shows no margin benefit in eval, the
fixed cone is simpler and deploys identically.

### 3.4 Everything else frozen + the anti-damping proof

R2 passage (+10), T1 collision (−25), T2 miss (−15), T3 OOB (−25), R3 time (−0.02/step), R5 dact
(0.25), R6 rate (0.05), R7 corner (0.0 OFF). **No reward term reads the injected obs error.** The
**asymmetric critic sees the TRUE gate-relative state** (§1.3): so when the policy brakes under low
confidence but the TRUE margin was fine, the advantage is negative (it gave up T4 time for nothing) and
PPO un-learns the reflex; when it brakes under low confidence AND the true error was large enough that
staying fast would have hit T1/T2, the advantage is positive. **The truth-seeing critic is the
mechanism that makes "careful when genuinely uncertain" emerge WITHOUT a hand-coded damping reward** —
the policy learns the *calibrated* amount of caution, no more. This is the doctrinally-correct
"NEVER reward damping": caution is an emergent best-response to honest perception noise + honest contact
penalties, never a shaped term.

---

## 4. ENVELOPE LADDER (gated on the gate-4 metric) + the MANDATORY spin-margin gate

### 4.1 BSR3 spin-margin gate — MANDATORY BEFORE retrain (index-rl-training inc8 sub-task 3)

Before any inc8 retrain, widen the spin aborts so the relaxed cone does not auto-abort legitimate
high-rate transients:
- `spin_rate_abort: → 9.0–10.0 rad/s` (from current; the super-rate plant legitimately commands ~11
  rad/s — index-rl-training plant note).
- `spin_time_abort: → 3.0 s`.
This is a HARD prerequisite — running the ladder without it aborts the very envelope the ladder opens.
(BSR3 = the spin-margin gate; widen first, retrain second.)

### 4.2 The ladder (planning §3 step-3; ordered, each rung GATED on gate-4)

| rung | change | expected | GATE (must hold before next rung) |
|---|---|---|---|
| L0 | inc8 baseline: R1' + T4 + R4' (cone 60°), rw_tilt 96 | reproduce inc7-class validity, gate-relative obs | gate-4 contact-valid (§5) |
| L1 | **rw_tilt 96 → 48** | ~1.4 s/lap on non-binding segments | gate-4 margin ≥ 0 @ r=0.38 AND per-fix σ at achieved speed ≤ c5 threshold |
| L2 | **free-cone 60° → ~70°** (`tilt_free_hi = 70°`) | ~1.5 s/lap on 3 high-κ corners | gate-4 margin ≥ 0 AND speed-rung σ-gate (§5) |

- **rw_tilt FIRST, then free-cone** — planning §3 proved rw_tilt-alone is necessary-but-insufficient;
  the free cone is the binding kinematic lever. Do not relax both at once (cannot attribute a regression).
- **Cap at 70°, not 75–80°.** Planning §3 step-3 says "→~70°"; the 75–80° band enters the M-1 corner-cut
  regime that is the decomposed-fallback trigger. Stop at 70° for inc8; 75–80° is a future increment
  gated on the M-1 watch.
- **Each rung re-verified against the estimator at THAT speed** (c5 §3): a faster rung raises the gate-4
  approach speed, which loosens cadence-variance and (long-exposure) blur. The gate is the **achieved
  gate-relative per-fix σ at the rung's speed vs the c5 table** (§5).

---

## 5. SEEDS + SPEED-LADDER PORTFOLIO SELECTION METRIC

### 5.1 Seeds — ≥5 (narrow basin)

inc7 was **2/3 viable** (narrow basin; index-rl-training 🚩). Budget **≥5 seeds per ladder rung**
(planning M-3 says ≥4; the gate-relative obs + confidence channel + cone-gating is a *harder* basin than
inc7, so ≥5). The portfolio is {L0, L1, L2} × 5 seeds = 15 policies; plus the §3.3 fixed-cone ablation
and §1.4 all-gate-lookahead ablation as 5-seed side-runs if budget allows. Throughput: 2048-env PPO
~88.9 K steps/s (index-rl-training); prefer A100 but do not idle (GPU-preference 🚩).

### 5.2 The selection metric = fastest contact-valid policy the gate-relative ESTIMATOR can support

This is the speed-ladder portfolio rule (CONTEXT.md / index-rl-training RL-SPEED-LADDER-PORTFOLIO).
For each candidate policy, run the eval (§5.3), then SELECT:

> **The policy with the lowest finish time AMONG those whose gate-4 contact-true margin ≥ 0 @ r=0.38
> (SIMSTART) AND whose achieved gate-4 approach speed `v*` satisfies the c5 σ-gate at the per-fix σ the
> component-3 estimator delivers at `v*`.**

The c5 σ-gate (tie per the prompt — "tie the contact_true_eval gate-4 margin to the component-3 achieved
per-fix sigma at each speed rung"):

| achieved gate-relative per-fix σ (component-3, at v*) | MAX estimator-valid gate-4 speed (0.155 m margin) |
|---|---|
| ≤ 0.10 m | > 55 m/s (full ladder valid) |
| 0.20 m | 17 m/s |
| 0.265 m (measured 0–12 m avg, today) | ~13–15 m/s (interpolated; margin held at 0.139 m RMS only at 37 m/s with WARM prior — see note) |
| 0.30 m | edge of margin |

> **Note (the conditional).** c1 shows that at the MEASURED per-fix σ=0.265 m with a WARM (lap-converged)
> velocity prior, gate-4 in-plane RMS is **0.139 m < 0.155 m margin at 37 m/s** — but **p90 0.203 m is
> OVER** the margin, and a COLD prior gives 0.17–0.21 m (CONTEXT.md straddle). So a policy that achieves
> v* = 37 m/s is **CONDITIONAL-GO**: valid in RMS, marginal at p90, prior-sensitive. The portfolio
> selection must therefore use the **component-3 achieved σ at v*** (not the candidate value) and prefer
> the policy whose v* keeps gate-4 p90 in-plane (not just RMS) inside 0.155 m. If component-3 delivers
> per-fix σ ≤ 0.10 m, the full ladder (L2, ~37+ m/s) is GO; if it stalls at 0.20–0.265 m, the selected
> policy is the FASTEST one whose v* ≤ the σ-gate speed (likely L1, not L2). **The estimator, not the
> policy, sets the ceiling** — this is the whole point of the portfolio.

### 5.3 Exact eval invocation + pass criteria

**Instrument:** `rl/contact_true_eval.py` (the selection instrument; gate-4 0.155 m margin guard,
S_stable, per-gate margins, gate_yaw support). Run **map-ON** (default `--plant mixer` — the 🚩 retrain
footgun: evals default legacy flat plant; mixer is the fully-measured plant).

```
.venv\Scripts\python.exe rl/contact_true_eval.py ^
    --ckpt rl/checkpoints/<inc8_rungL_seedK>_actor.pth ^
    --plant mixer --body-radius 0.38 --frame-depth 0.30
```

(Note: `--body-radius 0.38` sets the binding-gate radius; default 0.33 is the nominal mid-DR. The gate-4
margin is quoted @ r=0.38 — the worst-case body radius — so eval AT r=0.38 to read the binding margin.
`PASS_BAND = _HALF_OPEN − 0.38 = 0.37 m`; margin = 0.37 − linf.)

**PASS CRITERIA (all must hold for a candidate to be portfolio-eligible):**
1. **SIMSTART outcome = FINISHED** (the competition-representative full-course rollout; only SIMSTART
   carries the realistic high-speed post-gate-3 approach — `contact_true_eval.py` line ~613).
2. **gate-4 SIMSTART `margin_min ≥ 0.0` @ r=0.38** (the primary binding guard; from the per-gate margin
   table, SIMSTART filter). Gate-5 should hold its clean 0.314 m.
3. **`S_stable ≥ 2/3`** (`compute_s_stable`, threshold 0.90) — at least the inc7 narrow-basin bar; prefer
   ≥ 3/5.
4. **D-offset probe (gates 3,4,5) verdict does NOT flip pass→collision under ±1.5 m** — confirms the
   gate's margin is not map-confounded at the achieved line (`gate_d_offset_probe`).
5. **Speed-σ tie:** the candidate's achieved gate-4 approach speed `v*` (read from the SIMSTART
   trajectory: `|v|` at the last pre-gate-4 fix window) must satisfy the §5.2 c5 σ-gate at
   component-3's delivered per-fix σ. If component-3 σ is not yet measured, record `v*` and FLAG the
   candidate CONDITIONAL-GO pending the at-speed σ measurement.

**SELECT** = the FINISHED, margin≥0, S_stable-passing candidate with the **lowest `finish_t`** whose
`v*` clears the σ-gate. Crown ONLY after a ≥3–5-lap fresh-reset live batch (planning M-4 winner-
validation rider; fold the yaw-active segment in per index-rl-training NEXT ⑦).

### 5.4 Eval extension needed (build note for component-3 / integration)

`contact_true_eval.py` currently runs on TRUE pose (`obs_from_truth`). To make the selection metric
honest about the gate-relative estimator, the eval must run the **same §2 estimator-emulation layer** on
the obs the policy consumes during eval (so the eval drone flies on noised gate-relative pose, not
truth). This is a thin wrapper: inject `e_g(range, N_eff, L)` into the obs inside `run_episode`'s loop
(after `obs_from_truth`, before `policy_step`). Without it, the eval over-reports validity (flies on
perfect pose). **This wrapper is REQUIRED for criterion 5 to be meaningful** and is the one new eval
element inc8 needs. (Keep the truth-pose eval as the optimistic upper-bound control.)

---

## 6. Build order (the inc8 critical path)

1. Fix P0 bugs #1 (navigator `use_given_position` guard) + #2 (TIMESYNC) — `age_norm` and case-C
   validation depend on them (CONTEXT.md). PREREQ for honest training/eval.
2. Component-1 delivers the 20-dim obs + CALIBRATED gate-frame P (the confidence channel). Interface gate.
3. Rebuild the corrected-aero contact-safe reference line (planning §3 step-2) for R1'.
4. Widen BSR3 spin aborts (§4.1) — MANDATORY before any retrain.
5. Add the §2 estimator-emulation DR layer + §3 reward changes to the training env (new env, never edit
   the diffaero clone — registered via `ENV_ALIAS` per peregrine_racing doctrine).
6. Add the §5.4 eval wrapper to `contact_true_eval.py`.
7. Launch the {L0,L1,L2}×5-seed ladder on Adroit (map-ON; parity gate already cleared job 3270602).
8. Select per §5; crown after the ≥3–5-lap fresh-reset live batch.

---

## 7. Residual risks / open (carry to commander)

- **R-HIGH: per-fix σ at 37 m/s is UNMEASURED** (c1/c5 pool is ~5 m/s data; blur extrapolated 4–7×).
  The 0.139 m margin clearance is a BEST-CASE lower bound. The whole speed-ladder ceiling rides on
  component-3's at-speed σ. Decisive resolver = ShadowPC at-speed gate-4 recording.
- **R-MED: residual one-signed PnP/extrinsic bias** (gate-relative removes MAP bias but not a chain-
  correlated PnP systematic; c1 bound ≤0.19 m mag, sign unmeasured). DR injects ±0.10 m zero-mean as
  insurance (§2.1c), but a real one-signed 0.19 m would erode the margin. SHADOWPC-VISION-CAL resolves.
- **R-MED: confidence-channel calibration** — if component-1's P is overconfident (NEES≫3, as the
  ABSOLUTE KF was), the policy trusts a lie and flies too aggressively. The §2.2 NEES∈[0.8,1.3] unit
  test is the gate; component-1 MUST deliver a calibrated gate-relative P.
- **R-MED: narrow basin** — gate-relative + confidence + cone-gating is harder than inc7 (2/3). ≥5
  seeds; if <2/5 viable at L0, simplify (drop confidence-gated cone → fixed cone, §3.3 ablation) before
  blaming the obs.
- **R-LOW: cold-vs-warm velocity prior** — case-C velocity is observable only through fix differencing;
  cold prior gives 0.17–0.21 m (over margin). The lap-converged warm prior (0.11–0.14 m) is the design
  point; the standing-start FIRST gate-4 pass is the cold-prior worst case (mitigated: gate-4 is the
  5th gate, prior is warm by then on a full lap).
- **CASE CONDITIONALITY:** entire spec is the case-C worst case. If VQ2 streams LOCAL_POSITION_NED/
  ODOMETRY (case A/B), pose is pristine and the confidence channel reads ~1 always (still valid — the
  policy trained with variable confidence degrades gracefully to the certain regime). Conditional on
  organizer Q①, but build gate-relative REGARDLESS (organizer-pivot: Q① only decides load-bearing-vs-
  insurance — CONTEXT.md / MEMORY NOW).

---

## MEMORY-DELTA (≤10 lines)

- d5 INC8 SPEC DONE (`d5_inc8_spec.md`): gate-relative obs **17→20 dim** (+`c_inplane`,`c_along`,
  `age_norm` confidence triple from CALIBRATED gate-frame KF P; critic gets truth+same triple).
- DR = MEASURED case-C error classes on the OBS (not arbitrary): per-fix lateral σ **U[0.08,0.30] m/axis**
  (measured 0.265), KF-smoothed via N_eff U[4,9] (→filtered 0.095 @0.265), RANGE-COLLAPSING σ(r),
  along-track-only latency v*L (L U[6,125]ms, rewind-residual <5cm), MAP bias DROPS OUT (do NOT inject;
  only ±0.10m zero-mean PnP-residual insurance). Reproduced c1: rel inplane 0.139, E/D_bias −0.000/−0.004.
- REWARD: R1→arc-length progress over REBUILT contact-safe line (closes corner-cut M-2 by construction);
  keep T4 finish-time (speed); R4→confidence-gated free-cone tilt_free(c_inplane) lo60°→hi-ladder
  (fast-when-confident, ZERO inside cone = NOT damping). NEVER-reward-damping enforced by truth-seeing
  asymmetric critic, not a shaped term.
- LADDER: BSR3 spin-gate MANDATORY first (spin_rate_abort→9–10, spin_time_abort→3.0s); rw_tilt 96→48
  THEN free-cone 60→70° (cap 70, 75–80 is M-1 corner-cut regime), each rung GATED on gate-4 margin@r=0.38.
- SELECT: ≥5 seeds (inc7 2/3 narrow); fastest FINISHED policy with gate-4 SIMSTART margin≥0 @r=0.38 +
  S_stable≥2/3 whose achieved v* clears the c5 σ-gate at component-3's delivered per-fix σ (σ≤0.10→>55m/s;
  0.20→17m/s). Eval = `contact_true_eval.py --plant mixer --body-radius 0.38` + a NEW estimator-emulation
  obs wrapper (else over-reports validity). Crown after ≥3–5-lap fresh-reset live batch (winner rider).
- BINDING RISK = component-3 at-speed per-fix σ (UNMEASURED >5 m/s) + component-1 P calibration (NEES≈3).
```
