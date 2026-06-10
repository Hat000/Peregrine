# 2nd-order inner-rate-loop re-system-ID (ShadowPC, 2026-06-10)

**Deliverable: validated 2nd-order rate-loop model + parameters + integration plan.**
Plant code NOT touched (per tasking) — `twin.py` / `rl_plant.py` / `diffaero_dynamics.py`
integration is specced in §5 for the follow-up session to act cold.

Reproduce: `.venv\Scripts\python.exe handoff\shadowpc-2ndorder-resysid-2026-06-10\fit_2nd_order.py`
(needs the actor checkpoint at `C:\Users\Shadow\Downloads\stage1_inc1_actor.pth` for the
tumble-command reconstruction; recordings under `data/runs/`).

## 1. Problem

The shipped first-order inner-loop model (`target = rate_gain·rate_sign·cmd`, first-order lag
τ=19 ms) cannot exceed its target: at saturated commands (±3.14 rad/s → target 7.85) the live sim
peaked at **9.77 rad/s** (S1.2 tumble, `data/runs/20260610_205414_rl_s12_f1`), and the rate kept
**rising ~35 ms after the command flipped sign** (momentum) — structurally impossible for a
first-order lag. This was S1.2 transfer-blocker #1 (the RL policy's bang-bang attitude control
diverged live while passing cleanly in the twin).

## 2. Data (all known-input; wire cmd = FRD body-rate setpoint; output = ODOMETRY raw rate ·[−1,−1,1])

| dataset | source | input | regime |
|---|---|---|---|
| rate1, rate2 | `2026-06-03 rate_sysid` commands.jsonl | open-loop ±0.3 rad/s doublets | small-signal (the shipped fit's home) |
| course | `20260607_194615_course_60s` commands.jsonl (100 Hz) | closed-loop CTBR, ≲3 rad/s cmds | mid |
| tumble | `20260610_205414_rl_s12_f1`, commands **reconstructed** | bang-bang ±3.14 | saturated |

Tumble-command reconstruction: the deployment policy is deterministic (`tanh(actor_mean)`), so
replaying it over the recorded telemetry at every ODOMETRY sample recovers the wire command; the
policy ran **bang-bang** (outputs pinned at ±1) so the command is unambiguous wherever consecutive
samples agree (±13–33 ms ambiguity at sign flips vs the 30 Hz live tick). Cross-checked against the
live console print at t=10.75 (`[+3.14,−3.14,+3.14]` — exact match). Fit window **10.77–10.99 s
only**: beyond ~10.99 the drone is past 90° tilt = the sim's known yaw-spin anomaly regime
(VQ2 open item A, deliberately NOT modeled here).

## 3. Model + method

Per axis, keeping the existing DC structure (`target = rate_gain·rate_sign·cmd`, **G fixed** at the
shipped steady gains [2.501, 2.504, 2.231]):

```
omega_ddot = wn² · (target − omega) − 2·zeta·wn · omega_dot
```

plus a shared input transport delay `d` (ZOH command shifted), fitted per regime by simulation-error
RMSE (Nelder-Mead multistart, delay on a grid).

**Why G is fixed** (lesson from the first fit attempt): with G free, the optimizer inflated G to
3.2–3.8 to reach the 9.77 peak — but the tumble never reaches steady state (commands flip every
~100 ms), so saturated steady gain is unobservable there, and G=3.8 contradicts the repeatedly
measured small-signal steady gain ≈2.5. Fixing G forces the overshoot to be explained by dynamics,
which the data supports (see the momentum-after-flip signature).

## 4. Results

### Recovered parameters (G fixed at shipped)

| regime | axis | wn (rad/s) | zeta | d (ms) | τ_eq=2ζ/wn (ms) | step overshoot |
|---|---|---|---|---|---|---|
| **SATURATED** (tumble) | roll | **21.0** | **0.393** | 15 | 37 | **26%** |
| | pitch | **28.3** | **0.467** | 30 | 33 | **19%** |
| | yaw | 11.3 | 0.273 | 0 | 48 | 41% (caveat §6) |
| small/mid (rate1+2+course) | roll | 67.6 | 0.824 | 0 | 24 | 1% |
| | pitch | 66.9 | 0.699 | 0 | 21 | 5% |
| | yaw | 109 | 5.3 (overdamped; slow pole ≈ τ 97 ms) | 5 | — | 0% |

### Fit error, before → after (RMSE rad/s, simulation error)

| dataset | axis | shipped 1st | 2nd (sat-fit) | 2nd (small-fit) |
|---|---|---|---|---|
| **tumble** | roll | **7.70** | **0.25** | 7.17 |
| | pitch | **5.02** | **0.39** | 4.82 |
| | yaw | **6.92** | **0.28** | 2.66 |
| rate1+2 (worst axis) | pitch | 0.07 | 0.22 | 0.07 |
| course | roll | 0.26 | 0.82 | **0.22** |
| | pitch | 0.26 | 0.68 | **0.23** |
| | yaw | 0.83 | 1.28 | **0.41** |

### Tumble peak |rate| validation (the headline number)

| axis | measured | shipped 1st | 2nd (sat-fit) |
|---|---|---|---|
| roll | **9.77** | 7.82 | **10.14** |
| pitch | **9.72** | 7.84 | **9.50** |
| yaw | 8.14 | 7.00 | 7.75 |

The saturated-regime 2nd-order model reproduces the overshoot the first-order model is structurally
blind to (tumble RMSE improves **20–30×**), while the small/mid-regime fit shows τ_eq ≈ 21–24 ms ≈
the shipped τ=19 ms with ≤5% overshoot — i.e. the first-order model was adequate where it was fit.

### The key structural finding: the loop is AMPLITUDE-DEPENDENT

No overshoot at 0.75 rad/s targets; ~20–26% at 7.85. wn drops ~3× and zeta halves from small to
saturated. One linear 2nd-order cannot fit both regimes (the cross-eval columns above show each
regime's fit degrading the other). Mechanistically this is the signature of an inner PI(+D) rate
controller with torque saturation / integrator windup at large errors. The two fitted regimes
BRACKET the true nonlinear behavior.

## 5. Proposed integration (the three plant reps)

**Recommendation: do NOT hard-swap to one 2nd-order. Two-tier plan:**

**(a) For RL training (the urgent use — robustness):** keep the plant nominally 1st-order and add a
2nd-order mode under **DR**: per env at reset sample a regime blend λ ~ U[0,1] and set, per axis,
`wn = lerp(wn_sat, wn_small, λ)`, `zeta = lerp(zeta_sat, zeta_small, λ)` (λ shared across axes,
small per-axis jitter ±10%). Bounds from §4: roll/pitch wn ∈ [21, 68], zeta ∈ [0.39, 0.85]. A policy
robust across that envelope is robust to the real amplitude-dependence without modeling the windup
curve. This supersedes the cruder "+30% rate_gain band" proxy proposed in the S1.2 handoff (that
inflates DC gain, which the steady-gain anchor contradicts — prefer the 2nd-order DR).

**(b) For the deterministic offline twin (verify/replay):** amplitude-SCHEDULED params:
`wn(|target|), zeta(|target|)` linear between anchors at |target| = 3.0 (small/mid values) and
7.85 rad/s (saturated values), clamped outside. Matches both regimes' data with one model.

### Equations per representation

Common replacement for the rate-update step (per axis; new state `omega_dot`):

```
target    = rate_gain * rate_sign * cmd_rate              # unchanged
omega_dot += dt * (wn^2 * (target - omega) - 2*zeta*wn * omega_dot)
omega     += dt * omega_dot                                # then the existing |omega| clamp
```

Stability: explicit Euler needs wn·dt_sub ≲ 0.3 → at dt=0.0333 and wn up to 68 use **n_substeps ≥ 8**
(or the exact ZOH discretization: x=[ω−tgt, ω̇], x⁺=Φx with the standard damped-oscillator matrix
exponential — exact at any dt, branch on ζ<1 (sin/cos) vs ζ≥1 (sinh/cosh); use this in rl_plant/twin
where determinism matters, substeps in torch where simplicity matters).

1. **`src/racer/twin.py` (`CtbrPlant.step`) — change FIRST (twin = ground truth):** replace the
   `alpha`-lag rate update with the block above; carry `omega_dot` in the twin state; params
   `rate_omega_n`/`rate_zeta` (per-axis, or the §5b schedule); `None` ⇒ legacy first-order path
   (keeps every existing test green).
2. **`src/racer/rl_plant.py`:** `PlantParams` gains optional `rate_omega_n`, `rate_zeta` (None ⇒
   legacy); `PlantState` gains `omega_dot` (...,3) (default zeros; `hover()` zero-inits); `step()`
   §1 replaced as above. Extend `tests/test_rl_plant_parity.py` cell-for-cell vs the twin.
3. **`rl/diffaero_dynamics.py` (`_step_torch` + `_step_numpy`):** numpy path delegates (free);
   torch path mirrors the block with an extra `_omega_dot` state tensor (detach() it too);
   DR tensors `_dr_wn`, `_dr_zeta` resampled in `_resample_dr` per §5a; legacy scalar path must
   stay bit-identical when disabled to preserve the 4.4e-16 `check_against_rl_plant` gate, and the
   gate should ALSO be run once with the 2nd-order enabled (rl_plant is the reference for both).
4. **Deployment (`rl/fly_rl.py`):** no change — the model is plant-side only.

Do not double-count latency: the fitted d (0–30 ms) is the same physical transport the S1.3
`transport_delay_steps` DR covers. Keep delay in the transport-delay mechanism, not baked into wn/zeta.

## 6. Caveats + recommended follow-up data

- The saturated regime rests on ONE maneuver (17 telemetry samples/axis at 75 Hz) — but with an
  enormous, structurally-unambiguous signal (0→9.77 rad/s with a mid-trajectory command flip).
  Roll/pitch peaks occur at ≤47° tilt (clean regime). **Yaw's** peak (8.14 at t=10.94) is at ~112°
  roll — anomaly-adjacent; treat yaw sat-params as indicative. Yaw is also the worst-modeled axis
  in the shipped twin (course RMSE 0.83; the small/mid fit prefers an effective τ≈97 ms, 5× the
  shipped 19 ms) → yaw deserves its own pass.
- Command reconstruction carries ±13–33 ms transition-time ambiguity (replay cadence vs tick phase).
- ≥90°-tilt data excluded by design; the yaw-spin anomaly remains a SEPARATE untwinned phenomenon
  (VQ2 item A).
- **Cheap definitive follow-up:** `scripts/rate_sysid.py` open-loop step sweep at
  `--mag 0.3,1.0,2.0,3.14` (per axis, a few seconds each on ShadowPC). That gives clean steady
  state AT saturation (resolving saturated DC gain, here held at 2.5 by assumption) and maps the
  windup curve wn(|target|), zeta(|target|) directly. ~10 minutes of sim time; do it before the
  integration lands if convenient.

## 7. Files

- `fit_2nd_order.py` — loaders (sysid/course jsonl, tumble policy-replay reconstruction), the two
  models, two-regime fitting, cross-eval + peak validation. Output above is its verbatim print.
