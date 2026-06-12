# S17: motor-mixer coupling modeled + integrated (parity-gated) + INC6 retrain

**Session LAPTOP-S17-MIXER-INC6 (2026-06-11/12).** Implements
`handoff/shadowpc-live-deploy-diag-2026-06-11/WRITEUP.md` Section 8: the unmodeled MOTOR-MIXER
coupling that broke the inc4/inc5 live transfers (0/20 flights) is now a measured, fitted,
parity-gated feature of all three plants (twin -> rl_plant -> DiffAero adapter), and inc6
retrains against it with a style-regularizer A/B and transport-delay DR centered on the
measured live latency. Commits: `dafdc69` (model + integration), `eed0444` (review fixes),
`ab634c8` (R7 corner penalty), ship commit (checkpoint).

**SHIPPED: `rl/checkpoints/stage1_inc6_actor.pth` = inc6_c16_s0 (md5
`8fb8855e07d4fd01045e7b2ecbc5acd3`) + `.json` sidecar {3.765, 3.14}.** Mixer-ON: held-out VQ1
sr 1.000 / median 9.86 s / roll max 57.4 deg; generalization 0.982; ACTRATE thr_p95 0.061 /
yaw_p95 0.009 / flips 0%; 16/16 laptop deployment matrix (all start modes, latency 0-3,
perturbed seams to 10 m/s) with NO mitigation flags. The inc5 datum on the same plant: sr
0.000 / 0.000 (Section 8).

---

## 1. The mixer model

The sim's per-motor commands are collective +- rate-PID differentials, clipped to [idle, 1]
(diag Section 2, measured live in isolation). The S17 model, in commanded-collective units
(identity motor-units==collective-units — see Section 2 for why):

```
d_ax   = kappa_err * (target_ax - omega_ax) + kappa_hold * omega_ax      [signed, per axis]
d_yaw *= zeta / (zeta + c)            yaw torque effectiveness falls with rotor speed
u_i    = clip(c + S_i . d, idle, 1)   S = X-quad sign matrix (4x3, columns orthogonal)
c_eff  = mean(u)                      -> the S16 knot table -> a_up     [parasitic lift / sag]
delta  = (S^T u) / 4                  realized per-axis differential
Q_ax   = clip(delta/d, 0, 1) / r_fit_ax    scales the S14 slew limit alpha_max
```

* `target` = the S14 super-rate steady target (the rate loop's existing internal quantity).
* The error term (`kappa_err`) is the dither regime — the policy's per-tick sign flips keep
  |target - omega| permanently maxed; the hold term (`kappa_hold`) is the settled-spin
  differential (damping compensation).
* **`r_fit` normalisation (the key subtlety):** the S14 slew limits (260/260/80 rad/s^2) were
  measured at (hover collective x single-axis pi command) — WITH the mixer already throttling
  (r_fit roll/pitch 0.580/0.579, yaw 0.763 at the nominals). Scaling by raw `r` would
  double-count the throttle; `Q = r/r_fit` keeps the fit point bit-exact (pinned by
  `test_slew_fit_point_preserved`). Unclipped flight gets `Q = 1/r_fit` — the implied
  UNCONSTRAINED torque (rp ~448, yaw ~105 rad/s^2) — and the slew never binds small-signal.
* Benign envelope (no clipping): `c_eff == c` to 1 ULP and the slew never binds — the mixer
  cannot perturb the validated flight envelope (pinned by `test_benign_envelope_matches_mixer_off`).
* Known simplification: clip-induced CROSS-AXIS torque leakage (a deep 2-axis saturation
  corner) is not injected; DR covers it better than a sloppy term would.

## 2. The fit (`fit_mixer.py`; data = diag WRITEUP Section 2 aggregates)

| constant | value | source | DR band |
|---|---|---|---|
| `MIXER_KAPPA_ERR_MEASURED` | **0.073** | yaw rail 0.73/10.006 = 0.0730; r+p rail 0.64/8.763 = 0.0730 — **two independent probes agree to 0.2%** | [0.060, 0.085] |
| `MIXER_KAPPA_HOLD_MEASURED` | **0.046** | settled spin 0.46/10.006 (single point; r/p unmeasured — tumbles) | [0.030, 0.060] |
| `MIXER_ZETA_YAW_MEASURED` | **0.34** | zhov_y31 settled mean 0.287 -> d(hover) = 0.258 -> eta 0.561 | [0.20, 0.55] |
| `MIXER_IDLE_MEASURED` | **0.05** | z00_r0 mean 0.064; rail low-pair 0.08 (carries r/p stabilisation riding); diag's own ~0.05 | [0.04, 0.08] |

Model vs measured, every probe row (fit_mixer.py output):

| probe row | model | measured |
|---|---|---|
| z00_y31 early motors | [0.05, 0.05, **0.73**, **0.73**] | [0.08, 0.08, 0.73, 0.73] |
| z00_y31 settled mean -> a_up | 0.255 -> **8.8 m/s^2** | 0.258 -> 9.36 |
| z00_y12 settled mean | 0.095 | 0.126 (phase avg incl. transient) |
| zhov_y31 settled mean | **0.287** | 0.287 (the zeta anchor) |
| z00_rp15 early max motor | **0.640** | 0.64 |
| z00_r0 mean -> a_up | 0.050 -> **0.00** (free fall) | 0.064 -> 0.64 |
| c100 (re-level d~0.25) mean/max | **0.875** / 1.00 | 0.874 / 1.0 |

**Identity motor-units adoption:** the c100 row discriminates — identity + asymmetric top-clip
predicts mean 0.875 and a_up K(0.875)=65.3 ~= 58.9 after the observed re-level tilt wobble;
an affine motor map (base 0.874 at cmd 1.0) would predict K(1.0)=78.3 — refuted. Pure
desaturation-rescaling at the bottom is refuted by the sustained [0.08,0.73,0.73,0.08] spin
pattern (it would zero the differential at c=0). Simple per-motor clip matches everything.

**Under-constrained corners (REPORTED, not silently guessed)** — `mixer_probe2.json` in this
directory is the ~10-min ShadowPC errand profile that nails them:
1. **(collective 1.0 x large rate demand)** — the top-rail authority depth was never probed
   directly (c100 had only re-level differentials). `c100_r31`/`c100_y31`/`c60_r31` phases.
2. **zeta for roll/pitch** — `zhov_r31` (the roll analog of zhov_y31). Currently r/p get NO
   collective-dependent effectiveness (conservative both ways: over-predicts parasitic lift at
   mid-collective, under-predicts authority — the policy avoids the corner either way).
3. **kappa_err/kappa_hold split** — `z00_y31_long` (1.5 s) for a clean settled tail.
4. Documented tension: zhov max-motor 0.76 vs the mean-anchored model's 0.68 — the mean
   constraint wins (averages beat phase maxima); the zeta DR band [0.20, 0.55] brackets it.

## 3. KEY VALIDATION — inc5 through the mixer-ON twin reproduces the live failure

`offline_rollout.py --plant mixer --start handoff --handoff-speed 5.1` (inc5 checkpoint):

| | aero plant (mixer-blind) | **mixer plant** | live (diag Section 1c) |
|---|---|---|---|
| outcome | **6/6 FINISHED 8.19 s** | gate-0 pass (bridge momentum), then **+8.4 m vertical miss at gate 1 — never descends** | gate-0 region top-frame strike / loft |
| collective, first 13 ticks | 0.000 (pulses later) | **0.000 all 13** | 0.000 |
| yaw command | +-3.14 rail | **+-3.14 rail** | +-3.14 rail |
| vertical | free-falls, catches itself | **POWERED CLIMB, vz -0.97 at tick 12** | climbing +3.2 m/s |
| speed by tick 12 | — | **5.1 -> 9.8 m/s** | **5.1 -> 9.8 m/s** (exact match) |

The twin-blind-spot mechanism is reproduced **quantitatively**: the policy sees "climbing,"
never pulses thrust, and the parasitic mean (~0.39 collective -> ~2.1 g through the knot
table) is exactly the live "2 g uncommanded" signature.

**Mode B (the residual gate-2 lateral miss under `--yaw-scale 0`) is NOT discriminably
reproduced:** mixer-ON at the exact nominal seam threads 6/6 at latency 0/2/3 (as the aero
twin did). Under realistic seam perturbation (speed 4.0-6.5, dist 2.5-3.5) BOTH plants fail
uniformly at gates 2-3 — so ys0-fragility off-seam is a property of the yaw-mutilated policy,
not a mixer discriminator. The remaining candidate for the live gate-2 depth is the unmeasured
(top-rail x large-demand) corner — errand item 1 above. Training-side this does not block
inc6: the trained policy avoids both corners entirely (it feels sag + authority loss +
parasitic lift in-plant), and styles that avoid the corners live in the regime where the model
is exact.

## 4. Integration (S14/S16 pattern: defaults OFF, twin -> rl_plant -> adapter, gates at every seam)

* **`src/racer/twin.py`** — 4 config fields (`mixer_idle/kappa_err/kappa_hold/zeta_yaw`,
  set-together validated, range-validated, requires `alpha_max_rps2`); the realised-collective
  block moved AHEAD of the rate loop (independent computation — legacy floats unchanged,
  pinned by test_measured_aero's inline-legacy test); mixer block per Section 1; module helper
  `_mixer_r_fit`.
* **`src/racer/rl_plant.py`** — same 4 on `PlantParams` (+ validation + cached
  `_mixer_r_fit`), step mirrors twin operation-for-operation; exports the 4 measured nominals
  + `mixer_r_fit`.
* **`src/racer/twin_fit.py`** — `faithful_config(super_rate, measured_aero, mixer)`; the fully
  sim-faithful twin as of 2026-06-11 = all three True (mixer requires super_rate).
* **`rl/diffaero_dynamics.py`** — torch mirror in `_step_torch` (scalar branch bit-identical
  to rl_plant); **`+dynamics.dr_mixer=true`** opt-in (requires `dr_aero`; per-env idle/kerr/
  khold/zeta resampled at reset + per-env `r_fit` recomputed against the envs' sampled
  super-rate s); **`dr_latency_min_steps`** knob (default 0 = legacy; inc6 trains {1,2,3}
  centered on the measured 2-tick live latency).
* **`rl/check_diffaero_gate.py`** — configs {mixer, mixer_full}; rebuild_params rebuilds
  `_mix_rfit` at the gate dtype; random_traj crafts deep-rail actions at T-4..T-1 so they
  drain through the transport-delay buffer (review fix — T-2/T-1 alone never APPLY under k=2).
* **`rl/offline_rollout.py` / `rl/peregrine_eval.py`** — `--plant mixer` (aero + measured
  mixer, fixed params); eval prints ACTION_RATE / YAW_FLIP_RATE + `ACTRATE_SUMMARY` line.
* **`rl/peregrine_racing_precheck.py`** — DR MIXER + latency-band prints.
* **`rl/fly_rl.py`** — stale "0/6 raw standing start" comment replaced (inc5+: 6/6 from all
  start modes; standing start = deployment target per INC5-LIVE Appendix B).
* **`rl/run_parity.sh`** — md5 tripwire -> `cdf79613267644f94d3e89f5277bcf5f`.

## 5. Parity / gates (all PASS)

* **Test suite: 528 -> 562 green** (+14 `tests/test_mixer.py` anchors, +20 parity-battery
  cases: {mixer, mixer_full} x 2 dt x 5 sequences; omega/thrust bit-identical, pos <= 1e-9).
* **Anchor tests** pin every Section 2 row + the fit-point slew preservation + benign-envelope
  equality + free fall + both rails (parasitic climb at thr-0 dither; authority/sag at c=1).
* **Local CPU-torch gate** (torch 2.12.0+cpu, float64): 8 configs, worst **7.1e-15** PASS.
* **Negative controls (teeth):** uncorrupted 1.8e-15; torch kappa_err biased +1e-6 ->
  **1.4e-04 CAUGHT**; torch silently dropping the mixer -> **10.45 CAUGHT**.
* **Adroit V100 gate (SLURM job 3268647, GPU node, torch 2.5.1+cu121): GATE_PASS, worst
  7.105e-15 float64** over 8 configs x 6 seeds x 8 steps; md5 tripwire matched.
* **DR smoke (CPU):** latency sampled exactly {1,2,3}; mixer bands sampled per-env; per-env
  r_fit tracks sampled s (roll [0.47, 0.67], yaw [0.67, 0.91]); thr-0 yaw-dither under DR
  climbs (+4.9 m/s after 0.5 s) where legacy free-falls (-4.4) — the parasitic rail is IN the
  training plant.
* **Adversarial review (25-agent workflow, 5 lenses + per-finding refutation):** 2 confirmed
  minors, both fixed in `eed0444` (twin range guard mirroring rl_plant; gate rail-steps
  through the delay buffer). All four design invariants independently confirmed; notable
  verified non-issues: twin/rl_plant `r_fit` bitwise equal; a fresh 400-step rail-heavy
  parity battery re-run clean; Q at the fit point exactly 1.0 (not just ULP-close).

## 6. INC6 training config (rl/peregrine_racing_inc6.sbatch)

The inc5 winner recipe (t96: RW_TILT=96, validity weights, 6000 updates, 2048 envs, PPO,
random courses + 30% standing starts, sidecar 3.765/3.14) plus:

1. `+dynamics.dr_mixer=true` (with `dr` + `dr_aero`) — the S17 bands above.
2. `+dynamics.dr_latency_min_steps=1 +dynamics.dr_latency_max_steps=3` — {1,2,3} ticks,
   CENTERED on the measured live 2 (was {0,1,2}: trained for a latency the live system never
   has and not for the one it sometimes does).
3. **STYLE-REGULARIZER A/B (Part 3 amendment — two candidates + the combination):**
   * **(a) RW_DACT** — the blunt R5 ||delta action||^2 span-normalised penalty (inc5 trained at
     1.0 and still railed; the twin made rails free, the mixer now makes them expensive, and
     the weight accelerates style convergence). Taxes EVERY fast correction equally.
   * **(b) RW_CORNER (NEW, R7)** — the TARGETED mixer-corner tax
     `|a_thr - 0.5| * ||2(a_rate - 0.5)||` (span units: |thr-mid| is 0.5 at either thrust
     rail, ~0.23 at hover; rate magnitude 1 per railed axis). Prices exactly the two mixer
     rails — (thr~0 x high rate) parasitic lift, (thr~1 x rate) authority/sag — WITHOUT
     suppressing mid-range thrust corrections. Default 0 = OFF (legacy reward float-identical;
     pinned by test). On COMMANDED actions like R5, so it shapes style even where the mixer-ON
     plant already prices the realized physics.
   * Decision rule: **ship whichever arm is FASTEST at compliant style** (the ACTION-RATE gate
     is the equal-live-compatibility bar; lap time decides among compliant candidates).

**Acceptance gates (Section 8):** S15 gates mixer-ON (held-out VQ1 success ~1.0 + median lap,
generalization, style envelope, deployment rollouts incl. latency 2-3) **+ the new
ACTION-RATE gate: thr_p95 <= 0.5 span/tick, yaw_p95 <= 0.5 span/tick, yaw flip rate <= 5%**
(inc5 datum: ~1.0 rails / per-tick flips). Standing start is the deployment target.

## 7. Training rounds (LIVE — updated as rounds complete)

All evals MIXER-ON. VQ1 = held-out acceptance; gen = random courses; style = ACTRATE gate
(thr_p95/yaw_p95 <= 0.5 span/tick, flips <= 5%).

| round | job | tag | DACT | CORNER | seed | VQ1 sr / t_med | gen sr | thr_p95 / yaw_p95 / flips | style |
|---|---|---|---|---|---|---|---|---|---|
| R1 (a) | 3268648 | inc6_d1_s0 | 1 | 0 | 0 | 0.997 / 10.92 s | 0.727 | 1.000 / 0.333 / 0.2% | FAIL (thr) |
| R1 (a) | 3268649 | inc6_d4_s0 | 4 | 0 | 0 | 1.000 / 11.12 s | 0.649 | 0.998 / 0.093 / 0% | FAIL (thr) |
| R1 (a) | 3268650 | inc6_d16_s0 | 16 | 0 | 0 | 1.000 / 10.56 s | 0.571 | 0.181 / 0.039 / 0% | PASS |
| R1 (a) | 3268651 | inc6_d4_s1 | 4 | 0 | 1 | 0.999 / 12.09 s | 0.689 | 0.997 / 0.083 / 0% | FAIL (thr) |
| R1 (b) | 3268718 | inc6_c4_s0 | 1 | 4 | 0 | 1.000 / 10.39 s | 0.940 | 0.993 / 0.049 / 0% | FAIL (thr) |
| **R1 (b)** | **3268719** | **inc6_c16_s0** | **1** | **16** | **0** | **1.000 / 9.86 s** | **0.982** | **0.061 / 0.009 / 0%** | **PASS** |
| R1 (a+b) | 3268720 | inc6_d4c8_s0 | 4 | 8 | 0 | 0.999 / 9.76 s | 0.633 | 0.121 / 0.014 / 0% | PASS |
| R2 | 3268876 | inc6_c16_s1 | 1 | 16 | 1 | (follow-up: seed variance; not a ship blocker) | | | |

**A/B VERDICT — the targeted corner penalty wins on every axis simultaneously.** The blunt
||delta a||^2 arm trades smoothness against generalization monotonically (gen 0.727 -> 0.649 ->
0.571 as dact 1 -> 4 -> 16) because it taxes every fast correction, including the ones hard
random courses need. The corner arm decouples them: c16 is the FASTEST (9.86 s), the BEST
generalizer (0.982 — above even inc5's 0.939 aero-plant number, on the harder mixer plant),
AND the smoothest (thr_p95 0.061, yaw_p95 0.009) — pricing only the (thr-rail x rate) corners
removed the incentive to live near the rails at all, without flattening responsiveness.
At w=4 the corner tax is too weak to move collective off the rails (c4 thr_p95 0.993), though
its gen 0.940 already beats every dact arm. Yaw dither died in ALL seven arms (mixer physics).
The COMBO (d4+c8) confirms the attribution: style PASS and VQ1 9.76 s (0.10 s faster than c16
on the known track), but gen collapses to 0.633 + 3 VQ1 collisions — the dact component costs
generalization wherever it appears, and a 0.10 s single-course edge inside overfit territory
does not buy back a 35-point gen gap (META gap #1) or the validity delta (0.999 vs clean
1.000). **SHIP = c16_s0.** R2 (3268876) = one confirmatory seed of corner=16 for variance
data before the live session; follow-up, not a blocker (deploy matrix, gen, and style all
green on the shipped seed).

**TRAIN_RC=1 note (benign):** every job trains the full 6000/6000 updates and saves
checkpoints, then a post-training teardown step crashes (`ValueError: Unknown action frame:
body`, a diffaero export/test path). The chained evals load the final saved checkpoint.

**Early R1 findings:** ① the YAW DITHER — the #1 live killer — is DEAD in every mixer-ON arm
(yaw_p95 <= 0.11, flips 0%, even at dact=1-class weights: the mixer PHYSICS killed it, not the
regularizer); ② collective bang-bang SURVIVES dact=4 (thr_p95 0.997, 97% thrust saturation —
straight-line pulsing is mixer-cheap) and dies at dact=16, but dact=16 costs generalization
(0.571 vs 0.689); ③ note d16 is simultaneously FASTER on VQ1 (10.56 vs 12.09) — smoothness is
not costing lap time on the known course, it costs adaptation to hard random courses.

**d16_s0 early deployment matrix (laptop, `--plant mixer`, pulled ckpt md5 0badba6e...):
13/13 FINISHED** — simstart lat {0,2,3} (10.3-10.7 s), racestart lat 2 (10.0 s), trainreset
lat 2 (9.3 s), handoff lat {2,3} (8.8-9.2 s), PLUS the full perturbed-seam grid (speed
4.0-6.5 x dist 2.5-3.5 at lat 2: 6/6 FINISHED, 8.8-9.3 s) — the exact grid where inc5+ys0
died uniformly at gates 2-3. No mitigation flags anywhere.

The (a) arm jobs started before the R7 code landed on Adroit — harmless: their cfgs carry no
`rw_corner`, and R7 defaults to 0 (their chained evals import the new file with identical
reward). Corner-weight sizing: at the inc5 bottom-rail style (thr 0 + one railed rate axis)
R7 = 0.5/tick x w, so w=4 matches the rail pressure of dact~4 while leaving hover-band
corrections ~free; w=16 = the strong arm.

## 8. Eval vs inc5 — everything MIXER-ON (the measured plant)

The decisive datum (job 3268841): **inc5 evaluated on the mixer plant scores sr 0.000 on BOTH
courses** (VQ1: 16,601 episodes, 16,549 misses; random: 15,137 episodes, 0 finishes) with its
signature in the open — yaw_p95 = 1.000, yaw_flip = 0.817 (the per-tick rail dither). The live
0/20 transfer failure, reproduced wholesale at eval scale. Same checkpoint, same courses, only
the plant's two mixer corners priced in.

| metric (mixer-ON eval) | inc5 (t96_s1 datum) | **inc6 (c16_s0, shipped)** |
|---|---|---|
| held-out VQ1 success | **0.000** (16,549/16,601 miss) | **1.000** (2,560 eps, 0 coll, 0 miss) |
| median lap (VQ1) | — (no finishes) | **9.86 s** (p90 9.92, min 9.76) |
| generalization (random) | **0.000** | **0.982** (2,261 eps) |
| roll / tilt succ max (VQ1) | — | 57.4 / 62.9 deg |
| pass offset med / p90 / max (VQ1) | — | **0.145 / 0.271 / 0.277 m** (aperture 0.75; zero-contact margin everywhere) |
| cmd saturation (any axis) | yaw-railed by design | **0.0%** (the policy never touches a rail) |
| ACTRATE thr_p95 / yaw_p95 / flips | 0.186 / **1.000** / **81.7%** | **0.061 / 0.009 / 0.0%** |
| deployment rollouts (laptop, mixer plant) | 1/6 gates typical, needs --yaw-scale 0 | **16/16 FINISHED**, no mitigation flags |

*(Context: inc5's banked 1.000/9.52 s was an AERO-plant eval — the plant that could not see
the mixer corners its style exploited. The aero-plant numbers were real; the plant was not.)*

**inc6 c16 deployment matrix (laptop, `--plant mixer`, ckpt md5 8fb8855e...): 16/16 FINISHED**
— simstart lat {0,2,3} (9.5 s), racestart lat {2,3} (9.6 s), trainreset lat 2 (9.2 s), handoff
lat {2,3} (8.6-8.7 s), + the perturbed-seam grid speed {4.0-10.0} x dist {2.5-3.5} at lat 2
(8.3-8.7 s; the grid where inc5+ys0 died uniformly). Native latency tolerance (trained {1,2,3})
shows: lap time varies < 0.15 s across lat 0 -> 3.

## 9. Deployment notes (for the next ShadowPC session)

* **Fly inc6 with NO mitigation flags** (no --yaw-scale, no --max-rate) — the shipped style
  never saturates a command axis (0.0% on all four) and holds 0.27 m worst-case pass margins.
  `fly_rl.py --checkpoint rl/checkpoints/stage1_inc6_actor.pth` (default still points at inc4
  — pass explicitly; the sidecar auto-loads).
* **Standing start first** (`--no-bridge`, the deployment target per INC5-LIVE Appendix B —
  deletes the non-deterministic bridge seam); bridge as comparison. In-twin both are 6/6 at
  lat 0-3.
* `--debug-obs` stays ON; the live-reset/spin guards from the diag session remain mandatory;
  full ESC->Enter reset between flights.
* **Run the 10-min `mixer_probe2.json` errand FIRST** (Section 2): it nails the top-rail x
  large-demand corner and the r/p zeta — derisks a model re-fit if inc6 shows any residual
  live gap, and is independent of flight outcomes.
* Expected live behavior if the mixer model is right: clean gate-0 passes from standing start,
  laps ~10-12 s (twin 9.9 + live overheads). If it fails, the failure MODE is the signal:
  parasitic-climb-like = bottom-rail model wrong (unlikely — anchored); lateral-late at speed
  = top-rail depth (the errand's corner); anything at step 0 = deployment layer (byte-correct
  per the diag, but re-verify with replay_obs.py).
* R2 seed-variance job (3268876, corner=16 seed 1) lands ~1 h after this writeup; check
  `/scratch/network/fl3689/inc6_run_3268876.out` before the live session for the variance
  picture. inc5 (`stage1_inc5_actor.pth`) stays in the repo as the pre-mixer datum; inc4
  remains retired.
