# P2 INC8-RL Worker #3 — torch train-env port (Deliverable 2)

**Built:** the inc8 case-C training env — the torch-vectorised production twin of the numpy
escape-hatch reference `rl/estimator_emul.py` (GREEN on main). Camera-pointing changes fix-density
IN-LOOP over the GPU envs, so a PPO policy learns to point. Built in PARITY-GATED STAGES; every gate
is GREEN. **Branch: `claude/admiring-leavitt-7c331e`** (pushed).

## TL;DR
- **S1–S4 parity GATES all GREEN** (torch == numpy reference, float64 ≤ 1e-4 — achieved 0 … 7.6e-6).
- inc8 env (`PeregrineRacingInc8`) wired: 20-dim obs from the in-loop KF, R1'/GT-anchor/conf-shaping/
  R5'(A,B,C), critic 33→36, BSR3 spin-gate. **Default-OFF == byte-identical inc7** (structural proof).
- **CPU integration smoke GREEN** (the laptop proxy): pointing → fix-density↑ → KF-confidence↑ → R5'↑,
  no NaN, obs-dim 20, BSR3 ignores the ~11 rad/s super-rate transient.
- **Full suite: 762 passed / 42 skipped / 0 failed** (the 42 new inc8 tests included; 42 skips are the
  pre-existing diffaero/cluster-only tests). No regression.
- **GPU smoke = BUILT + ready** (`rl/peregrine_inc8_smoke.sbatch` + `rl/peregrine_train_inc8.py`) but
  **NOT yet executed** — it needs Adroit (no diffaero/GPU in this worker's laptop session). **Hand up to
  run on Adroit** (escape hatch: do not launch the ladder until the GPU smoke shows learning).

## Files (all on the branch; `git add` of only these paths)
| path | what |
|---|---|
| `rl/inc8_estimator_emul.py` | S1 batched fix-surrogate geometry/accept/cov/sample · S2 batched LinearKF · S3 BatchedEstimatorEmulator + the 20-dim `obs_zup_torch` seam. numpy+torch only (no scipy/diffaero → loads on Adroit). |
| `rl/reference_line_torch.py` | S4 `BatchedReferenceLine.progress` (R1' arc-length primitive over Γ). |
| `rl/inc8_reward.py` | R1' / GT-anchor / confidence-shaping / R5'(A,B,C COWORK-3 2-axis terminal-lock) / BSR3 — pure torch. |
| `rl/peregrine_racing_inc8.py` | `PeregrineRacingInc8(PeregrineRacing)` — wires it all; OFF == inc7. |
| `rl/peregrine_train_inc8.py` | launcher: `ENV_ALIAS["peregrine_racing_inc8"]`, sidecar carries obs_dim (S7). |
| `rl/peregrine_inc8_smoke.sbatch` | the Adroit GPU smoke (1 seed, arm A, GPU-util sampler, AUP-compliant). |
| `tests/test_inc8_*` (7 files, 42 tests) | the S1–S4 parity gates + reward/BSR3 + integration smoke + OFF-identity. |

## Parity gates (the non-negotiable: torch obs == the verified selection metric)
Reproducer: `handoff/p2-inc8-rl-trainport-2026-06-14/_measure_parity.py`. **float64 is the gate**
(matches the numpy reference dtype); float32 is the GPU-env deploy dtype (informational).

| gate | quantity | float64 worst | float32 worst | status |
|---|---|---|---|---|
| **S1** | geometry range | **0.0** | 6.8e-6 | ✅ |
| **S1** | p_accept band-pass | **1.1e-16** | 6.9e-7 | ✅ |
| **S1** | `in_image` (boolean) | **exact** | exact | ✅ |
| **S1** | fix_sigma / cov / sample z | ≤1e-4 (test) | — | ✅ |
| **S2** | KF state x (60-step seq) | **1.6e-13** | 3.1e-4¹ | ✅ |
| **S2** | KF covariance P | **7.1e-15** | 1.0e-5 | ✅ |
| **S3** | frame-seam identity (obs == obs_from_truth) | **7.6e-6** | 1.5e-5 | ✅ |
| **S3** | full-step injected-randomness (KF == numpy recon) | ≤1e-4 (test) | — | ✅ |
| **S4** | refline progress (vs numpy `.progress`) | **2.8e-14** (p99 0) | p99 1.9e-5² | ✅ |

¹ float32 KF state over 60 random steps accumulates to ~3e-4; the **gate is float64** (1.6e-13). The env
runs float32 natively and is self-consistent — the obs-normalizer + the reward DELTA absorb it.
² S4 float32 absolute arc-length loses ~mm over ~190 m; R1' uses `s_curr − s_prev` where it cancels.

### S3 — the three GREEN conditions reproduced in torch (the escape-hatch contract)
- **(a) FRAME-SEAM IDENTITY** — KF seeded at truth → full obs == `obs_from_truth` ≤ 7.6e-6 over 250
  random states × both virtual-flips. Pins NED↔Z-up↔gate-frame wiring AND **OBS SIGN = +L** (a wrong
  flip is ~24 m). *(Caught a real bug here: my first `R_camera_from_body` had the Ry(−20°) sin-signs
  flipped — the parity gate flagged it immediately. Now == `frames.R_camera_from_body()` to 1e-12.)*
- **(b) POINTING → FIX-RATE monotone** — accept density is monotone-decreasing in crab; centred gate
  fixes readily (>0.5), off-pointed (70°) leaves the frame and ~never fixes (<0.05).
- **(c) obs[17:20] non-degenerate** — bounds [0,1]; pooled NEES ∈ [0.8,1.3] bias-off (the confidence
  channel is HONEST); c_inplane rises across a fix; age_norm → 1 on a dropout.

## inc8 deltas over the inc7 spine (S5/S6/S7)
**SHARED SPINE (all arms)** — everything inc7 frozen except:
- **R1'** `rw_progress·(s_curr − s_prev)` over Γ (`reference_line_inc8.json`) — replaces R1-to-centre
  (closes the corner-cut M-2 by construction). `rw_progress=10`.
- **T4 finish-time KEPT** (the dominant speed term, unchanged).
- **GT-estimator-error anchor** `−rw_estimerr·|KF_pos − truth|_inplane-gateframe` — **FLAT** weight, no
  proximity schedule, reward sees TRUTH (actor sees the noisy KF obs). `rw_estimerr=2`.
- **Confidence/staleness dense-shaping** `+rw_conf_shape·anneal·(c_inplane − age_norm)` from obs[17:20],
  ANNEALED late (control-step ramp). Small (`rw_conf_shape=0.05`), ablatable.
- **R4 fixed 60° cone** (inc7 unchanged — the confidence-gated cone is an optional side-ablation, NOT
  the spine). R2/T1/T2/T3/R3/R5/R6/R7 frozen.

**ARM-SPECIFIC R5'** = the COWORK-3 perception reward (`handoff/cowork-2026-06-14/perception-reward.md`):
`rw_perc · w_term(d) · v(α,β) · max(Δs,0)`, gated to in-image. `v(α,β)=exp[−((α/45°)⁴+(β/29.5°)⁴)]`
(SWIFT/Geles exp(−δ⁴) generalised to Qin azimuth/elevation — the narrower V-FoV penalises elevation
more tightly). `α=atan2(X,Z)`, `β=atan2(Y,√(X²+Z²))` from the surrogate's `t_cam`.
- **A (default)** terminal-locked `w_term(d)=w0+(1−w0)·clip((d_acq−d)/(d_acq−d_lock),0,1)`, d_lock=5 m,
  d_acq=24 m, w0=0.15 (the Azhari λ(d) ramp).  **B** same with `w_term≡1` (flat).  **C** R5' off.
- `rw_perc=0.5` (~5% of progress, per SWIFT/Geles/Song). Progress-gating `max(Δs,0)` kills the
  slow-to-look loiter (anti-farming). The visibility plateau gives ~zero gradient when centred (CAPS:
  nothing to chatter against).

**CRITIC 33→36** — `get_state` (TRUTH gate-relative, 3-gate lookahead) + the confidence triple. The
truth-seeing asymmetric critic is the anti-damping mechanism (no shaped damping term, per d5/COWORK).

**BSR3 spin-gate (S6)** — sustained-spin termination on the **REALIZED** rate (`self._w`, NOT the
command): accumulate dt while |ω|>spin_rate_abort, reset otherwise, abort past spin_time_abort.
Defaults OFF (byte-identical); the launcher sets `spin_rate_abort=10`, `spin_time_abort=3.0` (d5 §4.1).
**Verified it does NOT false-abort a legitimate ~11 rad/s super-rate transient** (the plant amplifies
the command; a <3 s burst at 11 rad/s never aborts; a sustained >3 s spin does).

**S7 obs_dim=20 / critic 36 / sidecar** — `obs_dim=20` when inc8 on (else 17). The launcher writes
`obs_dim`+`inc8`+`r5_arm` into the checkpoint `actor.json` sidecar so deploy/load gates on it (inc7
17-dim checkpoints still load). *Note: `rl/fly_rl.py` is P4-owned (import-only here) — the sidecar is
written by the inc8 launcher; the deploy-side READ of it is P4's wiring.*

**Default-OFF == byte-identical inc7** — `step`/`get_observations`/`get_state` each short-circuit to
`super()` on the first statement when `inc8` is off; `__init__` returns before building any inc8 state.
Pinned by an AST structural test (`test_inc8_off_identity.py`). The numerical byte-identity check (run
inc7 vs inc8-OFF env over N steps) is one line for the Adroit smoke.

## Integration smoke (CPU, laptop — the proxy for the Adroit GPU smoke)
`tests/test_inc8_env_integration.py` runs the FULL inc8 signal chain (truth→NED→batched KF emulation→
20-dim obs→reward deltas+BSR3) over a scripted head-on approach, no diffaero. GREEN:
- **no NaN/inf** anywhere; **obs is 20-dim**; reward terms finite; BSR3 silent on the ~11 rad/s transient.
- **pointing PAYS** — centred camera (crab 0) vs off-pointed (60°): fix-rate +>0.05, in-frame >0.5 vs
  lower, terminal-window confidence higher, R5' higher. This is the precondition for *pointing emerges*.
- the **terminal fix-DROUGHT is reproduced** in-loop: confidence rises from cold into the 16–28 m
  accept window then FALLS below 16 m (band-pass cutoff) — exactly the phenomenon inc8's terminal-lock
  reward targets, so confidence is correctly non-monotone, not rising-to-the-gate.
- *Subtlety documented:* with the one-signed bias ON (binding case-(b)), a fix pulls the KF toward
  truth+bias, so absolute |KF−truth| floors at ~the bias (the irreducible case-C error the GT-anchor +
  boresight-calib target) — NOT to 0. On a short constant-velocity script the truth-synthesised IMU
  dead-reckons near-perfectly, so absolute-error REDUCTION is a long-horizon/maneuvering property left
  to the Adroit PPO smoke; the honest laptop signal is the (NEES-calibrated) CONFIDENCE.

## GPU smoke — STATUS: BUILT, ready, NOT executed (needs Adroit)
`rl/peregrine_inc8_smoke.sbatch` (1 seed, arm A, 2048 envs, map-ON measured plant + contact-true
geometry + BSR3; AUP-compliant: SLURM-only, /scratch output, accurate --mem, a `nvidia-smi` util
sampler, a 3-update precheck then a 300-update smoke). It logs `inc8_pointing_rate /
inc8_terminal_pointing / inc8_fix_rate / total_reward` per `log_freq` for the *reward-rising +
pointing-emerges* read, and the GPU-util summary for the *saturation* check.

**This laptop worker session has no diffaero/GPU**, so I could not run it. Acceptance criterion #4
(GPU saturated, reward rising, pointing emerges) is the one open item → **run on Adroit**. Per the
escape hatch: **do NOT launch the {L0,L1,L2}×{A≥5,B3,C1} ladder until this smoke shows learning** (no
NaN, GPU saturated, the pointing metric improving).

## Footgun checks honored
- torch obs == numpy reference (parity GATE, non-negotiable) — GREEN.
- NEVER `R_w2g@(gate_map − p_KF)`: `gate_pos` is TRUTH (no map bias); perception bias enters ONLY via
  the one-signed fix bias (mirrors `estimator_emul`). OBS SIGN = +L (frame-seam identity enforces it).
- Extends the ENV (subclass via ENV_ALIAS); the diffaero plant clone is untouched; inc7 byte-identical
  when OFF. NO_GRAD emulation (PPO confirmed; reward detached). BSR3 gates the REALIZED rate.
- course_mode=vq1 REQUIRED for inc8 (R1' is along the VQ1 Γ) — asserted in `__init__`.

---

## MEMORY-DELTA (≤10 lines — do NOT edit memory/; commander banks)
- **inc8 torch TRAIN-ENV port DONE & PUSHED** (branch `claude/admiring-leavitt-7c331e`). S1–S4
  torch==numpy parity GATES all GREEN (float64 ≤1e-4: range 0, p_accept 1.1e-16, KF 1.6e-13, frame-seam
  7.6e-6, refline 2.8e-14). 42 new tests; full suite **762 passed / 42 skipped / 0 fail** (no regression).
- New modules (numpy+torch only, no scipy/diffaero → load on Adroit): `rl/inc8_estimator_emul.py`
  (batched fix-surrogate+LinearKF+emulator+`obs_zup_torch`), `rl/reference_line_torch.py`,
  `rl/inc8_reward.py`, env `rl/peregrine_racing_inc8.py`, launcher `rl/peregrine_train_inc8.py`.
- Spine: R1' arc-progress over Γ + T4 + FLAT truth-seen GT-estimator-error anchor + annealed conf-shaping
  + R4 fixed 60°. R5' = COWORK-3 2-axis terminal-lock (arm A default / B flat / C off). Critic 33→36
  (truth + triple). BSR3 spin-gate on REALIZED ω (10 rad/s / 3 s) — verified NOT to false-abort ~11 rad/s
  transients. obs_dim 17→20; sidecar carries obs_dim (deploy gates on it; fly_rl is P4, untouched).
- **OFF == byte-identical inc7** (AST-proven; numerical check = 1 line in the Adroit smoke).
- 🚩 **GPU smoke BUILT (`rl/peregrine_inc8_smoke.sbatch`) but NOT run — needs Adroit** (no diffaero/GPU
  in the worker session). Run it; **do NOT launch the ladder until it shows learning** (no NaN, GPU
  saturated, pointing metric rising). CPU integration smoke GREEN (pointing→fix→confidence→R5', no NaN).
- Bug caught by the S1 gate: `R_camera_from_body` Ry(−20°) sin-sign flip — fixed (== frames to 1e-12).
- inc8 REQUIRES course_mode=vq1 (R1' along the VQ1 Γ); asserted. One-signed bias floors absolute
  |KF−truth| at ~bias (irreducible case-(b) error the GT-anchor/boresight-calib target) — honest laptop
  signal is the NEES-calibrated CONFIDENCE, absolute-error reduction is the Adroit long-horizon property.
