# inc8 Deterministic-Stability Re-train — σ_p0 measurable (inc8-deterministic-retrain, 2026-06-19)

**Worker:** inc8-deterministic-retrain (opus-4.8, EFFORT max). **Branch:** `inc8-deterministic-retrain`
(local, NOT pushed). **Status:** LIVE — engineering done + laptop-validated; Adroit training launching.

## Goal
Produce an inc8 policy whose DETERMINISTIC mean (test=True) flies the 2-axis camera-pointing course
(look-at ON, g_pitch high enough to seat fixes), then MEASURE its binding gate-4 terminal-centering σ_p0
with the validated torch instrument. Today σ_p0 is NO-DATA: rc1 flies only STOCHASTICALLY; its
deterministic mean crashes ~1 s post-spawn. Deliverable = a measured 2-axis σ_p0 (GO or NO-GO), or a
supported negative that the look-at magnitude is incompatible with a stable deterministic mean.

## ROOT CAUSE — quantified (the new finding)
rc1's checkpoints, read directly (`handoff/.../probe_logstd.py`, run on Adroit):

| seed | action_std (4 CTBR dims) | note |
|---|---|---|
| **0** | **[7.39, 0.16, 0.18, 7.39]** | dims 0,3 SATURATED at exp(2)=7.39 (LOG_STD_MAX); raw logstd 9.0/4.3 |
| 1 | [0.26, 0.26, 0.24, 0.29] | healthy ~0.25 |
| 2 | [0.25, 0.23, 0.24, 0.28] | healthy ~0.25 |

diffaero samples rollout actions `tanh(mean + std·N(0,1))` with a **state-independent learnable
`actor_logstd`** (network/agents.py). rc1 seed0 (the lone deterministic yaw-flyer, σ=0.198) trained with
**two action channels at the MAX std** → near-random bang-bang in the rollout. The policy learned to fly
*in expectation over that chaos* (feedback masks it), so its **mean is an unvisited, unstable operating
point**. The `entropy_weight=0.01` bonus (constant, no schedule) with no std ceiling drove the
saturation. This is the mechanism behind "stochastic 50% → deterministic crash", now localized to the std.

## The fix — three levers (TRAINING-CONFIG only; reward + look-at primitive untouched)
1. **NOISE-ANNEAL (load-bearing)** — `rl/inc8_noise_anneal.py`. Clamp `actor_logstd` to a decaying std
   CEILING (loose hold `std_hold=0.6` from update 0 to kill the 7.39 saturation; geometric decay
   `std_hold→std_floor=0.03` over the back `1-hold_frac` of training to collapse the rollout onto the
   mean) + anneal `entropy_weight→0` over the same window. The policy is then trained on near-deterministic
   actions ⇒ the mean flies. CLAMP not SET (PPO may still go lower). Wired into the existing
   `step_with_periodic_save` monkeypatch in `peregrine_train_inc8.py`, gated `+algo.noise_anneal=true`;
   OFF == byte-identical inc8. Zero diffaero-clone edits.
2. **DETERMINISTIC SELECTION** — `rl/inc8_snapshots.py` (dense numbered snapshots during training,
   `+snapshot_every=100 +snapshot_from_frac=0.4`) + `rl/inc8_select_ckpt.sbatch` (post-hoc: cheap
   deterministic reach-rank over all snapshots via the VALIDATED σ_p0 instrument, then full σ_p0 on the
   top-K). Replaces the runner's STOCHASTIC high-water "best", which did not transfer to the mean.
3. **g_pitch sweep** — find the highest look-at pitch gain a deterministic mean both FLIES and POINTS at
   (rc1's deterministic ceiling was 0.5, where fix_rate=0; we need ≥1 to seat fixes). `GPITCH` submit knob.

## Verification done (laptop .venv, no diffaero)
- `tests/test_inc8_noise_anneal.py` + `tests/test_inc8_snapshots.py`: **29 passed**. Pins the squash
  round-trip to diffaero's constants (init param 0 ↔ std 0.2231), the schedule shape (hold flat / geometric
  decay / floor), OFF==None resolves, and the REAL `actor_logstd` clamp reproducing the rc1 fix
  (7.39→0.6→0.03 on the saturated channels; healthy channels untouched; clamp only lowers).
- `py_compile` of the trainer; new modules import clean (no torch/diffaero at module level).
- diffaero PPO/agents/runner read from the Adroit clone to wire the lever against the real API.

## Campaign plan
**Batch 1** (seed0, parallel, noise-anneal ON, snapshots on) — brackets g_pitch × cap-strength:
- `ds_gp1p0_h06`: GPITCH=1.0 STD_HOLD=0.6 STD_FLOOR=0.03  (lever test + aggressive cap)
- `ds_gp1p0_h15`: GPITCH=1.0 STD_HOLD=1.5 STD_FLOOR=0.05  (cap-strength bracket)
- `ds_gp1p5_h06`: GPITCH=1.5 STD_HOLD=0.6 STD_FLOOR=0.03  (g_pitch headroom)

Each: precheck-gated (3 updates, exits before the 8 h burn on a wiring bug), stochastic FLIGHTCHECK early
kill, monitored mid-run via the deterministic snapshot probe (scancel non-flyers at the anneal tail).
**Batch 2** (informed): best deterministic config → seeds 1,2 (the ≥3-seed GO) + push g_pitch; then
`inc8_select_ckpt` + the full σ_p0 GO/NO-GO.

## Adroit pipeline — PROVEN (smoke job 3278163, NUPD=3)
End-to-end on the real diffaero/Adroit stack: `PRECHECK_RC=0`; noise-anneal engaged with exact scheduled
values (update0/1 std_ceil=0.60 entropy=0.010 hold; update2 std_ceil=0.221 entropy=0.0067 anneal) — the
`actor_logstd` clamp ran on the real agent, no error; snapshots `upd00001/2/3` written; only "error" the
known-cosmetic onnx-guard (exit 0); no NaN/traceback. Files synced byte-exact (CR-stripped LF) + sha-verified.

## Merge safety — GREEN
`scripts/green_gate.py --full` on the branch: all 3 load-bearing invariants GREEN (OFF==inc7 AST parity,
+L sign-faithfulness, VQ1 import guard), test-count sentinel **1077 ≥ 933**, **full suite 1077 passed**
(4:53). My changes are additive + gated (trainer only; env/obs/fly_rl untouched) ⇒ safe to report up.

## Autonomy / ops
Driving Adroit via the adroit-connector `serve` daemon (`adroit.py x "<cmd>"`, port 8765) — alive; needs
Fengyou's Duo only on an SSH-session drop (daemon self-heals on next `x`). File sync = base64 write + sha
verify (NOT `upload`, which re-prompts the passphrase). All login-node cmds are short/AUP-safe; compute on SLURM only.

## Results — Batch 1 (LIVE, launched 2026-06-19 ~20:45 ET; ~8h each)
| job | RUNTAG | g_pitch | std_hold→floor | precheck | stoch. flightcheck | det. reach (snap probe) | σ_p0 |
|---|---|---|---|---|---|---|---|
| 3278164 | ds_gp1p0_h06 | 1.0 | 0.6→0.03 | pending | — | — | — |
| 3278165 | ds_gp1p0_h15 | 1.0 | 1.5→0.05 | pending | — | — | — |
| 3278166 | ds_gp1p5_h06 | 1.5 | 0.6→0.03 | pending | — | — | — |

All 3 completed (~50 min each on A100 — the 8h was the SLURM *limit*). All **FLEW stochastically**
(success_rate_max 0.65/0.66/0.65, beating rc1's 0.50–0.54); 25 dense snapshots each; precheck RC=0.

## ⭐ HEADLINE — Objective A ACHIEVED, Objective B MEASURED = NO-GO (deterministic selection, seed0)
**Objective A — first deterministically-flyable 2-axis inc8 policy:** gp1.0 snapshot `upd01700` reaches
gate-4 at **det reach_rate 0.467** under test=True (rc1 = **0.000** at gp1.0 — it could not). The
noise-anneal/std-cap lever closes the "un-measurable" gap.

**Objective B — measured 2-axis gate-4 σ_p0 (gp1.0, the deliverable):**
| snapshot | det reach | σ_p0_lat | lat_p99 | σ_p0_vert | fix_rate | verdict |
|---|---|---|---|---|---|---|
| upd1700 | **0.467** | 0.1999 | 0.403 | 0.204 | **0.000** | NO-GO |
| upd1900 | 0.420 | 0.1731 | 0.386 | 0.199 | **0.000** | NO-GO |
| upd1800 | 0.320 | **0.1550** | 0.368 | 0.186 | **0.000** | NO-GO |
gp1.5 (cap0.6): only weakly det-flyable (best upd1900 reach **0.129**, σ_p0 0.139, fix_rate 0); final
upd4000 reach 0. **2-axis σ_p0_lat ≈ 0.15–0.20 m = NO-GO vs 0.08** (lat_p99 0.37–0.40 vs 0.24).

**ROOT of the NO-GO = `fix_rate=0.000` at EVERY deterministically-flyable gain (gp1.0 AND gp1.5).** The
look-at at gains that fly on the mean is too weak to seat a single fix → σ_p0 stays at the no-fix baseline
(≈ rc1's yaw-only 0.198). Fix-seating needs band_el≈20° (gp≈3 in the rc1 sweep), which does not fly
deterministically. **The deterministic-stability problem is SOLVED; the binding wall is now fix-seating-vs-flight.**

**Lever-1 refinement:** det reach is BEST at the std-cap hold phase (upd1700, std=0.6) and DEGRADES through
the anneal (upd4000 reach 0.044). ⇒ the std CEILING (killing rc1's exp(2)=7.39 saturation) is the
flight-stability win; the aggressive anneal→0.03 is unnecessary/slightly harmful. Selection-on-det-reach
correctly picks the hold-phase snapshot.

## Batch 2 — fix-seating frontier test = AIRTIGHT NEGATIVE (jobs 3278250/51 selection, seed0)
gp2.0 and gp3.0 were TRAINED with the std cap, then deterministically selected at their OWN gains:
**both = deterministic reach_rate 0.000 across ALL 25 snapshots (NO-DATA).** The mean does not fly at the
fix-seating gains, even with the noise-anneal.

## FULL BRACKET (seed0, deterministic reach @ trained gain) — the wall, quantified
| g_pitch | det reach_rate | σ_p0_lat | fix_rate | flies on mean? | seats fixes? |
|---|---|---|---|---|---|
| 1.0 | **0.467** | 0.16–0.20 | 0.000 | ✅ | ❌ |
| 1.5 | 0.129 | 0.139 | 0.000 | ~ | ❌ |
| 2.0 | 0.000 | — | — | ❌ | ❌ |
| 3.0 | 0.000 | — | — | ❌ | ❌ |
Deterministic flyability falls monotonically with g_pitch; fix-seating needs band_el≈20° ⇒ g_pitch≈3
(rc1 sweep). **The two are MUTUALLY EXCLUSIVE** — the gain that flies on the mean is too weak to fix; the
gain that could fix does not fly. This is the goal's "well-supported negative" across a 5-run bracket.

## CONCLUSION
- ✅ **Deterministic-stability problem SOLVED** (the lever works): first deterministically-flyable 2-axis
  inc8 (gp1.0 det reach 0.467; rc1 = 0.000). Root cause (rc1's saturated actor_logstd) found + fixed.
- ❌ **Gate-4 NOT closed by the look-at bet: 2-axis σ_p0_lat 0.16–0.20 m = NO-GO**, because fix_rate=0 at
  every deterministically-flyable gain. Definitive across the bracket.
- 🚩 **RECOMMENDED NEXT LEVER (not more look-at RL):** the NEAR-FIELD-GATE ESTIMATOR (lower the ~12 m PnP
  floor so a fix seats at a gain that flies), or a reward/architecture reconsideration of fix-seating.

REMAINING (optional firming, BLOCKED on serve-daemon restart): seeds 1,2 of gp1.0 to confirm objective A
reproduces (N=3). Does NOT change the gate-4 verdict (the negative does not depend on A's reproducibility).

## MEMORY-DELTA
1. **NOISE-ANNEAL/std-cap lever WORKS — first deterministically-flyable 2-axis inc8** (gp1.0 det reach
   **0.467**, rc1 was 0.000). Root cause of rc1 found+fixed: 2 action channels saturated at std=exp(2)=7.39
   (entropy bonus, no ceiling) → mean unflyable; clamping actor_logstd to a ceiling fixes it.
2. **Measured 2-axis gate-4 σ_p0 = 0.15–0.20 m = NO-GO vs 0.08** (lat_p99 0.37–0.40; σ_vert ~0.20). The
   un-measurable gap is CLOSED — but the verdict is NO-GO.
3. **AIRTIGHT NEGATIVE (5-run g_pitch bracket, seed0): det reach 0.47(gp1.0)→0.13(gp1.5)→0.00(gp2.0)
   →0.00(gp3.0), all fix_rate=0.** Deterministic flight and fix-seating (needs gp≈3) are MUTUALLY
   EXCLUSIVE ⇒ the look-at magnitude needed for fixes is incompatible with a stable deterministic mean.
4. ⇒ **NEXT LEVER = near-field-gate estimator** (lower the ~12 m PnP floor so a fix seats at a flyable
   gain) or a reward/architecture reconsideration of fix-seating — NOT more look-at-gain RL.
5. **Lever-1 refinement: the std CEILING is the flight win; anneal→0.03 DEGRADES det reach** (best at
   hold std=0.6). Select on det reach, not the stochastic high-water "best".
6. Caveat: Objective A single-seed (seed1/2 optional firming — does NOT change the gate-4 verdict).
   Branch `inc8-deterministic-retrain`, green_gate GREEN (1077 passed). Provisional on #37 emul fidelity.
