# SHADOWPC-INC6-DIAG — Root cause of the inc6 0/15 live failure

**Session:** SHADOWPC-INC6-DIAG (2026-06-12). **Model:** claude-fable-5.
**Input failure:** `handoff/shadowpc-inc6-live-2026-06-12/WRITEUP.md` — inc6 (`stage1_inc6_actor.pth`,
md5 `8fb8855e…`) 16/16 on the laptop deploy matrix, 0/15 live on ShadowPC.
**Commits this session:** `bcc93f9` (the fix), `325e191` (rate_sysid tilt-abort), this writeup.

---

## TL;DR

**ROOT CAUSE: DEPLOYMENT LAYER — the ODOMETRY/command ROLL-axis convention was
misidentified.** Three coupled errors, self-consistent at level attitude and physically
impossible when tilted:

1. **The raw ODOMETRY quat IS the true attitude.** fly_rl's "roll-inversion artifact undo"
   (`R(-roll, pitch, yaw)`) was undoing an artifact that does not exist on this message.
2. **Raw→true rate signs are [+1, −1, +1]**, not [−1, −1, +1] — only the PITCH rate report
   is inverted.
3. **The live sim inverts the ROLL command as well as yaw**: command→rate sign is
   **[−1, +1, −1]**, while the twin/training plant uses [+1, +1, −1].

The old bookkeeping (roll-negated attitude + roll-negated rate feedback + un-negated roll
command meeting a roll-inverting plant) is a **mirror of the real world**. In closed loop at
near-level attitude the mirror is dynamically self-consistent — every prior verification
(VQ1 flights, the 2026-06-10 sign re-verification, the inc5 byte-correct replay, the CTBR
bridge whose controller signs were tuned end-to-end) lived in that regime and could not see
it. A roll-only mirror is an improper transform: rigid-body kinematics breaks at tilt. inc6
is the first policy that flies tilted (−55° pitch, banked) from step 1 — the policy's roll
perception and roll authority were both sign-flipped the moment it left the pad, producing
the deterministic lateral sweep → left-right oscillation → spin. The operator's visual was
exactly right.

**The laptop matrix could not catch it by construction**: `telemetry_from_truth` applied the
same assumed artifact model that `build_obs` undoes — any misidentification round-trips to
zero. Only live MAVLink exercises the real convention. (The S17 asymmetry note called it.)

**Fix (committed `bcc93f9`, no retrain needed):** `build_obs` uses the quat as-is;
`_ODO_RATE_SIGN = [1,−1,1]`; wire command = `rate_flu * [−1,−1,−1]`. Counterfactual against
the emulated real sim: buggy mapping reproduces the live signature (0 gates, lateral sweep,
OOB at 2.5 s); corrected mapping **finishes 6/6 at 9.50 s** (lat 0 and 2).

**Live confirmation (2 flights, the authorized budget): the failure mode is transformed but
standing start is still 0/2.** No spin, no oscillation — smooth, coordinated, correct-signed
flight — but a ~5 m lateral (+y) miss at the gate-0 plane, then un-recovering OOD wander.
The residual is a **second, smaller, ordinary plant gap** (translational; Section 6), now
characterized offline from today's recordings. Fix path at the end.

---

## 1. H0 — version/artifact skew: CLEAN

* Clone at flight time = `972cebb` (reflog 20:48–21:11 PDT), post-S17 pull. Twin on THIS
  clone finishes simstart 6/6 → code functionally identical to the laptop.
* ckpt md5 `8FB8855E07D4FD01045E7B2ECBC5ACD3` ✓; sidecar `{3.765, 3.14}` present and loaded
  (header row of every debug_obs.jsonl) ✓; `meta.json` shows `--checkpoint` passed explicitly
  (no inc4 default), `bridge:false` for standing runs, yaw_scale 1.0, rate 30 Hz ✓.
* Track map: **not a live variable** — the RL obs gates are hardcoded constants in fly_rl
  (`_GATE_POS_ZUP`); `--map` only feeds the bridge navigator.

## 2. H1/H2 — obs encoding & action path at step 0: CLEAN (hypothesis from the live session FALSIFIED)

For all 10 standing runs (`scripts/diag_h1h2.py`):

* live step-0 obs == rebuilt-from-telemetry obs (max 4e-5, rounding of the logged telemetry);
* live step-0 obs == twin `simstart` obs to 7e-4 (worst dim rpy_g_y — the −179.9° convention);
  the −17.8° pad pitch is in the twin's simstart definition already;
* on-disk actor reproduces the recorded `actor_mean` to 3e-6; recomputed wire == logged wire;
* the twin takes the **identical first action** [0.595, −1.68, 1.081]/thr 0.320 — and finishes.

So the policy saw the right world at step 0 and the laptop twin agrees with its first move.
The "ShadowPC obs encoding mismatch" hypothesis is dead; divergence opens after step 0.

## 3. H3 — timing: CLEAN

All 15 runs: wall loop median 34.2 ms (29.2 Hz; 2.7% slow vs the 33.3 ms training dt),
sim/wall ratio 1.000 (no sim-rate skew; `--rate 30` pinned in meta), odo age med 2.2 ms /
p95 5.6 ms, **0.0% of ticks staler than the trained 67 ms**. ShadowPC CPU in-loop latency is
a non-issue; the "too low Hz" operator read is falsified. Channel fit (Section 6) puts
effective command→response delay at 2 ticks — exactly the trained DR center.

## 4. H4 → the discovery chain

1. Open-loop twin under the recorded live actions diverges in z/x by 1 m within 16–17 ticks
   (`diag_h4b.py`) — with realized RATES roughly matching. Physics, not policy.
2. Attitude comparison (`diag_h4c.py`): live "true" (artifact-undone) attitude holds roll ≈ 0
   for a full second while the live rate channel (as build_obs reads it) claims +1.0–1.5 rad/s
   sustained body roll. Self-contradiction in the deployed interpretation at high pitch.
3. Convention test (`diag_h4d.py`): quat-FD body rates of the **raw** quat match `w_raw`
   under **[+1,−1,+1]**, gain 1.00, corr 0.93–0.98, in level AND tilted phases, 3 flights.
   The roll-undone model collapses in tilted flight (corr 0.0–0.45). The raw quat also
   rotates `v_body` onto the position derivative (med 0.43 m/s) — it is the true attitude.
4. Command sign, feedback-free (`diag_probe_signs.py`, mixer_probe2 recordings): `c100_r31`
   wire roll **+3.14** → raw-quat roll **−1.34 rad in 0.18 s** (≈ −10 rad/s; super-rate ×3.2
   with inverted sign). `z00_y31_long`: yaw +3.14 → −9.76 rad/s (known). `climb`: pitch cmd +
   → pitch angle + with raw pitch-rate report inverted. **S_live = [−1, +1, −1].**
5. `mavlink_client.py` itself documents that the sign-inversion lore belongs to the ATTITUDE
   euler message, NOT the ODOMETRY quat (and derives state.roll from the raw quat — which is
   why the CTBR bridge, whose gains were sign-tuned end-to-end, always flew).
6. The 2026-06-10 re-verification measured corr(raw, reported-attitude-FD) = [+,−,+] — the
   same measurement as (3) — but inherited "reported roll is inverted" as a prior and
   concluded [−1,−1,+1]. The data could not discriminate at level attitude; tilted data can.

**Closed-loop counterfactual (`diag_counterfactual.py`)** against the emulated real sim
(plant rate_sign [−1,+1,−1], true reporting artifacts): BUGGY mapping → 0 gates, lateral
sweep, OOB 2.4–2.6 s (the live signature). FIXED mapping → **FINISHED 6/6, 9.50 s** (lat 0
and 2). Root cause demonstrated in both directions.

## 5. The fix (`bcc93f9`)

* `rl/fly_rl.py`: `build_obs` attitude from the raw quat; `_ODO_RATE_SIGN=[1,−1,1]`;
  `_ACT_FLU_TO_FRD=[−1,−1,−1]` (training FLU→FRD flip ∘ live roll inversion).
* `rl/offline_rollout.py`: `telemetry_from_truth` synthesizes the TRUE artifacts;
  all plants get the measured live `rate_sign=[−1,+1,−1]` (with fly_rl's corrected wire map
  this closes the loop exactly as training did — verified: simstart and handoff mixer
  rollouts still finish 6/6, `check_build_obs` 0.00e+00).
* `rl/replay_obs.py`: artifact undo corrected to match.

## 6. Live confirmation flights (2/2 budget) + residual gap

`inc6_rollfix_f1/f2` (standing start, no mitigation flags): **TIMEOUT, 0 gates, both — but
the failure mode is completely different**: no spin guard trips, no oscillation, smooth
coordinated flight; reaches the gate-0 plane in ~2 s, misses ~5 m to +y, then wanders OOD
(never re-acquires from far off-course — expected, those obs are untrained).

Residual forensics (offline, today's recordings):

* **Rate channel verified end-to-end now**: best-fit transport d=2 ticks, tau=0.019 s — the
  twin's own constants — per-axis gain live ≈ 0.94–0.97 × model, RMSE 0.13–0.24 rad/s
  (`diag_transient_fit.py`). The convention fix is confirmed live; rates are no longer the gap.
* **The gap is translational** (`diag_obs_diverge.py`, `diag_thrust_lapse.py`): the
  counterfactual gains forward speed ~50% faster than live (vel_gx 8.8 vs 5.7 m/s by tick 12).
  Smooth-tick audit of measured thrust-axis specific force vs `K(collective)`: ratio
  **0.74–0.88 at 3–12 m/s** (n≈1700; the standing-start acceleration regime), ≈1.0 at
  12–18 m/s, noisy 0.4–1.3 above, with axial-speed structure the model lacks. The collective
  map was fit at near-zero airspeed; live thrust at low-mid speed runs 15–25% below it, which
  displaces the approach line by meters inside the first 2 s — enough to turn the twin's
  +0.2 m gate-0 pass into the observed +5 m miss from the tilted pad.
* Secondary note: obs[8] (gate-frame yaw) sits exactly on the ±π wrap from the spawn — the
  MLP input chatters between +3.14 and −3.06 (numerically 6.2 apart). Training shares the
  same wrap behavior so it is not implicated by itself, but a wrap-robust encoding (sin/cos)
  is cheap insurance for inc7.

## 7. Reinterpretation of prior art (flagged for memory)

* **The S17 mixer law stands** — it was measured open-loop at the motor level and the fit
  rows are sign-independent. But "the obs/action pipeline was proven byte-correct" (inc5
  diag) must be read as *deterministically reproducible*, NOT *convention-correct*: replay
  verifies the code against itself; only tilted-phase kinematic consistency (the
  `diag_h4d.py` method) validates conventions against the sim. Recommend adopting it as the
  permanent convention gate.
* **inc5's uniform gate-2 lateral miss** (dy −1.0…−1.8 m, mid-bank, under ys0) now has a
  simpler explanation than top-rail authority depth: gate 2 is the first real banked y-move
  on the course, and inc5's roll channel was mirrored. The mixer rails and the roll mirror
  both contributed to the inc5-era 0/20; their relative weight is now untestable and idle —
  inc6 + the fix supersede that stack.

## 8. Tool fix + open probe rows

* `rate_sysid.py` Euler-singularity aborts: fixed via per-phase `no_tilt_abort` (`325e191`),
  same rationale as anomaly mode (the spin IS the measurement). `mixer_probe2.json` spin
  phases tagged. **Probe rows c100_y31 / c60_r31 / zhov_r31 remain uncaptured** (the 2-flight
  budget went to the rollfix confirmation; the next sim session can run the full probe in
  ~10 min unattended).
* Bridge-mode "180° then turn back" oddity (VQ1-era, non-critical): observed in prior
  sessions, not reproduced/investigated here (bridge was not flown today). Logged only.

## 9. Recommended path

1. **Keep the convention fix** (necessary regardless; live-confirmed: spin eliminated,
   correct-signed coordinated flight).
2. **Next ShadowPC session, first test: bridge mode ×5 with the fix.** Strong prediction:
   pre-fix, bridge passed gate 0 by inertia and orbited at gate 1 (the first banked move);
   with the roll channel fixed, the policy should now thread gates. Bridge bypasses the
   standing-start low-speed regime where the thrust-map error is largest — it is the
   cheapest discriminator between "translational refit needed for standing start only" and
   "deeper gap".
3. **Laptop: joint refit of the translational model from TODAY's recordings** (17 flights of
   debug_obs at 3–30 m/s — no new flights needed): collective map vs airspeed (thrust lapse)
   + drag, then re-run the inc6 deploy matrix on the refit plant. If inc6 is robust to the
   refit → fly as-is; else inc7 retrains with the refit + DR over the lapse (+ optional
   sin/cos yaw-wrap encoding).
4. Standing start stays the deployment target, gated on (3).

## 10. Recording index

* `data/runs/20260612_044219_inc6_rollfix_f1`, `…_044438_inc6_rollfix_f2` (post-fix, std start)
* Forensic scripts + method: `handoff/shadowpc-inc6-diag-2026-06-12/scripts/`
* Yesterday's 15 failure runs + 3 mixer_probe2 runs: see the INC6-LIVE writeup index.
* Sim exited to HOME at session end (screenshot-verified).

---

MEMORY-DELTA:
- NEW (project): AI-GP true conventions — ODOMETRY quat = true attitude; raw→true rates
  [+1,−1,+1]; live command rate_sign [−1,+1,−1] (roll AND yaw inverted). fly_rl fixed
  `bcc93f9`; validate any future convention with tilted-phase quat-FD consistency, never
  level-flight correlation.
- SUPERSEDES: aigp-sim-ops-gotchas item 3's sign lore (the [−1,−1,1]/roll-undo bookkeeping
  is a level-attitude alias); magnitude guidance (use raw ODO) stands.
- AMEND aigp-mixer-coupling-law: "pipeline byte-correct" ≠ convention-correct; inc5 gate-2
  lateral miss now co-attributed to the roll mirror; mixer law itself unchanged.
- NEW (project): residual inc6 gap = thrust-map overprediction 15–25% at 3–12 m/s (lapse
  vs airspeed, unmodeled); refit from 2026-06-12 recordings; bridge ×5 is the next live test.
- inc6 checkpoint remains the ship; no retrain implied by the convention fix itself.
