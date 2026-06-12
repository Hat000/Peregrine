# LAPTOP-FRAME-AUDIT — the third mirror, root-caused; conventions unified; inc6 salvaged

**Session:** LAPTOP-FRAME-AUDIT (2026-06-12). **Model:** claude-fable-5.
**Input:** S18's open-loop smoking gun — twin reproduces live attitude/rates/speed exactly,
East velocity opposite-signed (−33 vs +30 m/s² East thrust accel at the live attitude).
**Commits:** `93023cf` (fix + armor), this writeup.
**Data:** `handoff/shadowpc-refit-dataset-2026-06-12/` (17 runs; pristine `vel_ned` FD = the
external ground truth throughout). Scripts: `handoff/laptop-frame-audit-2026-06-12/scripts/`.

---

## TL;DR

1. **ROOT CAUSE (one defect explains everything): the sim's ODOMETRY quaternion is not the
   FRD→NED attitude — it is the attitude expressed in an R_y(π)-conjugated frame pair.**
   `q_true = q_raw * [1,−1,1,−1]` (wxyz; negate x,z). Euler: roll AND yaw negated, pitch
   intact. Because R_y(π) conjugation is a PROPER rotation, the reported quat passes every
   internal-consistency test (quat-FD vs rate channel, twist round-trip, all level-flight
   checks) — only comparison against an EXTERNAL invariant (FD of pristine `vel_ned`)
   discriminates. That is how it survived bcc93f9's tilted-phase validation: the diag
   validated rotational coherence, which a conjugation preserves; it never tested force
   direction against measured world acceleration.

2. **Decisive measurement** (`audit_candidates.py`, all 17 runs, 28,195 banked ticks):
   force model from the conjugated attitude matches measured world accel on ALL axes
   (corr +0.97…+0.99, med residual ~1 m/s²); the as-is reading ANTI-correlates on East at
   bank (−0.84, med error 24 m/s²); R_x(π)/R_z(π) conjugations fail North (pitch flip).
   Yaw pinned independently by turn direction: course rate from velocity matches only the
   conjugated yaw rate (`audit_yaw_focus.py`).

3. **Everything downstream simplifies.** True body rate = `−w_raw` (all axes; quat-FD gain
   0.999/0.999/0.996). Live command→rate sign = **[+1,+1,+1]** — the sim is a completely
   vanilla CTBR plant with NO command inversion on any axis. Every historical "roll/yaw
   inversion" was this telemetry conjugation read back as physics.

4. **🚩 S18 THRUST LAPSE VOIDED.** Re-fit with the true attitude on the same 17 runs:
   K_eff/K ratio ≈ 1.00 at 3–6 m/s, 1.04–1.10 above (`audit_lapse_recheck.py`). The
   "15–25% deficit recovering by 12 m/s" was the mirrored-b3 projection error (≈20° East
   bank during the early climb gives exactly 0.74; it "recovered" as the trajectory
   straightened). There is NO residual translational plant gap: regime-binned residuals
   (speed × tilt × collective) are ≤~2 m/s² everywhere on the **mixer** plant, lapse OFF.

5. **Fix shipped** (deploy layer only; CTBR untouched): `fly_rl` conjugates the quat,
   `_ODO_RATE_SIGN=[−1,−1,−1]`, `_ACT_FLU_TO_FRD=[+1,−1,+1]`; emulation
   (`offline_rollout`) emits conjugated telemetry and models the live plant as
   `rate_sign=[+1,+1,+1]`. **Acceptance met:** open-loop replay reproduces live East
   velocity correctly SIGNED on 4 banked windows (end |vE err| ≤ 0.72 m/s; the old reading
   gives −22 m/s on the flare); force selfcheck passes all axes at tilt.

6. **inc6 VERDICT: SALVAGED.** The training world (rl_plant/DiffAero, proper-rotation
   bridges both sides) is internally self-consistent — no mirror was ever inside training.
   Only the deploy mapping was wrong (twice). Deploy matrix REBUILT with fixed tools:
   **16/16 FINISHED** (simstart/racestart/handoff-4/handoff-10 × latency 0–3, mixer plant,
   8.2–9.6 s). Counterfactual: fixed mapping 6/6 @ 9.50 s; bcc93f9 mapping against the same
   corrected emulation diverges before gate-0 (OOB 2.2 s).

7. **Regression armor:** `tests/test_frame_conventions.py` (9 golden-file tests pinning
   per-axis handedness against recorded live data, incl. an open-loop East-sign replay and
   a mirror canary that must keep FAILING for the as-is reading) + standing
   `scripts/frame_residual_report.py` for every future live session. **598 tests green.**

8. **A FOURTH (latent) seam found, not fixed here:** the vision chain pairs the REPORTED
   attitude with TRUE-world camera pixels (`navigator.py:295` → PnP world fixes). Harmless
   near-level/yaw≈π (VQ1 — and it may explain part of the "roll-correlated wander" the
   1.4° attitude-noise fit absorbed), wrong at VQ2 bank. Queued as VISION-FRAME-FIX.

---

## 1. The per-layer handedness map

Anchors: **[A]** force projection vs FD of pristine `vel_ned` (28k banked ticks);
**[B]** course/turn direction vs velocity; **[C]** quat-FD internal relation; **[D]** pos-FD
vs vel at bank (corr ≥0.994, gain ≈1.0); **[E]** open-loop replay of recorded wire commands;
**[P]** parity tests / code identity. Level-flight evidence inadmissible and unused.

| Layer | Quantity | Receives / assumes | Verdict | Evidence |
|---|---|---|---|---|
| Sim wire LPN | pos, vel (world NED) | TRUE | ✅ truth anchor | [D]: pos-FD ≡ vel through 60° bank, all axes |
| Sim wire ODOMETRY | quaternion | R_y(π)-conjugated frame pair (roll,yaw Euler negated) | 🚩 CONJUGATED | [A]: conj wins all 3 axes (E: +0.99 vs −0.84 as-is); [B]: only conj yaw matches turn direction; pitch intact (X/Z-conj fail North) |
| Sim wire ODOMETRY | angular_rate | `w_raw = −ω_true` (all axes) | 🚩 NEGATED | [C]: quat-FD(conj) = −w_raw, gain 0.999/0.999/0.996 |
| Sim wire ODOMETRY | twist (body vel) | reported-frame body velocity: raw-quat rotation ⇒ TRUE world vel | ✅ consistent PAIR | [D]: `vel_ned` (LPN ⊕ rotated twist interleaved) never chatters at bank |
| Sim wire ODOMETRY | accel_body | reported-frame (pairs with raw quat) | ✅ consistent PAIR | same structure as twist; KF IMU-predict safe |
| Sim physics | command→rate sign | **[+1,+1,+1]** — vanilla CTBR, no inversion | ✅ TRUE | [E]: rate_sign [1,1,1] reproduces conj-attitude trajectory from recorded wire cmds |
| mavlink_client | velocity_ned | raw-quat rotation of twist | ✅ CORRECT (keep) | [D]; do NOT "fix" |
| mavlink_client | state.roll/pitch/yaw, quat | REPORTED frame, passed through | ALIAS consumers below | code [P] |
| CTBR stack (controller/twin_fit/fly_vq1) | all signs | reported frame, end-to-end-tuned closed alias | ✅ ALIAS — do not touch | VQ1 6/6; per-axis alias closure verified analytically (§3) |
| fly_rl build_obs (pre-audit) | attitude/rates | as-is quat + [1,−1,1] rates = a SECOND self-consistent mirror, mixed with TRUE pos/vel/gates | 🚩 THE SEAM | [A],[E]; S18 smoking gun |
| fly_rl build_obs (fixed) | attitude/rates | conj quat; rates `−w_raw` | ✅ TRUE | golden tests |
| fly_rl action path (pre-audit) | wire = rate_flu·[−1,−1,−1] | mirror-consistent with the mirrored obs | 🚩 same seam | [E] |
| fly_rl action path (fixed) | wire = rate_flu·[+1,−1,+1] | preserves trained semantics g·[+1,−1,+1]·rate_flu against live S=[1,1,1] | ✅ | composition pinned in tests |
| rl_plant / twin step | physics | proper rotations, single frame, quat R_world_body | ✅ self-consistent | [P] parity suite; code |
| DiffAero adapter | world/body bridges | R_x(π) both sides (proper), XYZW↔WXYZ | ✅ self-consistent | code + V100 parity 7.1e-15 (S17) |
| Training world (inc6) | rate_sign [+1,+1,−1] + adapter [1,−1,−1] | TRAINED-WORLD convention (not live physics) — harmless, anchor of trained semantics | ✅ keep | deploy map compensates; documented in rl_plant.py |
| offline_rollout emulation (fixed) | telemetry synth + live plant | conj quat, −rates, S=[1,1,1] | ✅ matches wire model | `check_build_obs` 0.00e+00; 16/16 matrix |
| Vision chain (navigator/PnP) | R_wb from reported euler + TRUE pixels | 🚩 LATENT 4th seam (bank-regime) | NOT fixed here | code `navigator.py:295`; queued VISION-FRAME-FIX |

**Why a fully mirrored world is harmless and ours wasn't:** both pre-audit deploy configs were
rotationally self-consistent mirrors. The failure lived at the seam with the unmirrored
quantities: TRUE pos/vel/gate obs (and live, the TRUE physics). The policy banked correctly in
its perceived frame; the true velocity (fed back as obs) evolved East-opposite to its
expectation → the deterministic +5 m East gate-0 displacement, then OOD divergence — including
the "commanded yaw spin" S18 saw, which in the TRUE frame is a turn the other way with
physical (≤35°) sideslip, not −72°.

## 2. What was wrong with each prior verification (do-not-relitigate ledger)

* **bcc93f9 quat-FD tilted validation:** validated the quat against its own rate channel —
  conjugation-invariant. Verdict mechanism now understood; its rate/wire fix was the same
  mirror expressed differently.
* **"Counterfactual 6/6 @ 9.50 s" + "16/16 matrix" (pre-audit):** emulation mirrored exactly
  like the deploy code → false passes, as S18 suspected. The REBUILT matrix (fixed tools) is
  16/16 — and now means something because the emulation model is pinned to live data by
  golden tests.
* **"Live rate channel verified end-to-end (gains 0.94–0.97)":** true but conjugation-blind
  (measured in the mirrored frame both sides).
* **S18 lapse:** voided (§TL;DR-4). The S18 *open-loop replay method* itself was sound — it
  found the mirror.
* **mixer_probe2 vs S17 contradiction (κ_hold ~4×, yaw top-rail ~3×):** measured via rate
  channel/quat magnitudes — |·| is conjugation-invariant, so the S19 question STANDS
  unchanged.
* **CTBR "yaw command inverted" + super-rate yaw sign, S1.2 [−1,−1,1], 2026-06-10
  re-verification:** all reported-frame aliases of the single conjugation; none were
  independent confirmations.

## 3. Why the CTBR alias closes (documentation, no change)

CTBR reads reported euler/rates and was sign-tuned end-to-end. Per axis at its near-level
regime: roll — odo_att_sign undoes the reported-roll negation, command sign +1 matches true
+1. Yaw — perceives −yaw_true with rate −r_true and commands through body_rate_sign [1,1,−1];
wire −c realizes true +g·c, perceived as −g·c: consistent within its own books. Pitch — true
everywhere except the rate report, undone by odo_rate_sign. A closed, self-consistent alias;
VQ1-proven; leave it alone.

## 4. Fix inventory (commit `93023cf`)

* `src/racer/frames.py`: canonical `ODO_QUAT_TRUE_CONJ_WXYZ` + `true_attitude_from_odo_quat_wxyz`
  + `true_rate_from_odo_angular_rate` + the full convention note (single source of truth).
* `rl/fly_rl.py`: `_ODO_QUAT_TRUE_CONJ=[1,−1,1,−1]` applied in `build_obs`;
  `_ODO_RATE_SIGN=[−1,−1,−1]`; `_ACT_FLU_TO_FRD=[+1,−1,+1]`.
* `rl/offline_rollout.py`: `_RATE_SIGN_LIVE=[1,1,1]`; `telemetry_from_truth` emits conjugated
  quat + negated rates; `--plant lapse` marked voided-historical.
* `rl/replay_obs.py`: true-state extraction conjugated; `v_client` documented CORRECT;
  `v_artifact` repurposed as a deliberate wrong-frame canary.
* `src/racer/rl_plant.py`: LAPSE constants annotated VOIDED; `rate_sign` default documented
  as trained-world convention (DO NOT change — trained-semantics anchor).
* Armor: `tests/test_frame_conventions.py` (+9, golden fixture `tests/data/frame_audit_golden.json`
  from rollfix_f1 + bridge_f1), `scripts/frame_residual_report.py` (standing per-session report:
  regime-binned residuals + mirror canary + rate canary + optional open-loop replay).
* 598 tests green. (`stage1_inc1_actor.pth`, previously untracked, swept into the commit per
  the small-checkpoints-in-git policy.)

## 5. inc6 verdict and what offline now claims

Training world internally consistent ⇒ **checkpoint salvageable; salvaged.** The corrected
deploy mapping reproduces the trained closed-loop composition exactly
(`wire∘S_live = [+1,−1,+1] = adapter∘rate_sign_train`), so offline results carry over:
rebuilt matrix 16/16 (mixer plant — the correct plant now; lapse OFF), counterfactual fixed
6/6 @ 9.50 s vs bcc93f9-mapping OOB pre-gate-0. **No retrain required. Do NOT train inc7
with dr_lapse on the S18 curve — the lapse is voided.** The remaining honest gap between
offline and live: the telemetry/plant model itself — now pinned per-axis to live recordings
by the golden tests, and the residual report shows no systematic force gap ≥~2 m/s² in any
visited regime.

Caveat for the record: offline can never *prove* live transfer — it now asserts "deploy
mapping consistent with every external invariant in 17 live recordings," which is the
strongest claim short of flying.

## 6. ShadowPC live confirmation spec (next session, sonnet)

**Flights (4 total, in order):** ① standing ×2 (`fly_rl.py --checkpoint
rl/checkpoints/stage1_inc6_actor.pth --no-bridge --flights 2 --label rl_inc6_frameaudit_std`),
② bridge ×2 (`--bridge --flights 2 --label rl_inc6_frameaudit_brg`). Defaults otherwise
(rate 30, debug-obs ON, full-reset ON, spin guards ON). Pull main first; verify
`git log -1` ≥ `93023cf` and `pytest tests/test_frame_conventions.py -q` passes on the clone.

**Telemetry confirmations (run `scripts/frame_residual_report.py --replay 36 <session>` on
every run dir afterwards):**
* PRIMARY (pass/fail): gate progression. Prediction: standing start now clears gate 0
  centred (offline E at plane ≈ −0.2 m) and threads the course; both prior failure modes
  (pre-fix spin; post-fix +5 m East miss) are explained and removed.
* Mirror canary OK (TRUE East corr ≫ as-is, as-is negative at bank) — confirms the sim's
  convention unchanged.
* Rate canary gains ≈ +1.0 all axes.
* Open-loop replay vel err ≤ ~2 m/s per axis over 27 banked ticks.
* In the first 2 s of standing start: no East drift beyond ±1.5 m at the gate-0 plane.

**Abort criteria:** any spin-guard trip or SIM_RESET on flight 1 → stop, capture, do not
burn flights 2–4; mirror canary TRIPPED on any run → stop everything (sim convention
changed — re-audit, do not iterate flags); 0/2 standing AND 0/2 bridge with canaries green →
the gap is NOT frames; capture recordings, hand back to laptop (no live tuning).

## 7. Queued follow-ups (commander triage)

1. **VISION-FRAME-FIX** (post-confirmation, before any VQ2 banked vision work): use
   `frames.true_attitude_from_odo_quat_wxyz` for all world-geometry projections
   (`navigator._maybe_run_vision` / PnP prior / `gate_mapper` case A). Touches the
   VQ1-proven stack → needs VQ1-replay regression. Also re-examine the 1.4°
   ATTITUDE_NOISE_STD_RAD fit (part of its "roll-correlated wander" is plausibly this seam).
2. **S19 mixer probe2 contradiction** — unchanged by the audit (sign-invariant measurements).
3. **V100 config-matrix gate** at next Adroit contact (unchanged; lapse configs present but
   voided — do not select them for training).
4. inc7 planning: the "lapse-DR retrain" branch is dead; if inc6 confirms live, S2
   architecture work resumes per the master plan.

---

MEMORY-DELTA:
- 🚩 NEW (project, HEADLINE — supersedes S18 §4 "candidate roots" and bcc93f9 convention claims):
  3rd mirror ROOT-CAUSED: ODOMETRY quat = attitude in an R_y(π)-CONJUGATED frame pair. TRUE
  conventions: q_true=q_raw·[1,−1,1,−1] (roll+yaw Euler negated, pitch intact); ω_true=−w_raw
  (all axes); live cmd→rate sign [+1,+1,+1] (NO inversion — all historical inversion lore =
  telemetry alias). vel_ned/pos_ned remain pristine; ODO twist pairs with RAW quat (keep).
  Pinned vs FD of vel_ned (28k banked ticks). Fixed in fly_rl/offline_rollout/replay_obs +
  frames.py helpers, commit 93023cf.
- 🚩 SUPERSEDES S18 lapse verdict: THRUST LAPSE VOIDED — artifact of mirrored-b3 projection
  (true-attitude refit ratio ≈1.0–1.1 all bands; no plant gap ≥~2 m/s² in any regime, mixer
  plant). Do NOT use --plant lapse or dr_lapse for training. inc7 lapse-DR branch dead.
- inc6 SALVAGED: training world proper/self-consistent; deploy matrix REBUILT with fixed
  tools 16/16 (mixer, latency 0–3 × 4 start modes); counterfactual fixed 6/6 @9.50 s vs
  bcc93f9-mapping diverges pre-gate-0. No retrain. Next = ShadowPC live confirm (spec §6:
  standing ×2 + bridge ×2 + frame_residual_report canaries; abort rules).
- NEW (durable): regression armor = tests/test_frame_conventions.py (golden live-data
  handedness pins incl. East-sign open-loop replay + mirror canary) + standing
  scripts/frame_residual_report.py to re-run after EVERY live session. 598 tests green.
- 🚩 NEW (latent 4th seam, queued VISION-FRAME-FIX): vision chain pairs REPORTED attitude
  with true-world pixels (navigator.py:295 PnP/world-fix path) — fine near-level (VQ1),
  wrong at bank; fix via frames.true_attitude_from_odo_quat_wxyz + VQ1-replay regression
  before VQ2 banked vision. May explain part of the 1.4° attitude-noise fit.
- mixer_probe2-vs-S17 contradiction (S19) UNCHANGED by audit (measurements sign-invariant).
- CTBR legacy alias: closure now derived per-axis (audit §3) — still ALIAS, still untouched.
