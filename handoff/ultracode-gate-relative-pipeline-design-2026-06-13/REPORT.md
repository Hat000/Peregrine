# Gate-Relative Case-C VQ2 Pipeline — DESIGN BLUEPRINT (commander deliverable)

**Fengyou** — commander deliverable for `ULTRACODE-GATE-RELATIVE-PIPELINE-DESIGN` (opus-4.8, ultracode).
Peregrine / AI Grand Prix. 2026-06-13. Design + offline prototype/sim only (no live sim, no src edits,
no SLURM, no memory edits). Workflow `wf_50b5bcc2-b9f`: **19 opus agents, 2.74M tok, ~183 min**, 3 phases
(Design 6 components → adversarially Verify each → Completeness-critic → Synthesize). The margin component
got a 3-lens skeptic panel. **Every load-bearing number was re-derived this session** (sim re-run / source
read / verifier re-run), AND I (commander) ran an **independent cold-velocity margin cross-check** before
reading the d3 agent's sim — it corroborates and sharpens the headline (§2).

**The full build-ready plan is `BLUEPRINT.md`** (34 KB, 6 components reconciled, the offline gauntlet
G0–G7, the live ladder L0–L4, the confidence ledger). This REPORT is the commander framing + the
make-or-break verdict + the cross-check. Read `BLUEPRINT.md` for the implementation detail.

---

## 1. TL;DR — the verdict in six lines

1. **The gate-relative OBSERVATION fix is SOLID and MUST be built.** It removes the per-track map/
   registration bias EXACTLY (re-run: rel-arm E_bias **−0.000 m** vs absolute/submap **+0.176/+0.174**;
   in-plane RMS 0.139 m vs abs 0.279 / submap 0.228). It reuses the shipped P4-C05 `get_gate_rotmat_w2g`
   path, degrades gracefully to case A/B as the high-confidence limit, and is necessary regardless of
   organizer Q①. This is the single highest-value change and it is proven.
2. **But the gate-4 0.155 m WORST-CASE margin does NOT close offline in ANY regime.** A margin is a contact
   (p90/p99) gate, not an RMS gate. Across the whole {σ_v × accel-bias} grid **every cell is p90-FAIL**
   (best: RMS 0.126 / p90 0.190 / 24% contact). The only "GO" numbers anywhere (c1-warm RMS 0.139,
   d4v-visvel 0.144) are **RMS-clears that are simultaneously p90-FAILS** (0.203 / 0.213).
3. **Margin-closure verdict = CANNOT-SETTLE-OFFLINE (escape hatch invoked).** The binding swing is the
   **cold case-C velocity prior / true effective accel-bias** — NOT speed, NOT latency. Both are pinnable
   only by the **ShadowPC at-speed (~37 m/s) gate-4 recording (L3)**.
4. **RewindKF solves latency** (re-confirmed three ways incl. my cross-check): GPU 15 ms vs CPU 115 ms
   in-plane error near-identical once measurements are correctly capture-timed. Latency is second-order.
5. **The d4v "cheap velocity lever" is REFUTED and demoted to P2 insurance:** honest inter-frame PnP-delta
   σ_v ≈ 2.81 m/s (not the assumed 0.3–1.0) + the mandated L≈115 ms → cold 0.19–0.35 m NO-GO. Closure must
   rest on **accel-bias/attitude control (≤~0.6°) + the uncertainty-aware speed-down**, not this channel.
6. **Build the whole pipeline (right architecture, observation fix is real); select on a p90 gate, not RMS;
   gate the speed ladder on the live recording.** The estimator — not the policy — sets the speed ceiling.

---

## 2. COMMANDER INDEPENDENT CROSS-CHECK (the make-or-break number, re-run by me)

I built my own cold-velocity margin sim (`handoff/_commander-xcheck/margin_xcheck_v2.py`) **before reading
the d3 agent's**, to keep it independent (as prior commanders did). Real `LinearKF` + `RewindKF` + the
measured 0.265 m/axis per-fix lateral σ. I caught and fixed **two of my own sim bugs** along the way (a
piecewise-corner artifact, and a measurement-timing bug that made the L=115 ms arm falsely blow up —
documented in `handoff/_commander-xcheck/CROSSCHECK_FINDINGS.md`); the corrected sim:

| regime (37 m/s) | in-plane RMS | p90 | clears 0.155 (RMS / p90) |
|---|---|---|---|
| warm (σ_v=0, = c1) | 0.132–0.140 | 0.193–0.218 | **YES / NO** |
| cold σ_v=0.3 m/s | 0.160–0.194 | 0.244–0.288 | NO / NO |
| cold σ_v=0.6 m/s | 0.203–0.267 | 0.304–0.412 | NO / NO |
| cold σ_v=1.0 m/s | 0.231–0.324 | 0.354–0.495 | NO / NO |

**It tri-confirms the agents.** My warm baseline (RMS 0.132–0.140, p90 ~0.20) reproduces c1's 0.139 and is
**over the margin on the p90 tail even warm**. My cold sweep crosses the margin at σ_v≈0.3 m/s and is
decisively over by σ_v≥0.6 — matching the d3 agent's cold p90 0.235 (zero bias) → 0.338 (measured 1.4°
attitude bias) and the v3 verifier's independent 0.227–0.234 → 0.338. My framing adds one complementary
angle: I swept the velocity-prior σ_v *directly* as the control variable (the agents swept accel-bias),
isolating that **σ_v is the binding lever and the margin crosses at σ_v ≈ 0.3 m/s**. My cross-check also
independently re-confirms RewindKF compensates latency correctly (L=15 vs 115 nearly identical once `z` is
capture-timed). **No divergence from the workflow's headline; it strengthens it.**

**Honest note on my warm vs d3's warm:** d3's warm (RMS 0.084) is lower than mine/c1's (0.13–0.14) because
d3 continuously re-seeds velocity over a full lap; mine and c1 are single-leg warm. All three agree on the
direction and on the cold p90 failure — the spread is sim-construction, not disagreement.

---

## 3. THE BUILD BLUEPRINT (6 components — full detail in `BLUEPRINT.md`)

The integrated pipeline: **wire telemetry → [C2 estimator: RewindKF + gate-relative +L fix + calibrated P]
→ [C1 obs: 20-dim gate-relative + confidence triple] → [C5 inc8 actor] → CTBR wire**.

- **C1 — Gate-relative obs (+ uncertainty channel).** The 17-dim obs is already gate-relative; case C feeds
  `pos_g = R_w2g @ (+L_seen)` (the PnP lever to the SEEN opening, map bias dropped) into the same slot the
  P4-C05 yaw-aware path uses → **one case-agnostic policy** (A/B = the high-confidence limit). **FROZEN at
  20-dim**: append a bounded-[0,1] confidence triple `[c_inplane, c_along, age_norm]` (σ_ref=0.05 m,
  τ_stale=0.10 s) from the calibrated gate-frame KF covariance — fast-when-confident, NOT logstd-as-
  uncertainty. Dims 0:17 byte-identical (verified 0.0). 🚩 **SIGN FOOTGUN the adversary caught:** the slot
  is **+L = (gate_pos − pos), NOT −L** (d1/d2 spec text said −L; verified a full **24 m flip**). d1's
  "0.0 bit-exact unification" was a tautology; the real gate is `v_obs_adversarial_check.py` end-to-end +
  a −L sign-flip negative control (in G1).
- **C2 — Case-C estimator chain.** Detector→PnP→associate (unchanged) → gate-relative in-plane fix
  (`+L_seen`, anisotropic cov: in-plane σ=0.265 m, **NO 0.40 m floor in-plane** — the floor is bias-
  absorption, gone with the bias) → RewindKF OOSM (horizon 0.5 s, **strictly > L** or it drops 100% of
  fixes) → calibrated P → confidence triple. AUGMENTS the absolute KF (which keeps planning/feed-forward).
  **+ a REQUIRED relative-innovation outlier gate** χ²(2,0.999)=13.82 (reproj alone passes 93% of depth-
  flips). **3 P0 bugs fixed:** (a) `_initialize` seed guarded on `config.use_given_position` not
  `ds.position_ned` (navigator.py:255-271) → true case C seeds origin@5 m; (b) TIMESYNC `delta_epoch`
  reconciliation to the IMU clock (navigator.py:426); (c) velocity = KF pos/vel coupling (no new vision-vel
  surface). Predict-forward (constant age) ships first, sidesteps TIMESYNC for the cheap 80%.
- **C3 — Margin closure (the load-bearing empirical).** → §2/§4. CANNOT-SETTLE-OFFLINE.
- **C4 — Velocity acquisition.** Position-fix-differencing IS the KF (free baseline). Direct vision-velocity
  **REFUTED as a margin lever → deferred P2 insurance** (honest σ_v 2.81 m/s + L=115 ms → NO-GO).
- **C5 — inc8 retrain spec (Adroit-ready, §below).**
- **C6 — Integration + validation (§below).**

## 4. inc8 RETRAIN SPEC (Adroit-ready — `BLUEPRINT.md §2`)

- **Obs:** 20-dim (consumes C1); train env populates `pos_g` as ground-truth +L offset **+ the measured
  estimator residual** (never pristine truth, never the `R_w2g@(gate_map−p_KF)` anti-pattern); critic 36-dim.
- **MEASURED-error DR** (`+env.estim_dr=case_c`, on the OBS only — true dynamics/contact unperturbed):
  per-fix lateral σ~U[0.08,0.30] m/axis, N_eff~U[4,9]; range-collapse σ(r); along-track latency staleness
  v·L (L~U[6,125] ms, RewindKF residual ~U[0,0.05] m); **one-signed PnP/extrinsic bias U[0,0.19] m with a
  per-episode constant RANDOM SIGN** (the adversary showed ±0.10 zero-mean cannot cover a one-signed 0.19 m
  that gives p90 0.337). MAP bias is NOT injected (it drops out). NEVER reward damping — caution is an
  emergent best-response to honest noise + honest contact penalties (the asymmetric critic sees true state).
- **Reward (3 changes):** R1' arc-length progress over the **rebuilt** corrected-aero contact-safe line
  (C4 line; `reference_line_vq1.json` is drag-infeasible+170° inverted → MUST rebuild); T4 finish-time KEPT
  (the confidence-blind speed pressure); R4' **fixed relaxed cone is the L0 default** (the confidence-gated
  per-env cone is a real code change — `tilt_free_rad` is a scalar — so it's a portfolio ablation).
- **BSR3 spin-gate MANDATORY first** (spin_rate_abort→9–10, spin_time_abort→3.0 s). Ladder: L0 → L1
  (rw_tilt 96→48) → L2 (free-cone 60→70°), each gated on the p90 metric.
- **≥5 seeds** (narrow basin), **design point = COLD** (not warm). **SELECT on a p90 gate:** lowest finish
  time among policies with **gate-4 SIMSTART in-plane p90 < 0.155 m @ r=0.38** AND S_stable ≥2/3 AND whose
  achieved v\* clears the c5 σ-gate at C2's delivered per-fix σ. 🚩 **The `contact_true_eval` selection
  needs an estimator-emulation obs wrapper + a v\* instrument FIRST** (it runs `obs_from_truth` = perfect
  pose today; selecting on it crowns a fiction).

## 5. INTEGRATION + VALIDATION (`BLUEPRINT.md §3`)

- **Wiring:** thread live `make_gate_map(TRACK_INFO)` + the estimator into `_fly_armed` (today's judged
  standing-start path never runs the KF — `build_obs` on raw wire pose; the rebuild inserts the estimator
  stage). **Guard swap is SAFE:** keep `_assert_vq1_constants_consistent` + `assert_gate_map_allpi`-on-None;
  **replace** `_assert_live_course_is_vq1` with `gate_map` threading + an estimator-readiness assert — any
  unwired non-π course still LOUD-ABORTS. `submit_rl.py` structurally unchanged (pin inc7→inc8 + ship the
  `.json` sidecar; no `MAV_CMD 31000` on the judged wire).
- **Offline gauntlet (cheapest fails first):** G0 import/sidecar → **G1 obs bit-exactness + the +L sign-flip
  negative control** → G2 estimator unit checks (OOSM bit-id, horizon>L, σ=0.265 no floor) → **G3 margin sim
  (p90 gate; EXPECT p90>0.155 → escape-hatch, by design)** → G4 contact_true_eval selection (needs the §4
  wrapper) → G5 S_stable+BSR3 → G6 speed-ladder → G7 full 692-suite.
- **Live ShadowPC ladder (zero gate-contact = THE validity rule):** L0 case-A smoke (no VQ1 regression) →
  L1 case-C cold-start activation → L2 at-speed fresh-reset batch → **L3 at-speed gate-4 recording (RESOLVES
  the escape hatch)** → L4 eval-HW latency + TIMESYNC wire-trace confirmation. L3/L4 HARD-blocked on TIMESYNC.

## 6. RANKED RESIDUAL RISKS (top of `BLUEPRINT.md §4`)

| # | Risk | Sev | Resolver |
|---|---|---|---|
| 1 | **Gate-4 p90 margin does NOT clear offline in any regime** (cold p90 0.234 @ bias0 → 0.338 @ measured 1.4°; even best cell p90 0.190). RMS-clears are p90-fails. | **HIGH** | p90 selection gate + uncertainty speed-down + slower-rung fallback. DECISIVE = L3. |
| 2 | **Cold velocity prior unpinned & HIGH** (variance/window-driven, NOT init-driven → "warm by gate-4" is FALSE). Dominant = effective accel/attitude bias (1.4°→0.24 m/s²→65% contact). No measured IMU accel-bias exists. | **HIGH** | Attitude/ESKF-bias-state is in the MARGIN critical path; keep bias ≤0.1 m/s²; L3 pins the slope. |
| 3 | **In-loop latency L on eval HW UNMEASURED** (CPU ~125 ms → 2.3–4.2 m staleness → speed thesis flips). | **HIGH** | One eval-HW timing run (L4); predict-forward fallback. |
| 4 | **TIMESYNC `delta_epoch` uncheckable offline** — full RewindKF indexed by it; unreconciled → C2 chain INVALID by construction. | **HIGH** | Predict-forward ships first; full rewind blocked on a live wire trace (L4). |
| 5 | **Per-fix σ at 37 m/s UNMEASURED** (pool ≤8.4 m/s; blur 1.0→2.0× MODELED) — every σ is a best-case lower bound; `c_cal=1.0` placeholder. | **HIGH** | L3 sole resolver. |
| 6 | **Velocity channel NOT the cheap lever** (honest σ_v 2.81 m/s + L=115 ms → NO-GO). | **MED** | Demoted to P2 insurance; closure rests on bias control + speed-down. |

(Plus: residual one-signed PnP bias ≤0.19 m unsigned [MED]; confidence over-confidence NEES 1.96<χ²(2) [MED];
narrow inc8 basin [MED]; detector 4→2 corner clip [MED]; obs-dim 17→20 consumer break [LOW].)

## 7. BUILD SEQUENCE (dependency-ordered)

`P0 bug fixes → C2 estimator → C1 obs → C5 inc8 retrain (LONGEST POLE, ≥5 seeds) → C6 gauntlet G0–G7 →
C6 live L0–L4.` **C3 margin sim** and **C4 corrected line** are OFF the critical path — parallel feeders
into C5 selection. **BSR3 spin-gate is MANDATORY before any retrain.** Everything upstream exists to
de-risk the one expensive run (C5). Full per-step depends-on/effort table in `BLUEPRINT.md §5`.

## 8. THE ESCAPE HATCH — exact live data that resolves what cannot be settled offline

**ShadowPC at-speed (~37 m/s) gate-4 vision recording (L3):** the current best policy, **≥5 laps** (full
g0→g4, or ≥ a g2→g3→g4 segment reaching 37 m/s on the g3→g4 straight), **all fields time-synced on ONE
clock** (the TIMESYNC P0 prereq): per-frame ≥30 Hz 4-corner detections + PnP `t_cam_gate` + reproj; HIGHRES_
IMU `accel_body` ≥90 Hz; given ATTITUDE quat; **ground-truth position + velocity** at IMU rate; ~60–100 s.
It pins: (a) true accel-bias/attitude-error entering g4 (open-loop KF-on-IMU vs GT-velocity → drift slope);
(b) the realized cold velocity-prior distribution; (c) achievable σ_v from real fix-differencing; (d) per-fix
gate-relative lateral σ at real blur (confirm/break 0.265 m); (e) 4-corner coverage vs range; (f) the
one-signed PnP bias magnitude AND sign. Plug (a)+(b)+(d) into the §2/`v3b` tables → the p90 verdict becomes
MEASURED. **Plus** one eval-HW latency run (L4) + a TIMESYNC wire trace.

---

## Confidence ledger
- **MEASURED (re-run this session, incl. commander cross-check):** rel margin RMS 0.139 / p90 0.203, E/D_bias
  −0.000/−0.004, abs 0.279 / submap 0.228; the +L identity 4.8e-7 vs −L 24.0; every v3b cell p90_clear=false
  (best 0.126/0.190/24%); cold p90 0.234@bias0 / 0.338@1.4°; commander cross-check warm 0.13/0.20, cold-σ_v
  sweep crosses at 0.3 m/s; velchannel honest σ_v 2.81 m/s, L=115 ms cold 0.19–0.35 NO-GO; RewindKF latency-
  invariance; `obs_dim=17`, `tilt_free_rad` scalar, `contact_true_eval` on `obs_from_truth` (source reads).
- **EXTRAPOLATED:** per-fix σ at 37 m/s (blur MODELED, lower bound); `c_cal=1.0`; L on eval HW; cold velocity
  prior entering g4.
- **ASSUMED:** const-v 37 m/s, 90 Hz IMU, ~14 Hz fix rate, the one-signed PnP bias magnitude (≤0.19 m) + sign.

## Artifacts (under `handoff/ultracode-gate-relative-pipeline-design-2026-06-13/` unless noted)
- **Blueprint:** `BLUEPRINT.md` (the full build plan). **This report:** `REPORT.md`. Shared brief: `CONTEXT.md`.
- **Component specs:** `d1_obs_spec.md`, `d2_estimator_chain_spec.md`, `d3_margin_findings.md`,
  `d4v_velocity_spec.md`, `d5_inc8_spec.md`, `d6_integration_spec.md`.
- **Sims/checks:** `d1_obs_faithfulness_check.py`, `d2_horizon_cov_check.py`, `d2_relinnov_gate_check.py`,
  `d3_margin_closure.py` (+results), `d4v_velocity_channel.py`, `v_obs_adversarial_check.py` (the +L/−L sign
  control), `v3_margin_refute.py`, `v3b_closure_refute.py`, `verify_velchannel_v6.py`.
- **Commander independent cross-check:** `handoff/_commander-xcheck/margin_xcheck_v2.py` (+results),
  `CROSSCHECK_FINDINGS.md` (incl. the two sim-bugs I caught + fixed).

---

## MEMORY-DELTA (≤12 lines — supersessions flagged)
- GATE-RELATIVE-PIPELINE-DESIGN DONE (wf 19 agents / 2.74M tok). BLUEPRINT.md = the build plan. Gate-relative
  OBSERVATION fix is SOLID + MUST be built (map bias drops EXACTLY: rel E_bias −0.000 vs abs/submap +0.176;
  RMS 0.139). Reuses P4-C05 path; degrades to case A/B; build regardless of Q①.
- 🚩 **HEADLINE: gate-4 0.155 m WORST-CASE margin DOES-NOT-CLOSE offline → CANNOT-SETTLE-OFFLINE.** A margin
  is a p90/p99 gate, NOT RMS. EVERY {σ_v×accel-bias} cell p90-FAIL (cold p90 0.234@bias0 → 0.338@1.4°; best
  0.190). Tri-confirmed (d3 + v3 verifier + commander independent cross-check). SUPERSEDES the c1-warm "0.139
  clears" framing — that's an RMS-clear / p90-FAIL.
- 🚩 **SIGN FOOTGUN:** obs slot = R_w2g @ (+L) = (gate_pos − pos). "estimator delivers −L" (d1/d2 text) is
  WRONG = 24 m flip. d1's "0.0 unification" was a tautology; faithfulness gate = end-to-end +L identity +
  −L negative control.
- Binding swing = **cold velocity prior / effective accel-bias** (NOT speed, NOT latency). RewindKF SOLVES
  latency (tri-confirmed; L=15 vs 115 ms ~equal once capture-timed). Cold-prior risk = **HIGH** (SUPERSEDES
  d5 R-LOW; "warm by gate-4" is FALSE — variance/window-driven).
- VELOCITY CHANNEL (d4v) **REFUTED + demoted to P2 insurance** (honest σ_v ~2.81 m/s + L=115 ms → cold
  0.19–0.35 NO-GO; was GT-cheat @ L=0). Closure rests on attitude/accel-bias control (≤0.6°) + uncertainty
  speed-down.
- FROZEN obs = **20-dim** (bounded triple c_inplane/c_along/age_norm; σ_ref=0.05, τ_stale=0.10 s); actor 17→20,
  critic 33→36 lockstep; append leaves [0:17] at 0.0. SUPERSEDES d6 18-dim / d1 19-dim-raw-metre.
- inc8 SELECT on a **p90 gate** (gate-4 in-plane p90 < 0.155 @ r=0.38), design point COLD, ≥5 seeds, BSR3
  spin-gate FIRST. DR: one-signed PnP bias U[0,0.19] m random-sign-per-episode (SUPERSEDES ±0.10 zero-mean).
  `contact_true_eval` runs `obs_from_truth` → estimator-emulation wrapper + v\* instrument are HARD G4/G6
  prereqs. `tilt_free_rad` is scalar → fixed cone is L0 default, gated-cone is an ablation.
- ESCAPE HATCH = ShadowPC at-speed gate-4 recording (≥5 laps, one clock, GT vel) + 1 eval-HW latency run +
  TIMESYNC wire trace. Resolves the p90 verdict, the accel-bias, and whether the speed thesis holds.
- Build order: P0 → C2 → C1 → C5 (longest pole) → C6 gauntlet → live L0–L4; C3 margin + C4 line = parallel
  feeders. NEES: c1's 1.96 = in-plane 2-DOF (mildly OVER-confident), NOT the χ²(3)=3 target — calibrate c_cal
  on L3. Estimator (not policy) sets the speed ceiling.
