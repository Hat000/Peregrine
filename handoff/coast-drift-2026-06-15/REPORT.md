# COAST-DRIFT — does the informed terminal RewindKF coast close gate-4 r=0.30 @30 m/s?

**Pathway:** P3/estimator worker (coast-drift). **Date:** 2026-06-15. **Model:** opus-4.8, high effort.
**Mode:** laptop, offline — PRODUCTION KF + existing data; NO sim / GPU / re-fly. **Branch:**
`worker/coast-drift-2026-06-15`. **Engine:** the production case-C stack `racer.state_estimator.LinearKF`
+ `racer.kf_rewind.RewindKF`, driven via `handoff/margin-closure-envelope-2026-06-14/margin_envelope.py`
(`ME`) + `boresight-closure-2026-06-14/margin_driver_v2.py` (`MD`). Artifacts under
`handoff/coast-drift-2026-06-15/`: `coast_drift.py` (A1/A2/A3 + `fly_lap_coast`), `diag.py`, `probe.py`
(the coast-decomposition + over-convergence diagnostics); **`final.py`/`final_results.json`** (the A1
centering boundary + VIO trigger — the deliverable); `coast_cheap_confirm.py`/`coast_cheap_results.json`
(54-cell coast-cheapness robustness); `extra.py`/`extra_results.json` (gate-difficulty + VIO back-solve
v1); `sweep.py`/`sweep_out.txt` + `realism.py`/`realism_out.txt` (the MC over-convergence evidence; the
full `sweep.py` run was killed ~60% in — superseded by `coast_cheap_confirm.py`).

---

## 0. VERDICT

> **The informed terminal coast is NOT the blocker — gate-4 r=0.30 @30 m/s is a CONDITIONAL-CLOSE whose
> single binding knob is the TERMINAL CENTERING of the last accurate fix at ~12 m (σ_p0), which the coast
> preserves rigidly. The prompt's hypothesised binding input — terminal velocity σ_v — is robustly NON-
> binding (the dense pointed [12–24 m] fix stream + IMU pins σ_v0 ≈ 0.02 m/s, so the velocity coast-drift
> is ~0.008 m). The 0.4 s coast behaves NOTHING like the boresight 0.3 s drought: a drought is zero-info;
> the informed coast dead-reckons on a tight velocity and adds <0.02 m total (velocity + process + a
> 0.6° accel bias).**

**Closure boundary (the deliverable, A1 production-Q coast, v=30, r_floor=12 → t_coast=0.404 s):**

| terminal-miss contributor | magnitude | binding? | lever |
|---|---|---|---|
| **terminal centering σ_p0** (last accurate fix ~12 m, rigidly coasted) | **≈ σ_p0** | **YES (dominant)** | per-fix σ + band fix-DENSITY; near-field floor |
| velocity drift σ_v0·t_coast (σ_v0 achieved ≈ 0.02 m/s) | 0.008 m | no | (VIO's domain — already fine) |
| KF process noise over the coast (DWNA 0.3 + attitude-Q at \|sf\|=13.7) | 0.009 m | no | — |
| accel/attitude bias 0.6° over the 0.4 s coast (½·g·sin0.6°·t²) | 0.008 m | no | (ESKF — irrelevant to the *coast*) |

> **r=0.30 (B=0.235) closes IFF the terminal centering σ_p0 ≲ 0.075–0.08 m** (lateral, per-axis, at the
> ~12 m handoff). The pointed fix stream's per-fix σ_lat is 0.075–0.11 m (accept-geometry 12–18 m) to
> 0.13–0.17 m (at-speed CTBR); averaged over the ~3–5 accurate band fixes plus the non-averaging
> per-approach bias σ_b (~0.02–0.05 m, at-speed ε_lat≈0), σ_p0 lands **~0.05–0.07 m (pointed/accept-geom →
> CLOSES with margin)** to **~0.08–0.13 m (raw at-speed σ → KNIFE-EDGE to FAILS).** So closure hinges on
> the **inc8 per-fix accuracy AND band fix-density at ~12 m**, not on the coast.

**VIO (parked #70) verdict:** VIO is the **WRONG lever** — visual-inertial *odometry* shrinks σ_v0 (already
tiny, ~0.02 m/s) but is **BLIND to the gate-relative centering** σ_p0 (the binding term). Its velocity
trigger would fire only at **σ_v0 ≳ 0.18 m/s for gate-4** (≳0.11 for B=0.155, ≳0.08 for B=0.12) — the IMU
already delivers ~0.02 m/s (×9 margin). **#70 stays parked;** the centering levers are per-fix σ, band
fix-density, and a **near-field GATE estimator** (lower the 12 m floor) — a *gate anchor*, not odometry.

**Best gate-4 speed (coast-drift view):** **30 m/s.** The coast cost is negligible and ~flat across 22–37
m/s (faster only trims the already-tiny velocity drift); terminal centering is speed-flat (at-speed σ_lat
blur-free, flat to 30 m/s). Pushing past 30 buys nothing for the coast and re-opens the sim-to-real blur σ
risk. The lever is centering, not speed.

---

## 1. METHOD — faithful to the production KF, terminal-lock geometry corrected

**Terminal-lock reframe in the engine.** `ME.fly_lap` fixes THROUGHOUT the last `FIX_WINDOW_M=12 m` (the
OLD "gate becomes usable within 12 m"). accept-geometry-2026-06-15 **inverts** this: accurate fixes EXIST
in [~12, ~24] m and DIE inside ~12 m (near-field PnP/association floor). So `fly_lap_coast` fixes only in
each gate's ACCURATE band `[r_floor, r_acc_max=24]` (the g3→g4 leg is exactly 24 m, so the gate-4 band
opens at gate-3) and **COASTS the inner `r_floor` predict-only (NO fix updates)**. Everything else is
`ME.fly_lap`/`MD.fly_lap_v2` verbatim (LinearKF init, RewindKF horizon 0.5, accel-bias model, accel white
0.3, OOSM capture-time latency queue, Catmull-Rom truth + real per-step gate-4 posture/specific-force/dt,
in-plane reduction onto u34). In-plane basis `MD._label_inplane_axes(u34)`: e1=[0.176,0.984,0] (\|z\|=0,
**lateral = binding**), e2 (\|z\|=0.999, vertical).

**Three analyses + a realism correction:**
- **A1** (`a1_coast_propagate`) — transparent predict-only propagation of a SPECIFIED (σ_p0, σ_v0) over the
  REAL gate-4 coast steps (production Q). **This owns the centering boundary** (below).
- **A2** (`a2_achieved_sigma_v`) — deterministic covariance pass (P is value-independent) → the σ_v0 / σ_p0
  the dense stream's covariance *claims*.
- **A3** (`fly_lap_coast` MC) — full-lap Monte-Carlo (fix noise + accel noise + systematic bias + AR(1)
  fix correlation), gate-4 in-plane MISS, bootstrap CIs vs `MARGIN(r)=W_EFF−r` (W_EFF=0.535 →
  **M(0.30)=0.235, M(0.38)=0.155**; prompt cites 0.245@0.30, within 0.01 — reported against both).

**⚠️ The MC over-converges (a logged limitation, drives the method).** Fed the iid synthetic stream, the
production KF collapses σ_p to ~1 mm and **rides the IMU, nearly ignoring band fixes** — so the MC miss is
**flat in per-fix σ, in AR(1) correlation τ_c (0→1.0 s), AND in per-approach centering bias σ_b (0→0.15
m)** (`sweep.py`/`realism.py`). The MC therefore **cannot transmit a realistic centering error into the
coast**; its valid, narrow finding is that the **coast DYNAMICS are cheap**. For the binding CENTERING term
the right tool is **A1 seeded with the measured per-fix σ as the coast-start floor** (the centering the
coast preserves rigidly). _(This KF over-confidence on a dense correlated stream is itself an estimator-
tuning flag — see §6 / NEW parked.)_

**Inputs pinned from the two recordings (not invented):** pointed per-fix σ_lat — accept-geometry 12–18 m
MAD 0.05–0.075 (→σ 0.075–0.11) and at-speed-sigma accepted σ_lat 0.13–0.17 (CTBR, 10–22 m); accept
~0.78\|in-FoV → effective fix-rate ~0.6 of 30 Hz; lateral chain bias ε_lat≈0; accel-bias ≤0.6° (g·sin →
0.103 m/s²).

---

## 2. THE COAST ITSELF IS CHEAP — process + bias + velocity over 0.4 s are all floors

Production Q over the 37 real gate-4 coast steps (`a1_coast_propagate`, isolated terms):
- **Pure process-noise coast** (σ_p0=σ_v0=0): terminal σ_lat = **0.0094 m**. The DWNA white-accel (0.3) +
  attitude-lever Q (1.4°×\|sf\|=13.7 → 0.335 m/s² phantom) over only 0.40 s contributes <1 cm (90 Hz IMU +
  tiny dt make the DWNA Q negligible on this horizon).
- **Accel/attitude bias 0.6° over the coast** = **0.0084 m** (= ½·g·sin0.6°·0.404²). A full-budget bias
  adds <1 cm *to the coast*.
- **Velocity drift** σ_v0·t_coast with the achieved σ_v0≈0.02 m/s (§3) = **0.008 m**.
- **The marginal "coast cost"** (A1, F1: p99 going σ_v0 0.01→0.20 m/s at fixed σ_p0) is only **+0.04…+0.13 m**
  — and at the *achieved* σ_v0≈0.02 it is ~**+0.003 m**.

**This is WHY the informed coast ≠ the boresight drought.** Boresight's Lens-A drought broke cells because
at fr=0.07 the COLD velocity was sparsely conditioned AND a bias acted over the whole ~10 s lap. Over the
0.4 s *terminal* coast in isolation, neither process noise nor a 0.6° bias nor the (well-conditioned)
velocity moves the needle.

**Robustness — coast DYNAMICS are cheap across the whole grid** (`coast_cheap_confirm.py`, 54 cells, 0
errors; this is the MC's over-converged/centering-BLIND floor, so it speaks ONLY to coast dynamics): across
**v∈{22,30,37} × r_floor∈{10,12,14} × accel-bias∈{0,0.3,0.6°} × σ_lat∈{0.10,0.15}** (τ_c=0.10), the coast-
floor **p99 ∈ [0.068, 0.114] m — ALL clear r=0.30 AND r=0.38.** The trends are exactly the physics: p99
rises with accel-bias (0° ~0.07–0.097 → 0.6° ~0.10–0.114) and with r_floor (deeper = longer coast), and
**falls with speed** (faster = shorter coast: v=37 ~0.083–0.098 < v=22 ~0.07–0.106). Worst cell (v=30,
r_floor=14, 0.6°) = 0.114, still ½ the gate-4 budget. The coast adds little no matter how you push it —
**confirming the coast is not the lever.** (Centering is invisible here; see §4.)

---

## 3. VELOCITY IS ROBUSTLY NON-BINDING (the prompt's σ_v hypothesis, refuted-as-binding)

Achieved terminal velocity 1-σ at the ~12 m coast-start (`probe.py`/`diag.py`, ACTUAL MC error, not the
covariance): **σ_v0_lat ≈ 0.012–0.024 m/s**, essentially flat in fix-rate (fr=0.2 ≈ fr=0.6) and per-fix σ.
Why so good, and why it's REAL (not the over-convergence artifact): velocity is **IMU-short-term-observable**
— σ_v over one fix interval (1/18 s) from accel noise 0.3 m/s² is 0.3·(1/18) ≈ **0.017 m/s**, matching the
MC. The position-fixes only correct the slow accel-bias velocity drift; the IMU owns short-term velocity.
A pure position-slope regression would (wrongly) predict ~0.5 m/s — it ignores the IMU. So **σ_v0 ≈ 0.02
m/s is a robust, IMU-floored number**, and the coast velocity-drift σ_v0·t_coast ≈ 0.008 m is genuinely
negligible. **The prompt's "σ_v is the key uncertain input" does not hold — σ_v is the well-pinned input;
the uncertain, binding input is the terminal CENTERING σ_p0.**

---

## 4. THE BINDING TERM — terminal centering σ_p0 — and the GATE-DIFFICULTY boundary

The coast is a **rigid carry** of the last-fix state, so terminal miss ≈ **σ_p0 (centering) ⊕ 0.012 m
(velocity+process+bias floors)**. A1 sweep of σ_p0 (per-axis in-plane centering at the ~12 m handoff) →
2-D in-plane miss p99, vs the budget B (the commander's gate-difficulty axis; B=0.235=gate-4 r0.30,
0.155=r0.38, ≤0.14 = tighter VQ2-like gates):

| σ_p0 (m) | miss p99 | closes B≥ | gate-4 r0.30 (B=0.235) | r0.38 (B=0.155) | tight VQ2 (B=0.12) |
|---|---|---|---|---|---|
| 0.04 | 0.127 | **0.14** | ✅ | ✅ | n (B=0.12 fails, 0.14 ok) |
| 0.06 | 0.185 | **0.20** | ✅ | n | n |
| **0.075** | **~0.235** | **~0.235** | **knife-edge** | n | n |
| 0.08 | 0.245 | 0.30 | ✗ | n | n |
| 0.10 | 0.305 | none | ✗ | n | n |
| 0.15 | 0.456 | none | ✗ | n | n |

(σ_p0 here is isotropic in-plane; the binding axis is lateral-dominated, so the true gate-4 threshold is
σ_p0_lat ≈ 0.077–0.091 — between the isotropic 2-D Rayleigh p99 and the 1-D half-normal p99. **Use σ_p0_lat
≲ 0.08 m as the gate-4 r=0.30 closure boundary.**)

**General terminal-approach physics (not gate-4-specific):** closure scales linearly — **B_min ≈ 3.0·σ_p0**
(2-D) — so each gate's difficulty maps directly to the per-fix-σ + fix-density budget it demands at ~12 m.

**Is σ_p0 ≲ 0.08 achievable?** σ_p0 ≈ σ_lat/√N_eff ⊕ σ_b (band-averaging of the per-fix σ over N_eff
effective accurate fixes ⊕ the non-averaging per-approach bias):
- **pointed / accept-geometry** (σ_lat 0.075–0.11, N_eff 3–5, σ_b 0.02): σ_p0 ≈ **0.05–0.07 → CLOSES r=0.30
  with margin** (and reaches B≈0.16).
- **raw at-speed CTBR** (σ_lat 0.13–0.17, N_eff 3–5, σ_b 0.03–0.05): σ_p0 ≈ **0.08–0.13 → KNIFE-EDGE to
  FAILS r=0.30.**

So **gate-4 r=0.30 @30 m/s is a CONDITIONAL CLOSE gated on inc8 delivering σ_p0 ≲ 0.08 m at ~12 m** — via
(a) tighter per-fix σ_lat (pointing/calibration drives at-speed 0.15 → accept-geom 0.10) and/or (b) more
accepted band fixes (higher N_eff). This is the SAME constraint boresight named ("σ_lat ≲ 0.245 + terminal
gate-lock"), now sharpened to a single number on the terminal centering.

---

## 5. VIO-TRIGGER (parked #70) — the back-solved spec, WITHOUT building VIO

VIO/VINS observes self-velocity → shrinks σ_v0; it is **blind to the gate-relative centering** σ_p0
(odometry has no gate anchor). So it can only ever buy back the (tiny) velocity-drift term:

| budget B | VIO-immune centering floor (σ_p0=0.10) p99 | σ_v0 that would breach B (the trigger) |
|---|---|---|
| 0.120 | 0.304 (> B → VIO can't close) | **0.081 m/s** |
| 0.155 | 0.304 (> B → VIO can't close) | **0.113 m/s** |
| 0.235 | 0.304 (> B → VIO can't close) | **0.183 m/s** |

**Trigger spec for #70:** VIO becomes worth building for the terminal coast **only if the terminal velocity
1-σ σ_v0 exceeds ~0.18 m/s (gate-4 r0.30) / ~0.11 (r0.38) / ~0.08 (tight VQ2)** — equivalently, if the
coast-velocity-drift would exceed ~0.07 m. **The IMU+fix stream delivers σ_v0 ≈ 0.02 m/s (×9 below the
gate-4 trigger)**, so #70 stays parked. It would only fire if the dense band disappears (a long fix-free
run-in with no [12–24 m] accurate fixes) or speed rises so far the band yields too few fixes to pin
velocity — neither is the gate-4 @30 m/s case. **Whenever the binding term is centering (σ_p0), the lever is
a near-field GATE estimator / better per-fix σ, NOT VIO.**

---

## 6. WHAT TO INVEST IN (the prompt's decision)

1. **inc8 per-fix accuracy + band fix-DENSITY at ~12 m → shrink σ_p0. HIGHEST value** — this is the single
   binding knob (σ_p0 ≲ 0.08 closes r=0.30). Pointing/calibration (at-speed σ_lat 0.15→0.10) and more
   accepted [12–24 m] fixes (higher N_eff) both push σ_p0 down.
2. **Near-field GATE estimator (lower the 12 m floor) → MEDIUM-HIGH.** A closer last *gate-anchored* fix
   directly reduces σ_p0 and the coast length. (This is a gate tracker / visual servoing — NOT VIO/odometry.)
3. **ESKF accel-bias → LOW for the *coast* specifically** (0.6° over 0.4 s = 0.008 m). It remains a
   whole-lap lever (boresight), but it does ~nothing to the terminal coast.
4. **VIO (#70) → LOW / parked** — addresses velocity (already fine), blind to centering. Trigger σ_v0 ≳ 0.18 m/s.
5. **Speed → not a coast lever.** Coast cheap + flat 22–37 m/s; centering speed-flat. Keep 30 m/s; the
   speed risk is the sim-to-real blur σ (separate, physical-drone item), not the coast.
- **NEW estimator-tuning flag (NEW parked, surfaced UP):** on a *dense* gate-relative fix stream the
  production LinearKF over-converges (σ_p→~1 mm) and starts **ignoring fixes / riding the IMU** — so a real
  IMU bias would accrue uncorrected and a real per-approach fix bias would NOT be pulled out. A **covariance
  / R floor** (don't let σ_p drop below the systematic per-fix floor ~0.05 m) keeps the filter weighting
  fixes. Revive-trigger: any inc8 terminal-lock deploy or case-C estimator-tuning pass.

---

## 7. CAVEATS / VALIDITY

1. **σ_p0 is a MODELED boundary input** (the prompt framed this as feasibility/boundary, no 30 m/s data).
   σ_p0 ≈ σ_lat/√N_eff ⊕ σ_b is bounded from the two recordings (accept-geometry + at-speed); the inc8
   race-posture terminal density (N_eff) is uncrowned, so the 0.05–0.13 m σ_p0 band — straddling the 0.08
   closure boundary — is the honest spread. **The 30 m/s gate-4 at-speed recording remains the pin.**
2. **A deterministic ~0.056 m velocity-integration offset** (Catmull-Rom-spline-vs-constant-accel) sits in
   every miss; it is COMMON to the boresight engine (so the budget comparison is apples-to-apples) and is
   partly synthetic — the random spread (what σ_v / σ_p0 control) is the physically-meaningful signal.
3. **MC over-convergence** (§1) — centering-blind; A1 (with measured per-fix σ) is the centering tool. The
   MC's valid contribution is the coast-cheapness across speed/floor/bias (`coast_cheap_results.json`).
4. **iid/AR(1) fix model** — the per-approach σ_b captures the dominant non-averaging lateral term; depth
   bias is along-track (owned by the absolute fix+IMU), out of the in-plane budget.
5. **Sign/frame:** OBS +L, R_y(π) ODOMETRY conjugation, contact-radius semantics all upstream/unchanged;
   this is a covariance+miss propagation on the production stack, no estimator code edited.

---

## MEMORY-DELTA (≤10 lines — for the overall commander to bank)

```
- COAST-DRIFT RESOLVED (P3 worker, 2026-06-15; production LinearKF+RewindKF coast, A1/A3): the INFORMED
  terminal coast is NOT the gate-4 blocker. The 0.4 s coast (12 m @30 m/s) adds <0.02 m total — process
  0.009, accel-bias 0.6deg 0.008, velocity-drift 0.008 (achieved sig_v0~0.02 m/s, IMU-short-term-floored).
  It is NOTHING like the boresight 0.3 s DROUGHT (zero-info): an informed coast dead-reckons on a tight velocity.
- PROMPT's sig_v HYPOTHESIS REFUTED-as-binding: sig_v0 is robustly ~0.02 m/s (IMU owns short-term velocity;
  fixes only correct slow accel-bias drift). The BINDING uncertain input is TERMINAL CENTERING sig_p0 = the
  last accurate fix's gate-relative lateral accuracy at ~12 m, which the coast preserves RIGIDLY.
- CLOSURE BOUNDARY (A1, gate-difficulty B-sweep): miss p99 ~ 3.0*sig_p0. gate-4 r0.30 (B=0.235) CLOSES IFF
  sig_p0_lat <= ~0.08 m; r0.38 (B=0.155) needs <=0.05; tight VQ2 (B=0.12) needs <=0.04. Achievable sig_p0 ~
  sig_lat/sqrt(Neff) (+) sig_b: pointed/accept-geom (sig_lat 0.075-0.11) -> 0.05-0.07 CLOSES w/ margin; raw
  at-speed (0.13-0.17) -> 0.08-0.13 KNIFE-EDGE/FAILS. => CONDITIONAL-CLOSE gated on inc8 per-fix sigma + band
  fix-DENSITY at ~12 m (NOT the coast, NOT velocity, NOT accel-bias).
- VIO-TRIGGER (#70 back-solve, no build): VIO shrinks sig_v0 (already fine) but is BLIND to centering ->
  WRONG lever. Trigger = terminal sig_v0 >= ~0.18 m/s (gate-4) / 0.11 (r0.38) / 0.08 (B=0.12); IMU delivers
  ~0.02 (x9 margin) -> #70 STAYS PARKED. Centering lever = near-field GATE estimator (lower the 12 m floor) +
  per-fix accuracy, NOT odometry.
- BEST gate-4 speed (coast view) = 30 m/s (coast cheap+flat 22-37; centering speed-flat; faster re-opens the
  sim-to-real blur-sigma risk). Reconciles boresight ("sig_lat<=0.245 + terminal gate-lock") -> sharpened to
  sig_p0_lat<=0.08 at 12 m.
- NEW PARKED (surfaced UP): production LinearKF OVER-CONVERGES on a dense gate-relative stream (sig_p->~1 mm)
  and starts IGNORING fixes/riding the IMU -> add a covariance/R floor (~0.05 m) so it keeps weighting fixes;
  real IMU bias would otherwise accrue uncorrected. Revive: any inc8 terminal-lock deploy / case-C tuning.
  -> [[index-vision-estimator]] [[index-rl-training]] [[index-control-sim]] [[project-parked-backlog]]
```
