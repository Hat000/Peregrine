# synth — ESTIMATOR-RACESPEED synthesis (gate-4 @ ~37 m/s, case-C)

Agent **synth** (synthesis lead, opus-4.8). Offline only. Reads all prior outputs
(a1–a4, b1–b2, c1–c5). Every number below is traced to a prior agent's MEASURED data or
sim cell and reproduces from their JSON (spot-verified: c1 `variance_sweep`, c5
`gate_relative_floor_removed`). The commander refines this into REPORT.md.

**The mission question:** can the vision-fed KF reach **<0.05 m 1-sigma in-plane position
error at the post-gate-3 ~37 m/s gate-4 window** for a zero-contact run, in the worst-case
**case-C** (vision-only pose)? Gate-4 in-plane axes = **E (lateral) + D (vertical)**;
along-track = **N**. Binding miss = `sqrt(E^2 + D^2)`. Contact-true margin **0.155 m @ r=0.38**.

---

## (1) MEASURED filtered 1-sigma at the gate-4 ~37 m/s window

All cells: REAL `LinearKF` + `RewindKF` + `gate_pose_to_world_position` fix-cov; residuals
resampled from the MEASURED `off_ned` pool (perception-char, n=109 good fixes), range-band
conditioned; const-v 37 m/s on the level g3->g4 leg; 47% accept (~14 Hz, ~9 landed fixes).
**RAW** = bias included (deployable real-world). **DE-BIASED** = global VISION-CAL de-bias
applied. **"deployable de-biased"** = global de-bias applied but the per-gate gate-4 residual
left in (the truly shippable case-C number — b1's correction to a1).

### A. ABSOLUTE world-fix path (current architecture)

| cell (edge L, 47%, short shutter) | filtered in-plane 1-sigma | in-plane RMS | clears 0.05 m? | clears 0.155 m? |
|---|---|---|---|---|
| DE-BIASED, idealized (per-gate bias fully removed) — **variance-only floor** | **0.239 / 0.239 (E/D)** | 0.33 | NO (~5x) | NO |
| **deployable de-biased** (per-gate residual intact) — **b1-corrected** | — | **~0.55** | NO (~11x) | NO (~3.5x) |
| RAW (no de-bias) | — | 0.46 | NO | NO |
| honest-worst (A3 long-exposure + leak + per-gate residual) | — | **~0.78** | NO | NO |

- **Along-track (N) vs in-plane (E,D):** the −0.346 m D bias and the lateral E systematic
  fall on the **in-plane** axes (binding). The depth/gate-size bias (−0.355 m, corr off_N vs
  range_err = 0.994, c3) falls almost entirely on **N (along-track)** — it perturbs *when* the
  plane is crossed, not the in-plane miss. Latency staleness `v·L` is also ~98.4% along-track
  (a4), so it does not enter the binding miss at edge HW.
- **DE-BIASED vs RAW:** de-bias removes the −0.33 m D bias and the global E offset, dropping
  RAW 0.46 → de-biased-idealized 0.33 in-plane RMS. But the **per-gate** gate-4 residual
  survives global de-bias (b1/a2), so the **deployable** de-biased number is ~0.55 m, not 0.33.

- **Verifier flags (load-bearing):**
  - **b1 CONFIRMED** a1's no-go but corrected the headline number UPWARD: a1's 0.24 m
    de-biased 1-sigma is a **variance-only floor**, NOT the deployable error. b1 reproduced
    a1's cells exactly (0.239/0.239, RMS 0.328, RAW 0.464, floor-probe, per-fix σ ~0.50 m) —
    the sim is sound — then showed the deployable de-biased in-plane RMS is **~0.55 m** because
    the per-gate gate-4 bias is not global. **Carry 0.55 m (not 0.33 m) as the deployable
    absolute number.**
  - **b2 WEAKENED** a2's bias headline: the gate-4 in-plane bias is **range-collapsing, not a
    fixed 0.52 m constant.** Independently re-measured: ≤27 m window 0.523 m → ≤12 m 0.337 m →
    **≤9 m (last-usable accepted-fix band) 0.191 m, CI [0.11, 0.25]**, last accepted fix at
    ~8.25 m = **0.109 m**. The plane-crossing fix itself (range 0.10 m) locks gate-5 (maha 127.8)
    and is rejected by the chi² gate, so 8.25 m is the operative last fix. **Carry 0.191 m as
    the defensible near-band per-track bias floor** (a2's 0.52 m would overstate the deficit
    ~2–3x). b2's two failed attacks (de-bias is NOT circular per LOGO hold-out; NOT a
    depth-scale artifact, corr in-plane vs |range_err| = −0.03) make a2's *direction* robust.

### B. GATE-RELATIVE path (the candidate fix — c1, variance reproduced from JSON)

Observe the offset to the *seen* gate-4 opening (`−L` from PnP); the map-registration bias
`db` drops out of the arithmetic exactly (proven, no MC). Filtered in-plane 1-sigma scales as
≈ per-fix-lateral / ~2.8 (KF averages ~9 fixes):

| per-fix lateral 1-sigma | filtered in-plane 1-sigma | in-plane RMS | clears 0.05 m? | clears 0.155 m? |
|---|---|---|---|---|
| 0.265 m (0–12 m avg, MEASURED) | 0.095 | 0.135 | NO | **YES** |
| 0.20 m (near-band, MEASURED) | 0.080 | 0.114 | NO | **YES** |
| 0.105 m (last-fix, MEASURED) | 0.061 | 0.086 | NO | **YES** |
| 0.08 m (achievable goal) | 0.049 | 0.070 | **YES** | YES |
| 0.05 m | 0.035 | 0.050 | YES | YES |

### THE BINDING TERM
- On the **absolute path**: the binding term is **VARIANCE** (the per-fix ~0.50 m measurement
  floor, which the KF averages only ~6–9 effective fixes deep because **velocity is
  unobservable in case C** — vision is position-only). A per-gate **BIAS** floor (≥0.19 m
  near-band, un-filterable, survives global de-bias) makes RAW and the honest de-biased path
  fail *additionally*. **Latency is NOT binding in-plane** at edge (5 mm leak; 41 mm at CPU);
  its real cost is along-track (a4). **Both terms are speed-flat** (c5).
- On the **gate-relative path**: bias is removed; the *residual* binding term is pure
  **VARIANCE** (per-fix lateral PnP scatter), which IS averageable and reaches the **margin**
  today but not the **bar**.

---

## (2) CASE-C race-speed VALIDITY — single honest verdict

The FACTS three-way split, each evaluated at the gate-4 ~37 m/s in-plane (E,D) window:

| component | absolute case-C | gate-relative case-C |
|---|---|---|
| **(i) filtered VARIANCE** | ~0.24 m (idealized) / floor-pinned; cadence saturates at 0.106 m even at impossible 240 Hz (c2); floor cannot be lowered (c4 TRAP); detector already sub-pixel so PnP accuracy is a dead end (c3) | crushes to per-fix/2.8; reaches 0.155 m margin at per-fix ≤0.30 m (MET today), 0.05 m bar at per-fix ≤0.08 m |
| **(ii) un-filterable per-track in-plane BIAS** | **≥0.19 m near-band** (b2), survives global de-bias; un-removable in pure case C (a2 §4.2 chain-circularity) | **→ 0 by construction** (db drops out); residual = unmeasured one-signed extrinsic systematic only |
| **(iii) latency** | along-track (a4); in-plane leak 5 mm edge / 41 mm CPU — sub-bar; RewindKF mandatory on CPU, ~free on edge | identical (latency is path-independent at the in-plane axis) |

### VERDICT

- **ABSOLUTE world-fix (current architecture): NO-GO at race speed, by a wide and
  speed-independent margin.** No knob clears it: not acceptance, not cadence (c2: floors at
  0.106 m), not the cov floor (c4: a TRAP that re-grows bias 2.5x and re-opens the 13–35%
  chi² over-rejection wound), not detector/PnP accuracy (c3: a perfect detector moves in-plane
  by 0.000 m — it is lever+floor limited, not pixel limited). Deployable in-plane error is
  **~0.55 m (b1), ≈3.5x the 0.155 m contact margin and ≈11x the 0.05 m bar**, and it is
  **floor-dominated and speed-flat** — already invalid at 8 m/s (c5). **No inc8 cone rung is
  estimator-valid on the absolute path at any speed.**

- **GATE-RELATIVE observation (the fix): CONDITIONAL-GO on the 0.155 m CONTACT MARGIN; NO-GO
  on the strict 0.05 m bar** at the currently-measured per-fix lateral accuracy (~0.10–0.27 m
  near-band). It pulls the gate-4 in-plane miss to **0.11–0.14 m RMS — inside the 0.155 m
  contact margin at 37 m/s** — by removing the per-track bias and converting absolute-NED
  variance to close-range corner reprojection.

**Honest single verdict:** *Case-C absolute pose cannot thread gate-4 at race speed. The
gate-relative observation is the architecture that can — it clears the contact margin (the
true validity criterion: zero-contact transit) at 37 m/s today, and clears the stricter 0.05 m
1-sigma bar only if per-fix lateral PnP reaches ~0.08 m.* The operative gate is therefore the
**0.155 m contact margin** (which a zero-contact run actually requires), with 0.05 m as a
stretch target.

**Conditions on the GO:**
1. The **0.155 m margin**, not the 0.05 m bar, is the operative validity criterion (the bar is
   margin/3, a conservative design target; the physical requirement is zero contact = stay
   inside 0.155 m).
2. A **relative-innovation outlier gate** must be built (reproj alone does NOT separate
   depth-flips: flip p50 0.71 px < clean 1.02 px — c1 adversarial item 2). This is the one
   genuinely new pipeline element.
3. Keep the absolute LinearKF for planning/feed-forward; gate-relative **augments, does not
   replace** it (c1 — preserves the global anchor for the min-snap/TOPP line and g4→g5 hand-off).
4. The per-fix lateral accuracy and the absence of a one-signed residual extrinsic bias must be
   confirmed at race speed (ShadowPC at-speed gate-4 recording) — until then 0.11–0.14 m is a
   **best-case lower bound** (5.35 m/s data; 37 m/s blur unmeasured).

---

## (3) GAP-CLOSING PLAN — the 5 branches ranked by leverage

Leverage = (achievable in-plane error) × (feasibility) × (bias-vs-variance fit to the binding
term). The binding terms are a per-track BIAS (≥0.19 m, un-filterable on absolute) + a VARIANCE
floor; the winning branch must attack **both**, and only one does.

| rank | branch | attacks | achievable in-plane | feasibility | verdict |
|---|---|---|---|---|---|
| **1** | **GATE-RELATIVE observation** (c1) | **BIAS (decisive) + VARIANCE** | **0.11–0.14 m RMS** (clears margin) → 0.07 m if per-fix lateral ≤0.08 m | HIGH (controller-side IBVS/relative-centering term; +1 relative-innovation gate) | **THE FIX. Clears the contact margin at 37 m/s.** |
| 2 | per-gate de-bias via privileged-pose CAL LAP (a2 §4) | BIAS | ~0.07 m residual (σ-limited) | MED on absolute path, but **needs case A/B pose**; pure case-C cal lap CANNOT reach it (chain-circularity) — and leaves the variance floor | PARTIAL. Only viable if Q1=A/B (which moots the whole risk). Not a case-C fix. |
| 3 | gate-size recalibration 1.5→~1.535 m (c3) | BIAS (depth) | removes ~0.18 m of ALONG-TRACK (N) bias at 12 m | HIGH (one-line) | LOW-LEVERAGE for the binding axis — it is **off-axis** (N, not E/D). Do it for crossing-timing, not the in-plane miss. |
| 4 | higher detector CADENCE / multi-fix fusion (c2) | VARIANCE only | floors at 0.106 m even at impossible 240 Hz; 0.34 m at realistic 30 Hz | LOW (>30 Hz needs edge HW; CPU caps ~8 Hz; 0.05 m needs ~1–2 kHz) | DEAD END. Variance-only, saturates above the bar, bias-blind. |
| 5 | lower the 0.40 m COV FLOOR (c4) | (intended VARIANCE) | NET NEGATIVE: ≤15 mm variance gain, +51 mm bias regrowth, 14–36% over-rejection | feasible but harmful | TRAP. Do NOT do it. The floor is correctly sized. |
| — | detector/PnP sub-pixel accuracy (c3) | VARIANCE (pixel) | 0.000 m (pixel = 0.15–0.7% of in-plane variance; detector already sub-pixel) | — | DEAD END. In-plane is lever(1.4°)+floor limited, not pixel limited. |

**THE RECOMMENDED PATH:** adopt **gate-relative observation** as the gate-4 terminal
observation (controller-side IBVS / relative-centering on the seen opening, augmenting the
absolute KF), plus a relative-innovation gate. **Residual in-plane error achieved: 0.11–0.14 m
RMS at 37 m/s — inside the 0.155 m contact margin.** To additionally clear the 0.05 m bar,
drive per-fix lateral PnP to ~0.08 m (sub-pixel corner refine — 2.6x easier than the 0.03 m the
absolute path would need). Gate-size recalibration (rank 3) is a free along-track add-on for
crossing-timing. Branches 4–5 and detector accuracy are dead ends / traps — do not spend on them.

**Speed-ladder coupling (c5, reproduced from JSON):** on gate-relative, the estimator imposes a
*finite* speed ceiling that gates the inc8 cone ladder:
- per-fix lateral **≤0.10 m → 0.155 m margin held past 55 m/s** (full inc8 ladder valid on margin).
- per-fix **0.05 m → 0.05 m bar holds only to ~26 m/s**; per-fix **0.03 m → bar holds to ~50 m/s**.
- **GATE: to fly ~37 m/s inside the contact margin, gate-relative per-fix lateral must reach
  ≤~0.10 m; for the 0.05 m bar, ≤~0.03 m.** Every cone-relaxation rung must be re-verified
  against the *achieved* gate-relative per-fix sigma at that rung's speed (cadence loosens,
  long-exposure blur inflates, reaction time shrinks ~1/v: 166 ms @8 → 36 ms @37 → 24 ms @55).

---

## (4) DEPENDENCIES

- **Organizer Q1 (VQ2 streams pose? case A/B vs C) — THE master gate.** If VQ2 streams
  `LOCAL_POSITION_NED`/`ODOMETRY` (**case A/B**), pose is pristine, the per-fix bias+variance
  wall does not exist, and **this entire risk is MOOT** — gate-relative is unnecessary and the
  absolute path is irrelevant. Everything in (1)–(3) is the **case-C worst case**. This must be
  resolved before committing engineering to the gate-relative pipeline. (Also folds in the
  rank-2 cal-lap branch: a privileged-pose per-gate de-bias only exists if A/B.)
- **Organizer Q5 (eval-HW latency / GPU class) — flips the latency margin.** Latency L is an
  ESTIMATE never measured on eval HW: edge 6 ms p50 / 16 ms p90 (GPU/~100-TOPS), CPU 112–125 ms
  (laptop upper bound). On **edge HW** latency is in-plane-benign (5–7 mm leak) and along-track
  staleness is 0.22–0.60 m. On **CPU-class HW** the naive arm is catastrophic (in-plane 0.83 m,
  +4.3 m along-track, NEES 411) — **RewindKF becomes MANDATORY**, and even with rewind the CPU
  path is independently unacceptable on the along-track ground (and caps cadence at ~8 Hz, c2).
  A CPU-class eval flips the latency posture from "free insurance" to "ruling constraint." Both
  L bands sit far under the RewindKF 0.5 s horizon (≥0.37 s margin even at CPU p90) — **no
  horizon-divergence risk at these L** (b1/a4 confirmed: mean dropped fixes ~0.49, the first
  pre-buffer fix only). RewindKF should ship regardless; on edge it is cheap, on CPU it is the
  difference between 0.38 m and 0.83 m in-plane.
- **THE single ShadowPC-at-speed measurement that resolves the extrapolation:** an **at-speed
  (~37 m/s) recording through gate-4 with track_map ground truth.** ALL prior numbers rest on
  ~5.35 m/s data (range ≤23.3 m); there is NO measured per-fix error at 37 m/s, so the 37 m/s
  blur multipliers (×1.0 short shutter → ×1.3–2.0 long exposure) are MODELED extrapolation 4–7x
  beyond data. This one capture collapses the **exposure/shutter pivot** (a3 escape-hatch
  E1–E5), re-measures per-fix and per-fix-lateral noise, acceptance, first-accept range, and —
  critically for the rank-1 fix — tests whether the **gate-relative per-fix lateral noise**
  stays at ~0.10–0.20 m and whether it carries a **one-signed residual extrinsic bias** (the
  one floor gate-relative cannot remove, currently UNMEASURED).
- **A3 escape-hatch items carried forward (resolved by the same at-speed capture):**
  E1 camera exposure/shutter type (the ×1 vs ×2 blur pivot); E2 body angular rate of the
  trained inc8 policy on g3→g4 (the dominant blur term, depends on a policy that does not yet
  exist); E3 per-fix accuracy at 37 m/s; E4 at-speed detection ceiling / first-accept range;
  E5 at-speed acceptance / chi² leak rate.

---

## (5) CONFIDENCE FLAGS — MEASURED vs EXTRAPOLATED vs ASSUMED

**MEASURED (high confidence):**
- Per-fix bias [−0.285, +0.064, −0.346] m and noise std [0.82, 0.58, 0.44] m NED (n=109).
- Gate-4 geometry, in-plane = (E,D), along-track = N (b2 confirmed independently).
- Detector is sub-pixel (reproj median 0.50 px) and world-fix noise is decorrelated from
  reproj (corr −0.145) → detector accuracy is a dead end for in-plane (c3).
- Cov-floor lowering is net-negative (c4 — direction independent of bias magnitude).
- Range-collapsing per-track bias: ≤9 m band 0.191 m, last fix 0.109 m (b2, n=3 — small sample
  but the monotone trend across all bands corroborates).
- a1 sim reproduced bit-for-bit by b1 (independent re-run, 600 seeds); c1/c5 anchors reproduce.

**EXTRAPOLATED (medium-low confidence — the dominant residual uncertainty):**
- **Per-fix (and gate-relative per-fix-lateral) accuracy AT 37 m/s.** All data is ≤8.4 m/s
  (median 5.35), range ≤23.3 m; speed↔reproj correlation is 0.004 (no dynamic range to see
  blur). The 37 m/s noise multipliers are MODELED. **Every filtered-sigma number — absolute and
  gate-relative — is a best-case LOWER BOUND.** The long-exposure column (in-plane noise ×1.3–2.0)
  is the honest worst-case band.
- Effective-fix count (~7–9) and cadence collapse at 37 m/s (MEASURED 30 Hz cadence × MODELED
  geometry window).
- The c5 gate-relative max-valid speeds (per-fix 0.10 m → margin to 55 m/s, etc.) — the per-fix
  accuracies are CANDIDATES, not measured; c5 tells you WHICH accuracy buys WHICH speed, not
  that gate-relative achieves it.

**ASSUMED (stated):**
- Const-v 37 m/s, drag-hold pitch 38.4°, 90 Hz IMU, 47% acceptance. a1/b1 showed the in-plane
  verdict is insensitive to the accel profile (<0.02 m) and the motion model (process growth ~mm).
- The 37 m/s cruise speed itself (memory/INC7-live, not re-measured here).
- The per-gate residual sigma [0.21, 0.24, 0.03] m NED used in c4's deployable arm (FACTS value;
  the verdict direction is independent of this magnitude).

**THE BIGGEST RESIDUAL UNCERTAINTY:** whether the **gate-relative per-fix lateral noise holds at
~0.10–0.20 m at 37 m/s** (vs blowing up under motion blur), and whether it carries a **one-signed
residual extrinsic/centroid bias** (the only floor gate-relative cannot filter — currently bounded
only by magnitude ≤0.19 m at n=3, sign UNMEASURED). If the noise holds and the residual is
zero-mean, the rank-1 fix clears the 0.155 m margin with headroom. If blur doubles it or a
one-signed extrinsic systematic exists, the margin tightens. **A single ShadowPC at-speed gate-4
recording resolves both** — and it is gated upstream by organizer Q1 (if A/B, none of this matters).

---

## Verifier-refutation ledger (where B1/B2 changed an A claim)

- **b1 vs a1:** CONFIRMED no-go; **corrected the deployable number UP** (a1's 0.24 m de-biased
  1-sigma is a variance-only floor, not deployable; deployable ~0.55 m, ≈11x bar). a1's "P is
  honest" overstated for the deployed arm (NEES 8.5, P under-reports ~2.8x). a1's catastrophic-leak
  omission immaterial (0.328→0.342). a1's slow-speed stream = the optimistic short-shutter end.
- **b2 vs a2:** WEAKENED — the bias is **range-collapsing, not a 0.52 m constant** (≤9 m band
  0.191 m). a2's headline 0.517 m overstates the deficit ~2–3x. Two a2 attacks survived (de-bias
  NOT circular per LOGO; NOT a depth-scale artifact), so a2's direction is robust. **The synthesis
  carries b2's 0.191 m near-band number, not a2's 0.52 m.**

---

## MEMORY-DELTA:
- synth ESTIMATOR-RACESPEED SYNTHESIS DONE. Single verdict: **case-C ABSOLUTE pose CANNOT thread
  gate-4 at 37 m/s** (deployable in-plane ~0.55 m, b1-corrected; ≈3.5x the 0.155 m margin, ≈11x
  the 0.05 m bar; floor-dominated + SPEED-FLAT — invalid even at 8 m/s). Binding term = VARIANCE
  (per-fix ~0.50 m floor, vel unobservable → KF averages only ~6–9 fixes) + un-filterable per-track
  BIAS (b2: range-collapsing, ≥0.19 m near-band, last fix 0.109 m). Latency in-plane-benign at edge.
- **GATE-RELATIVE observation = THE FIX (rank 1):** removes the per-track bias (db drops out exactly),
  pulls gate-4 in-plane to **0.11–0.14 m RMS — CLEARS the 0.155 m contact margin at 37 m/s**, NOT
  the 0.05 m bar (needs per-fix lateral ≤0.08 m). CONDITIONAL-GO on the margin criterion. Augment
  the absolute KF, don't replace; +1 relative-innovation gate (reproj doesn't separate depth-flips).
- DEAD ENDS/TRAPS confirmed: cadence (c2, floors 0.106 m), cov-floor lowering (c4, TRAP +51 mm bias
  / 14–36% over-rej), detector sub-pixel accuracy (c3, 0.000 m — lever+floor limited). Gate-size cal
  (c3) = along-track only (off binding axis). Privileged-pose cal lap (a2) only if Q1=A/B.
- SPEED LADDER (c5): per-fix lateral ≤0.10 m holds the margin past 55 m/s; 0.05 m holds the 0.05 m
  bar only to ~26 m/s; 0.03 m to ~50 m/s. Re-verify every inc8 cone rung vs achieved gate-rel sigma.
- DEPENDENCIES: Q1 (A/B → risk MOOT) is the master gate; Q5 (CPU-class → RewindKF MANDATORY, flips
  latency edge→ruling). ONE ShadowPC at-speed gate-4 recording collapses the 37 m/s blur/exposure
  extrapolation + the one UNMEASURED residual (gate-rel one-signed extrinsic bias). All filtered
  sigmas are best-case LOWER BOUNDS (5.35 m/s data; 37 m/s blur MODELED).
