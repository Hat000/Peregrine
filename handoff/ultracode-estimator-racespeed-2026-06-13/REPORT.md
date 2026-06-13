# ESTIMATOR @ RACE SPEED — can the vision-fed KF thread gate-4 at ~37 m/s? (case-C)

**Fengyou** — commander deliverable for `ULTRACODE-ESTIMATOR-RACESPEED` (opus-4.8, ultracode).
Peregrine / AI Grand Prix. 2026-06-13. Offline analysis + prototype/sim only (no live sim, no src
edits, no SLURM, no memory edits). Workflow `wf_fdc8b274-762`: 12 opus agents, 1.44M tok, ~71 min,
4 phases (Measure → Verify → CloseGap → Synthesize). Every load-bearing number was reproduced by an
adversarial verifier AND independently re-run by me (commander) in `.venv`; the spots where verification
or my own re-run **changed** a worker's headline are flagged explicitly.

**The question.** At the post-gate-3 ~37 m/s window the gate-4 contact-true margin is 0.155 m @ r=0.38,
but the vision world-fix East σ is 0.47 m (unfiltered) = 3.0× the margin. Can the vision-fed KF (with
the case-C OOSM rewind buffer / range-anisotropic R / latency compensation prototypes ② already built)
reach **<0.05 m 1-σ in-plane** there — the bar ③ derived as margin/3? Gate-4 in-plane axes = **E
(lateral) + D (vertical)**; along-track = **N**. Binding miss = √(E²+D²). CONDITIONAL on case C
(vision-only pose): if VQ2 streams `LOCAL_POSITION_NED`/`ODOMETRY` (case A/B, organizer Q① unanswered),
pose is pristine and this entire risk is **moot** — everything below is the case-C worst case.

---

## TL;DR — the verdict in five lines

1. **Absolute world-frame KF nav is NO-GO at race speed, by a wide and speed-flat margin.** Measured
   filtered in-plane error at gate-4 = **~0.33 m RMS de-biased-idealized / ~0.55 m deployable** — ≈2–3.5×
   the 0.155 m contact margin, ≈7–11× the 0.05 m bar. **No knob clears it**: not acceptance, not cadence,
   not the cov floor (a trap), not detector accuracy (the detector is already sub-pixel). It is invalid
   even at 8 m/s — this is not a speed problem on the absolute path, it is an architecture problem.
2. **The binding terms are (i) a per-fix VARIANCE floor** (~0.50 m/axis, which the KF averages only
   ~3–9 fixes deep because **velocity is unobservable in case C**) **and (ii) an un-filterable per-track
   BIAS** (range-collapsing: ≥0.19 m near-band, 0.11 m at the last fix). Latency is **in-plane-benign at
   edge HW** (5 mm leak) — its cost is along-track; at CPU HW the naive update is catastrophic and the
   OOSM rewind buffer becomes mandatory.
3. **GATE-RELATIVE observation is THE fix** (the highest-leverage branch, the only one that fits the
   gate-4 miss inside the margin): observe the offset to the **seen** gate-4 opening (−L from PnP), which
   removes the per-track map/registration bias *exactly* (`db` drops out of the arithmetic), and converts
   the absolute-NED variance to close-range corner-reprojection noise.
4. **Gate-relative clears the 0.155 m CONTACT margin at 37 m/s (CONDITIONAL-GO), but not the strict 0.05 m
   bar.** Achieved in-plane miss ≈ **0.11–0.21 m RMS** at the measured near-band per-fix lateral
   (0.10–0.27 m). **Commander refinement (load-bearing): the margin-clear is sensitive to the velocity
   prior** — c1's 0.11 m assumes a *warm* (lap-converged) velocity; my cold-prior re-run gives 0.17–0.21 m
   (over the margin). The truth straddles the margin and is resolved only by (a) a full-lap sim and (b)
   the at-speed recording. Clearing the 0.05 m bar needs per-fix lateral ≤ ~0.08 m (2.6× easier than the
   absolute path's ~0.03 m).
5. **Master gate = organizer Q①.** If VQ2 is case A/B, pose is pristine and none of this matters. Build
   the gate-relative pipeline only after Q① confirms case C. Q⑤ (eval-HW class) flips the latency posture
   (edge = free / CPU = ruling, rewind mandatory). One ShadowPC at-speed gate-4 recording collapses the
   single dominant residual uncertainty (37 m/s motion-blur is currently MODELED, not measured).

---

## (1) MEASURED filtered 1-σ at the gate-4 ~37 m/s window

All cells: the **real** `racer.state_estimator.LinearKF` + the case-C `RewindKF` + the **real**
`localization.gate_pose_to_world_position` fix covariance; per-fix errors **resampled from the measured
`off_ned` pool** (perception-char-2026-06-08, n=109 good fixes, range-band conditioned, reproduces the
FACTS per-band bias/std exactly); const-v 37 m/s on the level 24.4 m g3→g4 leg (0.66 s transit); 47%
acceptance (~14 Hz → ~9 landed fixes). RAW = bias included (deployable). DE-BIASED = global VISION-CAL
de-bias applied. The sim (`a1_sim.py`) was reproduced bit-for-bit by an independent verifier (b1, 600
seeds) and again by me.

### Absolute world-fix path (current architecture)

| cell (edge L=6 ms, 47%, short shutter) | filtered in-plane 1-σ | in-plane RMS | <0.05 m? | <0.155 m? |
|---|---|---|---|---|
| DE-BIASED, **per-gate bias idealized away** — variance-only floor | 0.239 / 0.239 (E/D) | **0.33** | NO (~5×) | NO (~2×) |
| **DEPLOYABLE de-biased** (per-gate gate-4 residual intact) — *b1 correction* | — | **~0.55** | NO (~11×) | NO (~3.5×) |
| RAW (no de-bias) | — | 0.46 | NO | NO |
| honest-worst (a3 long-exposure blur + leak + residual) | — | **~0.78** | NO | NO |

- **NO cell clears 0.05 m** — across 18 sweep cells (L ∈ {6, 16, 112 ms} × acceptance {47, 25, 10%} ×
  {raw, de-biased}). The best (de-biased, edge, 47%) is 0.33 m in-plane RMS.
- **Floor probe** (pure variance, 14 Hz): per-fix σ 0.50 → filtered in-plane 0.32; 0.30 → 0.23; 0.10 →
  0.098; 0.05 → 0.051; 0.03 → 0.032. **Only per-fix ≤ ~0.03 m clears the bar at 14 Hz** — a ~16×
  per-fix accuracy improvement the absolute path cannot get.
- **Along-track vs in-plane:** the −0.346 m D bias and the lateral-E systematic land on the **in-plane**
  (binding) axes. The gate-size depth bias (−0.355 m, corr(off_N, range_err)=0.994) and the latency
  staleness v·L (~98% along-track, a4) land on **N (along-track)** — they perturb *when* the plane is
  crossed, not the in-plane miss.
- **Filter honesty (NEES vs χ²(3)=3.0):** the deployed rewind+de-biased path is consistent-to-mildly
  overconfident (3.3–4.3). The RAW arm is overconfident (4–7.5: P does not model the per-fix bias). The
  **naive in-place update at CPU L=112 ms is BROKEN** (NEES 86–411, +4.3 m along-track staleness) — the
  rewind buffer is what keeps that cell sane.

### Verifier corrections (both load-bearing — this is the adversarial layer working)

- **b1 CONFIRMED a1's no-go but corrected the number UP.** a1's 0.24 m de-biased 1-σ is a *variance-only
  floor*, not the deployable error: the per-gate gate-4 residual is not global, so it survives global
  de-bias → **deployable in-plane RMS ~0.55 m, not 0.33 m.** Carry **0.55 m** as the deployable absolute
  number (≈3.5× margin, ≈11× bar).
- **b2 WEAKENED a2's bias headline (toward the *favorable* direction).** The gate-4 in-plane bias is
  **range-collapsing, not a fixed 0.52 m constant**: ≤27 m → 0.523 m, ≤12 m → 0.337 m, **≤9 m
  (last-usable accepted band) → 0.191 m (CI [0.11, 0.25])**, last accepted fix at 8.25 m → **0.109 m**.
  Carry **0.191 m** as the defensible near-band per-track bias floor (a2's 0.52 m overstates the deficit
  ~2–3×). b2's two attacks *failed* (the global de-bias is NOT circular per a leave-one-gate-out hold-out;
  the bias is NOT a depth-scale artifact, corr(in-plane, |range_err|) = −0.03) — so a2's *direction* is
  robust even though its magnitude shrank.

### Commander independent cross-check (I re-ran this myself, not relayed)
Driving the real `LinearKF` over the realistic 0.66 s / ~9-fix window: absolute de-biased in-plane RMS =
**0.28 m** (vs a1's 0.33 — same verdict, both decisively over both thresholds). **Absolute NO-GO
independently confirmed.** My earlier steady-state anchor (0.11 m at σ=0.47/14 Hz) was a loose lower
bound; the finite-window transient is worse, which is why a1's 0.24–0.33 m is the right number, not my
anchor. The KF cannot reach steady state in 0.66 s, and case-C velocity unobservability caps the averaging.

---

## (2) CASE-C race-speed VALIDITY — the three-way split + single verdict

The bar is margin/3 (a conservative design target). The *physical* validity criterion for a zero-contact
run is **stay inside the 0.155 m contact margin** — that is the operative gate; 0.05 m is a stretch.

| component (gate-4 in-plane E,D) | ABSOLUTE case-C | GATE-RELATIVE case-C |
|---|---|---|
| (i) filtered VARIANCE | ~0.24 m idealized, floor-pinned; cadence saturates at 0.106 m even at 240 Hz (c2); cov floor can't be lowered (c4 trap); detector already sub-pixel (c3) | crushes to per-fix-lat / ~1.6–2.8; reaches 0.155 m margin at per-fix ≤ ~0.13–0.30 m, the 0.05 m bar at ≤ ~0.08 m |
| (ii) un-filterable per-track BIAS | **≥0.19 m near-band** (b2), survives global de-bias; un-removable in pure case C (chain-circularity, a2 §4.2) | **→ 0 by construction** (`db` drops out); residual = one unmeasured one-signed extrinsic systematic only |
| (iii) latency | along-track (a4); in-plane leak 5 mm edge / 41 mm CPU — sub-bar; RewindKF mandatory on CPU | identical (latency is path-independent at the in-plane axis) |

### VERDICT

> **Absolute world-frame case-C pose CANNOT thread gate-4 at race speed.** Deployable in-plane error
> ~0.55 m ≈ 3.5× the 0.155 m margin, floor-dominated and **speed-flat** (already invalid at 8 m/s) — so
> **no inc8 cone rung is estimator-valid on the absolute path at any speed.** No tuning knob clears it.
>
> **The gate-relative observation is the architecture that can.** It pulls the gate-4 in-plane miss to
> **~0.11–0.21 m RMS — CONDITIONAL-GO on the 0.155 m contact margin** at 37 m/s, by removing the
> per-track bias and converting world-fix variance to close-range PnP scatter. It does **NOT** clear the
> strict 0.05 m bar at the currently-measured per-fix accuracy; that needs per-fix lateral ≤ ~0.08 m.

**Conditions on the GO** (all four are hard, not nice-to-haves):
1. Treat the **0.155 m margin** as the operative validity criterion (zero-contact), 0.05 m as a stretch.
2. Build a **relative-innovation outlier gate** — the one genuinely new pipeline element. Reproj alone
   does NOT separate depth-flips (flip p50 0.71 px < clean 1.02 px); the gate must test relative
   innovation, not a reproj threshold.
3. **Augment, do not replace** the absolute LinearKF: keep it for planning / feed-forward / the g4→g5
   hand-off; the relative-centering term owns only the terminal in-plane miss. Critically — observe the
   offset to the **seen opening** (−L from PnP). The "subtract gate_map pos" form is a proven
   ANTI-PATTERN: it *relocates* the bias to the control target (estimator E-bias still +0.17 m), it does
   not remove it.
4. **Confirm at race speed.** All numbers rest on ≤8.4 m/s data (range ≤23.3 m); the 37 m/s motion-blur
   multipliers (×1.0 short-shutter → ×1.3–2.0 long-exposure) are MODELED. Every filtered σ — absolute and
   gate-relative — is a **best-case lower bound**.

### Commander refinement to c1's headline (the margin-clear is velocity-prior-sensitive)
c1 reported gate-relative in-plane RMS 0.11–0.14 m (per-fix/~2.8) → clears the margin comfortably. My
independent re-run gets **per-fix/~1.6 (in-plane RMS 0.17–0.21 m, over the margin) with a *cold* velocity
prior**, recovering c1's ~per-fix/2.5 only with a **warm (lap-converged) velocity prior** (vel_std≈0.15).
The fork is real: **case-C velocity is observable only through position-fix differencing**, so the
effective velocity entering the gate-4 window sits between cold and warm, and the margin-clear at the
measured near-band per-fix (0.20–0.265 m) **straddles the 0.155 m margin**. This is why the verdict is
CONDITIONAL-GO, not a comfortable clear. Two cheap follow-ups resolve it: (a) a **full-lap** sim (not just
the gate-4 window) to measure the actual velocity-prior quality entering gate-4 in case C; (b) it mildly
reopens the deferred **vision-velocity channel** (case-C P3-1) — velocity is the swing variable, so a
weak vision-velocity assist may buy more margin headroom than ② credited. Flag, don't build yet.

---

## (3) GAP-CLOSING PLAN — 5 branches ranked by leverage

Leverage = achievable in-plane error × feasibility × fit to the binding term (a per-track BIAS +
a VARIANCE floor). The winner must attack **both** — only one does.

| rank | branch | attacks | achievable in-plane | feasibility | verdict |
|---|---|---|---|---|---|
| **1** | **GATE-RELATIVE observation** (c1) | **BIAS (decisive) + VARIANCE** | **0.11–0.21 m RMS** → clears 0.155 m margin (vel-prior-sensitive); 0.07 m if per-fix lat ≤0.08 m | HIGH (controller-side IBVS / relative-centering term + 1 relative-innovation gate) | **THE FIX. Adopt.** |
| 2 | per-gate de-bias via privileged-pose cal lap (a2 §4) | BIAS | ~0.07 m residual (σ-limited) | needs case A/B pose; pure case-C cal lap CANNOT reach it (chain-circularity) | PARTIAL — only if Q①=A/B (which moots the whole risk). Not a case-C fix, and leaves the variance floor. |
| 3 | gate-size recalibration 1.5→~1.535 m (c3) | BIAS (depth) | removes ~0.18 m of **along-track (N)** bias | HIGH (one-line) | LOW-LEVERAGE for the binding axis (off-axis). Do it for crossing-timing, not the in-plane miss. |
| 4 | higher detector CADENCE / multi-fix fusion (c2) | VARIANCE only | floors at 0.106 m even at impossible 240 Hz; 0.34 m at realistic 30 Hz | LOW (>30 Hz needs edge HW; CPU caps ~8 Hz; bar needs ~1–2 kHz) | DEAD END. Variance-only, saturates above the bar, bias-blind. |
| 5 | lower the 0.40 m COV FLOOR (c4) | (intended VARIANCE) | NET-NEGATIVE: ≤15 mm gain, +51 mm bias regrowth, 14–36% over-rejection | feasible but harmful | TRAP. Do NOT do it. The floor is correctly MLE-sized to 0.407 m. |
| — | detector/PnP sub-pixel accuracy (c3) | VARIANCE (pixel) | **0.000 m** — detector already sub-pixel (reproj 0.50 px); in-plane noise is lever(1.4°)+floor limited, not pixel limited | — | DEAD END. |

**Recommended path.** Adopt **gate-relative observation** as the gate-4 terminal observation
(controller-side IBVS / relative-centering on the seen opening, augmenting the absolute KF) + a
relative-innovation gate. Residual in-plane miss **~0.11–0.21 m RMS at 37 m/s** — inside the contact
margin if the velocity prior is warm and per-fix lateral holds at the measured near-band. To additionally
clear the 0.05 m bar, drive per-fix lateral PnP to ~0.08 m (sub-pixel corner refine — 2.6× easier than
the absolute path's ~0.03 m). Gate-size recal (rank 3) is a free along-track add-on for crossing-timing.
Branches 4, 5, and detector accuracy are dead ends / traps — do not spend on them.

**Speed-ladder coupling (c5) — the estimator gates the inc8 cone ladder.** On gate-relative the estimator
imposes a *finite* speed ceiling:
- per-fix lateral **≤0.10 m → 0.155 m margin held past 55 m/s** (full inc8 ladder valid on the margin).
- per-fix **0.05 m → 0.05 m bar holds only to ~26 m/s**; per-fix **0.03 m → bar holds to ~50 m/s**.
- **To fly ~37 m/s inside the margin, gate-relative per-fix lateral must reach ≤ ~0.10–0.13 m** (the
  vel-prior fork sets the exact threshold). **Re-verify every cone-relaxation rung against the *achieved*
  gate-relative per-fix σ at that rung's speed** — cadence loosens, long-exposure blur inflates, reaction
  time shrinks ~1/v (166 ms @8 → 36 ms @37 → 24 ms @55).

---

## (4) DEPENDENCIES — organizer questions + the one decisive measurement

- **Organizer Q① (does VQ2 stream pose? case A/B vs C) — THE master gate.** If VQ2 streams
  `LOCAL_POSITION_NED`/`ODOMETRY` (case A/B), pose is pristine, the bias+variance wall does not exist, and
  **this entire risk is MOOT** — the gate-relative pipeline is unnecessary. Everything above is the
  case-C worst case. **Resolve Q① before committing engineering to gate-relative.** (Q① also gates the
  rank-2 cal-lap branch — a privileged-pose per-gate de-bias only exists if A/B.)
- **Organizer Q⑤ (eval-HW latency / GPU class) — flips the latency posture.** L is an ESTIMATE never
  measured on eval HW: edge 6 ms p50 / 16 ms p90 (≥100-TOPS), CPU 112–125 ms (laptop upper bound). On
  **edge HW** latency is in-plane-benign (5–7 mm leak; along-track staleness 0.22–0.60 m). On **CPU-class
  HW** the naive update is catastrophic (in-plane 0.83 m, +4.3 m along-track, NEES 411) → **RewindKF
  becomes MANDATORY**, and even with rewind the CPU path is independently unacceptable on the along-track
  ground and caps cadence at ~8 Hz. Both L bands sit far under the RewindKF 0.5 s horizon (≥0.37 s margin
  even at CPU p90) → **no horizon-divergence risk** at these L (the piece-B "horizon < L → diverge to
  21 m" failure is not triggered). **Ship RewindKF regardless** — cheap on edge, decisive on CPU.
- **The single decisive measurement — one ShadowPC at-speed (~37 m/s) gate-4 recording with track_map
  ground truth.** All numbers rest on ~5.35 m/s data (range ≤23.3 m); there is NO measured per-fix error
  at 37 m/s (speed↔reproj corr 0.004 — no dynamic range to see blur). This one capture collapses the
  exposure/shutter pivot (×1 vs ×2 blur), re-measures per-fix and **gate-relative per-fix-lateral** noise,
  acceptance, first-accept range, AND tests for the **one-signed residual extrinsic systematic** — the
  single floor the gate-relative obs cannot remove (currently bounded only by magnitude ≤0.19 m, n=3,
  sign unmeasured). Fold into SHADOWPC-VISION-CAL.
- **a3 escape-hatch items (all resolved by that one capture):** E1 camera exposure/shutter type (the
  ×1↔×2 blur pivot); E2 the inc8 policy's body angular rate on g3→g4 (the dominant blur term — depends on
  a policy that does not yet exist); E3 per-fix accuracy at 37 m/s; E4 at-speed detection ceiling /
  first-accept range; E5 at-speed acceptance / χ² leak rate.
- **Commander-added follow-up:** a **full-lap** (not gate-4-window-only) case-C sim to pin the velocity
  prior entering gate-4 (the swing variable in the §2 refinement) — pure offline, no new data needed.

---

## (5) CONFIDENCE — measured vs extrapolated vs assumed

**MEASURED (high confidence):**
- Per-fix bias [−0.285, +0.064, −0.346] m and noise std [0.82, 0.58, 0.44] m NED (n=109; reproduced
  natively by me, a1, a2).
- Gate-4 geometry: in-plane = (E,D), along-track = N (independently confirmed by b2).
- Detector is sub-pixel (reproj median 0.50 px); world-fix noise decorrelated from reproj (corr −0.145)
  → detector accuracy is a dead end for the in-plane miss (c3).
- Cov-floor lowering is net-negative (c4, direction independent of bias magnitude).
- Absolute NO-GO and the gate-relative bias-removal (`db` drops out exactly) — both re-run by me.

**EXTRAPOLATED (medium-low — the dominant residual uncertainty):**
- **Per-fix (and gate-relative per-fix-lateral) accuracy AT 37 m/s.** All data ≤8.4 m/s. The 37 m/s noise
  multipliers are MODELED → every filtered-σ number is a best-case **lower bound**; the long-exposure
  band (×1.3–2.0) is the honest worst case.
- The effective-fix count (~3–9) and thus the gate-relative per-fix/RMS ratio (**1.6 cold ↔ 2.8 warm
  velocity prior** — my cross-check) — this is what makes the margin-clear conditional.
- The c5 max-valid speeds map *which* per-fix accuracy buys *which* speed; they do not assert
  gate-relative *achieves* a given accuracy.

**ASSUMED (stated):** const-v 37 m/s, drag-hold pitch 38.4°, 90 Hz IMU, 47% acceptance (a1/b1 showed the
in-plane verdict is insensitive to the accel profile <0.02 m and the motion model ~mm); the 37 m/s cruise
itself (memory/INC7-live).

**THE BIGGEST RESIDUAL UNCERTAINTY:** whether the gate-relative per-fix lateral noise holds at ~0.10–0.20 m
at 37 m/s (vs blowing up under blur), whether it carries a one-signed residual extrinsic bias, **and**
whether the case-C velocity prior entering gate-4 is warm enough for the per-fix/~2.5 averaging. If the
noise holds, the residual is zero-mean, and the velocity prior is warm → the rank-1 fix clears the margin
with headroom. If any of the three goes the wrong way → the miss tightens against the margin. **One
ShadowPC at-speed gate-4 recording + one full-lap offline sim resolve all three — and the whole question
is gated upstream by organizer Q① (if A/B, none of this matters).**

---

## Artifacts (all under `handoff/ultracode-estimator-racespeed-2026-06-13/`)
- **Measure:** `a1_sim.py` / `a1_results.json` / `a1_findings.md` / `a1_floor_probe.{py,json}` (closed-loop
  sim); `a2_bias.py` / `a2_findings.md` (bias decomposition); `a3_realism.py` / `a3_findings.md` (37 m/s
  fix-stream + escape-hatch); `a4_latency.py` / `a4_findings.md` (latency geometry split).
- **Verify:** `b1_verify.py` / `b1_findings.md` (CONFIRMED, corrected deployable ↑); `b2_verify.py` /
  `b2_findings.md` (WEAKENED, corrected bias ↓ to near-band 0.19 m).
- **CloseGap:** `c1_*` (gate-relative — THE FIX); `c2_*` (cadence — dead end); `c3_*` (PnP accuracy — dead
  end + along-track bias find); `c4_*` (cov floor — trap); `c5_*` (speed-validity Pareto).
- **Synthesize:** `synth_synthesis.md`. **This report:** `REPORT.md`. **Inputs:** `FACTS.md`.
- Commander cross-check sims were run inline (not committed); reproducible from the snippets in the
  session transcript (steady-state anchor + realistic-window absolute/relative + cold/warm velocity fork).

---

## MEMORY-DELTA: (≤12 lines; for the commander to bank — supersessions flagged)
- ESTIMATOR-RACESPEED DONE. **Case-C ABSOLUTE world-frame pose CANNOT thread gate-4 at 37 m/s**: deployable
  in-plane ~0.55 m (b1-corrected up from a1's 0.33 variance-only floor) ≈3.5× the 0.155 m margin, ≈11× the
  0.05 m bar; **floor-dominated + SPEED-FLAT (invalid even at 8 m/s)** → no inc8 cone rung is estimator-valid
  on the absolute path. NO cell clears (18-cell sweep, real KF, verified + commander-re-run).
- Binding terms: VARIANCE (per-fix ~0.50 m floor, **velocity unobservable in case C** → KF averages only
  ~3–9 fixes) + un-filterable per-track BIAS (b2: range-collapsing, **≥0.19 m near-band, 0.11 m last-fix** —
  ⚠️ SUPERSEDES a2's 0.52 m constant). Latency in-plane-BENIGN at edge (5 mm); CPU naive catastrophic →
  RewindKF MANDATORY (ship it regardless; no horizon<L divergence at these L).
- **GATE-RELATIVE observation = THE FIX (rank 1):** observe offset to the SEEN opening (−L from PnP) → map
  bias `db` drops out EXACTLY → in-plane **0.11–0.21 m RMS, CONDITIONAL-GO on the 0.155 m CONTACT margin**
  at 37 m/s (NOT the 0.05 m bar; that needs per-fix lat ≤0.08 m). "subtract gate_map" = ANTI-PATTERN
  (relocates bias to control). AUGMENT the absolute KF, don't replace; +1 relative-innovation gate (reproj
  doesn't separate depth-flips). Determinism-per-track preserved trivially.
- ⚠️ COMMANDER REFINEMENT to c1: the margin-clear is **velocity-prior-sensitive** — per-fix/2.8 (warm,
  lap-converged vel) vs per-fix/1.6 (cold) → in-plane 0.11 m vs 0.21 m, straddling the margin. Resolve via
  a full-lap case-C sim; mildly reopens the deferred vision-velocity channel as a margin lever.
- DEAD ENDS/TRAPS confirmed: cadence (c2 floors 0.106 m, bias-blind), cov-floor lowering (c4 TRAP +51 mm
  bias / 14–36% over-rej), detector sub-pixel (c3 = 0.000 m, in-plane is lever+floor limited). Gate-size
  recal (c3) = along-track only (off binding axis); do for crossing-timing.
- SPEED LADDER (c5): per-fix lat ≤0.10 m holds margin past 55 m/s; 0.05 m bar to ~26 m/s; 0.03 m to ~50 m/s.
  Re-verify every inc8 cone rung vs achieved gate-rel σ at that speed.
- DEPENDENCIES: **Q① (case A/B → risk MOOT) = master gate**; Q⑤ (CPU-class → RewindKF mandatory, flips
  latency edge→ruling). **ONE ShadowPC at-speed (~37 m/s) gate-4 recording** collapses the 37 m/s
  blur/exposure extrapolation + the UNMEASURED one-signed gate-rel extrinsic bias (n=3). All filtered σ are
  best-case LOWER BOUNDS (5.35 m/s data). Confirms the cross-cutting finding: **estimator is the binding
  VQ2 risk; gate-relative obs is the highest-leverage fix.**
