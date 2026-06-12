# LAPTOP-TRAINING-DOCTRINE — are we training the right thing? (2026-06-12, fable)

**Session:** LAPTOP-TRAINING-DOCTRINE. **Checkpoint interrogated:** `stage1_inc6_actor.pth` on the
S17 mixer plant. **Probes:** `scripts/doctrine_probes.py` (q1, q2q6, q3, q4, q5, corridor) +
`scripts/q2_live_forensics.py` (real live-confirm data — `handoff/shadowpc-postfix-dataset-2026-06-12`
landed mid-session, so Q2 is answered from data, NOT provisional). Question ledger: `QUESTIONS.md`
(committed first, 7ecc575).

---

## TL;DR — the headline answer

**We have been training against a fictional gate.** The training env scores a point-mass L-inf
< 0.75 m as a clean pass; the live sim's contact envelope is the aperture minus the drone's body
extent, enforced volumetrically (you can hit the frame *before* the plane on a steep approach).
The four standing gate-3 crashes all terminated at L-inf **0.37–0.49 m — positions the training
env calls comfortable passes.** The policy is not under-powered (it uses ~16% of its rate
authority in the final 1.5 s, zero saturation), not unable to recover (it re-threads from 1.5 m
displacements, 3× the live error, 12/12), and not force-model-fragile (it absorbs ±12% global
collective scale, ±25% drag, and the full measured climb-bin residual in closed loop, finishing
centered every time). It flies its trained funnel with high confidence and crosses exactly where
trained-geometry says is safe — and trained geometry is wrong by the body radius. The climb-bin
plant residual (S20) supplies the ~0.2–0.5 m displacement; the geometry fiction converts it into
a crash. **Fix the geometry in training (body-radius inflation + frame extrusion), keep the c16
reward byte-identical, keep DR, fold in S20 when fitted — that is inc7.** No aggression penalty
anywhere: robustness comes from honest contact geometry + structured DR, not from damping.

Premise corrections found on the way (both flagged in the session brief as open):
1. **Q4's premise is stale:** inc6's `dr_aero` DID randomize force-model scale (collective table
   deltas ±10%, quad-drag c2 ≈ ±24% relative, residual linear drag [0, 0.08]). Force-scale DR was
   on, and it works — that is *why* the policy absorbs the climb-bin residual in closed loop.
2. **Q8's premise is stale:** inc6 already trains on procedural random courses
   (`course_mode=random`, turns to ±60°/segment = 3× VQ1's max, VQ1 held out) with
   `standing_start_frac=0.3`. META gap ① is closed at the training-distribution level; what
   remains is sampler-range adequacy vs the unknown VQ2 course, not single-track overfit.

---

## 2. The ledger, answered

### Q1 — the crab is a near-optimum, not a pathology (ANSWERED, computation)

Method: twin racestart rollout (FINISHES 9.62 s, reproducing crab-diag), then per cruise tick
compare the observed orientation's along-track quad-drag deceleration against every rotation
about the thrust axis (the trajectory-preserving DOF — same world thrust vector, same path,
different body axes in the airstream).

| regime | sideslip (obs) | along-track drag obs | optimum | excess |
|---|---|---|---|---|
| \|v\| ≥ 15 m/s (n=246) | +36.8° | 12.66 m/s² | 12.23 m/s² | **0.43 m/s² (3%)** |
| 12–15 m/s (n=12) | +44.0° | 8.23 m/s² | 8.21 m/s² | 0.03 m/s² (0%) |

At drag-bound cruise, dT/T ≈ dΔD/(2D) ≈ **1.7% ≈ 0.15 s on the 9 s RL segment**. The drag basin
over the thrust-axis rotation is nearly flat (lateral c2 0.055 vs nose-first 0.042/m, and much of
the airspeed projects on body-z at 55° tilt anyway) — PPO parked anywhere in a flat basin.
Coordinated flight is NOT reward-blocked: it needs only ~0.32 rad/s mean yaw rate (p95 0.74; cap
3.14 wire → ~×2.2 realized) costing rw_rate ≈ 0.016/step against an observed total 0.044/step.
**The operator's "surely banking toward your target is better" is worth ~0.15 s/lap — 15× smaller
than the measured 2.3 s/lap tilt-envelope lever.** Verdict: leave the crab alone; no reward term;
any styling intervention risks the S17 generalization trade for a 0.15 s prize.
(Caveat per escape hatch: the TOGT 4.27 s bound artifacts were not re-solved for an attitude
profile; the bound assumes thrust-binding, which the 3% drag delta cannot meaningfully move.)

Q13 rider (corner tax in the climb): corner-tax exposure along the lap is small (mean factor
0.013) and final-approach commands are mid-band; q5's clean recoveries show corrections are not
being suppressed. The c16 tax prices the mixer rails only, as designed. No change.

### Q2 — crash forensics: confident mid-range flight into a fictional aperture (ANSWERED, live data)

`q2_live_forensics.py` on the 4 valid standing crashes + 2 bridge finishes (control):

| run | final-1.5 s \|rate\| p50/p95 (cap 3.14) | thrust rails | corridor Linf x=−3/−1/0 m | terminal |
|---|---|---|---|---|
| std_f1 | 0.10 / 0.50 | none | 1.65 / 0.72 / **0.49** | crash, 17.3 m/s |
| std_ext_f1 | 0.10 / 0.54 | none | 1.64 / 0.72 / **0.37** | crash |
| std_ext_f3 | 0.10 / 0.52 | none | 1.65 / 0.72 / **0.49** | crash |
| std_ext_f4 | 0.10 / 0.52 | none | 1.65 / 0.72 / **0.44** | crash |
| brg_f1 (control) | 0.14 / 0.70 | none | 1.54 / 0.66 / **0.25** | threads g3, FINISHES |
| brg_f2 (control) | 0.14 / 0.71 | none | 1.54 / 0.66 / **0.25** | threads g3, FINISHES |

- **Neither hypothesis in the question survives.** Not saturated-corrective-losing-a-fight (84%
  of rate authority unused; no thrust rails). Not passive-drift-into-OOD (the trajectory follows
  the trained funnel tick-for-tick; all four crashes are identical to ±0.1 m — deterministic).
- The trained maneuver at gate 3 is a **swoop**: arrive ~2.6 m high at 5 m out, descend at slope
  |dz/dx| ≈ 0.45 into the aperture. Standing arrives ~+0.15–0.25 m higher than bridge at every
  station (the integrated climb-bin residual), and crashes at L-inf 0.37–0.49 — **inside the
  0.75 m pass band the policy was trained to trust.** The bridge, 0.2 m lower on the same funnel,
  threads at 0.25.
- Authority work (plant/mixer) is NOT the fix; training-coverage injection is NOT the fix (q5:
  12/12 recoveries from ±1.5 m displaced restarts 1.5 s before gate 3 — 3× the live error — all
  re-threading to ±0.07 m of nominal crossing). **The fix is the contact geometry the optimum is
  computed against** (Q9/Q10) plus the S20 residual refit (which removes the displacement source).

### Q3 — margin doctrine: tight tails against the wrong aperture (ANSWERED, computation)

48 jittered standing starts × latency {0–3}, mixer plant: **48/48 finish.** Crossing-offset
tails (L-inf p95): gates 0–4 = 0.16–0.26, gate 5 = 0.38. Against the env's 0.75 aperture: p5
margin ≥ 0.37 everywhere — looks safe. Against the MEASURED contact envelope (corner-pass 2026-06-11:
contact logged at 0.60 m; live crashes at 0.37–0.49 on a 0.45-slope descent): gates 3 and 5 have
**near-zero to negative true margin in the tail.** The "flies close to gate edges" observation is
correct in the corridor sense: gate 3/5 approaches converge from OUTSIDE the aperture cone
(twin Linf at 3 m out: 0.98 and 1.20) in the final ~2 m. Corner exploitation for distance is mild
(crossings 0.2–0.38 toward the turn-inside, far from the legal 0.5 line).

**Doctrine: margin is a GEOMETRY property, not a reward property.** Required: per-gate crossing
tails (p95 + certified-residual displacement bound) must clear the *contact-true* aperture
(≈0.75 − r_body ≈ 0.40–0.45 m), and corridor clearance must be positive over the last ~2 m, not
just at the plane. Enforced by training against honest contact geometry — never by a
gate-proximity penalty (which would tax exactly the corner efficiency we want).

### Q4 — DR doctrine: global force DR works; the premise was stale (ANSWERED, computation)

Premise correction: `dr_aero` already randomizes the force model (coll table ±10%, c2 ≈ ±24%
rel, d1 [0,0.08]); `dr_mixer` + latency {1,2,3} on top. Probes, deterministic offsets, simstart:

- Global collective ×0.88 / ×0.94 / ×1.06 / ×1.12 → **all FINISH, gate-3 offsets unchanged**
  (±0.26 ± 0.05). Drag ×0.77 / ×1.25 → same.
- Regime-binned emulation of the measured residual (additive +2.79 N / −2.36 D m/s² in the
  12–18 m/s × tilt-35–90 bin, at 0.5×/1.0×/1.2×; multiplicative in-bin collective deficit
  0.92/0.88; with live latency 2) → **all FINISH, centered** (the policy spends up to 11 s
  in-bin absorbing it).

So the trained policy already carries the closed-loop robustness DR was meant to buy; what it
cannot do is conjure crossing margin that trained geometry says it doesn't need. **DR width
doctrine: every force axis keeps a standing band ≥ the certified residual bound from
`frame_residual_report` regime bins (currently ≤ ~3 m/s²); add a STRUCTURED regime-binned
force-bias DR axis** (random speed×tilt bin, random ≤3 m/s² world bias per env) so the class
"sysid is wrong in one regime" is in-distribution — global scales cannot represent it. Cost:
historical evidence says adding DR families is cheap (inc5→inc6 added mixer DR; median 9.52→9.86 s
includes the corner-tax change too: ≤~0.3 s/lap upper bound); q4's unchanged offsets show no
margin was being bought with lap time.

### Q5 — recovery curriculum: dissolved at the relevant scale (ANSWERED, computation)

Displaced restarts 1.5 s before the gate-3 plane, ±0.75 / ±1.5 m on each axis (N/E/D), 18.6 m/s:
**12/12 re-thread to the nominal crossing ±0.07 m.** Gate-relative obs + random courses + jittered
resets already cover "off-trajectory at gate approach" — these states are not OOD in obs space.
State-perturbation injection is NOT load-bearing for inc7 (optional cheap insurance later; no
speed cost expected since it adds no reward term). The live failure is not a recovery failure:
the policy recovers what it perceives as error; a crossing at 0.45 is not an error under trained
geometry.

### Q6 — cold start: exposure, not coverage (ANSWERED, computation)

Standing is trained (30% of resets) and robust in-twin (48/48 jittered). What differs from
bridge: (a) residual-bin exposure before gate 3 = **2.76 s vs 1.47 s** (1.9×) → roughly double
the integrated displacement; (b) the slower/steeper arc arrives at gate 3 **+0.15–0.25 m higher**
on the same funnel (live corridor table, Q2). Reset-redistribution would not fix this — the same
funnel is flown either way. The fixes are S20 (shrink the displacement) and contact-true geometry
(make the funnel carry real margin). With those, standing becomes the robust mode because nothing
else distinguishes it.

### Q7 — 30 Hz: does not bind here (fold-in declined)

The live failures are deterministic plant-geometry events (4 identical crashes, mid-range
commands); nothing implicates decision quantization (0.57 m/tick at 17 m/s, but the funnel error
is systematic, not quantization noise). §SPEED-CEILING-ANALYTIC verdict stands: binds only at
(≥30 m/s × last-fix ≤10 m), a vision-case corner. 60/100 Hz retrain stays QUEUED behind the
envelope ladder and the SHADOWPC-VISION-CAL gating measurement. Note: training already
interpolates crossings, so gate classification is rate-robust.

### Q8 — single-track: premise stale; remaining exposure is sampler RANGE (ANSWERED + judgment)

inc6 trains `course_mode=random` (turn ±60°/segment, yaw-jitter ±12°, VQ1 interior to every
range and HELD OUT); gen 0.982/0.741 across seeds IS the random-course success. inc7 stays on
the same sampler. The honest residual risks: (a) gen seed-volatility → ≥3-seed averaging is
already doctrine; (b) sampler ranges vs the real VQ2 course — unknowable until organizers answer;
widening costs gen variance now for unmeasurable benefit. Decision: keep ranges; revisit on VQ2
info. (The crab generalizes across course headings — obs are gate-relative; the constant-heading
reading was a VQ1-specific projection.)

### Q9 — NEW: the aperture fiction (ANSWERED, the central finding)

Training pass band: point-mass L-inf < **0.75 m** (`gate_half_opening_m`). Measured contact:
corner-pass probe logged contact at **0.60 m** (near-level crossing); live standing crashes at
**0.37–0.49 m** on a 0.45-slope descent. The sim collides the drone's BODY (rotor halo ~0.3 m)
with a VOLUMETRIC frame; the env collides a point with a plane band. The env over-promises
0.15–0.4 m of aperture depending on approach geometry — and 100% of inc6's live-relevant margin
deficit fits inside that fiction. **Fix (inc7, env-side, no reward change):** inflate the
collision band by a body radius — pass requires L-inf ≤ 0.75 − r_body, collision band
(0.75 − r_body, 1.36 + r_body], with r_body sampled per env ∈ [0.28, 0.38] m (uncertainty about
the true halo becomes margin pressure, unobservable → policy trains to the worst case) — and
extrude the frame band over gate-frame |x| ≤ 0.30 m so steep approaches price frame depth.
Predicted effect: trained crossings tighten from ≤0.38 to ≲0.25 tails; live standing residual
displacement (+0.2–0.5 m) then lands INSIDE the real contact envelope. Predicted cost: small
(crossing 0.1 m closer to center at 6 gates; the funnel barely changes elsewhere).

### Q10 — NEW: corridor blindness (ANSWERED)

Plane-band collision cannot see "hit the frame before the plane." Twin nominal corridor (mixer
plant): gates 3/5 approach from OUTSIDE the aperture cone at 3 m out (Linf 0.98 / 1.20),
converging only in the last ~2 m at slope up to 0.34 (twin) / 0.45 (live). The frame extrusion in
Q9 prices exactly this. Doctrine addition: the checkpoint gauntlet reports corridor clearance
(Linf at x = −1 m) per gate, not just crossing offsets.

### Q11 — NEW: obs noise in training (judgment)

Training obs are pristine truth; live VQ1 case A feeds given pose — fine, and the live-confirm
flights validate the seam. This becomes load-bearing only for VQ2 case C (vision-only), where
estimation noise/latency enter the loop — already queued as the Stage-2 noise model
(ADVISOR-TRIAGE ④, VISION-PKG2 fix-cov as twin input). NOT an inc7 blocker; do not add noise
axes to inc7 (one change family per increment — attribution discipline).

### Q12 — NEW: selection metrics (judgment → doctrine)

sr/median-lap/gen are insufficient: gen is seed-volatile (0.741–0.982), and none of them measure
the transfer-relevant tails. The gauntlet below adds: per-gate crossing-offset p95 vs the
contact-true aperture, corridor clearance at x=−1 m, residual-injection and force-scale probe
outcomes, and twin-attitude extraction (crab-diag lesson: lap time without attitude is
inadmissible as transfer evidence).

### Q14 — NEW: where robustness comes from (judgment → doctrine)

In priority order, with the evidence that earned each slot:
1. **Honest contact geometry** (Q9/Q10) — margins emerge from the optimum itself; zero speed tax
   away from gates; cannot be reward-hacked.
2. **Structured + global force DR** (Q4) — closed-loop disturbance absorption, demonstrated.
3. **Reset/course diversity** (Q5/Q8) — recovery coverage, demonstrated (12/12, 48/48).
4. **Reward shaping — NEVER for robustness.** The c16 corner tax stays exactly because it prices
   a measured actuator nonlinearity, not a fear. No aggression penalty, no proximity penalty, no
   action damping beyond c16. "Slow is smooth" enters only through geometry honesty: the policy
   slows/centers precisely where contact physics says it must, and nowhere else.

---

## 3. Part 3 — the verdict (inc7)

**TRAIN inc7 NOW — do not wait for a bigger env redesign.** The env redesign needed IS small and
this session specced it: two geometry changes + one DR axis, all in `peregrine_racing.py` /
`diffaero_dynamics.py`, defaults OFF, parity-gated. Waiting costs fable days (sunset ~10 days)
on exactly the work fable is for; nothing in the evidence demands a larger redesign (reward is
validated, recovery is validated, course distribution is validated).

**inc7 spec (c16 lineage; sbatch at `rl/peregrine_racing_inc7.sbatch`, DO NOT SUBMIT until the
three gates below pass):**
- Reward: **byte-identical c16** (rw_corner=16, no dact, weights per inc6 winner).
- Env (new, to implement next session — LAPTOP-INC7-ENV, fable, ~half session + tests):
  1. `+env.body_radius_m` sampled per env ∈ [0.28, 0.38]: pass band 0.75−r, collision band
     (0.75−r, 1.36+r].
  2. `+env.frame_depth_m=0.30`: frame-band collision active over gate-frame |x| ≤ depth
     (segment-vs-slab test).
  3. `+dynamics.dr_force_bias=true`: per-env random world-frame bias ‖b‖ ≤ 3 m/s² active in a
     per-env random speed×tilt bin (the certified-residual class).
- Plant: S17 mixer (unchanged). **S20 refit folds in if fitted before launch** — its dataset is
  now LOCAL (`shadowpc-postfix-dataset` 6 valid runs + refit-dataset 17 runs); S20 improves the
  nominal but is NOT a launch blocker (Q4: the policy absorbs the residual; the geometry fix
  supplies the margin).
- DR: all inc6 axes unchanged (aero, mixer, latency {1,2,3}) + dr_force_bias.
- Spawns/courses: `course_mode=random`, `standing_start_frac=0.3` (unchanged — don't confound).
- Selection: ≥3 seeds; gen averaged across seeds; full gauntlet (doctrine §5) before any live.
- Launch gates: ① V100 config-matrix parity (standing flag since S18, now incl. the new env
  features' dynamics side); ② 598+ tests green incl. new geometry unit tests; ③ deploy matrix
  REBUILT on the new eval geometry (contact-true scoring).

**Prediction (on record):** inc7 standing start clears gate 3 with ≥0.3 m corridor margin;
crossing tails ≤0.25; posture unchanged (~55° tilt crab); lap cost vs inc6 ≤0.3 s; bridge parity.

**Ranked queue:**
1. **LAPTOP-INC7-ENV** (fable): implement Q9/Q10 geometry + dr_force_bias + tests; then S20 refit
   on the now-local datasets (same session if it fits); then 3-seed Adroit launch + V100 gate.
2. **VISION-FRAME-FIX** (parallel, unchanged).
3. Envelope ladder step 1 (rw_tilt 96→48) — gated on inc7 standing live confirm; the 2.3 s/lap
   lever, 15× the crab prize.
4. S19 mixer contradiction (unchanged priority); SHADOWPC-VISION-CAL bundle; 60/100 Hz retrain
   (queued, evidence-gated).

**REJECTED (no-re-litigate ledger, with reasons):**
- Anti-crab / sideslip / "bank toward target" reward term — worth 0.15 s/lap (Q1), risks the S17
  generalization trade. The crab is a near-optimal trained style.
- Aggression/action damping beyond c16 — S17 standing result; Q2 shows live failures use 16% of
  authority, damping is aimed at a non-problem.
- Gate-proximity penalty for margin — margin must come from contact geometry, not reward (Q3/Q14);
  a proximity term taxes legal corner efficiency everywhere.
- Recovery curriculum as inc7 blocker — Q5: 12/12 from 3× live error. Optional insurance later.
- Reset-distribution redesign for cold start — Q6: coverage is fine; exposure+geometry are the
  causes.
- 60/100 Hz retrain now — Q7: nothing binds at VQ1 speeds; queue intact.
- Course-sampler widening now — Q8: unmeasurable benefit until VQ2 info; gen variance cost real.
- `--plant lapse` / `dr_lapse` — remains VOIDED (frame-audit).

---

## 4. Files

- `QUESTIONS.md` — Part-1 ledger (Q1–Q14, tagged answerable/probe/judgment).
- `scripts/doctrine_probes.py` — q1 (crab aero), q2q6 (residual injection + exposure), q3
  (offset tails), q4 (force-scale sweep), q5 (displaced restarts), corridor.
- `scripts/q2_live_forensics.py` — live crash forensics on the postfix dataset.
- `scripts/q1_trace.npz` — nominal twin racestart full-state trace.
- `docs/training_doctrine.md` — Part-2 durable doctrine (in-repo).
- `rl/peregrine_racing_inc7.sbatch` — Part-3 spec (DO NOT SUBMIT; gated).

---

MEMORY-DELTA:
- **GATE-APERTURE FICTION = root cause of the standing gate-3 barrier's lethality** (supersedes
  "climb-bin residual → S20" as the WHOLE story; residual supplies ~0.2–0.5 m displacement, the
  geometry converts it to a crash): training env passes point-mass Linf<0.75; live crashes at
  Linf 0.37–0.49 (4/4, mid-range commands, 16% authority used, zero saturation); bridge threads
  same funnel at 0.25. Sim contact = aperture − body halo (~0.3 m), volumetric (strikes before
  plane on slope-0.45 approaches).
- **inc7 = c16 reward UNCHANGED + contact-true geometry (r_body ∈[0.28,0.38] inflation + 0.30 m
  frame extrusion) + dr_force_bias (regime-binned ≤3 m/s²) + mixer plant; S20 refit folds in if
  ready (dataset now LOCAL via shadowpc-postfix-dataset), NOT a launch blocker.** Gates: V100
  parity, tests, rebuilt deploy matrix on contact-true scoring. ≥3 seeds.
- **Crab DISSOLVED as inefficiency:** excess along-track drag 0.43 m/s² (3%) ≈ 0.15 s/lap vs
  drag-optimal thrust-axis rotation; flat basin; coordinated alt needs only 0.32 rad/s yaw. No
  reward change (15× smaller than envelope lever).
- **inc6 robustness MEASURED:** absorbs ±12% global coll / ±25% drag / full climb-bin residual
  (closed loop, centered); 12/12 recovery from ±1.5 m displaced restarts; 48/48 jitter×latency.
  Premise "DR omitted force scale" FALSE (dr_aero: coll ±10%, c2 ±24%).
- **Premise "single-track training" STALE:** inc6 trains course_mode=random (±60°/seg, VQ1
  held-out) + standing_start_frac 0.3.
- `docs/training_doctrine.md` = durable doctrine: margin from geometry, robustness from
  structured DR + diversity, NEVER from reward damping; checkpoint gauntlet incl. corridor
  clearance + offset tails + twin-attitude extraction; inadmissible-evidence list.
