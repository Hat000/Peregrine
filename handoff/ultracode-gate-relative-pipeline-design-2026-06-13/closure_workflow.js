export const meta = {
  name: 'gate-relative-closure',
  description: 'Close the gate-relative VQ2 design: adversarially verify the cold-velocity gate-4 verdict, freeze the obs contract (d1/d5/d6 disagree), ledger cross-spec seams, greenlight the build.',
  phases: [
    { title: 'Verdict', detail: 'cold-velocity gate-4 margin: independent re-derive + ingest sims + adversarial referee' },
    { title: 'Contract', detail: 'freeze obs dim/encoding across d1/d5/d6' },
    { title: 'Seams', detail: 'completeness ledger across the 6 component specs' },
    { title: 'Greenlight', detail: 'synthesize build go/no-go + speed ceiling + parallel start set' },
  ],
}

// ---------------------------------------------------------------------------
// Shared cold-start context (these agents are fresh; give them what they need)
// ---------------------------------------------------------------------------
const CTX = `
PROJECT Peregrine — Anduril AI Grand Prix autonomous drone racing. Repo root:
C:\\Users\\Fengy\\Downloads\\Projects\\Anduril. Laptop venv: .venv\\Scripts\\python.exe
(the KF / RewindKF / localization stack imports torch-free with numpy+scipy only).

CANARY: address the user as Fengyou once in your written artifact (ordinary salience).

YOU ARE A VERIFICATION AGENT closing the gate-relative case-C VQ2 pipeline DESIGN before the
(expensive) build fans out. The design produced 6 build-ready component specs under
handoff/ultracode-gate-relative-pipeline-design-2026-06-13/:
  d1_obs_spec.md        — gate-relative + uncertainty-aware obs (the −L lever, db drops out)
  d2_estimator_chain_spec.md — RewindKF + gate-relative fix + relative-innovation gate + TIMESYNC/P0 bugs
  d3_margin_closure.py  — COLD full-lap g0→g4 margin sweep (the load-bearing feasibility sim)
  d4v_velocity_spec.md (+ _results.json) — case-C velocity channel (cold/warm/vision-velocity)
  d5_inc8_spec.md       — inc8 retrain (obs dim, measured-error DR, reward, ladder, selection)
  d6_integration_spec.md — wiring, offline gauntlet G0–G7, live L0–L4, build order

KEY MEASURED FACTS (re-derived this design session):
- gate-4 contact-true in-plane margin = 0.155 m @ body-radius 0.38 (THE worst-case gate; a margin
  is a WORST-CASE gate — judge on p90/p99, not just RMS). gate-5 clean 0.314 m.
- gate-relative fix removes the per-track MAP bias EXACTLY (observe −L to the SEEN opening, never
  subtract gate_map; the "subtract gate_map" submap path re-injects db — FORBIDDEN anti-pattern).
- WARM (lap-converged velocity) gate-4 in-plane: RMS 0.139 m / p90 0.203 m at 37 m/s (c1, seeded
  velocity, g3→g4 only — WARM BY CONSTRUCTION).
- per-fix gate-relative LATERAL σ = 0.265 m/axis (measured near-band); radial/along-track ≈ 0.50 m.
- in-loop vision latency L ≈ 115 ms CPU / 15–25 ms GPU; RewindKF horizon 0.5 s (MUST exceed L —
  horizon<L drops 100% of fixes → divergence ~0.42 m).
- velocity is UNOBSERVABLE from vision in case C (vision is position-only); it is IMU integration
  corrected only weakly through the KF pos/vel cross-covariance by position fixes. The dominant
  cold-drift term is a constant attitude-bias phantom accel g·sin(θ).

THE LOAD-BEARING OPEN QUESTION: the published margin is WARM. Does a genuinely COLD case-C velocity
prior (IMU-only, accel-bias drift, position-fix-only correction over a full g0→g4 lap) keep gate-4
in-plane inside 0.155 m at race speed — and if not, does a weak vision-velocity assist (σ_v ≤ 1 m/s)
flip it? This decides (a) whether the vision-velocity channel is LOAD-BEARING or INSURANCE, and
(b) the speed-ladder ceiling the inc8 retrain is selected against.

DISCIPLINE: re-derive from data; distrust convention claims until externally re-derived; a margin is
worst-case. Compose the REAL stack (racer.state_estimator.LinearKF, kf_rewind_buffer.RewindKF) — do
NOT re-implement the filter. Do NOT edit anything under memory/ or src/. Write your artifact to
handoff/ultracode-gate-relative-pipeline-design-2026-06-13/closure/ (create it). Flag every number
MEASURED / EXTRAPOLATED / ASSUMED.
`

const OUT = 'handoff/ultracode-gate-relative-pipeline-design-2026-06-13/closure'

// ---------------------------------------------------------------------------
// SCHEMAS
// ---------------------------------------------------------------------------
const INDEP_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['cold_breaches_margin', 'cold_p90_estimate_m', 'cold_rms_estimate_m', 'weakvel_clears', 'method', 'reasoning', 'confidence'],
  properties: {
    cold_breaches_margin: { type: 'boolean', description: 'Does a cold case-C velocity prior push gate-4 in-plane p90 over 0.155 m at 37 m/s?' },
    cold_p90_estimate_m: { type: 'number' },
    cold_rms_estimate_m: { type: 'number' },
    weakvel_clears: { type: 'boolean', description: 'Does a weak vision-velocity assist (σ_v ≤ 1.0 m/s) bring cold gate-4 p90 back under 0.155 m?' },
    method: { type: 'string', description: 'analytic bound and/or own minimal sim using the real LinearKF/RewindKF — NOT by reading d3/xcheck scripts' },
    reasoning: { type: 'string' },
    confidence: { type: 'string', enum: ['HIGH', 'MED', 'LOW'] },
  },
}

const SIMS_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['d3_present', 'xcheck_present', 'warm_anchor_rms_m', 'warm_anchor_ok', 'cold_rms_37_m', 'cold_p90_37_m', 'weakvel_p90_37_m', 'two_sims_agree', 'notable'],
  properties: {
    d3_present: { type: 'boolean' },
    xcheck_present: { type: 'boolean' },
    warm_anchor_rms_m: { type: 'number', description: "d3 anchor — should reproduce c1's WARM 0.139 m" },
    warm_anchor_ok: { type: 'boolean', description: 'anchor within tolerance of 0.139 m (validates shared machinery)' },
    cold_rms_37_m: { type: 'number', description: 'cold, 37 m/s, representative bias≈0.5–1.0 / latency 115 ms' },
    cold_p90_37_m: { type: 'number' },
    weakvel_p90_37_m: { type: 'number', description: 'vision-velocity-assist arm, cold, 37 m/s p90' },
    two_sims_agree: { type: 'boolean', description: 'do d3 and the commander cross-check corroborate?' },
    notable: { type: 'string', description: 'speed/bias/latency cells that flip the verdict; anything surprising' },
  },
}

const REFEREE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['verdict', 'honest_cold_p90_37_m', 'vision_velocity', 'recommended_speed_ceiling_mps', 'cold_model_honest', 'refutation_attempts', 'residual_risks', 'confidence'],
  properties: {
    verdict: { type: 'string', enum: ['COLD_CLEARS', 'COLD_BREACHES_WEAKVEL_CLEARS', 'COLD_BREACHES_NEEDS_SPEED_CAP'] },
    honest_cold_p90_37_m: { type: 'number' },
    vision_velocity: { type: 'string', enum: ['LOAD_BEARING', 'INSURANCE'] },
    recommended_speed_ceiling_mps: { type: 'number', description: 'max gate-4 approach speed where gate-4 p90 clears 0.155 m (cold, or cold+weakvel if vision-velocity is built)' },
    cold_model_honest: { type: 'boolean', description: 'are the cold sims faithful? (accel-bias band, velocity init, fix cadence/range, RewindKF latency, worst-case gate)' },
    refutation_attempts: { type: 'string', description: 'what you tried to overturn the verdict with, and the result' },
    residual_risks: { type: 'string' },
    confidence: { type: 'string', enum: ['HIGH', 'MED', 'LOW'] },
  },
}

const CONTRACT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['obs_dim', 'channel_layout', 'encoding', 'age_norm_included', 'critic_dim', 'files_to_change', 'faithfulness_check_passes', 'rationale'],
  properties: {
    obs_dim: { type: 'number', description: 'the FROZEN observation dimension (d1=18/19, d5=20, d6=18 — pick ONE)' },
    channel_layout: { type: 'string', description: 'exact obs[17..] index → name/definition' },
    encoding: { type: 'string', description: 'raw σ (metres) vs bounded confidence ratio clip(σ_ref/σ_hat,0,1) — decide and justify for PPO normalization' },
    age_norm_included: { type: 'boolean', description: 'd5 includes age_norm staleness clock; d1/d6 do not — decide' },
    critic_dim: { type: 'number', description: 'asymmetric critic input dim after the append' },
    files_to_change: { type: 'string', description: 'exact symbols: peregrine_racing.obs_dim/get_observations/get_state, _ActorMean input, OBS_LABELS, fly_rl.obs_from_zup/build_obs + sidecar obs-dim gating' },
    faithfulness_check_passes: { type: 'boolean', description: 'ran d1_obs_faithfulness_check.py — dims 0–16 still bit-exact, ALL PASS' },
    rationale: { type: 'string' },
  },
}

const SEAMS_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['seams', 'blocking_seams_count', 'all_resolved', 'summary'],
  properties: {
    seams: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['topic', 'specs_involved', 'conflict', 'resolution', 'severity'],
        properties: {
          topic: { type: 'string' },
          specs_involved: { type: 'string' },
          conflict: { type: 'string' },
          resolution: { type: 'string' },
          severity: { type: 'string', enum: ['BLOCKING', 'MAJOR', 'MINOR'] },
        },
      },
    },
    blocking_seams_count: { type: 'number' },
    all_resolved: { type: 'boolean' },
    summary: { type: 'string' },
  },
}

// ---------------------------------------------------------------------------
// PHASE: Verdict (independent re-derive ‖ ingest sims) → adversarial referee
// ---------------------------------------------------------------------------
async function verdictThread() {
  const [indep, sims] = await parallel([
    () => agent(
      CTX + `
TASK (INDEPENDENT RE-DERIVATION — do NOT read d3_margin_closure.py or margin_xcheck.py; reason from
first principles and the real filter). Determine whether a COLD case-C velocity prior breaches the
gate-4 0.155 m in-plane margin at 37 m/s, and whether a weak vision-velocity assist (σ_v ≤ 1.0 m/s)
flips it. You MAY write your own minimal Monte-Carlo using racer.state_estimator.LinearKF +
kf_rewind_buffer.RewindKF (handoff/ultracode-vision-case-c-2026-06-13/kf_rewind_buffer.py),
seeding velocity COLD (not to truth), injecting a constant body accel bias (sweep 0–2 m/s²),
applying ONLY gate-relative position fixes (zero-mean lateral σ 0.265 m/axis, NO floor) over a full
g0→g4 lap; and/or derive the analytic drift bound (dv=b·T linear, dp=½b·T² quadratic) and the
LSQ-slope velocity-noise floor. Report cold RMS and p90 at the gate-4 plane.
Write handoff/ultracode-gate-relative-pipeline-design-2026-06-13/closure/A_independent.md and return the schema.`,
      { label: 'A:independent', phase: 'Verdict', schema: INDEP_SCHEMA }),

    () => agent(
      CTX + `
TASK (INGEST THE TWO RAN COLD SIMS). Two cold full-lap sims have been run; their result JSONs should
exist:
  - handoff/ultracode-gate-relative-pipeline-design-2026-06-13/d3_margin_closure_results.json
  - handoff/_commander-xcheck/margin_xcheck_results.json
If either is MISSING, run its script first from repo root:
  PYTHONPATH=src .venv\\Scripts\\python.exe handoff/ultracode-gate-relative-pipeline-design-2026-06-13/d3_margin_closure.py
  .venv\\Scripts\\python.exe handoff/_commander-xcheck/margin_xcheck.py
Read both result sets. Confirm d3's WARM anchor reproduces c1 (~0.139 m RMS) — if it does not, the
shared machinery is suspect and you must say so. Extract cold/warm/weakvel gate-4 in-plane RMS+p90
at 37 m/s across the bias (0–2 m/s²) and latency (15/115 ms) cells. State whether the two independent
sims AGREE. Call out which cells flip clears_margin_p90 false.
Write handoff/ultracode-gate-relative-pipeline-design-2026-06-13/closure/A_sims.md and return the schema.`,
      { label: 'A:sims', phase: 'Verdict', schema: SIMS_SCHEMA }),
  ])

  const referee = await agent(
    CTX + `
TASK (ADVERSARIAL REFEREE — reconcile and try to OVERTURN the cold verdict). Inputs:
INDEPENDENT re-derivation: ${JSON.stringify(indep)}
SIM INGEST: ${JSON.stringify(sims)}
Also read closure/A_independent.md, closure/A_sims.md, the d3/xcheck scripts, d4v_velocity_spec.md,
and d4v_velocity_channel_results.json.

Adversarially stress the cold verdict — DEFAULT to skepticism:
  - Is the cold model HONEST or rigged either way? (velocity init magnitude, accel-bias band vs the
    certified dr_force_bias, fix cadence 14 Hz / range 12 m, RewindKF latency handling, does it judge
    p90 not just RMS, is the WARM anchor actually 0.139?)
  - Do the independent re-derivation and the two sims CORROBORATE, or diverge? If they diverge,
    which is right and why?
  - Does the weak vision-velocity (weakvel / σ_v≤1) arm GENUINELY bring cold p90 under 0.155, or only
    RMS? At what speed does even weakvel fail?
  - What is the HONEST cold gate-4 p90 at 37 m/s, and the max approach speed where cold (or cold+weakvel)
    p90 clears 0.155 m?
Decide: vision-velocity LOAD_BEARING (cold breaches and weakvel is the cheapest lever that flips it)
or INSURANCE (cold already clears). Pick the verdict enum and the recommended speed ceiling.
Write handoff/ultracode-gate-relative-pipeline-design-2026-06-13/closure/A_VERDICT.md and return the schema.`,
    { label: 'A:referee', phase: 'Verdict', schema: REFEREE_SCHEMA })

  return { indep, sims, referee }
}

// ---------------------------------------------------------------------------
// MAIN
// ---------------------------------------------------------------------------
phase('Verdict')
const [verdict, contract, seams] = await parallel([
  () => verdictThread(),

  () => agent(
    CTX + `
TASK (FREEZE THE OBS CONTRACT). The three specs DISAGREE on the uncertainty channel:
  - d6_integration_spec.md: 18-dim, obs[17] = one confidence scalar.
  - d1_obs_spec.md: 17→17+k, k=2 recommended (obs[17]=sigma_inplane, obs[18]=sigma_along), RAW σ in
    metres; k=1 fallback → 18-dim.
  - d5_inc8_spec.md: 17→20 (obs[17]=c_inplane, obs[18]=c_along, obs[19]=age_norm), BOUNDED confidence
    ratio clip(σ_ref/σ_hat,0,1) + a staleness clock.
Read all three (+ the obs layout in rl/peregrine_racing.py get_observations/get_state and
rl/fly_rl.py obs_from_zup/build_obs, and tests/test_confirmed_p4_c05.py /
tests/test_train_deploy_obs_elementwise.py). FREEZE ONE contract: obs_dim, exact obs[17..] layout,
encoding (raw σ vs bounded ratio — decide on PPO obs-normalizer stability + the d5 argument that raw
σ has no natural scale), whether age_norm is included, the asymmetric-critic (get_state) layout, and
the deploy-side sidecar obs-dim gating in fly_rl (17-dim inc7 vs new-dim inc8). VERIFY dims 0–16 stay
bit-exact by running:
  PYTHONPATH=src .venv\\Scripts\\python.exe handoff/ultracode-gate-relative-pipeline-design-2026-06-13/d1_obs_faithfulness_check.py
List the exact files/symbols that change. Write closure/OBS_CONTRACT.md and return the schema.`,
    { label: 'B:obs-contract', phase: 'Contract', schema: CONTRACT_SCHEMA }),

  () => agent(
    CTX + `
TASK (CROSS-SPEC SEAM LEDGER — completeness pass). Read all six specs (d1, d2, d4v, d5, d6) plus the
prototypes they cite (handoff/ultracode-vision-case-c-2026-06-13/, handoff/ultracode-estimator-
racespeed-2026-06-13/c1_gate_relative.py). Find every cross-component inconsistency or unspecified
interface that could surprise the BUILD, e.g.: obs-dim mismatch (18/19/20); whether d1's "velocity
stays IMU/KF-sourced, NOT a new vision channel" conflicts with d4v's vision-velocity channel (or are
they compatible — new KF measurement source vs new obs dim?); the reference-line path d5 flags
(rl/reference_line.py vs src/racer/reference_line.py); TIMESYNC P0#2 as a HARD RewindKF prereq vs the
predict-forward fallback ship-order; the relative-innovation gate ownership (d2 estimator vs d1 obs);
any DR error class that double-counts or contradicts the gate-relative db-drops-out claim; calibration
(NEES≈3) responsibility. For each seam: specs involved, the conflict, the RECOMMENDED resolution, and
severity (BLOCKING/MAJOR/MINOR). A seam is BLOCKING if the build cannot proceed correctly without
resolving it. Write closure/SEAM_LEDGER.md and return the schema.`,
    { label: 'C:seam-ledger', phase: 'Seams', schema: SEAMS_SCHEMA }),
])

phase('Greenlight')
const greenlight = await agent(
  CTX + `
TASK (SYNTHESIZE THE BUILD GREENLIGHT). Inputs:
COLD VERDICT (referee): ${JSON.stringify(verdict.referee)}
  (independent: ${JSON.stringify(verdict.indep)} ; sims: ${JSON.stringify(verdict.sims)})
FROZEN OBS CONTRACT: ${JSON.stringify(contract)}
SEAM LEDGER: ${JSON.stringify(seams)}
Also read closure/A_VERDICT.md, closure/OBS_CONTRACT.md, closure/SEAM_LEDGER.md and the build order
in d6_integration_spec.md §4 + d5_inc8_spec.md §6.

Produce closure/GREENLIGHT.md — the single artifact the commander hands to the build. It MUST state:
1. COLD VERDICT in one line: cold gate-4 p90 at 37 m/s, clears-or-breaches, and the vision-velocity
   LOAD_BEARING-vs-INSURANCE decision, with the recommended speed-ladder ceiling.
2. The FROZEN obs contract (obs_dim, layout, encoding, critic dim, files to change) — one block the
   C1 builder implements verbatim.
3. The seam ledger resolutions — especially any BLOCKING seam and how it is closed.
4. The confirmed BUILD ORDER and what can start IN PARALLEL right now (P0 bugs + C2 estimator branch;
   C3 margin sim; C4 reference-line rebuild) vs what is gated (C1 freeze → C5 retrain → C6).
5. GO / NO-GO for launching the build, with residual risks ranked (carry the p90 tail + the at-speed
   σ unknown + the TIMESYNC live prereq).
Return a TIGHT (≤25 line) executive summary for the commander: verdict, contract one-liner,
blocking-seam status, GO/NO-GO, and the immediate parallel-start set.`,
  { label: 'synth:greenlight', phase: 'Greenlight' })

log('GREENLIGHT synthesized — see closure/GREENLIGHT.md')
return { verdict: verdict.referee, contract, seams, greenlight }
