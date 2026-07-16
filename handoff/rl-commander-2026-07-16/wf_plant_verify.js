export const meta = {
  name: 'plant-fidelity-verify',
  description: 'Adversarially verify the faithful_rate plant validation against real VQ2 flight logs; build the accel/aero channel; decide v1 scope',
  phases: [
    { title: 'Verify' },
    { title: 'Synthesize' },
  ],
}

// ------------------------------------------------------------------------------------------------
const SCR = 'C:/Users/Fengy/AppData/Local/Temp/claude/C--Users-Fengy-Downloads-Projects-Anduril--claude-worktrees-cool-heyrovsky-e08624/c864e5bb-f53f-4b54-b78a-a2216bb2c906/scratchpad'
const REPO = 'C:/Users/Fengy/Downloads/Projects/Anduril/.claude/worktrees/cool-heyrovsky-e08624'
const PY = 'C:/Users/Fengy/Downloads/Projects/Anduril/.venv/Scripts/python.exe'

const CONTEXT = `
CONTEXT — Peregrine VQ2 drone-racing RL. We are validating a plant fix (\`faithful_rate\`) before committing a
GPU-week retrain. The DEPLOY failure being fixed: policy over-rotates at aggressive banks (the "g2->g3 wall")
+ yaw oscillation. Root cause hypothesis: the TRAINING plant (DiffAero) used a FLAT cmd->achieved-rate gain
(~2.5x constant), but the REAL VQ2 plant is EXPANSIVE (gain rises with |command|). \`faithful_rate\` installs a
measured amplitude-dependent (super-rate) model: rate_gain=[2.359,2.363,2.163] (roll/pitch/yaw small-signal),
super_rate_s=[0.296,0.284,0.316], alpha_max=[260,260,80]. Model: gain = rate_gain/(1 - s*min(|cmd|,pi)/pi);
achieved_target = gain*rate_sign*cmd; rate_sign=[+1,+1,-1] (yaw -1 is a KNOWN legacy convention alias, do NOT
"fix" it). First-order lag rate_tau_s=0.019s, per-axis slew clip alpha_max.

CRITERION (set by the human lead): "if we get the same accelerations/gyro at every tick, the plant is accurate."
No ground-truth position/velocity exists (pose-blind wire: no ODOMETRY/mag/baro; only HIGHRES_IMU gyro+accel
are real sensor ground truth). So the whole validation is PER-TICK IMU match (gyro + specific-force accel).

DATA (already downloaded locally — do NOT need to re-fetch, but you MAY via 'gh api ... --Accept raw' from
repo Hat000/Peregrine branch sysid-handoff-2026-07-15 if you want more):
  * ${SCR}/vq2sysid/*.csv  — 20 CONTROLLED sysID captures. Each is SELF-ALIGNED, columns:
      k,phase,sim_time_ns, a_thrust,a_roll,a_pitch,a_yaw (raw command [-1,1]),
      cmd_wx,cmd_wy,cmd_wz,cmd_thrust,collective, gyro_x,gyro_y,gyro_z, accel_x,accel_y,accel_z (VQ2 IMU).
    'boot' phase rows have empty a_* (on the 17deg tilted pad); 'prog' phase is the maneuver.
    Key ones: yaw_ampsweep (STEPPED holds, clean), roll_sweep/pitch_sweep (continuous), roll_dyn/pitch_dyn
    (step responses -> for tau), yaw_bw (bandwidth), roll_fill/pitch_fill (amplitude fill),
    thrust_curve/thrust_ff/thrust_hover3g (collective sweeps), vzsweep/hover_vzsweep/vzsweep_lo (velocity
    sweeps for inflow-lapse), pitch_level/pitch_trulevel/pitch_30 (tilt-corrected).
  * ${SCR}/panels/{osc,deep}_*.jsonl — 6 REAL racing flights (deploy). Each tick (ego_obs.jsonl): obs (21-dim:
      [0:3] body vel, [3:5] roll/pitch, [5:8] BODY RATES=gyro, [8] prev thrust, ...), rate_frd (commanded body
      rate FRD), collective, kf_pos_ned. osc_* = the vpef8nc OSCILLATION; deep_* = vpeffs0 reaching gate 5 at
      13 m/s (the speed ACCUMULATION).

HARNESS (already written + working):
  * ${SCR}/plant_val.py  — replays each sysID capture's command through the numpy plant (flat vs faithful),
      overlays gyro + specific-force accel per tick. Frames: world NED (Z down, g=+9.80665 on +Z), body FRD;
      specific force f_body = R_world_body^T @ (a_world_kinematic - [0,0,g]). Run: cd ${SCR} && "${PY}" plant_val.py
  * ${SCR}/panel_val.py  — racing-panel rate replay. Run: cd ${SCR} && "${PY}" panel_val.py
  * Plant source: ${REPO}/src/racer/rl_plant.py (PlantParams, step, faithful constants
      RATE_GAIN_SMALLSIGNAL_MEASURED / SUPER_RATE_S_FAITHFUL / ALPHA_MAX_RPS2_MEASURED).
      Action adapter + FLU<->FRD flip [1,-1,-1]: ${REPO}/rl/diffaero_dynamics.py (_action_diffaero_to_ctbr_np, _FLIP).
  Use PYTHON: "${PY}"  (has numpy + the repo on path via sys.path.insert(src, rl)).

MY CURRENT CLAIMS (your job: independently CONFIRM or REFUTE with evidence — assume they may be WRONG):
  C1 Real VQ2 rate gain is EXPANSIVE: yaw ampsweep realized gain rises 2.13->2.33->2.52 across |cmd|; flat
     plant is dead-constant 2.20.
  C2 faithful_rate cuts per-tick yaw-rate RMSE 1.22->0.87 (-29%) on the yaw ampsweep and matches the rise.
  C3 On roll/pitch faithful is ~neutral in RMSE and slightly TOO STEEP at high |cmd| (bounded by ~5-8% creep);
     it errs HIGH (sim gain >= real).
  C4 Direction is SAFE: faithful sim gain >= real -> policy UNDER-rotates in deploy (recoverable); flat sim
     gain < real (esp. yaw) -> OVER-rotates -> the wall. So faithful strictly moves to the safe side.
  C5 The RACING PANELS are CONFOUNDED for per-tick plant-gain (closed-loop lag/phase/timing: |achieved|/|cmd|
     is dominated by phase not gain; e.g. VQ2 roll "gain" 2.94->1.59 DECREASING is a lag artifact). => the
     plant verdict must rest on the CONTROLLED sweeps, not the panels.
  C6 Frame handling in the harness is correct (world-NED/body-FRD, specific-force conversion, yaw -1 alias via
     per-axis sign detection).
  C7 (aero) The thrust/aero gap is real+known (legacy under-predicts full-stick thrust ~2.1x) but the corrected
     coll_map is HELD (tilt-corrected static not landed) -> v1 stays RATE-ONLY (+ a recovery-reward arm as
     stopgap for accumulation); a clean-thrust re-fly is the priority follow-up.
`.trim()

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['lens', 'verdict', 'key_findings', 'numbers', 'challenges', 'recommendation'],
  properties: {
    lens: { type: 'string' },
    verdict: { type: 'string', enum: ['CONFIRMED', 'REFUTED', 'PARTIAL', 'NEW_FINDING'] },
    claims_examined: { type: 'array', items: { type: 'string' } },
    key_findings: { type: 'array', items: { type: 'string' } },
    numbers: { type: 'array', items: { type: 'string' }, description: 'concrete quantitative results with units' },
    challenges: { type: 'array', items: { type: 'string' }, description: 'ways the claim could be wrong / confounds / risks found' },
    recommendation: { type: 'string' },
  },
}

const lenses = [
  { key: 'reproduce+frames', effort: 'high', prompt: `${CONTEXT}

YOUR LENS: REPRODUCE + FRAME/SIGN CORRECTNESS. Independently verify C1, C2, C6.
1. Run plant_val.py yourself; confirm (or refute) the yaw-ampsweep numbers (expansive VQ2 2.13->2.52, flat
   2.20, faithful rise, yaw RMSE 1.22->0.87).
2. AUDIT the frame/sign math by reading rl_plant.py + diffaero_dynamics.py + plant_val.py: is the specific-force
   conversion f_body = R^T(a_world - g) correct given the documented conventions? Is the yaw -1 alias handled
   right (per-axis sign detection)? Is the action adapter fed correctly? Independently re-derive the world/body
   frame + gravity sign and CHECK. A wrong convention would make faithful look good/bad spuriously.
3. Sanity: does the per-tick binned gain under-read steady gain equally for VQ2/flat/faithful (all lag-matched)?
Return the VERDICT_SCHEMA. Be adversarial: default to REFUTED if you cannot independently reproduce.` },

  { key: 'direction-safety', effort: 'high', prompt: `${CONTEXT}

YOUR LENS: DIRECTION-SAFETY / ERRATIC-RISK (adversarial). The lead's fear is a policy that "behaves erratically."
Try HARD to REFUTE C4 (faithful is safe) and scrutinize C3 (roll/pitch too steep).
1. Construct the argument: could faithful_rate make deploy WORSE than flat on ANY axis? Where faithful over-reads
   the gain (esp. roll/pitch high |cmd|), the policy under-banks in deploy -> is that actually safe, or could
   under-banking + then over-correcting create a NEW oscillation / miss pattern?
2. Quantify per-axis: at what |cmd| does faithful cross from below to above VQ2? By how much % at the extremes?
   Use plant_val.py + the roll_fill/pitch_fill/dyn captures for more amplitude coverage.
3. Is the "safe side" argument (under-rotate = recoverable) actually valid for a course that REQUIRES aggressive
   banks? Under-banking could miss gates (a different failure, not erratic, but still failure).
Return VERDICT_SCHEMA. Your job is to find the strongest case AGAINST safety.` },

  { key: 'racing-confound', effort: 'high', prompt: `${CONTEXT}

YOUR LENS: RACING-PANEL VALIDITY. Verify or refute C5 (panels confounded -> unusable for plant gain).
1. Run panel_val.py; confirm the incoherent gains (e.g. decreasing-with-|cmd|).
2. Try to SALVAGE a clean racing plant-read: implement a 1-STEP-AHEAD prediction test — at each tick re-init the
   plant omega to the REAL obs[5:8]_t, step ONE tick with rate_frd_t, compare plant omega_{t+1} to obs[5:8]_{t+1}.
   Does that remove the confound and give a clean flat-vs-faithful comparison IN THE OSCILLATION? Watch for
   command<->gyro timing alignment ambiguity (is rate_frd[t] applied before or after obs[t]'s gyro?). Test both
   alignments.
3. If a clean racing read IS possible, report whether faithful beats flat in the actual oscillation/accumulation.
   If not, confirm the panels are unusable and say why precisely.
Return VERDICT_SCHEMA.` },

  { key: 'accel-aero', effort: 'high', agentType: 'general-purpose', prompt: `${CONTEXT}

YOUR LENS: THE ACCEL/AERO CHANNEL (build it — this is the criterion the lead cares about most, and I have NOT
done it cleanly). Quantify the thrust/aero gap and decide if v1-rate-only will still ACCUMULATE SPEED erratically.
1. Build an accel analysis on the THRUST captures (thrust_curve, thrust_ff, thrust_hover3g, vzsweep,
   hover_vzsweep). Key physics: airborne specific force along body-z ≈ -thrust_accel (gravity-free, thrust is
   body-fixed -> attitude-independent). So VQ2 |accel_z| during airborne thrust HOLDS = the thrust accel at that
   collective. Compare to the DiffAero legacy coll_map (what v1 trains on: a_up = g*collective/hover_thrust, or
   the interp map in rl_plant.py) ACROSS the collective range. Restrict to steady airborne holds (use 'phase' +
   exclude big transients / the +accel_z tumble spikes).
2. Quantify: by how much (%) does legacy under/over-predict thrust vs VQ2 at low/mid/high collective? Is it the
   claimed ~2.1x at full stick?
3. The vzsweep captures: does thrust LAPSE with climb velocity (inflow)? By how much? Legacy floor is 0.78.
4. DECISION INPUT: given the deploy accumulation (drone builds 3->13 m/s over gates), does the thrust gap
   plausibly DRIVE the accumulation (deploy over-delivers thrust vs training)? Or is it small enough that v1
   rate-only is fine? Be quantitative.
Write your analysis script under ${SCR}/agent_accel.py. Return VERDICT_SCHEMA with the thrust-gap numbers.` },

  { key: 'rollpitch-tau', effort: 'high', agentType: 'general-purpose', prompt: `${CONTEXT}

YOUR LENS: ROLL/PITCH GAIN + RATE-LOOP TAU calibration. Determine if faithful's roll/pitch super_rate is TOO
STEEP and whether tau=0.019s is right.
1. Use the STEPPED / dynamic captures (roll_dyn, pitch_dyn, roll_fill, pitch_fill, yaw_bw) — NOT just the
   continuous sweeps. Extract the STEADY-STATE gain per amplitude (after the rate settles within each hold),
   and the RISE TIME (-> tau). Compare to faithful's model gain and tau=19ms.
2. Is faithful roll/pitch (super_rate_s 0.296/0.284) too steep vs the maneuver-timescale steady gain? Note the
   documented CREEP (+4-9% within a hold) — separate settled-gain from creep. Recommend a corrected roll/pitch
   super_rate_s (or "leave as-is, safe-high") with numbers.
3. Does DiffAero's tau=19ms match VQ2's inner-loop rise time? If DiffAero's tau is too fast, faithful leads VQ2
   in transients (would inflate the apparent gain). Measure VQ2 rise time from the step captures.
Write your script under ${SCR}/agent_rollpitch.py. Return VERDICT_SCHEMA with recommended constants (if any).` },

  { key: 'completeness', effort: 'high', prompt: `${CONTEXT}

YOUR LENS: COMPLETENESS CRITIC. What does this validation MISS that could still make v1 behave erratically?
Think hard and list gaps, each with why-it-matters + a cheap check if one exists. Candidates to evaluate (don't
limit to these): (a) 36Hz capture/deploy rate vs 40Hz training — does dt mismatch change the discrete rate
integration / does faithful's alpha_max slew interact with dt? (b) drag / velocity-dependent terms untested in
the rate replays (rate replays start from hover@0vel). (c) command<->gyro TIMING alignment in every comparison.
(d) the panel 'collective' scale vs the plant's expected collective (could bias the panel replay). (e) DR bands:
does the real plant run-to-run spread fall inside training DR? (f) the yaw -1 alias: is it possible the SIGN
issue actually flips a real behavior in deploy, not just the comparison? (g) obs[5:8] provenance — is it raw
gyro or leveler-processed? (h) whether the sysID captures' IC (17deg pad) confounds anything. (i) does faithful
change the plant's stability/limit-cycle behavior in a way that could create NEW oscillation? Return VERDICT_SCHEMA
with lens='completeness', verdict='NEW_FINDING', and the ranked list of gaps in key_findings/challenges.` },
]

phase('Verify')
const verdicts = (await parallel(lenses.map(L => () =>
  agent(L.prompt, { label: `verify:${L.key}`, phase: 'Verify', schema: VERDICT_SCHEMA,
                    effort: L.effort, ...(L.agentType ? { agentType: L.agentType } : {}) })
    .then(v => v ? { ...v, _key: L.key } : null)
))).filter(Boolean)

phase('Synthesize')
const SYNTH_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['faithful_rate_verdict', 'v1_scope', 'recalibration', 'erratic_risk', 'followups', 'confidence', 'summary'],
  properties: {
    faithful_rate_verdict: { type: 'string', enum: ['VALIDATED', 'VALIDATED_WITH_RECAL', 'NEEDS_MORE_DATA', 'REJECT'] },
    v1_scope: { type: 'string', description: 'rate-only vs rate+aero, and why' },
    recalibration: { type: 'string', description: 'concrete constant changes recommended, or "none"' },
    erratic_risk: { type: 'string', description: 'will v1 behave erratically? residual risks ranked' },
    followups: { type: 'array', items: { type: 'string' } },
    confidence: { type: 'string', enum: ['HIGH', 'MEDIUM', 'LOW'] },
    summary: { type: 'string' },
    disagreements: { type: 'array', items: { type: 'string' }, description: 'where the lenses disagreed + how you resolved it' },
  },
}
const synth = await agent(`${CONTEXT}

You are the SYNTHESIS judge. Below are ${verdicts.length} independent adversarial verification verdicts on the
faithful_rate plant validation. Weigh them (a REFUTED frame-check or a real erratic-risk outweighs confirmations).
Produce the final decision: is faithful_rate VALIDATED for the v1 retrain? Should v1 be rate-only or must aero
come in now? Any recalibration (roll/pitch super_rate, tau)? Will v1 behave erratically — residual risks? What
follow-ups (e.g. clean-thrust re-fly)? Resolve any disagreements between lenses explicitly.

VERDICTS:
${verdicts.map(v => `--- lens: ${v._key} [${v.verdict}] ---
findings: ${(v.key_findings||[]).join(' | ')}
numbers: ${(v.numbers||[]).join(' | ')}
challenges/risks: ${(v.challenges||[]).join(' | ')}
recommendation: ${v.recommendation}`).join('\n\n')}

Return SYNTH_SCHEMA.`, { label: 'synthesize', phase: 'Synthesize', schema: SYNTH_SCHEMA, effort: 'high' })

return { verdicts, synth }
