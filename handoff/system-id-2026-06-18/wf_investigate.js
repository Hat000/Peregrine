export const meta = {
  name: 'system-id-investigate',
  description: 'System-ID: register every diffaero-free estimator/obs/geometry pipeline pair vs its independent counterpart; measure divergence vs condition; draft pinning tests',
  phases: [
    { title: 'Investigate', detail: 'one agent per registration pair: drive both sides, measure divergence, draft pinning test' },
  ],
}

const VENV = 'C:/Users/Fengy/Downloads/Projects/Anduril/.venv/Scripts/python.exe'

const PREAMBLE = [
  'You are a worker on the Peregrine system-ID effort (laptop arm). Repo root is the CURRENT working directory (a git worktree on branch p2-system-id). The Python venv with numpy 2.4.6 + torch 2.12.0+cpu + scipy is at: ' + VENV,
  'RUN ALL python/pytest via that absolute interpreter, e.g.:  ' + VENV + ' -m pytest tests/test_foo.py -q',
  'diffaero is ABSENT on this laptop (torch IS present). If ANY counterpart needs diffaero or a GPU to import/run, mark that arm "deferred: Adroit arm" and register everything else that is laptop-runnable.',
  '',
  'GOAL OF THE WHOLE EFFORT: register each pipeline against its INDEPENDENT counterpart, and for every divergence either (a) prove one side wrong vs a known-correct reference and propose a LOCAL fix, or (b) document it + pin current behavior with a regression test.',
  '',
  'METHOD (this is what makes a registration valid): drive BOTH sides with IDENTICAL input and isolate ONE layer.',
  '  - Dynamics: replay the same action sequence open-loop.',
  '  - Obs/estimators/geometry: feed the same ground-truth trajectory + same DR seed + same injected noise.',
  '  - RNG: first try a BIT-EXACT per-step match (inject shared draws into both sides). If the two draw in different order/shape so bit-exact is impossible at the module API, say so explicitly and fall back to DISTRIBUTIONAL (K seeds each side, compare per-channel mean/std). State which mode you used and WHY.',
  '  - Report divergence vs CONDITION (range-to-gate, fix-accept, staleness, tilt/rate) -- NOT a bare scalar. Use real measured numbers from scripts you actually run.',
  '',
  'AUTONOMY RULES:',
  '  - You do NOT commit, do NOT git add, do NOT touch the memory/ directory, do NOT edit files outside your own scratch dir. The commander serializes commits.',
  '  - Write ONLY under your scratch dir: handoff/system-id-2026-06-18/scratch/<your pair_id>/ . Put measurement scripts + raw output there.',
  '  - If a NEW pinning test is warranted, return its FULL file content in new_test_content (ready for the commander to write verbatim into new_test_path under tests/). Make it import-correct (mirror the sys.path.insert pattern other tests in tests/ use), self-contained, fast (<10s), and assert the registration you measured. PREFER pinning current behavior over changing code.',
  '  - NEVER propose autonomously changing a load-bearing SIGN or FRAME convention. If you find a sign/frame divergence, set load_bearing_flag=true and verdict="flag-commander" with full evidence. Do not edit it.',
  '  - A FIX is only justified when one side is demonstrably wrong vs a pinned test / canonical numpy frame / spec constant AND the fix is local AND it keeps inc7 (obs_dim 17) byte-identical (you must state how to prove that). Otherwise document + pin.',
  '',
  'KNOWN FOOTGUNS (do NOT "fix" these -- register/document them as intentional):',
  '  - The legacy CTBR _rate_sign alias is self-consistent.',
  '  - +L is CORRECT (-L would be a ~24 m flip; pinned by tests/test_obs_sign_faithfulness.py).',
  '  - The sigma sqrt(2) gap: NavState sqrt(P_E+P_D) vs emulator sqrt((P_E+P_D)/2) is INTENTIONAL -- register it, do not reconcile.',
  '  - inc7 (obs_dim 17) must stay byte-identical.',
  '',
  'Before measuring, RUN the existing test(s) that already pin your pair (find them via grep) and record their GREEN/RED/SKIP status from the actual run. Do not claim a test passes without running it.',
  '',
  'Return STRICTLY the structured object (the schema is enforced). report_section must be clean GitHub-flavored markdown (a ## section) that the commander will paste into handoff/system-id-2026-06-18/REPORT.md. Be precise and quantitative.',
].join('\n')

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['pair_id', 'title', 'counterpart', 'existing_tests', 'drive_method',
             'divergence', 'verdict', 'load_bearing_flag', 'report_section'],
  properties: {
    pair_id: { type: 'string', description: 'short kebab id, e.g. obs-emulators' },
    title: { type: 'string' },
    counterpart: { type: 'string', description: 'the two sides being registered, with file paths' },
    existing_tests: {
      type: 'array',
      description: 'tests that already pin this pair, with the status you observed by RUNNING them',
      items: {
        type: 'object', additionalProperties: false,
        required: ['node', 'status'],
        properties: {
          node: { type: 'string' },
          status: { type: 'string', enum: ['GREEN', 'RED', 'SKIP', 'NOT-RUN'] },
        },
      },
    },
    drive_method: { type: 'string', enum: ['bit-exact', 'distributional', 'static-constant', 'deferred-adroit', 'mixed'] },
    drive_method_note: { type: 'string', description: 'why this mode; if not bit-exact, what blocks it' },
    divergence: { type: 'string', description: 'measured divergence vs CONDITION, with numbers (range bins / fix-accept / staleness / tilt). State the worst-case and where it occurs.' },
    verdict: { type: 'string', enum: ['already-pinned', 'new-pinning-test', 'fix-needed', 'deferred-adroit', 'flag-commander'] },
    load_bearing_flag: { type: 'boolean', description: 'true if this touches a load-bearing sign/frame (commander must review; never auto-fix)' },
    new_test_path: { type: ['string', 'null'], description: 'tests/test_sysid_*.py path if a new pinning test is warranted' },
    new_test_content: { type: ['string', 'null'], description: 'FULL file content, ready to write verbatim' },
    fix_description: { type: ['string', 'null'], description: 'if fix-needed: exact change + proof one side is wrong + how to prove inc7 byte-identical' },
    report_section: { type: 'string', description: 'a ## markdown section for REPORT.md (counterpart, method, divergence-vs-condition, verdict, evidence)' },
    memory_delta_line: { type: ['string', 'null'], description: 'at most ONE <=1-line text fact for the commander memory delta, or null' },
    scratch_dir: { type: ['string', 'null'], description: 'path to your scratch dir if you created measurement scripts' },
  },
}

function task(id, body) {
  return PREAMBLE + '\n\n========================================\nYOUR PAIR: ' + id + '\n========================================\n' + body
}

const PAIRS = [
  task('obs-emulators',
    [
      'THE LIVE BLOCKER (#1). Register the two OBS EMULATORS end-to-end:',
      '  numpy: rl/estimator_emul.py  class EstimatorEmulator  (.reset / .step / .obs / .confidence_channel)',
      '  torch: rl/inc8_estimator_emul.py  class BatchedEstimatorEmulator  (reset_idx / .step ~line 504 / .confidence_channel ~line 553 / obs_zup_torch / kf_pos_zup / kf_vel_zup)',
      '',
      'CONTEXT SEED FROM THE COMMANDER (verify, do not blindly trust):',
      '  - The existing parity test tests/test_inc8_estimator_emul_torch.py already proves: frame-seam identity (obs==obs_from_truth when KF seeded at truth, <=1e-4); FULL-STEP KF-STATE parity under SHARED injected draws (kf.x, kf.P, t_since match <=1e-4); pointing->fix monotone; NEES in [0.8,1.3]; obs[17:20] bounds. RUN it and confirm GREEN.',
      '  - THE RESIDUAL GAP: nobody has driven both modules end-to-end and compared the FINAL OBS VECTOR the policy actually consumes -- obs[0:17] (KF pos/vel + truth attitude) AND obs[17:20] (confidence triple) -- step by step under an identical drive. Build that. That is your #1 deliverable.',
      '  - RNG ORDER FINDING TO CONFIRM: numpy EstimatorEmulator.step draws per step: rng.standard_normal(3) [IMU accel], then inside FixSurrogate.sample_fix rng.random() [Bernoulli], then CONDITIONALLY rng.standard_normal(3) [fix noise, ONLY on accept]. The torch BatchedEstimatorEmulator.step takes accept_u, accel_noise, fix_noise UNCONDITIONALLY (env-injected). So a naive shared-seed drive of the two real .step() APIs diverges in RNG stream after the first accept/reject difference. Bit-exact is only achievable by INJECTING shared draws (replicating numpy unconditional order) -- which proves the MATH/ENCODING is identical. Confirm this is the situation and state it crisply.',
      '',
      'DO THIS:',
      '  1. Write a measurement script under handoff/system-id-2026-06-18/scratch/obs-emulators/ that drives a head-on (and a few crabbed/tilted) approaches to several gates with shared injected draws, using the REAL numpy objects (the real EstimatorEmulator.obs() and .confidence_channel(); drive its self.kf via the same predict/update the real .step does but with injected unconditional draws so states stay identical to torch) and the REAL torch BatchedEstimatorEmulator. Compare obs[0:17] and obs[17:20] element-wise.',
      '  2. Report max-abs-diff for obs[0:17] and obs[17:20] separately, BROKEN DOWN vs condition: range-to-gate bins, fix-accepted vs not, staleness (age_norm). Not a bare scalar.',
      '  3. ALSO run a DISTRIBUTIONAL check: run the REAL numpy EstimatorEmulator.step (its true conditional-draw path, internal rng) over K=64 seeds and the torch module over K=64 seeds, compare per-channel obs mean/std at a few fixed range bins. Confirm they agree distributionally despite the RNG-order difference.',
      '  4. DECIDE THE LIVE QUESTION: do the emulators DIVERGE at the obs encoding level (=> FIX-THE-GRADER) or are they faithful (=> the 0/200 numpy-grader death is a real POLICY-GAP and the answer is MEASURE-IN-TORCH)? State the conclusion with the numbers that support it.',
      '  5. Draft tests/test_sysid_obs_emul_parity.py: assert obs[0:17] and obs[17:20] parity <=1e-4 (or the tolerance you measure) under the shared-draw drive over an approach, AND a distributional-agreement assertion. Make it run in <15s on CPU.',
    ].join('\n')),

  task('confidence-constants',
    [
      'Register the confidence-triple CONSTANTS and the sigma-hat definitions across the three modules (#2):',
      '  rl/estimator_emul.py: SIGMA_REF_M, TAU_STALE_S, EmulConfig.sigma_ref/tau_stale, confidence_channel formula, _gate_frame_sigmas (sigma_inplane = sqrt((P_E+P_D)/2), sigma_along = sqrt(P_along)).',
      '  rl/inc8_estimator_emul.py: SIGMA_REF_M, TAU_STALE_S, EmulConfig.sigma_ref/tau_stale, confidence_channel, _gate_frame_sigmas.',
      '  src/racer/estimator_obs.py: note whether it appends obs[17:20] at all (it should NOT -- it is the inc7 17-dim deploy seam; confirm).',
      '  Also grep src/racer/ for any other confidence/sigma_ref/tau_stale/NavState sigma definition (e.g. navigator.py, kf_rewind.py, contracts.py NavState). In particular register the INTENTIONAL sqrt(2) gap: NavState exporting sqrt(P_E+P_D) vs the emulator using sqrt((P_E+P_D)/2). DOCUMENT it, do NOT reconcile.',
      '',
      'DO THIS: confirm the numeric equality of SIGMA_REF_M and TAU_STALE_S across both emulators (and any other definer). Confirm the confidence_channel and _gate_frame_sigmas formulas match between numpy and torch (you may reuse/inspect existing tests). Draft tests/test_sysid_confidence_constants.py that imports both emulator modules and asserts SIGMA_REF_M / TAU_STALE_S / EmulConfig defaults are equal, and pins the sqrt(2) relationship between the NavState sigma and the emulator sigma as a documented (asserted) invariant. Quantify where each constant lives.',
    ].join('\n')),

  task('frames-camera',
    [
      'Register the CAMERA geometry constants (#3a): the numpy scipy-based frames vs the torch self-contained reimpl.',
      '  numpy: src/racer/frames.py  R_camera_from_body(), IMAGE_WIDTH, IMAGE_HEIGHT, the camera intrinsics K, project_camera_point, vertical_fov_deg.',
      '  torch:  rl/inc8_estimator_emul.py  _r_camera_from_body_np()/R_CAMERA_FROM_BODY_NP, CAMERA_INTRINSICS_K_NP, IMAGE_WIDTH/HEIGHT.',
      '  also the fix_surrogate geometry (rl/fix_surrogate.py geometry()) uses frames.R_camera_from_body + project_camera_point -- the numpy reference the torch batched_geometry mirrors.',
      '',
      'Find + RUN any existing parity test (grep for R_camera_from_body, R_CAMERA_FROM_BODY_NP, batched_geometry, test_inc8_fix_surrogate_torch, test_frames, test_projection). Verify the torch _r_camera_from_body_np == frames.R_camera_from_body to ~1e-12, and K / image dims match. Confirm batched_geometry == fix_surrogate.geometry (range/az/el/in_image/t_cam) over random poses -- if an existing test covers it, run it and cite; if there is a gap, draft tests/test_sysid_camera_geometry.py pinning R_camera_from_body + K + image dims + a geometry parity sweep (numpy fix_surrogate.geometry vs torch batched_geometry) over random drone poses & gates.',
    ].join('\n')),

  task('gate-frame-plusL',
    [
      'Register the GATE-FRAME rotations and the +L sign (#3b):',
      '  numpy: rl/fly_rl.py _gate_rotmat_w2g(yaw) (world->gate, rows [[c,s,0],[-s,c,0],[0,0,1]]); rl/estimator_emul.py ned_gate_frame(yaw) (gate->world NED, columns [right,down,downrange]); the obs +L sign in fly_rl.obs_from_zup (pos_g = R_w2g @ (gate - pos)) and src/racer/localization.py.',
      '  torch:  rl/inc8_estimator_emul.py _gate_rotmat_w2g_torch and ned_gate_frame_torch.',
      '',
      'The +L sign is LOAD-BEARING (tests/test_obs_sign_faithfulness.py: +L ~4.77e-7, -L control breaks ~24 m). RUN that test, confirm GREEN, and confirm it actually ran (not skipped). Do NOT propose changing the sign.',
      'Register: _gate_rotmat_w2g (numpy) == _gate_rotmat_w2g_torch over random yaws; ned_gate_frame (numpy) == ned_gate_frame_torch over random yaws. Confirm the documented relationship between the world->gate obs frame and the NED contracts.Gate frame (the C1 identity: a pure in-plane NED-gate displacement -> zero obs along-track change). If a gap exists, draft tests/test_sysid_gate_frames.py pinning numpy==torch for both rotations + the +L convention restated as a numpy assertion (independent of the torch importorskip path). Set load_bearing_flag=true (you are touching the +L sign even just to register it) but verdict should be new-pinning-test or already-pinned -- only use flag-commander if you find an ACTUAL divergence.',
    ].join('\n')),

  task('action-sign-maps',
    [
      'Register the ACTION/RATE SIGN MAPS (#3c) -- this is the area of the recently fixed yaw-injection bug, so DOCUMENT + pin current behavior; NEVER auto-change. Set load_bearing_flag=true.',
      '  rl/fly_rl.py: _FLIP=[1,-1,-1] (world&body Z-up<->NED / FLU<->FRD flip, involutory), _ACT_FLU_TO_FRD=[1,-1,1] (policy_step wire map), _RZ_PI_BODY=diag(-1,-1,1).',
      '  rl/contact_true_eval.py: _RATE_SIGN_LIVE=[1,1,1] (the eval plant rate sign), _ACT_FLU_TO_FRD usage at ~line 365/367 (invert then re-apply to reconstruct FRD from the look-at FLU), _LOOKAT_FLIP/_FLIP_FRD_FLU=[1,-1,-1].',
      '  Find where _RATE_SIGN_LIVE is defined and any sibling rate_sign (the legacy CTBR _rate_sign alias -- do NOT fix; register as self-consistent).',
      '',
      'Register the RELATIONSHIPS as facts: _FLIP and _ACT_FLU_TO_FRD agree on roll/pitch (axes 0,1) but differ on YAW (axis 2: -1 vs +1); both are involutory (v*m*m==v). State exactly which map each call site uses and why the yaw-only difference was the bug (the look-at yaw realized rate was sign-flipped when reconstructed via _FLIP instead of _ACT_FLU_TO_FRD). RUN tests/test_super_rate.py, tests/test_mixer.py, tests/test_controller.py, tests/test_contact_true_eval.py (whichever exercise these maps) and record status. Draft tests/test_sysid_action_sign_maps.py that pins: the exact values of _FLIP / _ACT_FLU_TO_FRD / _RATE_SIGN_LIVE / _FLIP_FRD_FLU; their involution; and the roll/pitch-agree, yaw-differ relationship. This is a pure-constant registration (no diffaero). Quote the live-launch sign config from memory only if you can verify it in code.',
    ].join('\n')),

  task('odometry-ry-pi',
    [
      'Register the ODOMETRY R_y(pi) quaternion-conjugation convention (#3d).',
      '  Counterpart files: scripts/frame_residual_report.py (the internal-consistency check run after every live session), and wherever the ODOMETRY quat is consumed/conjugated (grep for R_y, Ry, conjug, odometry, ODOMETRY, _RY_PI, quat across src/racer/ and rl/ -- likely src/racer/mavlink_client.py or frames.py or contracts.py).',
      '  Independent counterpart candidates: tests/test_frame_conventions.py, tests/test_mavlink_velocity_single_rotation.py, tests/test_frames.py.',
      '',
      'RUN those tests, record status. Establish: WHAT the R_y(pi) conjugation is (the ODOMETRY quaternion is R_y(pi)-conjugated relative to the canonical body frame), WHERE it is applied, and the independent check that pins it (a numpy frame round-trip). The footgun is: internal consistency cannot catch a conjugation -- so the registration must compare against an INDEPENDENT canonical frame, not a self-consistent round-trip. If laptop-runnable, draft tests/test_sysid_odometry_quat.py pinning the conjugation against a canonical numpy attitude reference. If consuming the real ODOMETRY requires a live mavlink stream / diffaero, mark that arm deferred-adroit and pin whatever pure-numpy frame math is laptop-runnable.',
    ].join('\n')),

  task('plant-parity',
    [
      'Register the PLANT parity (#4): numpy-vs-numpy racer.rl_plant vs racer.twin.',
      '  Existing tests: tests/test_rl_plant_parity.py and tests/test_twin_diffaero_extreme_parity.py (both numpy-vs-numpy on this laptop). RUN both, record status + what regime they cover (read them).',
      '  src/racer modules: rl_plant.py (PlantState, step), twin.py.',
      '',
      'METHOD: replay the SAME action sequence open-loop through both plants and compare state trajectories. After confirming the existing tests GREEN, EXTEND coverage to the AGGRESSIVE regime the current tests may not exercise: near the rate rails (body rates near max), high tilt (large roll/pitch), and high collective. Write a measurement script under your scratch dir that drives both plants with an identical aggressive action sequence and reports max state divergence vs condition (tilt angle, rate magnitude). If they stay within tolerance, draft tests/test_sysid_plant_parity_aggressive.py pinning it. If they diverge in the aggressive regime, document precisely where (which state, which condition, how much) and whether it is a real bug (one plant wrong vs a known reference) or an expected modeling difference -- if a real load-bearing divergence, verdict=flag-commander.',
      '  If racer.twin or rl_plant needs diffaero, mark that arm deferred-adroit (they should be pure numpy -- verify).',
    ].join('\n')),

  task('localization-kf-navigator',
    [
      'Register the LOCALIZATION / KF / RewindKF / Navigator stack (#5).',
      '  src/racer/localization.py (gate_relative_inplane_fix, the +L anisotropic fix), src/racer/kf_rewind.py (RewindKF / OOSM), src/racer/navigator.py (NavState, _apply_gate_relative_fix), src/racer/state_estimator.py (LinearKF).',
      '  Independent counterparts: the torch BatchedLinearKF in rl/inc8_estimator_emul.py mirrors state_estimator.LinearKF (predict/update_position) -- pinned by tests/test_inc8_linearkf_torch.py. Existing tests: test_localization.py, test_kf_rewind.py, test_navigator.py, test_navigator_gate_relative.py, test_gate_relative_fix.py, test_state_estimator.py, test_state_estimator_pfloor.py, test_inc8_linearkf_torch.py, test_casec_foundation.py.',
      '',
      'RUN all those tests, record status. For each module identify whether it HAS an independent counterpart already registered (e.g. LinearKF numpy vs torch) or is pinned only against itself. The key NEW registration: confirm numpy state_estimator.LinearKF == torch BatchedLinearKF (predict + update_position) under a shared drive across BOTH benign AND stress conditions (near-singular S, large covariance, tiny dt, rapid fixes) -- extend test_inc8_linearkf_torch coverage if it only tests benign cases. Confirm gate_relative_inplane_fix produces the +L anisotropic cov consistent with the surrogate fix_covariance (both rotate a gate-plane diagonal to world). If a gap exists, draft tests/test_sysid_kf_localization.py. Note: RewindKF OOSM rewind has no torch counterpart (it is deploy-only) -- register it against its own analytic OOSM expectation if a clean independent check exists, else mark "self-pinned, no independent counterpart" and document.',
    ].join('\n')),

  task('production-vs-trained-obs',
    [
      'Register PRODUCTION-vs-TRAINED obs (#6): the deploy obs builder vs the trained 20-dim contract.',
      '  src/racer/estimator_obs.py: estimator_state_for_obs (replaces pos/vel with NavState) + estimator_obs (-> fly_rl.build_obs) = the 17-dim deploy seam.',
      '  Trained obs: rl/fly_rl.obs_from_zup / build_obs / obs_from_truth (17-dim); the 20-dim contract = obs[0:17] + the d5 confidence triple obs[17:20].',
      '  Existing tests: test_estimator_obs_wiring.py, test_train_deploy_obs_elementwise.py, test_confirmed_p4_c05.py. RUN them, record status.',
      '',
      'COMMANDER SEED (verify): memory claims deploy obs == trained obs to obs[0:17] delta 0.0 and obs[17:20] delta ~9.2e-9, and that the spike froze a deploy obs[17:20] builder (deploy_confidence_triple) that is NOT yet promoted into estimator_obs.py (there is no production obs[17:20] builder in src/racer/estimator_obs.py -- confirm this gap exists). ',
      'DO THIS: register estimator_state_for_obs + build_obs (deploy 17-dim) == obs_from_truth / obs_from_zup (trained 17-dim) element-wise under identical state (the inc7 byte-identity). Confirm whether a production obs[17:20] builder exists in src/racer (grep deploy_confidence_triple, confidence, obs[17, 17:20). If the 20-dim production builder is ABSENT, that is a real GAP to register (deploy currently cannot build the inc8 confidence triple from NavState the way training does) -- document it precisely (this is a carry-forward, not a test failure). Draft tests/test_sysid_production_obs.py pinning the 17-dim deploy==trained equality, and DOCUMENT the missing 20-dim production builder as a registered gap with verdict flag-commander if it is load-bearing for inc8 deploy.',
    ].join('\n')),
]

phase('Investigate')
log('System-ID: investigating ' + PAIRS.length + ' registration pairs in parallel')

const results = await parallel(PAIRS.map((p, i) => () =>
  agent(p, { label: 'reg:' + (i + 1), phase: 'Investigate', schema: SCHEMA })))

const ok = results.filter(Boolean)
log('Investigation complete: ' + ok.length + '/' + PAIRS.length + ' pairs registered')
return ok
