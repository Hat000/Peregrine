# VQ2 residual loop-choke fix — BUILD NOTES (incremental, survives a crash)

Build dive for R1 + R2 + Q3-instrumentation from ANALYSIS.md. Leave everything UNCOMMITTED.
CPU-only, no sim/flights/GPU. Do NOT touch src/racer/gate_seeker.py or deploy_profile.py
seeker_overrides.

## Status ledger
- [x] Read ANALYSIS.md + all touched files, established architecture
- [~] R1 worker-thread vp_yaw — VpYawWorker built (async_detect.py); navigator split
      _apply_vp_yaw -> sync/async + shared _apply_vp_heading_estimate; config flag vp_yaw_async;
      lazy _ensure_vp_yaw_worker + close_vp_yaw_worker. TODO: fly_rl exit stop + [async-vp-yaw]
      summary + perf_summary; profile flag; tests.
- [~] R2 short-circuit ACTUATOR_OUTPUT_STATUS — parse_actuator_output flag on MavlinkClient
      (default True=byte-identical), _handle branch guarded, fly_rl client built with it OFF.
      TODO: tests. Tlog verdict = SAFE (see above). R1 fly_rl exit stop also DONE.
- [x] Q3 per-tick phase timing + work_ms accounting fix + nav-record monotonic stamp
      - _phase_ms buckets {pump,nav,seeker,ctrl_send,log,pump_spin} + _phase_worst_tick snapshot
      - ACCOUNTING FIX: work_ms clock now starts at `now` BEFORE the pre-tick client.pump() (pump is
        its own bucket AND billed to work_ms/over-budget). in-spin pumps -> pump_spin (aggregate only,
        _tick_acc=None, NOT billed to work_ms = it is sleep not work).
      - [loop-phase] exit line (mean/max/n per bucket + worst work-tick breakdown) +
        perf_summary.json["loop_phase_ms"] (+ loop_phase_worst_tick_ms).
      - nav record gains rec["t_mono_ns"] = time.monotonic_ns().
- All 5 touched files py_compile OK.
- [x] Tests (R1, R2, phase-timing) — tests/test_residual_choke_fixes.py, 12 tests:
      R1 worker unit (compute+publish, latest-wins backlog collapse, error-counted-not-raised,
      same-fid-not-duplicated), R1 nav wiring (OFF byte-identical, ON honours monkeypatch + bounds
      drift, worker error safe), R2 (default parses, flag-false skips, other msgs unaffected, tlog tap
      independent via REAL pump), Q3 (phase buckets + nav t_mono_ns stamp end-to-end via _fly_gate_seeker).
      NOTE during test build: added _last_submitted_fid guard to VpYawWorker so a re-submit of the
      most-recently-accepted frame_id (pending OR already picked up) is dropped -> re-compute only on a
      NEW frame_id (mirrors AsyncDetectWorker.last_fid). None ids never de-duped.
- [x] Battery run: test_residual_choke_fixes + test_perf_summary_json + test_vision_compute_cuts +
      test_vision_yaw_wiring + test_gate_seeker + test_use_ahrs_wiring + test_mavlink_client
      = 145 passed, 1 skipped (recon fixture absent). 0 failures.
- [x] Full suite vs baseline: 11 failed, 1847 passed, 73 skipped (3m10s). EXACTLY the 11-failure
      baseline (test_ahrs_bench R_quat_roundtrip + test_diagnose_session end-to-end +
      test_sysid_camera_geometry bitclose + 8x test_sysid_plant_parity_aggressive). 0 NEW failures.
      FIRST full run had 2 extra fails in test_nav_estimate_log (EXPECTED_KEYS schema-lock) from the
      new t_mono_ns record field -> added "t_mono_ns" to that test's EXPECTED_KEYS set (intended
      schema maintenance; the test exists to force exactly this). Re-run clean at baseline.

## DONE — all items complete, everything UNCOMMITTED for coordinator review.
Files touched:
  src/racer/vision/async_detect.py  (+VpYawResult, +VpYawWorker)
  src/racer/navigator.py            (vp_yaw_async flag; _apply_vp_yaw split; _apply_vp_heading_estimate;
                                     _apply_vp_yaw_async; _ensure_vp_yaw_worker; close_vp_yaw_worker;
                                     _vp_yaw_worker field)
  src/racer/mavlink_client.py       (parse_actuator_output flag + guarded ACTUATOR branch)
  src/racer/deploy_profile.py       (vq2_case_c: vp_yaw_async=True)
  rl/fly_rl.py                      (R2 client parse OFF; R1 worker stop + [async-vp-yaw] summary;
                                     Q3 _phase_ms buckets + accounting fix + [loop-phase] + perf_summary
                                     keys; nav record t_mono_ns)
  tests/test_residual_choke_fixes.py (NEW, 12 tests)
  tests/test_nav_estimate_log.py     (EXPECTED_KEYS += t_mono_ns)

## COORDINATION NOTE (git state — READ THIS)
- Task started at HEAD 4cdd6ba. During the build a CONCURRENT dive (climb-approach) committed
  6eec0a7 "fix(vq2): climb-approach limit-cycle" and the branch advanced. That commit touched
  src/racer/deploy_profile.py (its OWN climb-approach seeker_overrides knobs) AND SWEPT IN my
  one-line `vp_yaw_async=True` nav_config edit + comment block (the shared worktree working tree).
  => `vp_yaw_async=True` is ALREADY COMMITTED in HEAD (6eec0a7), NOT in my uncommitted diff. It is
  present + correct (verified: vq2_case_c().nav_config.vp_yaw_async is True). I only edited the
  nav_config line (explicitly permitted); I did NOT touch seeker_overrides.
- My R1 nav/worker code, R2, Q3, and tests remain UNCOMMITTED (git status: 5 M files + 2 new test/
  notes files). HEAD does NOT contain the navigator vp_yaw_async wiring or VpYawWorker (grep=0) —
  those are in the working tree for coordinator review, as instructed.
- The full-suite green run (11 baseline failures, 0 new) and the battery were BOTH executed against
  the MERGED tree (post-6eec0a7 + my uncommitted changes), so the merged state is verified.

## Key facts established from the code (before writing anything)

### R1 coupling analysis (the make-or-break question)
- `navigator._apply_vp_yaw(frame, R_wb)` (navigator.py:858) does, ON THE LOOP THREAD:
  1. `roll, pitch, yaw_hat = self._current_true_rpy()`  (CURRENT ESKF attitude)
  2. `est = estimate_heading(frame.image_bgr, roll, pitch, ransac_iters=...)`  ← THE ~30-100ms COST
  3. branch-disambiguate `est.branch_headings_rad` against `yaw_hat`, gate on quality + branch_max
  4. `self._ahrs.eskf.update_yaw(yaw_meas, vp_yaw_noise_std)` + refresh attitude cache/R_wb
- CRITICAL: `estimate_heading(frame_bgr, roll, pitch, ransac_iters)` uses roll/pitch ONLY as the
  gravity-known tilt to back-project VP pixels (heading_vp.py:152 yaw_from_vp_pixel). It does NOT
  touch R_wb or the ESKF. It returns a HeadingEstimate (immutable frozen dataclass) or None.
- => The split is CLEAN: only step 2 (estimate_heading) moves off-thread. Steps 1,3,4 stay on the
  loop thread. The worker needs ONLY (frame, roll, pitch) captured at submit time; the branch snap
  uses the CURRENT yaw_hat (fresh, on-thread) so acceptance gates / disambiguation / update_yaw noise
  are BYTE-IDENTICAL to the sync path — none of them move. NO R_wb obstacle. Transplants cleanly.
- Applying a HeadingEstimate computed from a frame up to ~200ms old is fine: yaw drifts ~0.5 deg/s
  (=> ~0.1 deg error over 200ms), and the branch snap uses the fresh yaw_hat so a stale frame cannot
  cause a 90-deg branch flip. Documented as the design rationale in code.

### TEST MONKEYPATCH CONSTRAINT (load-bearing)
- tests/test_vision_yaw_wiring.py + test_vision_compute_cuts.py monkeypatch
  `racer.navigator.estimate_heading`. The async worker MUST call estimate_heading through a callable
  that resolves the navigator-module global at CALL TIME, so the patch is honored. Design: Navigator
  builds the worker with `compute_fn=lambda img,roll,pitch: estimate_heading(img,roll,pitch,ransac_iters=N)`
  defined INSIDE navigator.py — the lambda's free name `estimate_heading` is looked up in navigator's
  module globals at call time => sees the patched fn. Worker stays generic (no import of estimate_heading).

### async-detect proven pattern (racer/vision/async_detect.py, commit fe70406)
- daemon thread + threading.Lock single-slot latest-wins + `_prev` to close the publish/consume race
- stats: n_detects/total_ms/max_ms/n_errors/last_error (worker writes, loop reads, GIL-safe for logging)
- worker swallows+counts every exception; never dies; `stop()` joins briefly (daemon => exit-safe)

### R2 tlog-safety (VERDICT: SAFE)
- CONFIRMED SAFE. fly_rl.py:2475-2483 `_on_msg` captures `msg.get_msgbuf()` -> `rec.record_mavlink`.
  It is `client.on_message`, called at mavlink_client.py:254-255 INSIDE pump() BEFORE `_handle`
  (line 256). Raw wire bytes hit the tlog independent of the parse. scripts/analyze_flight.py:60 +
  extract_run.py:76 read ACTUATOR_OUTPUT_STATUS by RE-PARSING the tlog (parser-independent) — they
  never touch client.actuator_outputs. => short-circuiting the _handle branch cannot change the tlog.
- HOWEVER: there ARE two live readers of `client.actuator_outputs`: scripts/rate_sysid.py:574 (a
  sysid tool, NOT the flown fly_rl gate-seeker loop) and test_mavlink_client.py:319-320 asserts
  `_handle(_actuator())` populates `c.actuator_outputs["motors"]`. So I CANNOT unconditionally strip
  the parse. R2 => a `parse_actuator_output: bool = True` flag on MavlinkClient. Default True keeps
  test_mavlink_client + rate_sysid + every caller byte-identical; the flown loop (fly_rl.py:2335)
  constructs the client with it OFF. Msg stays counted: the raw tlog records every ACTUATOR via
  on_message regardless (there is NO in-client per-type counter — the tlog IS the record).
- The branch is NOT a dataclasses.replace (ANALYSIS said "replace" loosely) — it is a dict build +
  `[float(x) for x in list(msg.actuator)[:4]]` list-comp. Short-circuit = early-skip that build.

### R2 tlog-safety (old placeholder)
- mavlink_client.pump() (mavlink_client.py:250-256): `while True: recv_match; on_message(msg); _handle(msg)`
- `on_message` is the RAW tap the recorder sets to capture msg.get_msgbuf() for mavlink.tlog. It runs
  BEFORE _handle and is INDEPENDENT of parsing. => short-circuiting the ACTUATOR_OUTPUT_STATUS branch
  inside _handle CANNOT change what lands in the tlog. (Confirm the recorder wiring in build step.)
- ANALYSIS confirmed zero readers of client.actuator_outputs on the flown path — re-verify with grep.

### Q3 accounting hole
- fly_rl.py: work_ms measured from `now` (line 1605) AFTER the rate-limiter spin (1600-1603), so the
  two pump() calls (1601/1603) are NOT billed. Fix: start the work clock BEFORE the pre-tick pump
  (1603) so pump gets its own bucket and the over-budget accounting tells the truth. The nav record
  (_nav_estimate_record, fly_rl.py:1158) has sim_time_ns+tick_index but NO wall stamp — add
  time.monotonic_ns().

## Flag to turn R1 ON in vq2_case_c
- NavigatorConfig.vp_yaw_async: bool = False (default => byte-identical sync path)
- vq2_case_c sets vp_yaw_async=True (keeps vp_yaw_decimate=15 as the SUBMISSION cadence)
