# VQ2 Residual Loop-Choke Diagnosis — 2026-07-05

Run: data/runs/20260705_211007_loopchoke_confirm_f1 (branch vq2-gate2-turn-dive, HEAD 4cdd6ba;
confirm flight for commit f1c5db9 = floor_height OFF + vp_yaw decimate 15).
perf verdict: 23.38 Hz / 661 ticks, worst work 161.5 ms, 37.8% over 33 ms budget -> CHOKED.
Target: reliable >=27 Hz under sim load.

## 0. Loop accounting: what work_ms bills (fly_rl.py) [ESTABLISHED]
work_ms measured at line 1690 from `now` set at line 1605 -- AFTER the rate-limiter spin
(lines 1600-1603). So the two client.pump() calls (1601 in-spin, 1603 pre-tick) and the sleep
spin are NOT in work_ms. The per-tick status print (1697-1716) is measured AFTER 1690 and fires
only ~1/sec -> also not in work_ms.
work_ms COVERS (1605-1690): stop checks, sim-reset guard, RACE_STATUS read, vision_worker.latest()
snapshot, nav.update() [ESKF per-sample ingest + _maybe_run_vision], seeker.command_visual(),
send_command(), in-memory nav-log append.
CONSEQUENCE: pump() mavlink-drain cost is charged to the SPIN, invisible in worst_work_ms, but
still eats wall-clock inside the tick period -> lowers achieved Hz without showing in "worst work".
The 162 ms worst tick = vp_yaw tick (101 ms vp_yaw + ~60 ms same-tick nav/seeker). Steady sub-30 Hz
is pump()+nav that work_ms never bills.

## 1. Tick-interval distribution [PARTIAL: no wall-clock in log]
LIMITATION: nav_estimate.jsonl has sim_time_ns + tick_index but NO wall timestamp
(_nav_estimate_record fly_rl.py:1158). Only wall-rate signal = 1 Hz hz= prints.
hz= swings 15.4 -> 28.1 across flight (early gi0->1 heavy-vision ~19-22 Hz; late drift ~27 Hz).
VERDICT: regime-correlated / bimodal, NOT uniform slowness.
imu_samples_ingested hist (661 ticks): 4=62%, 5=17%, 6=7%, 7=5%, >=10 =3.9% (up to 20/tick),
mean 4.94, total 3264. Right-skew w/ backlog-drain tail; each sample = 1 ESKF step -> nav cost
4-20x per tick, spikes on heavy ticks.
seeker_regime: pursuit 437, bridge 100, pass_vis 58, settle 49, pass_wire 8, egress 6, hold 3.

## 2. mavlink mix = pump() load [ESTABLISHED]
tlog 8347 msgs / 661 ticks = 12.6/tick:
  HIGHRES_IMU 3957 (each: replace() w/ 4 fresh np arrays + ring.append)
  ACTUATOR_OUTPUT_STATUS 3185 (each: _handle + [float(x) for x in list(msg.actuator)[:4]]; UNUSED by loop)
  COLLISION 732, HEARTBEAT 335, ENCAPSULATED_DATA 136 (RACE_STATUS parse), COMMAND_ACK 2.
HIGHRES_IMU 3957 arrived vs 3264 ingested -> ~700 dropped (ring maxlen=64 overflow / dedupe).
ACTUATOR_OUTPUT_STATUS = 48% of IMU volume, ZERO useful work for this path.

(continued below as analysis proceeds)

## 3. Hypothesis verdicts (non-vision hogs)

(a) client.pump() unbounded drain per tick  -> [STRONGEST NON-VISION HOG, ESTABLISHED as costly]
    src/racer/mavlink_client.py:250-256. `while True: recv_match(blocking=False); _handle(msg)`.
    12.6 msgs/tick, each _handle does a dataclasses.replace() rebuilding the whole DroneState.
    HIGHRES_IMU branch (mavlink_client.py:279-317) builds 4 fresh np.array()s + ring.append per msg
    (3957 of these). ODOMETRY branch (336-360) builds ~5 arrays + euler_from_quat + world_vec rotate.
    ACTUATOR_OUTPUT_STATUS (408-412, 3185 msgs) does a list-comp per msg for data NOTHING in the
    control/estimate loop reads. CRUCIAL: pump() runs at lines 1601+1603, OUTSIDE the work_ms window,
    so this cost is INVISIBLE in worst_work_ms yet directly lowers achieved Hz. This is why "vision
    explains almost nothing" on typical over-budget ticks. Pump() is separate from the 1M video
    datagrams (video thread owns its own UDP socket; pump drains only self.conn on udp:14550) -- so
    the load is the 12.6 mavlink msgs/tick, dominated by IMU(3957)+ACTUATOR(3185)=86% of all msgs.

(b) ESKF rate-ingest cost per tick  -> [SECONDARY HOG, ESTABLISHED]
    navigator.py:769-813 _ingest_imu_ring. TWO costs:
      (i) list(ring) COPIES the whole maxlen=64 deque EVERY tick (line 790) + a list-comp scan of
          all 64 to extract the ~5 new (line 802) -> O(64) alloc+scan/tick to do O(5) work.
      (ii) per NEW sample: self._ahrs.ingest -> ESKF predict (eskf.py:426-456) allocates F(6x6),
           Q(6x6 diag), multiple np.eye(3), _skew, quat ops = ~10-15 small np allocs PER SAMPLE.
           At mean 4.94 samples/tick (tail to 20) -> ~50-75 small-array allocs/tick, spiking to
           ~250 on backlog-drain ticks. The imu tail (>=10 samples, 3.9% of ticks) aligns with the
           slowest hz readings. Small matrices so wall-cost is modest per op, but the ALLOCATION
           CHURN feeds hypothesis (e).

(c) per-tick print() through Tee on Windows  -> [NOT A PER-TICK HOG, ESTABLISHED negative]
    The status print (fly_rl.py:1697-1716) is GATED `if now - last_p >= 1.0` -> fires ~1/sec (27
    lines total in the console log), NOT per tick, and is measured AFTER work_ms. ~200 chars once/sec
    through Tee is negligible. NOT the choke. (Other prints -- gate-pass, collision -- are rare events.)

(d) recorder enqueue contention  -> [LOW RISK, LIKELY negative, PARTIAL]
    The nav-log is an in-memory list.append (fly_rl.py:1684, no I/O until exit). The frame recorder
    runs on the video thread (max_pub=0.3ms in the console -> publish is fast, no contention seen).
    commands.jsonl / video.bin writing is on other threads. No evidence of enqueue stalls in the log.
    [OPEN: not directly timed -- the proposed 'log' bucket in section 4 would confirm.]

(e) GC pauses from per-tick churn  -> [PLAUSIBLE CONTRIBUTOR, HYPOTHESIS]
    Per tick the loop allocates: ~12.6 DroneState replace()s (each w/ several np arrays) in pump +
    ~50-75 small np arrays in ESKF ingest + list(ring) 64-tuple copy + PnP arrays + nav-log dict.
    Order ~150-300 short-lived allocations/tick at ~23 Hz = ~3.5-7k allocs/sec. Enough to trigger
    CPython gen-0 GC sweeps periodically; a sweep over the large np-array population could explain
    part of the bimodal spike tail. UNPROVEN without gc.callbacks timing -- flag as hypothesis. Cheap
    de-risk: gc.disable() during the loop + manual gc.collect() at exit (or gc.freeze() after warmup).

(f) remaining on-thread CUDA/torch call  -> [FULLY NEGATIVE, ESTABLISHED]
    No torch/cuda/synchronize in gate_seeker.py or navigator.py. detect_cached on the async path is
    served by AsyncDetectorProxy.detect (async_detect.py:210-214) = dict lookup + list() copy, NEVER
    inference. Worker runs detect on its own daemon thread; latest()/detect() are non-blocking.
    vision_step_ms.detect count=542 @ ~0ms confirms cache hits only. The ONLY on-thread vision cost
    left is (1) vp_yaw (36 calls, 67ms mean, the 101ms worst tick -- CV RANSAC, real) and (2) the
    seeker's own cv2.solvePnPGeneric IPPE per candidate (gate_pose.py:144, ~1-3ms/new-frame tick,
    CPU, in work_ms). PnP is minor; vp_yaw is the single worst-tick spike.

## 4. Proposed instrumentation (design only -- do NOT implement here)
Mirror the existing navigator _time_step accumulate-in-memory / print-once pattern (navigator.py:840;
result-dict stash + [vision-timing] print at fly_rl.py:1778-1794). Add per-tick PHASE buckets in the
control loop itself:
  buckets = {"pump_pre","nav","seeker","ctrl_send","log"} each {count,total_ms,max_ms} + a worst-tick
  snapshot dict, exactly like vision_step_ms / _vision_worst_tick_ms.
Placement in fly_rl.py loop (around 1600-1690), each wrapped in perf_counter() deltas:
  - pump_pre : time the FINAL client.pump() at line 1603 (the pre-tick drain -- currently un-billed).
               [Also a separate 'pump_spin' counter summing the in-spin pumps at 1601 to see how
                much wall-clock the rate-limiter burns draining backlog.]
  - nav      : around nav.update(s, frame) at 1677 (splits ESKF+vision; vision already sub-timed).
  - seeker   : around seeker.command_visual at 1678 (captures the on-thread IPPE PnP).
  - ctrl_send: around client.send_command(cmd) at 1679.
  - log      : around the _nav_log.append(_nav_estimate_record(...)) at 1684 (confirms/kills hyp d).
Emit one "[loop-phase] pump=.. nav=.. seeker=.. send=.. log=.. (mean/max ms, n)" line at loop exit +
stash in perf_summary.json result["loop_phase_ms"]. ADD a per-tick wall-clock stamp to the nav-log
record (time.monotonic_ns()) so the NEXT run yields a true wall tick-interval distribution (the gap
this run could not answer). All logging-only, no behaviour change, swallow exceptions like _time_step.

## 5. Ranked fixes (expected Hz recovered vs risk)

R1. Worker-thread vp_yaw  [KILLS THE 101ms WORST TICK; MEDIUM RISK; scoped]
    vp_yaw is 36 calls, 67ms mean / 101ms max, and it IS the 162ms worst-work tick. Moving the VP
    RANSAC + Manhattan-line estimate_heading (navigator.py:877, _apply_vp_yaw) onto a worker thread
    (like async-detect) and consuming its latest yaw pseudo-measurement removes 101ms from the worst
    tick and ~2.4ms/tick amortized. Per the memory note vq2-loopchoke-cut-vp-yaw-sole-anchor, vp_yaw
    is the SOLE yaw anchor on the flown map-free path -- do NOT cut it, MOVE it. Risk: the yaw update
    would be applied one-frame-stale (like the detect); acceptable for a 5deg-noise pseudo-measure at
    healthy ~0.5 deg/s drift. Expected: removes the entire 37.8%-over-budget worst-tick class; the
    steady rate gain is smaller (~2.4ms/tick -> ~1-2 Hz) but the tail collapses (worst 162->~60ms).
    THIS is the single highest-value fix for "worst work" and the over-budget %.

R2. Skip/short-circuit ACTUATOR_OUTPUT_STATUS in _handle  [CHEAP, LOW RISK, ~1-2 Hz]
    3185 msgs/flight (48% of IMU volume) each pay _handle dispatch + a list-comp, for data the
    control/estimate loop never reads (actuator_outputs is diagnostics-only). Early-return/continue
    for this type in _handle (mavlink_client.py:408), or filter it in recv_match, removes ~4.8
    msgs/tick of pure-waste work. Near-zero risk (nothing on the flown path reads it). Confirm no
    consumer first (grep actuator_outputs).

R3. Bound the pump() drain per tick + de-alloc _handle  [MEDIUM VALUE, LOW-MEDIUM RISK]
    (a) Cap recv_match iterations per pump call (e.g. <=32) so a backlog burst cannot stall one tick
        unboundedly (defends the spike tail). (b) In the HIGHRES_IMU _handle, the replace() rebuilds 4
        np arrays every msg; the ring already holds (t,accel,gyro) -- the snapshot accel_body/gyro_body
        duplicate the ring newest entry. Consider a lighter update (mutate a small holder / reuse
        arrays) to cut allocation churn. Risk: DroneState immutability is load-bearing for concurrent
        readers (navigator on control thread, recorder on another) -- must preserve snapshot semantics;
        medium risk, needs care. Expected ~1-3 Hz + smaller GC pressure.

R4. Fix _ingest_imu_ring O(64) copy  [CHEAP, LOW RISK, ~0.5-1 Hz + less churn]
    Replace list(ring) + full-scan (navigator.py:790,802) with a bounded read of only samples newer
    than the watermark (iterate the deque from the right until t<=wm, or keep an index). Removes a
    64-tuple alloc + 64-element scan every tick to do ~5-sample work. Low risk (pure refactor of an
    already-guarded path). Modest Hz but cuts allocation churn (helps hyp e).

R5. gc.freeze()/gc.disable() around the loop  [CHEAP, LOW RISK, DE-RISKS hyp e]
    If phase-timing (section 4) confirms GC-correlated spikes, gc.disable() during the tick loop with a
    single gc.collect() at exit (or gc.freeze() after warmup so the big static objects skip gen-0
    sweeps) removes GC pauses from the hot path. Very cheap to try; revert trivially. Only pursue if
    the instrumentation implicates GC.

### Recommended order to hit >=27 Hz under load
1. R1 (worker-thread vp_yaw) -- collapses the worst-tick tail + the 37.8% over-budget class. Biggest
   single lever for the over-budget %; likely the difference between CHOKED and OK on the verdict.
2. R2 (drop ACTUATOR_OUTPUT_STATUS) -- free ~1-2 Hz, trivial, no risk.
3. Land phase-timing (section 4) to MEASURE the residual after R1+R2 and confirm/kill R3/R4/R5 with
   data rather than guessing (this run could not split pump vs nav because work_ms excludes pump).
4. R4 + (if implicated) R5 for the last Hz + tail-smoothing; R3 only if still short of 27 Hz.

### Distribution verdict (headline)
Slowness is REGIME-CORRELATED / BIMODAL, not uniform: loop swings 15.4->28.1 Hz, worst during the
vision-heavy gi0->1 turn. Two co-located mechanisms drive the over-budget ticks:
  - a 101ms vp_yaw spike (the 162ms worst tick) -- fixed by R1;
  - a steady pump()+ESKF-ingest tax (12.6 msg replace()s + ~5 ESKF steps/tick, INVISIBLE to work_ms
    because pump runs outside the work window) that pins the baseline at ~20-23 Hz -- fixed by R2/R3/R4.
The brief "something non-vision eats the loop" is real and is chiefly client.pump()'s per-message
DroneState rebuild (dominated by the useless ACTUATOR_OUTPUT_STATUS + IMU streams), a cost the
existing [loop-rate] accounting structurally hides.
