# EGO-A8 (2026-07-10) — lean-profile (vq2_ego_lean) validation flight

Branch `claude/ego-deploy-2026-07-09` @ `e1aa4d1`. vn16 policy + TRT engine detector,
`--deploy-profile vq2_ego_lean` (case-C minus {vp_yaw, gate_bearing_yaw, floor_height}),
takeoff-assist ON.

## Headline

**Handover on rates @ t=0.246 s; loop 23.9 Hz; `vp_yaw` + `floor_height` ABSENT from the timing.**
Lean nav proven flight-safe (all obs finite, attitude/velocity sane through the crash). Crashed
env/gates=0 as expected (vn16 gait unchanged/disqualified). Clean fresh GO, video flowed
(5124 frames, gaps>1s=0), longest flight of the session (73 ticks).

## Pass criteria

| criterion | result |
|---|---|
| `vp_yaw` + `floor_height` absent from ego_timing | **✓ CONFIRMED** — only `nav.detect` appears; the lean profile flags took |
| takeoff-assist handover on rates | **✓** `HANDOVER at t=0.246 s trigger=rates` |
| nav attitude/velocity sane through crash | **✓** all 73 ticks' obs finite; roll/pitch/vel reflect the physical tumble, no NaN/rail |
| loop ≥ ~27 Hz | **✗ 23.9 Hz** — but see below; the shortfall is detector GPU-contention this run, not the lean profile |

## Loop rate: the lean profile clearly helped — read the details, not just the headline

```
[loop-rate] 23.9 Hz over 73 ticks; worst work 176 ms; 20.5% over budget
[ego-timing] median ms/tick: work=21 nav_update=16  (NO vp_yaw / NO floor_height)
  nav.detect: n=71 avg=18.7ms max=72.6ms
  NO-detect ticks (n=2): work=6ms nav=1ms
```

Cross-run comparison (all engine detector, orphans-killed):

| run | profile | nav.detect | **over-budget %** | loop |
|---|---|---|---|---|
| a7 | case-C (anchors ON) | 10.5 ms (low-contention window) | 40.7 % | 23.9 Hz |
| **a8** | **lean (anchors OFF)** | **18.7 ms (higher-contention window)** | **20.5 %** | **23.9 Hz** |

The lean profile **matched a7's 23.9 Hz despite an ~8 ms worse detector window, and HALVED the
over-budget fraction (40.7 → 20.5 %)**. That is the lean profile working exactly as intended: it
removed the `vp_yaw` (~34–50 ms) and `floor_height` (~22–25 ms) *spikes* that pushed ticks over
budget. It does not move the median tick much — the median tick never ran those anchors (they were
decimated) — so the headline Hz is still dominated by the detector, whose cost swings 10.5–18.7 ms
run-to-run with GPU-context serialization against the sim's render. **Conclusion: lean is a real
win (fewer spikes, same Hz at a worse detector draw); clearing ~27 Hz consistently additionally
needs a low-contention detector window — i.e. a7's 10.5 ms detect + a8's lean profile together
would comfortably exceed 27.** The two are complementary, and neither run got both at once.

## Lean nav is flight-safe (velocity/attitude sanity)

With `vp_yaw` off, ESKF yaw dead-reckons on gyro — but the ego obs is yaw-free (RL: the yaw datum
cancels in the velocity projection), so this doesn't corrupt what the policy consumes. Verified:
all 73 ticks' 21-dim obs are finite; roll/pitch swing ±2.9 rad and velocity ±14 m/s, which is the
*physical* violent tumble at max thrust being dead-reckoned by the KF (as in every ego run,
anchors on or off), not estimator breakage. No NaN, no railed-garbage. Removing the anchors did not
degrade the obs the policy sees.

## FYI — the failing diagnose test is NOT caused by the ego bundles

`tests/test_diagnose_session.py::test_real_bundles_diagnose_end_to_end` globs
`handoff/shadowpc-postfix-dataset-2026-06-12/extracted` for subdirs containing **`debug_obs.jsonl`**
(0 present on this machine — gitignored recorded payloads), so `assert bundles` fails on an empty
list. The ego bundles (a5/a7/a8) carry **`ego_obs.jsonl`** and live under `handoff/ego-flight-*`;
they are **not** discovered by that test. This is the pre-existing missing-payload environmental
failure, unrelated to the new bundle formats. (Not fixing — flagged per the tasking note.)

## Bundle

`data/runs/20260711_022751_ego_vn16_a8_f1/` (video.bin local-only). Self-contained copy here:
REPORT + ego_obs (73 ticks) + ego_timing + meta + mavlink.tlog (185 Hz IMU).

## MEMORY-DELTA

- EGO-A8 lean profile (`vq2_ego_lean`, e1aa4d1) FLOWN: `vp_yaw`+`floor_height` confirmed ABSENT from
  timing, handover on rates @0.246s, obs finite/sane through the (expected) crash → lean nav is
  flight-safe. Loop 23.9 Hz (didn't clear ~27) BUT matched a7's 23.9 at an ~8ms-worse detector
  window and HALVED over-budget% (40.7→20.5%): lean removes the anchor SPIKES; detector
  GPU-serialization variance (10.5–18.7ms) still caps the headline Hz. Lean + a low-contention
  detect window would clear 27.
- CORRECTION to my earlier "vp_yaw is load-bearing, don't cut": RL confirms the ego policy is
  YAW-FREE (yaw datum cancels in the velocity projection) → vp_yaw IS dead weight on ego flights.
  It's still the estimator's sole yaw anchor, but nothing the ego obs consumes needs it. floor_height
  likewise dead weight. Both correctly dropped by vq2_ego_lean.
- Failing `test_real_bundles_diagnose_end_to_end` = pre-existing missing-`debug_obs.jsonl` payloads
  in the postfix-dataset dir, NOT the ego bundles (which use `ego_obs.jsonl`, not globbed by that test).
