# VQ2 loop-choke backstop verification (offline replay)

Generated 2026-07-05 11:22:21 on this machine (CPU-only, uncontended).
Config: vp_yaw_min_quality=0.3, vp_yaw_branch_max=35deg, ransac_iters=256; floor_height_min_quality=0.3, floor_height_max_std_m=0.5, floor_grid_cell_m=2.0.

### Run: 20260705_002939_rl_s1_f1

- video frames total: 4238
- frames in armed-flight window: 3603
- frames sampled (replayed): 1200 (effective stride ~3.00)
- frame mean-gray: median 1.34  p95 30.38  max 82.23  (dark warehouse)
- decode failures: 0

**vp_yaw** (estimate_heading, 256 RANSAC iters):
- n_tried=1200  n_would_apply=97  acceptance=8.08%
- rejections: no-estimate=936 (78.0%)  low-quality(<0.3)=140 (11.7%)  branch-reject(>35deg)=27 (2.2%)
- timing ms/call: mean 14.7  p50 11.8  p95 31.1  max 50.8
- quality of non-None estimates: n=264  median 0.288  p95 0.771  max 0.792  (gate needs >= 0.3)
- ACCEPTED yaw innovation |deg|: n=97  median 2.97  mean 8.92  p95 29.42  max 34.91

**floor_height** (estimate_floor_height):
- n_tried=1200  n_would_apply=36  acceptance=3.00%
- rejections: no-estimate=990 (82.5%)  low-quality(<0.3)=37 (3.1%)  high-std(>0.5m)=137 (11.4%)
- timing ms/call: mean 18.3  p50 17.3  p95 27.0  max 43.2
- quality of non-None estimates: n=210  median 0.494  p95 0.900  max 0.900  (gate needs >= 0.3)
- std_m of non-None estimates: median 0.970  p05 0.138  min 0.008m  (gate needs <= 0.5m)
- ACCEPTED heights m: n=36  median 1.62  range [0.13, 9.24]

### Run: 20260705_001321_rl_s1_f1

- video frames total: 4154
- frames in armed-flight window: 3595
- frames sampled (replayed): 1199 (effective stride ~3.00)
- frame mean-gray: median 26.17  p95 49.51  max 155.68  (dark warehouse)
- decode failures: 0

**vp_yaw** (estimate_heading, 256 RANSAC iters):
- n_tried=1199  n_would_apply=903  acceptance=75.31%
- rejections: no-estimate=29 (2.4%)  low-quality(<0.3)=95 (7.9%)  branch-reject(>35deg)=172 (14.3%)
- timing ms/call: mean 31.9  p50 31.5  p95 42.5  max 56.3
- quality of non-None estimates: n=1170  median 0.595  p95 0.779  max 0.843  (gate needs >= 0.3)
- ACCEPTED yaw innovation |deg|: n=903  median 18.07  mean 17.71  p95 32.62  max 34.97

**floor_height** (estimate_floor_height):
- n_tried=1199  n_would_apply=375  acceptance=31.28%
- rejections: no-estimate=291 (24.3%)  low-quality(<0.3)=233 (19.4%)  high-std(>0.5m)=300 (25.0%)
- timing ms/call: mean 26.7  p50 26.1  p95 36.2  max 56.0
- quality of non-None estimates: n=908  median 0.400  p95 0.898  max 0.900  (gate needs >= 0.3)
- std_m of non-None estimates: median 0.372  p05 0.049  min 0.005m  (gate needs <= 0.5m)
- ACCEPTED heights m: n=375  median 1.74  range [0.10, 8.92]

---

## The load-bearing finding: acceptance tracks SCENE BRIGHTNESS, not "always fails"

The two runs look like contradictory evidence (8% vs 75% vp_yaw acceptance) until you look at the
video content. **The clean run's camera goes essentially BLACK (mean-gray ~1.3) from frame ~1130 to
the end of the flight** — the drone stalled/crashed facing an unlit wall and stared at nothing for
~90% of the armed window. The choked run stayed in lit Manhattan structure (mean-gray 20-60)
throughout. Attitudes are sane in both (roll/pitch within +-30 deg, no garbage feeding the modules),
so the difference is purely what the camera saw.

Splitting the CLEAN run's frames by brightness (mean-gray >= 8 = "lit", < 8 = "dark"), same gate
logic, ~600 frames:

| bucket | n | vp_yaw accept | floor accept |
|--------|---|---------------|--------------|
| lit    | 60  | **75.0%** | 18.3% |
| dark   | 541 | **0.6%**  | 1.5%  |

The lit-frame vp_yaw acceptance in the clean run (75.0%) is **identical** to the choked run's overall
75.3% — because the choked run was lit the whole time. So the estimators are NOT intrinsically dead
weight. **vp_yaw acceptance is ~75% whenever the drone is looking at lit warehouse structure, and
collapses to ~0% only when the camera sees black.** The low headline number on the clean run is an
artifact of that specific flight failing early and pointing at a dark wall.

## Per-estimator read

### vp_yaw — evidence does NOT support cutting
- Accepts ~75% of frames with visible structure (both runs agree). This is a live, frequently-firing
  yaw correction, not dead weight.
- Innovation size is REGIME-DEPENDENT (full follow-up: INNOVATION_DISCRIMINATOR.md). In the
  HEALTHY-flight regime (clean run's lit head, vp_yaw pinning normally) the artifact-immune
  quasi-static innovations are ~1.5 deg — small trims, small BECAUSE the pin is working. In the
  DEGRADED regime (choked run = a 2-min crash-loop, collisions=200) they are a REAL ~18 deg median,
  verified artifact-immune (flat across yaw-rate buckets; quasi-static innovations move only
  0.77 deg under +/-300 ms mapping shifts; a +/-1.5 s lag sweep never collapses them) — vp_yaw was
  actively fighting genuine yaw-estimate error. Yaw is inertially unobservable on the VQ2 wire;
  wherever pinning stops (dark stretch / starved cadence) the error visibly grows to tens of
  degrees. Cutting it removes the only map-free yaw anchor.
  CAVEAT on the pooled tables above: the per-run "ACCEPTED yaw innovation" rows mix in a
  time-misalignment artifact on TURNING frames (the offline end-anchor clock mapping is +/- a few
  100 ms and yaw sweeps up to ~86 deg/s in turns). Trust the quasi-static split in
  INNOVATION_DISCRIMINATOR.md, not the pooled innovation numbers.
- The branch-reject bucket is meaningful (14% choked / 2% clean): when a real heading is available but
  >35 deg from the gyro estimate it is (correctly) refused — evidence the disambiguation is active.
- Timing is the real cost: mean ~15 ms on dark frames (early-out, few segments), but **mean ~32 ms /
  p95 43 ms on lit frames** uncontended. Under GPU/CPU contention in-flight this is worse. So vp_yaw
  IS a genuine loop-time cost when it matters most (lit = flying) — but it is EARNING that cost.

### floor_height — weaker case; a defensible cut candidate, but not "dead"
- Lit-frame acceptance ~18-31%, i.e. rejected ~70-80% of the time even with structure. The dominant
  killers are no-estimate (too few floor grid lines) and **high-std (>0.5 m): 11-25% of tries** — the
  near-horizon ill-conditioning the module's own docstring warns about, plus the placeholder
  grid_cell_m=2.0 (uncalibrated -> absolute heights are suspect anyway; accepted heights span
  0.1-9.2 m, physically implausible spread).
- It fires far less often than vp_yaw and its metric anchor is admittedly un-calibrated, so its z
  corrections are low-confidence. Timing mean ~18-27 ms.
- Read: floor_height is the more plausible cut of the two on a pure cost/value basis, BUT it is not
  "almost never accepts" (31% on the lit run). Before cutting, note it is the ONLY map-free z anchor
  during no-gate/wrong-map stretches (z is also inertially unobservable here). Recommend: gate it
  behind gate-visibility (only pay the compute when the primary gate-relative z fix is unavailable),
  or calibrate grid_cell_m and re-measure, rather than an unconditional cut.

## Surprises / caveats to flag
1. **The original hypothesis ("nearly always fail the gates -> dead weight") is refuted for vp_yaw and
   overstated for floor_height.** Both work well on lit structure; the clean run's low number is a
   dark-scene artifact, not a property of the estimators.
2. **The clean run is a degenerate sample for this question** — it crashed early and filmed a black
   wall. The choked run (lit throughout) is the representative "actually flying" case, and there
   vp_yaw is a 75%-hit, ~18-deg-correction workhorse.
3. Cost-of-compute is real and concentrated on lit frames (when flying): vp_yaw p95 ~43 ms, floor
   p95 ~36 ms uncontended. If the goal is loop headroom, the higher-value move is likely **cadence /
   threading** (move these off the hot loop, or drop vp_yaw to every-Nth with N larger) rather than
   deleting a working yaw anchor. floor_height is the more defensible outright cut.

## Method notes
- CPU-only (CUDA_VISIBLE_DEVICES=""), no torch/YOLO imported; classical OpenCV (cv2 4.13.0, LSD on).
- Gate logic replicated verbatim from navigator._apply_vp_yaw / _apply_floor_height; ESKF/KF update
  itself skipped (only would-apply vs reject + reason needed). Config = NavConfig defaults.
- Time mapping: nav_estimate (sim-boot clock, armed flight only ~120s) and video_index (epoch clock,
  full recording ~141s) are on different clocks but both END at disarm. Each video frame is mapped to
  nav-time by seconds-from-the-end (same end-anchor as handoff/tools/render_cmd_indicator.py), then
  fed the time-nearest nav roll/pitch/yaw. Pre-GO waiting-room frames dropped. Approximate (+-tail);
  roll/pitch vary slowly relative to the tail so this is adequate for gate-acceptance statistics.
- Sampled 1200 / 1199 armed-window frames per run (stride 3), n well above the 500 floor.
- Script: handoff/vq2_loopchoke_verification_2026-07-05/replay_backstop_gates.py (re-runnable).
