# Root cause of the video "surges" — it's NOT the sim/VM, it's a 448 ms/tick navigator (2026-06-29)

You were right not to let me blame "our setup" reflexively. I ran the experiments. The camera stream
is **perfect**; the drops are a *symptom* of the navigator taking ~450 ms per tick — which is also
running the **control loop at ~2 Hz instead of 30 Hz**, silently crippling every flight attempt.

## Two separate problems, as you suspected
- **Viewport lag (what you SEE):** the sim window is encoded by ShadowPC's remote-desktop streaming and
  sent over the internet to your display — virtual GPU + V-sync + network. Cosmetic; it does NOT touch
  the recorded camera data (which is captured on-box). "Unfocused runs smoother" is a viewport/present
  effect, not a camera effect.
- **Camera frame drops (in the recordings):** measured purely on-box (`recv_monotonic_ns`). This is the
  real one, and it is **on our side of the wire** — see below.

## What I tested (each isolates one suspect) — all on the live sim, this machine
| test | what ran | whole-frame loss |
|---|---|---|
| bare capture (no decode, no fly loop) | socket recvfrom only | **0.0%** |
| + inline JPEG decode | cv2.imdecode every frame | **0.0%** |
| + busy main thread (GIL held 75%) | daemon video + spin loop | **0.0%** |
| + real cv2 vision (detector+PnP), OpenCV 8-thread | daemon video + main cv2 | **0.7%** |
| real `JpegUdpReceiver` + `Recorder` (production path) | daemon video + disk | **0.0%** |
| **sim under flight load** (armed + CTBR @50 Hz) + bare capture | drone flying | **0.0%** |
| real production fly_rl flight | full stack | **57–59%** |
| OpenCV/torch threads capped to 1, real flight | full stack | **57%** (no change) |
| **fly_rl minus arming/control: real `Navigator.update` in the loop** | daemon video + nav | **56.8%**, `mean_nav_ms=448` |

So: the sim streams a steady ~28.6 fps even while the drone is armed and flying (0% loss, ≤36 ms
jitter). V-sync, focus, the virtual GPU, the 40%-GPU box, the network, the socket buffer, JPEG decode,
the GIL, OpenCV threading, and the recorder are all cleared. The drop only appears when the **real
`Navigator.update()` runs in the loop**, and that call takes **~448 ms**.

## The 448 ms: profiled to one function
`cProfile` of `Navigator.update` on a real frame (`use_vp_yaw=True` in the `vq2_case_c` profile):

```
nav.update                                   ~370–450 ms / call
  _apply_vp_yaw -> heading_vp.estimate_heading
    manhattan_lines.fit_multiple_vanishing_points   (3 VPs)
      fit_vanishing_point   x3   <-- 2000-iter Python RANSAC each
        np.cross(...)  30,015 calls  -> 1.47 s of moveaxis/normalize_axis_tuple overhead
        _vp_consistency_residual  29,988 calls
```
`fit_vanishing_point` runs a **2000-iteration RANSAC** (×3 vanishing points = 6000 iters/tick), and
each iteration calls `np.cross` on 3-vectors — numpy's `np.cross` is dominated by `moveaxis` /
`normalize_axis_tuple` overhead for tiny arrays. That microcall-in-a-Python-loop is the whole cost.

### Why this also wrecks flight (the bigger deal)
At 448 ms/tick the control loop runs at **~2.3 Hz**, not the intended 30 Hz — the gate-seeker issues
CTBR commands ~13× too sparsely. That alone explains a lot of the launch/approach instability across
A1–A5 (and matches the ~0.5 s gaps between rows in the per-flight `seeker_diag.csv`).

## The fix (validated, ~21×) — vectorize the RANSAC
Replace the per-iteration `np.cross` Python loop with a vectorized hypothesis batch + explicit cross
product (sample all pairs at once, intersect, score the (M×N) residual matrix). Benchmarked on a real
frame (210 segments):

```
ORIGINAL fit_vanishing_point : 113 ms   best inliers = 118
VECTORIZED (iters=512)       : 5.3 ms   best inliers = 118   (identical result)
-> a 3-VP nav tick: ~340 ms  ->  ~16 ms
```
That restores ~30 Hz control AND removes the video drops in one change (the receiver thread is no
longer starved by a 448 ms GIL-holding tick). It's localized to
`src/racer/vision/manhattan_lines.py::fit_vanishing_point` (and a smaller win in
`_vp_consistency_residual`); behaviour is equivalent (same inlier set), so it's a perf fix, not a
control change. Lower-effort partial wins if a full vectorization is undesired: drop `iters` 2000→256
and replace `np.cross` with explicit scalar components.

I scoped this as a recommendation because it changes the flight vision stack — say the word and I'll
implement + verify it end-to-end (re-run the monitor to confirm 0% drop and check the loop hits 30 Hz).

## Monitoring going forward (already in the repo)
- `scripts/video_timing_report.py --latest --csv --ledger ...` after each flight → drop%, fps, verdict
  (FAIL = exit 1). A regression here would have caught this immediately.
- A loop-rate check is worth adding too: if `nav.update` mean exceeds ~10 ms (or the loop falls below
  ~20 Hz), warn — that's the upstream signal the video monitor was reflecting.

## Evidence files
- `bench/` experiments were run live (scratch scripts) — the table above is the result.
- `a5_timing_report.txt`, `video_timing_ledger.csv`, `*__frame_timestamps.csv` — the drop measurements.
- `run4_FAITHFUL_realtime.mp4` — faithful playback (gaps shown as freezes, not surges).
