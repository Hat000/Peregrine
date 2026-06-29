# VQ2 onboard-video frame timing — diagnosis + monitoring (2026-06-29)

You noticed the run videos "surge forward then slow down." That is **real dropped frames**, not just a
playback artifact. This documents what's happening, the monitor I added so every run is checked, and
how to keep future runs consistent.

## What's wrong
The onboard camera is a chunked-JPEG UDP stream: the sim sends ~**28.6 fps** (each frame re-sent ~14×
for redundancy). Our receiver only keeps **~12–13 fps** — **~55% of frames are dropped** — and the loss
is **bursty**: a few frames arrive ~12–35 ms apart, then a **~430 ms hole** where ~10 consecutive
frames are lost, repeating. My old `make_vision_video.py` wrote every *kept* frame at a constant 8 fps,
so each 430 ms hole got squashed into one step → the motion **surges**.

Measured on the 4 attempt-5 runs (`a5_timing_report.txt`, `video_timing_ledger.csv`):

| run | kept/sent | dropped | eff fps | recv dt p95 / max | verdict |
|---|---|---|---|---|---|
| run1 2.0 | 190/460 | 58.7% | 12.3 | 432 / 480 ms | FAIL |
| run2 2.0 | 46/109 | 57.8% | 12.5 | 426 / 444 ms | FAIL |
| run3 3.0 | 41/95 | 56.8% | 12.8 | 432 / 433 ms | FAIL |
| run4 3.0 | 62/136 | 54.4% | 13.6 | 426 / 441 ms | FAIL |

Key tell: the **sender is steady** (sim-clock inter-frame dt p50 = **34.7 ms** = 28.8 fps, every run),
so the jitter is **entirely on our receive side**. The regular ~430 ms stall (not random) means the
Python video thread is being **starved in bursts**, the 4 MB socket buffer overflows, and whole frames
(all their chunks) are lost before they're ever seen — invisible to `meta.json`'s `"dropped"` (that
only counts the recorder's *queue* overflow, which was 0) and even to the receiver's `partials_evicted`
(a frame with zero chunks received never becomes a "partial"). The **only** complete evidence is the
gap in the `frame_id` sequence in `video_index.jsonl` — which is what the monitor reads.

## The monitor (use this on every run)
`scripts/video_timing_report.py` — reads a session's `video_index.jsonl` and reports drop %, effective
fps, inter-frame jitter, longest hole, and surge points, with a PASS/WARN/FAIL verdict (FAIL → exit 1,
so it can gate a post-flight check).

```
# after a flight — check the newest run, write its per-frame timestamps, append to the ledger:
python scripts/video_timing_report.py --latest --csv --ledger handoff/vq2-video-timing-2026-06-29/video_timing_ledger.csv

# or check specific sessions / globs:
python scripts/video_timing_report.py data/runs/2026*_f1
```
- `--csv` writes `frame_timestamps.csv` (frame_id, sim_ms, recv_ms, dt_recv_ms) into each session — the
  per-frame timestamp record you asked for (samples copied here as `*__frame_timestamps.csv`).
- `--ledger` appends one row per run to a CSV so you can watch **drop% and fps stay consistent** across
  runs (the cross-run consistency check). Thresholds: drop <5 % PASS, <20 % WARN, else FAIL.

## Faithful video renderer (no more false surges)
`scripts/render_vision_video.py` plays back on the real `recv_monotonic_ns` clock: each frame is held
for its true duration, so a dropped-frame hole is an honest **freeze with a red "STREAM GAP — held Nms
(~M frames dropped)" banner** (see `example_STREAM_GAP_banner.png`) plus a running "dropped so far"
counter — never a surge. Example: `run4_FAITHFUL_realtime.mp4` (4× slow-mo).

```
python scripts/render_vision_video.py <session_dir> out.mp4 --slowmo 4
```

## Why frames drop, and how to actually reduce it (recommended, not yet done)
Strong hypothesis from the evidence (steady sender, periodic ~400 ms receive-thread stalls): the single
Python **video thread is GIL-starved by the main control loop** and/or does too much on the hot receive
path, so the socket buffer overflows during each stall. Prioritized fixes:

1. **Get JPEG decode off the receive path.** The receiver `cv2.imdecode`s every completed frame inline
   (`jpeg_receiver._ingest`). Recording only needs the raw JPEG bytes (already what `video.bin` stores);
   only the live seeker needs a decoded image. Receive+reassemble+enqueue raw bytes fast, decode lazily
   on the consumer — drains the socket far quicker.
2. **Bigger socket buffer:** raise `SO_RCVBUF` 4 MB → 16–32 MB so a 400 ms stall doesn't overflow.
3. **Stop starving the thread:** the heavy nav/estimator work in the fly loop holds the GIL; either run
   the video receiver in its own **process** (shared-memory/queue the latest frame), or yield more often.
4. **Persist capture health into `meta.json`:** write the receiver's `frames_completed /
   partials_evicted / duplicate_datagrams` + the frame_id-gap drop count at close, so every run
   self-reports (today `"dropped"` is misleading — it's queue-only).

I scoped this change-set as recommendations because it touches the flight-critical capture path; say the
word and I'll implement #1/#2/#4 (low-risk, high-impact) and re-measure with the monitor.

## Files here
- `a5_timing_report.txt` — full monitor output for the 4 a5 runs.
- `video_timing_ledger.csv` — one row per run (cross-run consistency ledger).
- `*__frame_timestamps.csv` — per-frame timestamps for each a5 run.
- `run4_FAITHFUL_realtime.mp4` — faithful real-time render (gaps shown as freezes + banner).
- `example_STREAM_GAP_banner.png` — still showing the gap banner.
- Tools live in the repo: `scripts/video_timing_report.py`, `scripts/render_vision_video.py`.
