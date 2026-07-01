# VQ2 A17 — video-thread freeze diagnosis (instrument-only) — 2026-07-01

**1-LINE READ:** Freeze REPRODUCED (max_gap 1051ms, gaps>1s=1, gap_sum 9129ms). By the strict
decision tree the signature is **BRANCH D (unexpected — flag for re-exam):** small pub (0.1ms), NO
reconnects/idle/exceptions, the wire snapshot never fired, and frames keep **completing to the gap
edge** (fid increments through every gap). **BUT the NEW `[vision-timing]` line hands us the likely
ROOT the tree didn't anticipate: ~256 ms of synchronous vision compute PER tick (detect 150 ms +
vp_yaw 67 ms + floor_height 37 ms) starving the video-receiver thread — and detect at 150 ms is ~7×
the ~21 ms expected for single YOLO.** Indicated fix = the BRANCH-A remedy anyway (cut GPU/CV load).

HEAD flown: `307a126` (flight stack byte-identical to `a32938a` — the instrumentation commit; only a
memory commit since). Config identical to A16 (single YOLO course_L110, vq2_case_c). Instrument-only,
no behavior change. **NOTE:** armed via **LATE-JOIN `to_go=-2.51s`** — the race went GO ~2.5 s before
fly_rl armed (I did not manually GO; it caught a GO already in progress). Mild (near-start), but the
armed flight was only **22 ticks (~2.2 s)** before HARD COLLISION, gates=0.

---

## THE INSTRUMENTATION (console tail — full copy in `console_tail.txt`)

    [loop-rate]  10.0 Hz over 22 ticks (target 30); worst work 496 ms; 18.2% over -> CHOKED
    [vision-timing] detect: mean=149.7ms max=157.1ms n=3  vp_yaw: mean=66.9ms max=71.1ms n=3  floor_height: mean=36.7ms max=39.8ms n=3
    [vision-timing] worst tick sum=255.8ms breakdown: detect=147.8ms, vp_yaw=69.7ms, floor_height=38.4ms
    [seeker-diag] cmds=4 pursuit=4 none=0 (valid_empty=0)
    [video-thread] frames=190 max_gap=1051ms@fid=277 gaps>200ms=4 gaps>1s=1 gap_sum=9129ms max_pub=0.1ms reconnects=0(idle=0,exc=0) | wire: (no rx.metrics snapshot -- receiver never hit an idle timeout)
    [video-thread] ring buffer:
      ts=7219.861 fid=186 gap=401ms  pub=0.0ms exc=None
      ts=7220.575 fid=208 gap=714ms  pub=0.0ms exc=None
      ts=7221.507 fid=242 gap=932ms  pub=0.0ms exc=None
      ts=7222.558 fid=277 gap=1051ms pub=0.0ms exc=None

---

## CLASSIFICATION BY THE DECISION TREE

| branch | criteria | match? |
|---|---|---|
| **A** sim stopped emitting UDP | gap>1s + pub small **+ datagrams flat / reconnects(idle)>0** | ✗ reconnects=0, idle=0; **no wire snapshot to confirm datagrams-flat** |
| **B** packet loss shredding frames | gap>1s + pub small **+ datagrams arriving while completed stalled + evicted spike** | ~ datagrams-arriving *inferred* (receiver never idle) but **evicted/decode_failed UNCONFIRMED** (no snapshot) |
| **C** silent reconnect/exception loop | ring exc non-empty + reconnects(exc)>0 | ✗ exc=None, reconnects=0 |
| **D** unexpected — re-examine | pub small, **no reconnect + datagrams arriving + frames completing to the gap edge** | ✅ **best match** — fid increments (186→208→242→277) right through the growing gaps |

**Why the A/B discriminator is missing:** the wire snapshot (datagrams / completed / evicted /
decode_failed) only prints when the **receiver hits an idle timeout**, and it **never did** — i.e. the
receiver was never idle long enough → datagrams were very likely **still arriving** during the 1051ms
published-frame gap. That rules against a clean "sim stopped emitting" (A) and toward "frames not
completing while data flows" — but without the evicted/decode counters we cannot nail B vs D.

---

## THE REAL SIGNAL — `[vision-timing]` (new this build)

Per **vision tick**: **detect 150 ms + vp_yaw 67 ms + floor_height 37 ms ≈ 256 ms.** This is the
choke (loop 10 Hz, worst-work 496 ms) and — the key insight — it's synchronous work on the MAIN loop
that starves the video-receiver thread (GIL/GPU/CPU contention): while a 256 ms tick runs, the
receiver can't drain the UDP socket, frames don't complete, and the published-frame gaps grow
(401→714→932→1051 ms, in lockstep). pub=0.1 ms because when a frame DOES complete it publishes
instantly — the video thread itself is not the bottleneck; it's **starved**.

- **`detect` = 150 ms is ~7× the ~21 ms expected for single YOLO** → prime suspect: GPU contention
  with the co-located sim, a CPU-fallback inference, or a double-detect. Investigate first.
- **`vp_yaw` (67 ms) + `floor_height` (37 ms) = the CV backstops** (the `heading_vp` /
  `manhattan_lines` per-frame steps flagged in 5b3c131) — ~104 ms/tick, confirmed expensive.

---

## RECOMMENDED NEXT STEPS (for the commander)

1. **Cut the vision compute (the BRANCH-A remedy — indicated regardless of A/B/D):**
   (a) chase the 150 ms `detect` (≫21 ms): GPU contention / CPU fallback / double-detect;
   (b) decimate or share the `vp_yaw` + `floor_height` backstops (run every N ticks, not every tick).
   Cutting ~256 ms/tick should both un-choke the loop AND stop starving the receiver → freeze gone.
2. **Make the wire snapshot unconditional at exit (instrumentation gap).** It only fires on receiver
   idle-timeout, which didn't happen — so we never saw datagrams/completed/evicted and can't split
   B vs D. Snapshot rx.metrics at flight-exit unconditionally (or lower the idle threshold) so the
   NEXT run definitively confirms whether datagrams keep arriving (→ receiver/buffer starvation, B/D)
   vs go flat (→ sim stopped emitting, A).

---

## ARTIFACTS
- `console_tail.txt` (GO line + loop-rate + vision-timing + seeker-diag + video-thread + ring dump)
- `nav_estimate.jsonl` (22 ticks)
- Run dir on ShadowPC: `C:\Users\Shadow\Peregrine\data\runs\20260701_181739_vq2_slow_seeker_a17_f1`
