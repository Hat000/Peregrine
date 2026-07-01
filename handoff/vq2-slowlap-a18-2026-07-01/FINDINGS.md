# VQ2 A18 — vision load-cut (frame-supply fix) — 2026-07-01

**1-LINE READ (footage-grounded):** The load-cut **un-choked the loop (10 Hz → 28 Hz)** — real win —
but it **did NOT fix the vision frame-supply**: the detector still ran only **4×** in a 44 s flight,
`tsv=infinity` the whole time, `seeker-diag cmds=0 pursuit=0`. The drone flew an **unpiloted beeline**
straight through gate 0 on the single egress command (pilot: "straight line, no veer — one initial
command that happened to go through gate 0") and then coasted → **STALLED, gates=1**. **Nothing works
until the detector is fed proper ~30 Hz frames.** New wire smoking gun: **dup=234275 (26% of datagrams
are duplicates)** + published-frame gaps that **grow progressively to 40 s**.

HEAD flown: `e845699` (flight stack byte-identical to `180009b` — the load-cut commit; memory-only
since). Config: single YOLO `course_L110`, `vq2_case_c`. **Clean GO** this time (`to_go≈0`, no
late-join). ⚠️ First A18 attempt died in `wait_fresh_go` with `ConnectionResetError [WinError 10054]`
because the **sim crashed** at GO (pilot-recovered, relaunched); this is the successful re-fly (a18b).

---

## CONSOLE TAIL (full copy in `console_tail.txt`)

    [detector] GateDetector on device='cuda:0'          <- NOT a CPU fallback
    [loop-rate]  28.0 Hz over 1248 ticks; worst work 259 ms; 0.3% over -> OK     <- UN-CHOKED (was 10 Hz)
    [vision-timing] detect: mean=129.1ms max=184.4ms n=4   floor_height: mean=44.6ms n=1
    [seeker-diag] cmds=0 pursuit=0 none=0 (valid_empty=0)
    flight 1: STALLED  gates=1
    [video-thread] frames=256 max_gap=40356ms gaps>1s=4 gap_sum=102827ms max_pub=0.2ms reconnects=0
                   | wire: datagrams=910046 completed=256 evicted=2837 dup=234275 decode_failed=0 ...
    ring buffer gaps GROW over the flight: 360 -> 582 -> 6136 -> 12506 -> 28296 -> 40356 ms

---

## WHAT THE FIX DID vs DIDN'T

| | result | verdict |
|---|---|---|
| **Loop rate** | 10 Hz → **28 Hz**, worst-work 496 → 259 ms, "OK" not "CHOKED" | ✅ decimating the backstops worked |
| **`detect` cost** | still **129 ms on cuda:0** (max 184 ms); did NOT collapse | ❌ not a CPU-fallback; real GPU cost |
| **Detector fed?** | `detect n=4` in 44 s (~0.09 Hz); `tsv=inf` entire flight | ❌ vision essentially never ran |
| **Seeker pursuit** | `cmds=0 pursuit=0` — never left egress | ❌ no piloting; unpiloted coast |
| **Flight** | beeline through gate 0 (egress ballistic), then STALLED gates=1 | ❌ not progress (gate 0 ≈ luck since A15b) |
| **Video pipeline** | 256 completed / 910046 datagrams; recorder 23 usable frames; gaps → 40 s | ❌ still collapses (progressively) |

**Pilot (ground truth):** "It did pass gate 0, but many runs have — not celebrating. It did a beeline
toward gate 0, through it, and kept going straight — no veer. That straight line = it wasn't being
piloted; one initial command that happened to go straight through gate 0. And remember the sim renders
the world regardless of whether our controller reads the vision frames — a coherent live view is NOT
evidence our vision worked."

---

## THE CORE BLOCKER (pilot's directive): FEED THE DETECTOR ~30 Hz FRAMES

Everything downstream (pursuit, the yaw test, threading gate 1) is blocked on this. Right now the
detector sees ~4 frames per flight and the seeker gets ZERO poses (`tsv=inf`). Two independent things
BOTH have to be fixed to get usable vision:

1. **Frame DELIVERY is broken — new wire evidence:**
   - `datagrams=910046` arrive but only `completed=256` frames assemble (~6/s, collapsing to ~0).
   - **`dup=234275` — 26% of datagrams are DUPLICATES.** Prime suspect: the sim (or the receiver's
     reassembly) is re-sending massively, clogging the pipe so unique frames rarely complete.
   - **Progressive degradation:** published-frame gaps grow 360 ms → 6 s → 12 s → 28 s → **40 s** as
     the flight runs — suggests an accumulating backlog / buffer or a leak, not a steady drop.
   - (`decode_failed=0 size_mismatch=0 bad_chunkmap=0 short=0` — frames that DO assemble are clean;
     the loss is upstream of decode.)
2. **`detect` = 129 ms on cuda:0 caps vision at ~7 Hz even with perfect delivery.** It did NOT collapse
   with the device confirmed CUDA, so it's GPU contention with the co-located sim or a genuinely heavy
   model — not a CPU fallback. To reach ~30 Hz vision this also needs cutting (lighter model / smaller
   imgsz / resolve the GPU sharing).

**⚠️ Possible REGRESSION to check:** in A16 (pre-load-cut, same model) the seeker got `pursuit=17`
(vision fed 17×); A18 (post-load-cut) got `pursuit=0`, `detect n=4`. Did the load-cut also reduce the
DETECTOR call frequency (over-decimation / a frame-freshness gate), on top of the delivery problem?
Worth confirming the detector still runs every available frame, not on a decimated schedule.

---

## RECOMMENDED NEXT STEPS (commander)
1. **Fix frame DELIVERY to the detector (the real blocker).** Chase `dup=234275` (why 26% dupes?) and
   the progressive gap growth (backlog/leak). Goal: ~30 completed frames/s reaching the detector.
2. **Confirm the detector isn't over-decimated** by the load-cut (A16 pursuit=17 → A18 pursuit=0).
3. **Cut `detect` (129 ms → target ≤33 ms)** so vision can keep up once frames flow: GPU-sharing fix
   with the co-located sim, or a lighter model / smaller input.
4. Only THEN re-fly for pursuit / the ff_owns_horizontal hold / the 337c554 yaw test.

---

## ARTIFACTS
- `console_tail.txt` ([detector] + loop-rate + vision-timing + seeker-diag + video-thread + ring dump)
- `nav_estimate.jsonl` (1248 ticks — rate=[0,0,0] & tsv=inf throughout; position is dead-reckon fiction)
- Onboard mp4 near-empty (23 usable frames — itself proof of the collapse). Run dir on ShadowPC:
  `C:\Users\Shadow\Peregrine\data\runs\20260701_204806_vq2_slow_seeker_a18b_f1`
