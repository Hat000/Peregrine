# VQ2 A19 (a/b/c) — UDP drain-to-empty (frame-supply fix) — 2026-07-01

**1-LINE READ (footage-grounded):** The drain fix **RESTORED the detector frame-supply** — the seeker
is fed and actively pursuing (A19c `cmds=64 pursuit=22`, vs A18 `0/0`). First **fully-observed
active-pursuit flight** (A19c, clean 30fps recording GO→crash). **NEW dominant lever discovered: GPU
CONTENTION on `detect`** — cost swings **32 ms (un-contended) ↔ 234 ms (contended)**, and it drives
the loop rate, the freeze, AND the recorder cutout together. **NEW primary flight blocker (pilot's
eyes): the VERTICAL/alt control overshoots UP — thrust ~0.6 climbs the drone OVER the gate, dropping
it out of view → crash.** `ff_owns_horizontal` cured the *horizontal* nose-up; *vertical* is now the
problem. Yaw-sign test (337c554) still inconclusive (gate lost to the climb before a clean test).

HEAD flown: `c325585` (flight stack == `db4e016` drain-to-empty). Config: single YOLO `course_L110`,
`vq2_case_c`.

---

## THREE FLIGHTS — GPU CONTENTION IS THE MASTER VARIABLE

| | pre-GO wait | `detect` mean | loop | flight | seeker | recorder |
|---|---|---|---|---|---|---|
| **A19** | ~31 s (long) | **234 ms** | 3.1 Hz | 11 tk | cmds=5 pursuit=5 | **cut out ~1s into flight** |
| **A19b** | ~1 s | 150 ms | 4.2 Hz | 13 tk | cmds=7 pursuit=7 | caught start of movement, then cut out |
| **A19c** | ~1 s (clean protocol) | **31.9 ms** | **14.2 Hz** | **76 tk** | **cmds=64 pursuit=22** | **FULL 30fps GO→crash ✅** |

**The pattern:** low GPU load → `detect` ≈ 32 ms → loop un-chokes to 14 Hz → dense control (64 cmds) +
the recorder has headroom → **full flight captured**. High GPU load (long waiting-room accumulation of
the ~90% duplicate datagrams) → `detect` balloons to 234 ms → loop chokes to 3 Hz + recorder starves →
**cutout**. So the freeze, the loop choke, and the recorder cutout are ONE root: GPU contention on the
co-located sim+detector. `detect`=cuda:0 confirmed (not CPU-fallback); the 234 ms is contention, not
the model's true cost (~32 ms).

**Frame DELIVERY to the detector is fixed** in all three (drain-to-empty): `gaps>1s=0`, `max_gap`
52–166 ms, `evicted` 0–5. The A18 collapse (dup swamping the receiver) is gone. `dup` is still ~90%
(sim re-sends massively) but the drain receiver now completes frames regardless.

---

## A19c — FIRST FULLY-OBSERVED ACTIVE FLIGHT (pilot's eyes = ground truth)

**Pilot:** "Left the start gate, dipped a little (standard egress). Pitched up just a little as
commands rolled in, still flying forward. Then **throttle picked up to ~0.6 and the drone flew UP and
the gate went down and out of view** — odd, because pitch wasn't very forward and the gate was flying
out of view, yet it kept going up. Once the gate was out of view, it yawed left, couldn't find
anything, and crashed."

Nav arc corroborates (attitude/commands trustworthy; position is dead-reckon fiction):
- ticks 0–18: egress −17.8° → **eases to −6.6° nose-down hold** (ff_owns_horizontal ✅), mild forward.
- tick 24: acquires gate on the **RIGHT** (`yaw_des +6.6°`).
- **thrust bursts to 0.60 while pitch is only −2 to −6°** (near-level) → thrust goes mostly VERTICAL →
  the drone CLIMBS over the gate and loses it.
- ticks 36–75: with the gate lost/above, `yawRate` pins at −1.50 and the seeker flails
  (estimator yaw spins +16→+165→wrap — this is post-gate-loss SEARCHING, not a clean yaw test; pilot
  saw a single left yaw-search, so the estimator "360° spin" is defendant, not truth).

---

## NEXT-FIX LEADS

1. **[PRIMARY] Fix the VERTICAL / alt-align overshoot.** The seeker commands thrust ~0.6 (climb) while
   pitch is near-level, so the drone rises OVER the gate and loses it (pilot-confirmed). The vertical
   target should hold/track the gate's height, not climb above it. Suspect the vertical-align climb
   rate or the alt-hold target (the `velocity_ned[2]` path the ff_owns_horizontal fix deliberately
   left feeding alt-hold). This is now the flight-ending blocker.
2. **[A20 — still valuable] Cut `detect` + isolate it from GPU contention.** The imgsz/half cut helps,
   but note `detect` swings 32↔234 ms with sim GPU load — the contention itself is a big lever. Lower
   baseline cost AND reduce sim-detector GPU sharing so the loop stays ~14 Hz+ every flight, not just
   on lucky low-load runs.
3. **[RECORDER] The cutout is GPU-contention-driven, not a fixed bug.** The clean protocol (below)
   mitigates it (A19c full capture). A robust fix (decouple the disk recorder from GPU/compute load)
   would give consistent post-hoc video, but it's not flight-critical (the detector feed is separate
   and fixed).
4. **[YAW — still untested] 337c554 yaw-sign** remains unconfirmed: the drone loses the gate to the
   vertical climb before a clean yaw-toward-off-axis-gate test. Reachable once #1 keeps it on the gate.

---

## OPERATOR PROTOCOL (new — pilot-designed, use every flight)
Minimizes pre-GO recording AND GPU contention → best flight + full recording:
1. Pilot leaves the sim at the **main menu**.
2. Launch fly_rl FIRST; wait for `>>> Waiting PASSIVELY`.
3. **Down×2 → screenshot-verify "R2-TRAINING (VQ2)" highlighted** (never fly R1/VQ1) → Enter (waiting
   room) → Enter (GO). GO lands seconds after fly_rl is ready → short pre-GO wait.

---

## ARTIFACTS
- `console_tails.txt` (A19c + A19 + A19b: detector/loop-rate/vision-timing/seeker-diag/video-thread)
- `nav_estimate_a19c.jsonl` (76 ticks)
- A19c full-flight video (30fps, GO→crash, command overlay) on ShadowPC:
  `C:\Users\Shadow\Downloads\a19c_flight.mp4` (also in the run dir
  `data\runs\20260701_224640_vq2_slow_seeker_a19c_f1\onboard_go.mp4`)
- Render tool `handoff/tools/render_cmd_indicator.py` (on branch vq2-slowlap-a17; `--from-go` default).
