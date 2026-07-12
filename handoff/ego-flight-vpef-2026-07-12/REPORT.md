# EGO vpef flights (2026-07-12) — first no-GT, despin-trained multi-gate candidates on the wire

Branch `claude/ego-deploy-2026-07-09` @ `b6e9b5c` (≡ the pinned `6838713` behavior — the only commit
on top is `--ego-kp-persist`, `default 0 == byte-identical`, and it was **not** passed). Profile
`vq2_ego_lean`, negreal42 fp16 384x640 **TRT engine** detector, takeoff-assist ON, `--rate 30`,
**`--ego-yaw-clamp 0.7`** (mandatory despin mirror). Two ckpts from release
`ego-ckpts-vpef-2026-07-12` (both sha256-verified, both load clean at obs_dim=21):

- **vpefwh2** (PRIMARY, `28edf24c9bf0`)
- **vpeffs0** (behavioral, `0132d7537af8`)

## Headline

Both no-GT despin ckpts fly the **same, and much healthier, signature than vn16**: tame near-hover
thrust (**no 3.76 g punch**), continuous gate tracking, a **forward+climbing approach that flies
into gate 0**, a **bounded yaw dither (no corkscrew)**, and a **sane leveler with no GT**. Both
ended in **gate contact → INVALID** scored runs (`gates=0`). The despin *clamp* works; the sim's
"~0.4 rad/s level" promise did **not** transfer to the wire. Neither flight reached the 30 Hz bar
(27.6 and 20.9 Hz) — the second run is heavily dt-OOD and its rougher behavior is partly the choke.

## Per-flight

| | **vpefwh2 (PRIMARY)** | **vpeffs0 (behavioral)** |
|---|---|---|
| Validity | **❌ INVALID — gate contact** (2× COLLISION id=1001) | **❌ INVALID — gate contact** (11× COLLISION id=1001) |
| Outcome | CRASH, gates=0, into gate 0 @ 1.74 s (50 ticks) | CRASH, gates=0, into gate 0 @ ~2.1 s (56 ticks, span 2.59 s) |
| Takeoff / handover | assist ACTIVE; **HANDOVER 0.159 s trigger=rates** | assist ACTIVE; **HANDOVER 0.230 s trigger=rates** |
| Thrust (normed) | near hover, max **1.46 g** | mean 1.31 g, max **1.75 g** |
| Detection | **41/50 fresh levers, 0% blackout**, area 0.8–1.0 | 28/56 levers, **38% blackout** (choke-driven) |
| Spin — net heading | **+19.6°** over the flight | **−33.1°** over the flight |
| Spin — measured yaw | mean \|ω_z\| 1.35, max 2.56 rad/s | mean \|ω_z\| 0.78, max 1.89 rad/s |
| Yaw cmd at ±0.70 clamp | **31/50 ticks** (rails, alternating) | 16/56 ticks |
| Attitude (leveler) | roll ±0.35, pitch −0.31…−0.81 rad — sane, no NaN | roll −0.59…**+1.06**, pitch −0.12…−1.15 rad — bigger swing |
| Loop rate | **27.6 Hz**, worst 87 ms, **6% over budget**, detect 14.9 ms | **20.9 Hz**, worst 133 ms, **66% over budget**, detect 22.2 ms |
| Peak climb / \|vel\| | +5.8 / 10.6 m/s | +7.1 m/s / *(post-impact KF spike to 91 m/s — artifact, not flight speed)* |

## Findings

**1. Despin: the clamp holds, the level-flight promise does not.** Neither ckpt corkscrews — net
integrated heading is **< 35°** over each whole flight, and the measured yaw rate **alternates sign**
(19 flips/50 on the primary). But the policy **rails its yaw command at the ±0.70 clamp** (primary
31/50 ticks), and through the sim's ~2.5× rate gain the plant answers with a **±1.9–2.6 rad/s yaw
dither**. So "non-spin" holds in the sense of *no runaway rotation*, but the sim DET's "~0.4 rad/s ≈
level" did **not** transfer — there is a real, bounded yaw limit-cycle on the wire that the clamp
caps but does not remove. This is the S17 yaw-dither class, now clamp-bounded.

**2. Thrust is solved-adjacent — the vn16 failure mode is gone.** Both ckpts hold near-hover
collective (~1.0–1.3 g, peaks < 1.8 g). Neither reproduces vn16's punch-to-3.76 g-then-tumble.
Handover fires cleanly on measured rates on both (0.159 s / 0.230 s). Takeoff-assist works.

**3. Failure mode = over-climbing approach into the gate structure.** Both fly *forward and up*
(peak climb +5.8 / +7.1 m/s) while tracking gate 0 with high visible-area, then contact the gate —
i.e. they rise/press into the frame rather than thread the opening. This is **forward progress that
misses on centering**, not a tumble into the environment (vn16). A qualitatively better failure.

**4. The no-GT leveler is flight-safe.** With `vq2_ego_lean` (no vp_yaw / floor_height anchors) and
**no GT attitude ever**, the ESKF-leveler + velocity-KF obs stayed finite and physical on both runs
(roll/pitch reflect the real motion; no NaN, no railed garbage). The obs contract the policy
consumes is intact without ground truth.

**5. Loop rate: 30 Hz is not reliably reachable on this hardware.** Same setup, 0 orphans, back to
back: **27.6 Hz** (low-contention window, detect 14.9 ms) then **20.9 Hz** (high-contention window,
detect 22.2 ms, 66% over budget). That swing is pure GPU-context serialization between the sim's
render and the detector's CUDA on the one passthrough RTX 2000 — the session's standing finding.
**The ≥30 Hz hard requirement was met by neither flight.** The primary is only mildly dt-OOD (~8%
slow); **vpeffs0 at 20.9 Hz is heavily dt-OOD (~30% slow)**, so its rougher run (bigger roll
excursion, 11 gate contacts, 38% blackout) is **confounded by the choke** and should not be read as
purely the ckpt. The queued dt-hardening retrain is the right fix; flagged, not forced.

**6. Primary vs behavioral.** vpefwh2 is the cleaner candidate: faster loop, 0% blackout, tighter
attitude, fewer contacts. vpeffs0 actually **dithers yaw *less*** (mean \|ω_z\| 0.78 vs 1.35;
16/56 vs 31/50 at the clamp) — consistent with "the longer training run drifted it down/softer" —
but crashed messier. How much of the messier crash is the ckpt vs the 20.9 Hz choke is unresolved;
a low-contention re-fly of vpeffs0 would separate them (offered, not taken — escape-hatch: don't
force degraded reruns).

## Bundle

`handoff/ego-flight-vpef-2026-07-12/{vpefwh2,vpeffs0}/` — each: `ego_obs.jsonl`, `ego_timing.jsonl`,
`mavlink.tlog` (185 Hz IMU + collisions), `meta.json`, raw `console.log`. Video is local-only
(`data/runs/20260712_062431_ego_vpefwh2_2026-07-12_f1`, `..._063357_ego_vpeffs0_..._f1`) — pull
frames / render a clip on request.

## MEMORY-DELTA

- **vpef no-GT despin ckpts FLOWN 2026-07-12** (vpefwh2 primary / vpeffs0) @ `b6e9b5c` (≡6838713),
  `--ego-yaw-clamp 0.7` + `vq2_ego_lean` + negreal42 engine. Both: **tame thrust (~1.0–1.3 g, NO
  vn16 3.76 g punch)**, tracked gate 0, flew **forward+climbing INTO gate 0 → gate contact
  (INVALID), gates=0**. Large behavioral step up from vn16 (hits the gate it aims at, not the env;
  leveler sane, no tumble).
- **DESPIN clamp WORKS (no corkscrew — net heading < 35° both)**, but the sim's "~0.4 rad/s level"
  did **NOT** transfer: policy **rails yaw cmd at ±0.70** (fwh2 31/50, ffs0 16/56) → plant dithers
  **±1.9–2.6 rad/s** (wire ~2.5× gain). Bounded dither, not spin. (S17 dither, clamp-capped.)
- **Failure mode = over-climb approach** (peak +5.8/+7.1 m/s up) presses into the gate structure,
  doesn't thread the opening.
- **LOOP: neither hit 30.** fwh2 **27.6 Hz** (0% blackout, detect 14.9 ms, best ego loop yet);
  ffs0 **20.9 Hz** (38% blackout, detect 22.2 ms, 66% over budget = **heavily dt-OOD**). Pure
  GPU-serialization variance run-to-run; **30 Hz not reliable** on the shared passthrough GPU.
- **No-GT leveler flight-safe** (attitude finite/sane both). **fwh2 = better candidate.** ffs0
  dithers yaw *softer* but crashed messier — **confounded by its choke** (re-fly to separate).
- Footgun: TRT prewarm caches — 2nd launch prewarmed **4.68 s** vs 1st **10.27 s**.
