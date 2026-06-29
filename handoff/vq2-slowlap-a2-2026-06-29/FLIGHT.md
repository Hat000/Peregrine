# VQ2 self-localized SLOW-lap — ATTEMPT 2 (map-free visual servo) — 2026-06-29

**Branch flown:** `claude/loving-galileo-92f020` @ `5c62524`
("fix(vq2): map-free visual-servo gate-seeker — fixes the blind-launch U-turn").
**Sim:** AI-GP 1.0.3379, event **R2 - TRAINING = VQ2** — lit warehouse verified by screenshot
before every flight (`a2_verify_vq2_run1.png`, `a2_verify_vq2_run2.png`); menu navigated blind by
keyboard, screenshot used ONLY for the VQ1/VQ2 check (per instruction).
**Controller:** map-free visual-servo gate-seeker, `vq2_case_c` profile, `red_glow` detector.

## TL;DR scoreboard
| Question | Result |
|---|---|
| Launch-held (no blind ~180° slew)? | **YES** — the attempt-1 U-turn is GONE. tlog shows the drone holding level + still for ~0.75 s at launch (gyro≈0, accel=level 1g). |
| Vision fixes accepted? | **NO** — still **zero** (`tsv=inf` throughout), and on the live map-free path it is **structurally impossible** to accept one (see root cause A). |
| Gates reached | **0 / 6** |
| Contact? | **YES** — environmental (id **1002**), **64–71 contacts**, beginning ~1.5 s after launch (a real pitch-tumble into the gate, not just the spawn transient). |
| Failure mode | **Launch-hold pitch divergence.** The yaw-hold works, but the hold leaves roll/pitch to the cold case-C AHRS; ~0.75 s in, pitch diverges to a saturated ~3.4–9.7 rad/s and tumbles the drone into the start gate. The anchor never releases, so it never reaches pursuit. |

Ran **2.0 m/s** (run1) and **3.0 m/s** (run2) — **identical** failure. Speed is irrelevant here
because the drone never leaves the launch-hold regime.

---

## What the fix DID achieve
Attempt 1 slewed ~180° away from the visible gate at launch (absolute stale-VQ1-map steering).
**That is fixed.** Logs confirm the map-free path is active:
`"no live TRACK_INFO + self-localizing profile -> MAP-FREE flight (visual servo + vision yaw/z;
NO absolute map). The stale --map is IGNORED."` (`gates=0`). And the realized IMU (tlog,
`run*/frames/analysis.json`) shows a clean launch hold:

```
t=0.00–0.75 s : gyro≈[0,0,0],  accel settles to [~0, 0, -9.88]  -> level, still, hovering (HOLD OK)
```

The detector also sees the gate the whole pre-tumble window — **18/27** frames (run1) /
**14/25** (run2) with ≥1 detection, scene bright (meanBGR≈35); `frame_00_*` shows the start gate
dead ahead. So perception + the seeker's own relative-bearing steering input were never the problem.

## Why it still crashes — TWO root causes

### A. The anchor can NEVER release on the live map-free path (structural)
The visual servo has three regimes; it leaves the **launch-hold** (regime 1) only once
`_anchored` flips true, and that happens **only** when `nav.time_since_vision_update_s` becomes
finite (`gate_seeker.py:297-298`).

But `time_since_vision_update_s` is driven by `Navigator._last_vision_sim_time_ns`, which is set in
**exactly one place** — `navigator.py:861`, inside the gate-**map-associated** position-fix path
(detect → PnP → associate to a MAP gate → accept). On the live VQ2 path the navigator is built
**map-free** (`gates=[]`), so association can never occur, `_last_vision_sim_time_ns` stays `None`,
and `_nav_state` returns `tsv = inf` forever (`navigator.py:992-993`). The map-free vision yaw (VP)
and floor-height anchors do **not** stamp this timestamp.

**Net:** on VQ2, `_anchored` can never become true → the seeker is pinned in the launch-hold and
can never transition to pursuit, even with the gate centered in view. The anchor's release signal
is wired to the very map-fix the map-free path disables.

### B. The launch-hold is not actually safe (pitch/roll unclamped on a cold AHRS)
`_hold_command` (`gate_seeker.py:351-363`) zeros horizontal velocity and **clamps only the yaw
rate** — roll/pitch come straight from the attitude-leveling controller against the case-C AHRS
estimate (`_cap_yaw_rate` touches `body_rate[2]` only). At the in-gate spawn the AHRS is cold (no
mag, gyro-bias unestimated, initialized with an ~18° tilt: tlog t=0 accel `[-3.0, 0, -9.34]`).
After ~0.75 s of stable hold the attitude/velocity estimate diverges and the hold commands a
**saturated pitch**; the realized body rate (tlog) is the smoking gun:

```
run1 (2.0):  t=0.82–1.44 s  gyro_y ≈ -3.4 rad/s sustained (~195°/s)  -> pitch-over
             t≈1.54 s  peak |gyro| 4.46 rad/s ; t≈1.58 s  accel z = +32.8 m/s²  (IMPACT)
run2 (3.0):  t≈1.33 s  peak |gyro| 9.66 rad/s ; t≈1.36 s  accel z = +368 m/s²   (IMPACT)
```

Thrust also ramps during the hold (alt-hold chasing the weak/uncorrected case-C z), adding a climb
into the gate structure. Result: 64–71 environmental contacts (id 1002) as the drone thrashes.

## Recommended fixes (design — not done here)
1. **Wire the anchor to a map-free signal.** Release the launch-hold on the first *map-free*
   evidence the estimator is usable — e.g. the seeker's own `detect_gate_lever` returning a
   quality-gated pose for N consecutive frames, or have the map-free vision yaw/floor-height
   anchors stamp a "vision alive" timestamp. As written, `_anchored` is unreachable on VQ2.
2. **Make the hold genuinely attitude-safe.** Clamp roll & pitch rates in `_hold_command` (not just
   yaw), and/or hold on a gravity-direct accel-levelled attitude rather than the full case-C AHRS
   estimate until the AHRS has settled. Bound the alt-hold thrust authority during the launch hold.
3. **Let the AHRS settle before commanding.** Add a short post-arm settle (hold raw-levelled
   attitude + fixed hover thrust, no estimator-driven leaning) so gyro-bias/tilt converge before
   the controller trusts the estimate.

## Artifacts
- `run1_speed2/` (2.0 m/s), `run2_speed3/` (3.0 m/s): `fly_run*.log`, `analysis.log`,
  `frames/analysis.json` (per-frame detector + first-2.5 s HIGHRES_IMU), sample frames.
- `analyze_run.py` — the offline analyzer (detector replay + tlog IMU decode).
- `a2_verify_vq2_run1.png`, `a2_verify_vq2_run2.png` — the lit-warehouse VQ2 verification shots.
- Full recordings: `data/runs/20260629_180714_*` (2.0), `data/runs/20260629_181548_*` (3.0).
