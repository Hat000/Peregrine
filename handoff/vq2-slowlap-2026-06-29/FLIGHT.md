# VQ2 first self-localized SLOW-lap attempt — 2026-06-29

**Branch flown:** `claude/loving-galileo-92f020` (full stack: deploy_profile, gate_seeker,
fly_rl `--gate-seeker`, case-C navigator + red_glow detector).
**Sim:** AI-GP build 1.0.3379, event **R2 - TRAINING = VQ2** (lit warehouse — verified, see
`sim_vq2_confirmed_lit_warehouse.png`).
**Controller:** transparent slow gate-seeker (NOT RL), CTBR / `SET_ATTITUDE_TARGET` body-rate,
`vq2_case_c` deploy profile (`cmd_rate_scale=0.4`), `red_glow` detector.

## TL;DR scoreboard
| Question | Result |
|---|---|
| Armed + lifted off? | **YES** (after a 1-line arming-gate fix — see below) |
| Gates reached | **0 / 6** (`active_gate_index` never advanced past 0) |
| Self-localized? | **NO** — estimator position pinned at `(0,0,0)`, `time_since_vision_update = inf` (zero accepted vision fixes) |
| Contact? | **YES** — environmental (collision id **1002** = env; 1001 would be a gate), ~1.2–1.4 s after launch |
| Failure mode | **Blind launch U-turn**: stale VQ1 gate-map + origin-seeded estimator → seeker commands a ~180° yaw slew (saturated +4 rad/s) → drone spins off the start gate, loses visual lock, tumbles into the start-gate structure |

Ran at both **3.0 m/s** (run2) and **2.0 m/s** (run3) — **identical** failure, same first command,
same root cause. Speed is not the lever here; the launch geometry is.

---

## What happened, step by step

### 0. Arming was structurally blocked on VQ2 (FIXED to proceed)
First run (`run1_speed3/fly_run1.log`) ended **NO_GO** — never armed.
Root cause: `wait_fresh_go()` (rl/fly_rl.py) gates race-GO acceptance on
`live = s.position_ned is not None and s.sim_time_ns > 0`. **VQ2 denies raw position AND
attitude on the wire** (confirmed live — `wire_probe.log`: `position_ned=None`,
`attitude_wxyz=None`, IMU streaming ~95 Hz, RACE_STATUS @4 Hz, `started=True`, RTF≈1.0). So
`live` is *never* True on VQ2 and the loop can never accept the GO → times out → NO_GO.

The downstream seeker loop is already position-`None`-safe (rl/fly_rl.py:1004, 1034); only this
arming gate hadn't been updated for the position-denied wire.

**Fix applied (scoped, behavior-identical when position IS present):** treat the sim as live on
sim-clock advance, use `RACE_STATUS.active_gate_index` as the start-line authority, and guard the
`pos_off` computations against `None`. Diff saved as
`fixes/wait_fresh_go_position-denied.patch` (33 lines). **NOT committed to the stack** — left for
your review/landing. With it, the drone **late-joins the running training race and ARMs**
(`MAV_RESULT_ACCEPTED`).

### 1. Armed → immediate tumble (the real finding)
Both post-fix runs (`run2_speed3/fly_run2.log` @3.0, `run3_speed2/fly_run3.log` @2.0):

```
LATE-JOIN GO! ... armed=True ... pos=NO vel=NO mag=yes baro=nan
[arm] attempt 1/3: ACK=MAV_RESULT_ACCEPTED  armed=True
t=...s gi=0 pos=(+0.0,+0.0,+0.0) thr=0.600 rate=[+0.00,+0.00,+4.00] tsv=inf   <- tick 1
t=...s gi=0 pos=(-0.0,-0.0,-0.0) thr=0.600 rate=[...,...,~-3.8]    tsv=inf   <- tick 2
[gate-seeker] HARD COLLISION -> abort.   (env id 1002, ~1.2-1.4 s)
```

- `pos` stays **exactly (0,0,0)** — the case-C self-localizer never produced a position.
- `tsv=inf` — **no vision fix was ever accepted** by the navigator.
- tick-1 command is **always** `rate=[0,0,+4.00]` — a saturated yaw slew (max_body_rate_rps=4.0).

### 2. Perception DID see the gate at spawn — it was thrown away by the spin
Offline replay of the recorded video through the red_glow detector
(`run*/frames/detector_replay.json`, `replay_detector.py`):

- **Frames 0–7 (~first 0.25 s): detector fires (1–2 detections/frame)**, scene bright
  (meanBGR≈35), red gate clearly in view → `frame_00_id982_dets2.png` shows the drone staring
  straight down the course at the glowing red start gate.
- **Frame 8 onward: image goes near-black** (meanBGR 35→13→8→4, red_frac→0.000), **0 detections**
  → `frame_15_id1013_dets0.png` is the camera pointed at the dark ceiling grid. The drone has
  already rotated/tumbled off the gate.

So perception was working at launch; the **control law spun the gate out of frame** before the
estimator could anchor on it.

### 3. Why the bogus yaw — stale VQ1 map + origin-seeded estimator
VQ2 broadcasts **no gate map** on the wire (`track_gates=0`, confirmed live). fly_rl therefore
falls back to the default `--map data/runs/track_map_20260602_114630.json` — a **6-gate map
captured from the VQ1 wire on 2026-06-02**, when VQ1 still broadcast TRACK_INFO.

That map's **gate 0 sits at world NED `(-23.30, -0.40, -0.03)`**. The seeker
(src/racer/gate_seeker.py:143-154) computes `to_gate = gate_pos - position`, and at launch the
estimator's `position = (0,0,0)` with AHRS yaw 0:

```
to_gate = (-23.30, -0.40, -0.03) - (0,0,0)
yaw_setpoint = atan2(-0.40, -23.30) ≈ -179°
```

The drone is at yaw 0, so the controller demands a ~180° U-turn → **saturates yaw at +4 rad/s**
(exactly tick 1). The `launch_ramp` (0.6 s) eases *tilt* but **not yaw**, so nothing damps the
spin. Within ~0.25 s the drone has rotated the gate out of the 640-px FOV, gets no further vision
fixes, and tumbles into the start-gate structure → env collision at ~1.2–1.4 s.

This is a **frame/anchor mismatch**, not a tuning issue: a VQ1-world map can't localize a VQ2
spawn, and the absolute-position guidance fires at full authority *before* the gate-relative
estimator has a single vision fix to anchor it.

---

## Root cause(s) & recommended fixes (for the stack, not done here)
1. **Arming gate (DONE as a patch):** land `fixes/wait_fresh_go_position-denied.patch` so the
   submitted/training path can arm on a position-denied wire.
2. **Map-free gate-relative guidance (the real one):** the seeker must NOT steer to an absolute
   world map position at launch. Per the no-map design note
   (`memory: no-map visual-guidance assumption`), gate-relative guidance needs the **gate size +
   the live bearing/range from vision**, not a world map. The current path *requires* `--map`
   (aborts NO_MAP without it) and steers to absolute map coordinates — incompatible with VQ2.
3. **Don't command full authority before the first vision fix.** Gate the launch (hold attitude /
   small thrust, no yaw slew) until `tsv` is finite at least once, so the estimator anchors on the
   gate the camera already sees at spawn. Extend `launch_ramp` to also clamp the yaw rate.
4. If an absolute map is kept for VQ2, it must be a **VQ2** map in the **spawn frame**, not the
   2026-06-02 VQ1 capture.

## Artifacts
- `wire_probe.log` / `probe_wire.py` — live wire state (position/attitude denied, race live).
- `run1_speed3/` — pre-fix NO_GO.
- `run2_speed3/` (3.0 m/s), `run3_speed2/` (2.0 m/s) — fly logs + per-frame detector replay + sample frames.
- `fixes/wait_fresh_go_position-denied.patch` — the arming-gate fix (review before landing).
- `sim_vq2_confirmed_lit_warehouse.png` — VQ2 verification. `sim_post-crash_looking-up.png` — tumbled end state.
- Full recordings (mavlink.tlog + video.bin, not copied here, large) live under
  `data/runs/20260629_170125_*` (3.0) and `data/runs/20260629_170406_*` (2.0).
