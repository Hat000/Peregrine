# How to drive the AI-GP / VQ2 sim from a session (ShadowPC)

Verified 2026-06-29 on build **1.0.3379**. The whole thing is keyboard-driven; you do **not**
need computer-use grants. `mcp__computer-use__request_access` **cannot resolve the game window by
name** (it's an Unreal window titled "AI-GP" / process `DCGame-Win64-Shipping`). Use the repo's
`scripts/sim_focus.py` (ctypes: finds the window, force-foregrounds it, sends keys) for everything.

All commands below are for the **Bash tool** (Git Bash). Paths are absolute so cwd resets don't matter.

---

## 0. Connection facts (don't rediscover these)
- MAVLink: `udpin:127.0.0.1:14550` (client binds). Send TIMESYNC@10Hz to keep telemetry flowing;
  a GCS heartbeat appears to force ACRO — omit it unless you want ACRO.
- Video: JPEG-UDP `:5600`, 24-byte header `<IHHIIQ`, 640×360 @ ~30Hz (sim re-sends each frame ~14×).
- Reuse `src/racer/vision/jpeg_receiver.py` (frames) and the wire layout in
  `src/racer/mavlink_client.py`. A ready passive capture tool:
  `handoff/vq2-recon-2026-06-29/recon_capture.py`.
- Wire reality (build 3379): pose/attitude/odometry + gate-map are **BLOCKED in both training and
  competition**; IMU is **accel+gyro only** (mag/baro NaN). See `RECON.md`.

---

## 1. Launch the sim
```bash
"C:/Users/Shadow/Downloads/AI-GP Simulator v1.0.3379/AIGP_3379/FlightSim.exe" &
sleep 8
python /c/Users/Shadow/Peregrine/scripts/sim_focus.py status    # expect: title='AI-GP' foreground=True
```

## 2. Cold start → into a TRAINING flight
Each move needs load time, so sleep between them. (`sim_focus.py keys` already waits 0.35s per key.)
```bash
SF="python /c/Users/Shadow/Peregrine/scripts/sim_focus.py"
$SF keys enter ; sleep 4          # PRESS ANY BUTTON -> login page
$SF keys enter ; sleep 6          # submit cached login -> ACTIVE EVENTS menu
$SF keys down down down down enter ; sleep 5   # select R2-TRAINING (3rd event) -> confirm screen
$SF keys enter ; sleep 6          # RACE -> flight system up; MAVLink+video now streaming
```
The ACTIVE EVENTS menu lists: **R1**, **R2 - SUBMISSION**, **R2 - TRAINING**.

## 3. See what's on screen (window grab, not the camera feed)
```bash
python /c/Users/Shadow/Peregrine/scripts/sim_focus.py shot /c/Users/Shadow/AppData/Local/Temp/claude/sim.png
# then Read that PNG
```

## 4. Capture telemetry + camera frames (passive, ~50s)
```bash
python /c/Users/Shadow/Peregrine/.claude/worktrees/elated-poincare-a93225/handoff/vq2-recon-2026-06-29/recon_capture.py \
  --mode training --seconds 50 --save-every 50 \
  --outdir /c/Users/Shadow/AppData/Local/Temp/claude/cap
# add  --keepalive heartbeat --control fwd  to arm + nudge forward (induces motion/collisions)
```

## 5. Back to the main menu (from a live flight)
```bash
SF="python /c/Users/Shadow/Peregrine/scripts/sim_focus.py"
$SF keys esc ; sleep 2                       # pause menu: RESUME / RESTART / TOGGLE HUD / BACK TO MAIN MENU
$SF keys down down down enter ; sleep 4      # -> BACK TO MAIN MENU
```

## 6. Switch into COMPETITION (R2-SUBMISSION) from the main menu
On the main menu **R1** is highlighted at the top.
```bash
SF="python /c/Users/Shadow/Peregrine/scripts/sim_focus.py"
$SF keys down enter ; sleep 5    # R1 -> down1 = R2-SUBMISSION -> confirm screen   (down2 = R2-TRAINING)
$SF keys enter ; sleep 6         # RACE -> flight up
```
(Capturing the wire here does **not** submit a score; just don't complete a scored run.)

---

## Gotchas
- **Foreground first.** The sim auto-minimizes when it loses focus and then ignores keys.
  `sim_focus.py keys ...` re-foregrounds before each send, but if a send seems to do nothing, run
  `sim_focus.py focus` and retry.
- **Leave ample time** after login and after RACE (loading screens) — under-sleeping drops keys.
- Default flight mode is **ACRO**. Velocity/position setpoints don't fly cleanly in ACRO (drone
  tumbles). Reaching ANGLE/position needs the TIMESYNC-only (no-heartbeat) regime ± a mode handshake
  — unsettled as of this writing.
- ESC menu order is context-dependent (e.g. after FINISH, the items differ). Screenshot if unsure.
- One process at a time should bind 14550+5600; don't run two capture scripts concurrently.
- `scripts/vq2_loadday/` (referenced in some specs) **does not exist** — use `recon_capture.py`.
