# AI-GP Simulator — Unattended Operations Guide

How to drive the sim smoothly from a script/agent session with nobody at the keyboard.
Everything here was exercised end-to-end on 2026-06-11 (SHADOWPC-LIVE-DEPLOY-DIAG: 18
policy flights + 1 sysid probe, including mid-batch recovery from a minimized window),
on top of the mechanics established by the S1.2/twin-falsify sessions. Sim build:
v1.0.3364 on ShadowPC.

The four rules that prevent 90% of lost sessions:

1. **One sim instance, ever.** Two instances split the UDP stream → armed-but-deaf drone.
2. **Never send a key without verified foreground.** The fullscreen sim auto-minimizes
   when it loses focus; blind `keybd_event` then lands in whatever IS focused.
3. **MAV_CMD 31000 only works in-race.** At HOME or any parked screen it is a silent
   no-op — you must keyboard-kick instead.
4. **Never command through a sim autoreset.** The race can reset mid-flight (sustained
   gate contact); if you keep streaming SET_ATTITUDE_TARGET, throttle latches into the
   fresh race (observed: minutes of uncontrolled spinning, 12k collision events).

---

## 1. Processes and launch

| thing | value |
|---|---|
| Binary | `C:\Users\Shadow\Downloads\AI-GP Simulator v1.0.3364\AIGP_3364\FlightSim.exe` |
| Processes when healthy | `FlightSim` (launcher) **and** `DCGame-Win64-Shipping` (the game) — one pair |
| MAVLink | sim streams to `udp:127.0.0.1:14550`; our clients bind/receive there |
| Video | JPEG/UDP on `racer.vision.jpeg_receiver.VIDEO_PORT` (drone FPV camera only — menus do NOT render in it) |

**Pre-flight check (always, before the first arm):**

```powershell
Get-Process | Where-Object { $_.ProcessName -match "FlightSim|DCGame" } |
    Select-Object Id, ProcessName, StartTime
```

Exactly one FlightSim + one DCGame. If there are two pairs: you have the zombie
dual-instance — both stream to 14550, pymavlink re-learns its send peer per packet, the
drone arms (ACK fine) but motors pin at 0.05 and nothing moves. Telltale from telemetry:
`client.state.sim_time` flip-flops between an advancing and a frozen value within one
pump loop. **Killing either pair kills both** (shared parent) → full relaunch.

**Cold start:** launch FlightSim.exe → any key → login page → Enter (logins are cached)
→ HOME page. From HOME, Enter×2 starts a race (see §3). After launch give it ~30 s; the
login/HOME pages emit no MAVLink, so "no telemetry" right after launch is normal.

## 2. Window and keyboard control (the part that breaks silently)

The sim is a fullscreen Unreal app and **minimizes itself whenever it loses focus**
(another window stealing foreground, an RDP hiccup, a stray click). A minimized sim
still runs the race and still streams telemetry — but every key you "send" goes to the
wrong window, so all keyboard recovery silently no-ops. This exact failure ended a
10-flight batch at flight 7 on 2026-06-11 (NO_GO chain) until diagnosed by screenshot.

**Therefore: every key send must be wrapped in restore + verify.** The canonical
procedure (implemented as `_force_foreground` in `rl/fly_rl.py` and in
`scripts/sim_focus.py`):

1. Find the window: `EnumWindows`, title contains `AI-GP` / `FlightSim` / `AI Grand Prix`.
2. `ShowWindow(hwnd, SW_RESTORE)` — no-op if not minimized.
3. `SetForegroundWindow(hwnd)` **wrapped in a synthetic ALT press** (`keybd_event 0x12`)
   — Windows refuses foreground changes from background processes otherwise.
4. **Verify `GetForegroundWindow() == hwnd`**, retry ×3; if it never verifies, do NOT
   send keys (fall back to 31000 or report).
5. Only then `keybd_event` the keys, ~60 ms hold, 250 ms+ settle between keys.

**Standalone tool** (no repo imports, safe anywhere):

```text
.venv\Scripts\python.exe scripts\sim_focus.py status            # hwnd, visible, minimized, foreground?
.venv\Scripts\python.exe scripts\sim_focus.py focus             # restore + verified foreground
.venv\Scripts\python.exe scripts\sim_focus.py keys enter enter  # focus then send keys (enter/esc/down/up/left/right)
.venv\Scripts\python.exe scripts\sim_focus.py shot out.png      # screenshot of the sim window rect
```

**Verify screen state by screenshot, never by the video stream** — the video port
carries the drone camera; menus/HOME/countdown don't render in it. `sim_focus.py shot`
(window rect) or a full-desktop PIL grab both work. Reading the screenshot is the only
reliable way to know which page the sim is on.

## 3. Screen map and navigation

All navigation is Enter/ESC/arrows. Verified screens and transitions:

| screen | how you got here | what works | how to leave |
|---|---|---|---|
| **Login** | cold launch → any key | Enter (cached login) | → HOME |
| **HOME** | login; or exit-race | "AI-GP VIRTUAL QUALIFIER R1 — AVAILABLE!" hero page. **No telemetry, 31000 is a no-op.** | **Enter ×2** (HOME → waiting room → race), ~1.5 s settle between |
| **In race** | Enter×2 from HOME; or 31000 | ~3 s countdown then GO; full telemetry; 31000 restarts the race | **ESC → Down×3 → Enter** = BACK TO MAIN MENU → HOME |
| **Pause menu** | ESC during a race | RESUME / RESTART / TOGGLE HUD / BACK TO MAIN MENU | Down×3 + Enter selects exit; ESC resumes |
| **Parked/off-race** (post-race idle, results) | race ended long ago | stale RACE_STATUS, frozen sim_time, **31000 no-op** | focus + Enter×2 usually re-enters; else ESC-dance then Enter×2 |

Timing that works: 0.5 s after focusing before the first key; 0.6 s after ESC; 2.5 s
after the exit-confirm Enter; 1.5 s between the two race-entry Enters.

## 4. Race lifecycle from MAVLink

* **Fresh-GO predicate** (don't fly into a stale race): `RACE_STATUS.started` and
  `race_start_boot_time_ms ≥ 0` and `(race_start − sim_boot)` within the last ~2 s, and
  the drone within ~5 m of the origin. Implemented in `fly_rl.wait_fresh_go` /
  `rate_sysid._wait_fresh_go`.
* **MAV_CMD 31000** (`client.send_sim_reset()`): in-race only → fresh ~3 s countdown.
  **Never fire it into a ticking countdown** (the wait loops already guard this).
* **Escalation ladder** when no fresh GO appears (all automatic in `fly_rl` and
  `rate_sysid`): 31000 → wait `--reset-after` (8 s) → 31000 again → after 2–3
  ineffective attempts, **home-kick** = verified-focus + Enter×2.
* **Between flights, prefer the FULL reset** (`fly_rl --full-reset`, default ON):
  ESC→Down×3→Enter→Enter×2. A 31000 restart can carry residue across races —
  prior-race RACE_STATUS epochs and collision events leak into the next recording
  (race_outcome.py knows how to discard the residue, but clean is better).
* **Arming:** `MAV_CMD_COMPONENT_ARM_DISARM` p1=1; confirm via COMMAND_ACK + the
  HEARTBEAT armed flag (`wait_armed`). Force-disarm (`p2=21196`) on every exit path.
* Our GCS heartbeat puts the sim in ACRO (what CTBR/RL want). The reference client
  instead stays silent + TIMESYNC@10 Hz to keep ANGLE mode — only relevant for probes.

## 5. In-flight safety guards (default-on in `fly_rl.py`, port to any new flier)

* **Autoreset guard** — cut commands the instant any of: ODOMETRY `reset_counter` bump,
  `race_start_boot_time_ms` change, `active_gate_index` drop, >10 m position jump in
  one tick. Then disarm and start the next attempt cleanly.
* **Spin guard** — |body rate| > 6 rad/s sustained 2 s with no gate progress → abort.
* **Hard-collision abort** — any COLLISION with `threat_level ≥ 2` since the attempt
  started (snapshot the collision-list length at GO — earlier events are residue).
* **Stall detection** — `sim_time_ns` not advancing for 1.5 s → race ended/sim hung.
* **Per-step dump** — `--debug-obs` (default ON) writes `<session>/debug_obs.jsonl`
  (telemetry, labeled obs, actor internals, wire command). It has paid for itself every
  session; leave it on.

## 6. Telemetry: what to trust

| signal | trust | notes |
|---|---|---|
| ODOMETRY pos | ✅ | world NED, the canonical pose |
| ODOMETRY twist | ✅ via client | **body-frame**; `mavlink_client` rotates it with the raw quat — verified against d(pos)/dt to <1 m/s even mid-tumble. Use `state.velocity_ned`, don't re-derive |
| ODOMETRY quat | ✅ with care | reports roll INVERTED (`odo_att_sign [-1,1,1]`); undo before physics |
| ODOMETRY angular_rate | ✅ | raw→true FRD = `*[-1,-1,1]`. **Do NOT use quat-FD rates** for magnitudes — they read 12–30% high (mixed-rate clock artifact) |
| HIGHRES_IMU time_usec | ✅ | the ONLY sim clock; never mix with ATTITUDE.time_boot_ms |
| ATTITUDE euler | ❌ | pitch sign inverted; liveness only |
| baro | ❌ | NaN in this sim; z comes from the given pose |
| ACTUATOR_OUTPUT_STATUS | ✅ | per-motor outputs — the parser-independent thrust witness; idle floor ≈ 0.05 |
| RACE_STATUS / COLLISION | ✅ | the authoritative pass/collision oracle (`scripts/race_outcome.py`); beware prior-race residue at recording start |

**Plant surprise to remember:** the sim's mixer couples thrust↔rates at both rails
(parasitic ~hover lift at thr≈0 + large rate demands; no rate authority at thr≈1.0).
Don't command those corners and expect twin-like physics — details in
`handoff/shadowpc-live-deploy-diag-2026-06-11/WRITEUP.md`.

## 7. Recovery playbook (symptom → fix)

| symptom | likely cause | fix |
|---|---|---|
| `waiting: started=True pos=y` forever, 31000s do nothing | parked/off-race screen, or window minimized | `sim_focus.py status` → if minimized/not-foreground: `focus`, screenshot, then `keys enter enter` |
| No telemetry at all | sim at HOME/login (no MAVLink there), or sim dead | screenshot → navigate per §3; if no process, relaunch per §1 |
| Armed OK but motors pinned at 0.05, ignores commands | dual instance | §1 process check; kill+relaunch (killing one kills both) |
| Countdown visible but GO never accepted | drone far from origin = stale race | let the reset timer fire a fresh 31000 (never into the countdown) |
| Mid-flight: position teleports / gate index drops | sim autoreset | guards cut commands automatically; if hand-rolling, STOP STREAMING first, then disarm |
| Recording shows collisions/RACE_STATUS before your race | prior-race residue | snapshot collision count at GO; race_outcome discards leading epochs |
| Keys "sent" but nothing happens | focus not actually held | only send after VERIFIED foreground (§2); screenshot to confirm the screen changed |

## 8. Session etiquette

Exit the race to HOME when done (ESC→Down×3→Enter — `sim_focus.py keys esc down down down enter`)
and **verify with a screenshot**. Leave exactly one sim instance running. Recordings
land in `data/runs/<stamp>_<label>/`; certify flights with
`scripts/race_outcome.py <session>` before claiming results.
