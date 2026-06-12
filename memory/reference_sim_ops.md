---
name: reference-sim-ops
description: "Unattended FlightSim operation mechanics — launch/login chain, MAV_CMD 31000 restart semantics, window-kick recovery, the three idle states, zombie dual-instance, autoreset/spin guards, fullscreen auto-minimize, measurement footguns."
metadata:
  node_type: memory
  type: reference
---

# Sim ops — unattended FlightSim control (ShadowPC)

Facts moved verbatim from the MEMORY.md index (2026-06-11 restructure). Operational trail: [[project-phase2-rl-vision-decisions]] §S1.2 + §SHADOWPC-LIVE-DEPLOY-DIAG.

## Launch + login
- **SIM LAUNCH:** `"C:\Users\Shadow\Downloads\AI-GP Simulator v1.0.3364\AIGP_3364\FlightSim.exe"`; any button → login → Enter → homepage.
- **🚩 2026-06-10 — UNATTENDED SIM CONTROL proven:** S1.2 cold-launched FlightSim.exe + raced NO human.
- `rl/fly_rl.py --flights N` chains attempts.

## Race restart semantics + idle states
- **MAV_CMD 31000** restarts race once race context exists — **NO-OP from HOME**; from HOME: Win32 `SetForegroundWindow` + **Enter ×2**.
- **🚩 THIRD idle state** (~90 min post-race): 31000 NO-OP, stale telemetry; recovery = Win32+Enter×2 — wired as auto-escalation in `rate_sysid.py`.
- Full between-flight reset = **ESC→HOME→Enter** (mandatory, protocol (c) below).

## Failure modes / mandatory guards (the "NEW SIM OPS" list)
- **🚩 ZOMBIE DUAL-INSTANCE:** two instances on 14550 → drone arms but ignores; killing zombie kills BOTH → relaunch fresh.
- (a) sim **AUTORESETS on gate contact** → RL commanding through autoreset → throttle-latched spin — cut commands on reset/contact immediately; (e) autoreset guard cuts RL commands instantly (shipped).
- (b) spin detection + auto-flag.
- (c) full ESC→HOME→Enter between-flight reset.
- **(d) fullscreen sim AUTO-MINIMIZES on focus loss** — window control MUST verify foreground (shipped).
- (f) `--debug-obs` per-step dumps default ON.
- Sim PAUSES physics off-race (wire fact; [[reference-sim-interface]]).

## Measurement footguns
- **🚩 quat-finite-diff rate ALIASES** vs LPN/ODO stagger — use raw ODOMETRY rate.
- **🚩 run 200505 frozen-telemetry at gate 5** = footgun (from mixer-diag appendix).
