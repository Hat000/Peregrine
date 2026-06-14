---
name: reference-sim-ops
description: "Unattended FlightSim operation mechanics — launch/login chain, MAV_CMD 31000 restart semantics, window-kick recovery, the three idle states, zombie dual-instance, autoreset/spin guards, fullscreen auto-minimize, measurement footguns."
metadata: 
  node_type: memory
  type: reference
  originSessionId: 972b9a84-aa8d-44c1-b7d1-36e8c7f9a995
---

# Sim ops — unattended FlightSim control (ShadowPC)

Facts moved verbatim from the MEMORY.md index (2026-06-11 restructure). Operational trail: [[project-phase2-rl-vision-decisions]] §S1.2 + §SHADOWPC-LIVE-DEPLOY-DIAG.

## Launch + login
- **SIM LAUNCH:** `"C:\Users\Shadow\Downloads\AI-GP Simulator v1.0.3364\AIGP_3364\FlightSim.exe"`; any button → login → Enter → homepage.
- **🚩 2026-06-10 — UNATTENDED SIM CONTROL proven:** S1.2 cold-launched FlightSim.exe + raced NO human.
- `rl/fly_rl.py --flights N` chains attempts.

## Race restart semantics + idle states (REFINED 2026-06-13 — two state-machine corrections)
- **MAV_CMD 31000** restarts race once race context exists — **NO-OP from HOME**; from HOME: Win32 `SetForegroundWindow` + **Enter ×2**.
- **🚩 THIRD idle state** (~90 min post-race): 31000 NO-OP, stale telemetry; recovery = Win32+Enter×2 — wired as auto-escalation in `rate_sysid.py`.
- **🚩 REFINEMENT 1 — cold-launch waiting room takes TWO Enters:** HOME→Enter→waiting-room (started=False, drone at origin) → **Enter AGAIN** → GO. The extra Enter out of the waiting room is required and was previously the common gotcha.
- **🚩 REFINEMENT 2 — from FINISHED race, ESC+Down×3+Enter RESTARTS the race (does NOT exit to HOME):** this returns to the waiting-room, not HOME. Confirmed by fresh to_go + new race_start_boot_ms. Chain from FINISHED to a new race: ESC+Down×3+Enter (back in waiting-room) + Enter×2 (to GO). ESC+Down×3+Enter alone is context-dependent (restart from FINISHED; no-op from stale race / HOME).
- Full fresh-race chain (from HOME or after FINISHED): ESC+Down×3+Enter+Enter×2 (verified via fresh to_go + new race_start_boot_ms).

## Unattended orchestrator pattern (MASTERED 2026-06-13)
- **`partA_cycle.py` pattern (ShadowPC):** run `fly_rl --flights 1` per cycle so every state transition (arm, race, reset) is MAVLink-confirmed before the next cycle begins. Avoids blind between-flight resets. 10/10 clean autonomous cycles; 13 clean inc7 flights; 0 gate contact.
- All 4 failure modes handled live: stale-started, zombie dual-instance, spawn-artefact detection, off-race-pause.
- 🚩 Tooling in handoff/simops-mastery-2026-06-13/ on ShadowPC — **NOT yet on main (pending commit+push from ShadowPC).**

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

## Gotchas — respawn / collision artefacts
- **🚩 SPAWN-ARTEFACT AFTER GATE-3 CRASH (2026-06-12):** a gate-3 HARD COLLISION full-reset leaves residual collision geometry that traps the next respawn → 0-tick crash (obs header only, 0 usable ticks). Pattern: gate-3 crash → artefact → clean spawn → gate-3 crash → repeat. Affects per-batch stats; NOT a policy failure. If running multi-flight batches through gate-3 standing failures, expect every other flight to be artefact-contaminated. Workaround: add a 2–3 s sleep after full-reset before issuing the arm command (untested; may allow geometry to clear).

---

## §AUTONOMY-READINESS (2026-06-13 ultracode audit — read-only; fixes NOT applied)

Full report: `handoff/ultracode-autonomy-readiness-2026-06-13/REPORT.md`. Adversarial offline failure-injection (~80 scratch probes). **Verdict: shipped launch path NOT submission-safe under bare defaults.**

### Rank-1 DQ — auto-reset sends `MAV_CMD 31000` on judged wire (R1)
`fly_rl.py` default `auto_reset=True` → `wait_fresh_go` fires `client.send_sim_reset()` (a real `COMMAND_LONG` on the SAME UDP wire as arm/control, not a GUI action) whenever organizer start-telemetry diverges from our ShadowPC model (stale countdown / lagging pose / drone not at origin). **Spec §7: any sim-control cmd during timed run = DQ.** Fires on stale-GO, pos-None, or drone 50 m off-origin. `fly_vq1.py` NEVER auto-resets (contrast oracle). Fix: hard-disable auto-reset (`auto_reset=False` default; gate `send_sim_reset` behind `--dev-auto-reset`).

### Rank-2 — bare-default invocation is dead-on-arrival after arm (R2)
`--bridge` defaults True → loads gitignored `data/runs/track_map_...json` (absent in any clean checkout) → uncaught `FileNotFoundError` AFTER arm → dies ARMED. `[safety]` disarm is past the escape point (not in a `finally`; handlers are `KeyboardInterrupt`-only). Default `--checkpoint` still points at the retired inc4 actor.

### D1/D2 root cause — ODOMETRY per-field freshness absent
`DroneState.recv_monotonic_ns` is a single SHARED stamp bumped by HIGHRES_IMU, LPN, and ODOMETRY alike. An ODOMETRY-only drop while LPN/IMU survive → policy flies on frozen stale attitude/rate (~85° wrong) with **all guards blind** (D1 open-loop TIMEOUT/contact). If stale `|w|≥6` remains latched when ODOMETRY drops >2 s, the spin guard false-fires `SPIN_ABORT` mid-air (D2). Fix F-C: add `odo_recv_ns` to `DroneState`; gate command emission AND spin guard on `(monotonic_ns − odo_recv_ns) > ~0.15 s`.

### Fix summary (none require retrain)
| Fix | Closes | Action |
|---|---|---|
| F-A: committed wrapper `--checkpoint stage1_inc7_actor.pth --no-bridge --no-auto-reset --no-debug-obs` | R1, R2, R3, inc4 trap | Config only |
| F-B: `finally`-disarm + broaden handlers to `except Exception` | dies-ARMED half of R2/R4/n3 | One code change |
| F-C: `odo_recv_ns` + finite/zero-norm gate | D1, D2, R4, R5 | One struct + gate |
| F-D: late-join GO gate + bounded arm-retry | N1, AR1 | Two small logic changes |

**DONE: LAPTOP-FLYRL-AUTONOMY-HARDENING — worktree Anduril-wt-flyrl, branch flyrl-autonomy-hardening, commit 7210c1d; suite 657 green; NOT merged to main (held for ShadowPC live-verify).** F-C subsumes the earlier NaN-guard task. F-C VERIFY-FIRST CONFIRMED: all 5 degraded-input failures propagated on old code (np.clip(nan)=nan reached wire; D1 stale-attitude reached wire via shared stamp advancing 1.79 ms while quat froze). obs[12] relabeled prev_normed_thrust. **🚩 DEV-RIG:** ShadowPC dev batch now needs --dev-auto-reset. **MERGE GATE:** ShadowPC §4 checklist must pass (clean finish + no 31000 on wire + late-join GO + odo/finite gates silent on healthy run + crash→disarm).

### N1 interaction — F-A creates a silent NO_GO on late-joined race
`--no-auto-reset` + the strict `to_go > -2000 ms` fresh-GO gate → if the race is already running when our process attaches, we spin to 180 s deadline and exit without arming. F-D must widen the acceptance for the `--no-auto-reset` config (accept first-seen STARTED race at origin, gate 0, not finished).

### BSR3 forward gate — spin-margin for envelope-relaxation retrain
Healthy inc7 `|w|≥6` max run = 1.97–1.99 s vs 2.0 s window (0/23 false-fires, but a ≥1.5× proxy trips it). **Before rw_tilt 96→48 retrain, mandate a realized-`|w|` trace gate; widen `spin_rate_abort`→~9–10 rad/s and `spin_time_abort`→3.0 s for the relaxed policy.** Bank as rider on the inc8 envelope arm.

### REFUTED — do NOT re-litigate
Missing inc7 sidecar (both `.pth` and `.json` git-tracked + co-located — sidecar gitignore only matches `*.pt`, not `*.pth`); false-finish sentinels (`sim_finish_confirmed`); policy nondeterminism (`tanh(mean)`, `actor.eval()`, bitwise-reproducible); spin-guard NaN-blindness (spin guard fires at ~2 s even under sustained NaN rate); unbounded-wait/hang (every wait is hard-bounded); stale-armed second flight.

### Organizer Qs that gate residual
Add to critical-path email: (Q-A) race-start-vs-launch ordering (→N1); (Q-B) per-msg-type telemetry reliability (→D1/D2); (Q-C) submission interface / entrypoint contract (→whether default hardening or wrapper suffices); (Q-D) §7 sim-control prohibition confirmation.
