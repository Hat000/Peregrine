---
name: replay-ratchet-2026-07-17
description: "REPLAY RATCHET strategy SSOT: deterministic tape-replay banks gates 1..k, arrestor brake bridges to the champion policy which explores/extends the tape. Unlocked by Fengyou's A3/A2 resolution (eval sim deterministic, gates fixed). Legality confirmed (§7 = no human interaction DURING the run only)."
metadata: 
  node_type: memory
  type: project
  originSessionId: cd451162-1e4a-450b-987c-7c21ee7a5592
---

# REPLAY RATCHET — the deterministic bank strategy (2026-07-17, Fengyou-directed)

## The unlocking facts (Fengyou, 2026-07-17 — ground truth, supersedes my worst-case assumptions)
1. **The actual VQ2 eval sim's gates DO NOT MOVE** (fixed course) ⇒ **A2 RELAXED for the sim round**.
2. **The sim is DETERMINISTIC: same commands ⇒ same reactions** ⇒ **A3 RELAXED/CONFIRMED**.
3. The Track-A /goal's "randomized distance + xy shifts" = the TRAINING-robustness target, NOT eval behavior. 🚩 **Commander lesson: I conflated the training-generalization goal with eval randomization and wrongly killed the replay idea — Fengyou corrected. Do not re-make this conflation.**
- ⇒ Parked-backlog triggers FIRED: **#48** (between-attempt refinement exploiting determinism — Q④ made it legal, A3 makes it converge) and the command-replay half of **#75** (the twin-LOCALIZATION variant stays parked; open-loop EXECUTION is the revived part).

## Legality (checked 2026-07-17 against the spec dump)
§7 Compliance: "human interaction during the flight which the participants submit as a timed run is grounds for immediate disqualification" — the ONLY constraint. No vision mandate, no pre-programmed-trajectory prohibition. Tape prep/refinement happens BETWEEN attempts (Q④ explicitly legal). ⚠️ Re-verify the clause carried into VADR-TS-**003** when convenient (grep source = TS-002 dump at handoff/body-contact-reconcile-2026-06-13/spec_text_dump.txt:335).

## The ratchet loop
1. **TAPE**: replay the recorded per-tick commands of the champion's best flight ⇒ deterministic sim reproduces gates 1..k (k=5 today from vpeffs0's `28404fa` deep runs). No "extrapolation" — replay verbatim.
2. **BRIDGE = the arrestor (Fengyou-approved: "braking works too")**: at tape end, a stare-at-the-gate brake (yaw-hold on gate k+1, gentle pitch-back, vision-vertical hold) bleeds speed to ~spawn distribution (slow, level, gate-in-view ±FOV/3). 🚩 EPISODIC takeover≠the REJECTED continuous cap/governor ([[feedback-no-deploy-bandaids]] scope: what died was mid-maneuver fighting/clamping; a clean pause→re-spawn hands the policy a state INSIDE its training distribution). Trigger/handback on KF speed (>~7 in → <~2 out + gate-in-view + level). Risks: brake must not lose gate (over-pitch = perception wall) nor altitude (no abs-alt; gate holds altitude).
3. **EXPLORE**: champion policy released at gate k+1 in its proven regime; whatever it passes, HARVEST its command segment into the tape ⇒ next attempt banks 1..k+m. Monotone prefix growth; unlimited attempts; 8-min budget; zero-contact validity per attempt.
4. **SPEED (after the 20-bank)**: between-attempt segment refinement (ILC-style, #48/#53) — re-fly tape segments faster, keep only valid attempts. Fengyou's mapping point: every extension also surveys the course (gate positions from vision fixes) → feeds faster lines + coarse-map truth.
5. **v1's role unchanged**: a non-accumulating policy = a better EXPLORER (bigger bites) + the speed/physical-round base. Ratchet de-risks the BANK while GPUs are contended.

## The ONE caveat determinism doesn't cover → P0
Sim = deterministic given COMMANDS-PER-TICK; our tape rides the async UDP wire (fly_rl ≥30 Hz) — SEND JITTER can shift which sim tick consumes a command = the divergence channel. Also initial state must repeat (fixed pad spawn, 17° tilt — expected fixed).
**P0 = REPLAY-FIDELITY PROBE (ShadowPC, vision-commander relay):** replay ONE recorded 5-gate tape ×2 with the send clock locked to a reproducible anchor (e.g., first-HEARTBEAT/GO-relative tick schedule); PASS = same gate-pass sequence + per-gate timings both times (RACE_STATUS events; IMU traces for divergence rate). 5/5 twice ⇒ build the ratchet. Partial repeat ⇒ measure divergence-vs-time, shorten tape segments accordingly (re-anchor at each arrest).

## ✅ P0 ENABLERS BUILT (ratchet-builder agent, 2026-07-17) — branch `ratchet-tape-2026-07-17` @ `3389f68` (worktree `wt-ratchet` off ego-deploy `dac05a5`; PUSH pending Fengyou)
- **THRUST IS LOGGED — no champion re-fly needed.** `ego_obs.jsonl` carries the FULL 4-ch command: `rate_frd`+`collective` = EXACTLY what `send_command` received (post-clamp, **PRE-client-scale**; fly_rl.py:2312-2353); `act_raw`/`actor_mean` = pre-clamp levels.
- 🚩 **True wire = logged rate_frd × `ego_rate_scale` (client-side multiply, mavlink_client.py:486-495). Champion flew 1.2 ⇒ every replay MUST pass `--ego-rate-scale 1.2`** (cannot bake into the tape — ±3.14 span would clip).
- **THE champion tape source = `data/runs/20260714_202317_panel_run_f1`** (@28404fa; 5 gates, 1 collision = the terminal crash, 11.03 s): passes at k=76/132/159/193/243, KF speeds **6.4/9.1/8.1/11.6/15.2 m/s**. (213605 run = 49 collisions, unusable.)
- **Player = PRODUCTION `fly_rl.py --sysid-replay`** (:745-759, :2052-2197; present on ego-deploy + sysid-handoff): raw [-1,1] CSV rows, one/tick @ `--rate`, **t column IGNORED**; recipe `--no-virtual-flip`; `--sysid-climb-s 0 --sysid-settle-s 0` ⇒ program from tick ~1; program end = commands STOP (the arrestor's future slot).
- 🚩 **Recording under-ran: 26.43 Hz effective vs 40 target** (sync detect 16-31 ms choked the loop) ⇒ raw row-per-tick replays ~1.5× fast; **the extractor's ZOH 40 Hz resample IS the tape**; replay should run `--video-async-detect` to hold 40 Hz. ⇒ P0 measures TWO things: (a) tape-vs-tape determinism (×2 identical), (b) does the RESAMPLED tape still pass gates (it is an approximation of the original irregular timeline — trajectory may differ from the original flight; that's fine if gates still pass). Contingency if (b) fails: patch the player to honor raw timestamps (`.raw.csv` emitted alongside).
- **`rl/tape_extract.py` + 11 green tests**: exact algebraic inverse of the player map (golden cross-check vs `fly_rl.sysid_wire_from_action` itself); round-trip rates 1.1e-16, collective 5.7e-6 (log's 5-dp floor); `--validate` + `--truncate-at-gate K --margin-ticks M`; extra cols `gate,src_k,kf_speed` feed the arrestor's TAPE-RELATIVE divergence guard (🚩 absolute ">7 m/s" false-fires at k≈85 — champion rides 8-15 m/s from gate 2 on; guard = |KF−taped| >3 m/s for 5 ticks).
- **Arrestor DESIGN done** (docs/replay-ratchet-arrestor-DESIGN.md): stare-brake = yaw-hold ≤0.7 + one-sided nose-UP pitch-rate stopped ~level (camera +20°: over-pitch-back walks the gate out the frame BOTTOM) + collective 1 g + K_z·(rel_up−entry) ∈[0.7,1.4] g; aborts: ±2 m rel_up drift, gate-lost >1 s ⇒ HANDBACK. Integration: phase dispatch replaces the sysid tick :2148-2197; move gate bookkeeping :2199-2209 above it.
- **Probe ORDER: full-tape replay-only ×2 FIRST → arrestor at `--truncate-at-gate 1` (tape-end 8.3 m/s) → ratchet up.** Gate-5 tape-end 15.2 m/s exceeds the inter-gate brake distance — do NOT arrest there.
- Quirk banked: champion flew with nose-down pitch authority OFF (`ego_pitch_clamp=1.0` **degree** ⇒ fence zeroed rate_frd[1] on 252/287 ticks at rest pitch −17.8°) — tape reproduces it; possibly part of why the early gates were stable.

## Infrastructure map
- **Command-player EXISTS** (vision side, sysID era: ran `sysid_program.csv` through production fly_rl) = the tape player's skeleton; tape format should match it.
- **Tape source**: `28404fa` deep 5-gate runs (`data/runs/*_panel_run_f1/ego_obs.jsonl`, per-tick `rate_frd` cmd). ⚠️ VERIFY the THRUST channel is logged; if absent → one champion re-fly with full 4-ch command logging is P0's first step.
- Laptop-side build (agent, in motion 2026-07-17): tape_extract tool (jsonl→player-format tape) + arrestor state-machine DESIGN (build after P0 passes).
- Validity rule unchanged: gate contact = invalid attempt (deterministic replay of a clean tape stays clean).
