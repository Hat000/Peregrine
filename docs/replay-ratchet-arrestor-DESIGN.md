# REPLAY-RATCHET ARRESTOR — design (P0 doc, no implementation)

2026-07-17 · ratchet-tape branch. Companion to `rl/tape_extract.py` (built, tested) and the
existing command player `rl/fly_rl.py --sysid-replay`. All file:line cites are the deploy branch
`claude/ego-deploy-2026-07-09` @ `dac05a5` (== the champion-commit `28404fa` semantics for the
ego command path; the sysid commits only ADDED code).

**Premise.** The eval sim is deterministic with fixed gates. Replaying a champion recording's
per-tick wire commands (tape) reproduces its trajectory and BANKS gates 1..k without perception
risk. The tape ends hot (champion 202317: 6.4 m/s at the gate-1 pass, 9.1 at gate 2, 15.2 at
gate 5). The ARRESTOR bridges tape-end → a calm, gate-facing, near-hover state → hands the RL
policy a start it was trained for. Determinism ends at tape end by design: TAPE is open-loop,
ARREST/POLICY are closed-loop.

## 1. State machine

```
TAPE ──(tape exhausted | divergence guard)──▶ ARREST ──(calm+gaze)──▶ HANDBACK ──▶ POLICY
  │                                              │
  └────────(loop-level guards, all phases)───────┴──▶ existing stops: FINISHED / CRASH / SIM_RESET
ARREST ──(abort: gate lost >1 s | altitude drift | timeout)──▶ HANDBACK (policy owns recovery)
```

- **TAPE** — emit tape rows exactly like the sysid tick does today (fly_rl.py:2170-2197:
  `sysid_wire_from_action` → `client.send_command`), one row per tick at `--rate 40`.
- **ARREST** — closed-loop stare-brake primitive (§3). No policy, no tape.
- **HANDBACK** — one bookkeeping tick (§4), then
- **POLICY** — the stock ego tick (fly_rl.py:2280-2317), zero delta.

Implement as a pure, unit-testable class (`rl/replay_arrestor.py`), following the
`EgoTakeoffAssist` precedent (fly_rl.py:768: enable/latch state machine + per-tick `apply()`),
dispatched where the sysid tick sits today (§5).

## 2. Triggers

- **TAPE → ARREST (primary): tape exhausted.** Today the player `break`s with `SYSID_DONE`
  (fly_rl.py:2171-2174) and the drone goes ballistic; the arrestor replaces that break with a
  phase transition. Tapes for ratcheting are cut by `tape_extract.py --truncate-at-gate K`
  (pass + 12 source-tick margin), so "tape end" == "just banked gate K, pointed at K+1".
- **TAPE → ARREST (guard): divergence, NOT an absolute speed cap.** A fixed "KF speed > 7 m/s"
  trigger false-fires at tick ~85 of the champion (its recorded profile rides 8-15 m/s from the
  gate-1→2 segment on; per-gate pass speeds 6.4/9.1/8.1/11.6/15.2). The tape CSV carries the
  recording's own per-tick `kf_speed` column: guard on **|KF speed − taped kf_speed| > 3 m/s
  for ≥5 consecutive ticks** (KF speed = horizontal `nav_state.velocity_ned`, the same NavState
  already produced every tick at fly_rl.py:2220). Divergence ⇒ determinism broke ⇒ brake NOW
  with whatever gates are already banked. An absolute ceiling (e.g. 18 m/s > any recorded value)
  may back-stop the guard.
- **ARREST → HANDBACK (calm+gaze, all three):**
  1. KF horizontal speed **< 2 m/s**;
  2. gate in view: fresh slot-0 pose, `age_s < det_hold` (0.3 s), conf above the builder's
     accept floor — the signals already logged per tick (fly_rl.py:2340-2342);
  3. near-trim attitude: |roll obs[3]| < 0.17 rad AND pitch obs[4] ∈ [−0.45, −0.15] rad
     (hover trim rests at −0.31 rad = −17.8°, fly_rl.py:640).
- **ARREST aborts (→ HANDBACK immediately; the policy is the trained recovery agent, trained on
  blackout coasts — do NOT hover blind):**
  - gate lost: `age_s > 1.0 s`;
  - vision-vertical drift: |slot-0 rel-up − value at ARREST entry| > 2 m (§3, altitude bound);
  - timeout: ARREST > 4 s;
  - loop-level guards stay senior in ALL phases: hard collision / RACE finished / sim-time stall
    (fly_rl.py:2111-2124) and the reset-epoch guard (2126-2146).

## 3. ARREST: the stare-brake primitive

Goal: burn 6-15 m/s to <2 m/s **without losing the gate**, using only wire-legal signals
(POSE-BLIND wire: no mag, no baro, no absolute altitude — the gate is the altitude reference).
All commands go out the same `ControlCommand(BODY_RATE)` path (fly_rl.py:2312-2317); the
client's `cmd_rate_scale` (×1.2, src/racer/mavlink_client.py:486-495) applies as in any flight.

- **Yaw-hold on gate bearing (the "stare"):** yaw-rate = −K_yaw · bearing, bearing =
  atan2(rel_left, rel_fwd) from the builder's slot-0 relative position (obs[11:14], maintained
  and staleness-decayed by `EgoObsBuilder` through detection gaps). Cap at ±0.7 rad/s — the
  trained yaw clamp (fly_rl.py:623-632); never yaw faster than the lineage ever flew.
- **Gentle pitch-back (the "brake"):** nose-up pitch-rate, capped ≤ ~0.8 rad/s, and HARD-STOPPED
  by attitude: block further nose-up once body pitch obs[4] ≥ ~−0.05 rad (≈ level). Rationale:
  camera elevation = body pitch + 20° mount; the known perception wall is over-nose-DOWN pointing
  the camera at the floor (fly_rl.py:702-711 fence), but over-pitching UP walks the gate out the
  BOTTOM of frame — a level body already means +20° camera elevation. The stop is one-sided
  (mirror of the deploy pitch fence, opposite side); nose-down recovery back toward trim is
  always allowed. Expected decel ~0.4-0.7 g horizontal ⇒ 8 m/s → <2 m/s in ~1.5-2 s, 8-15 m of
  travel — fits the ~20-25 m inter-gate spacing for gate-1..3 speeds. At gate-5 speeds (15 m/s)
  it does not fit; ratchet from LOW truncations first (see §6).
- **Vision-vertical altitude hold:** collective = 1.0 g + K_z · (rel_up − rel_up_entry), bounded
  [0.7, 1.4] g, where rel_up is slot-0 relative "up" (obs[13]) — i.e. hold the drone's height
  RELATIVE TO THE GATE constant while braking. This is the same no-absolute-altitude doctrine as
  the trained gate-holds-altitude coupling; the governor precedent (fly_rl.py:674-689) shows the
  altitude-neutral thrust-shaping style. The ±2 m drift ABORT (§2) bounds a bad rel_up estimate.
- **Roll: command 0** rate; the symmetric roll fence pattern (fly_rl.py:712-729) is available as
  a backstop if entry banks past ~25°.
- Emitted g-units feed `last_normed` every tick (obs[8] honesty — the assist/floor precedent,
  fly_rl.py:2293-2311), so the policy's first obs after HANDBACK sees the true thrust history.

## 4. HANDBACK

One tick of bookkeeping, then the stock policy tick:
- `last_normed` already carries the arrestor's final emitted g (above) — nothing to fix up;
- builder/seeker are already warm because perception RAN through TAPE+ARREST (§5) — slot-0
  fresh, staleness decay live, sector latch valid;
- `gate_index` is already current (§5 moves gate bookkeeping above the phase dispatch);
- log a phase-transition record; POLICY phase = fly_rl.py:2280-2317 unmodified.

## 5. Integration points (fly_rl.py @ dac05a5)

1. **Phase dispatch replaces the sysid-only tick** at :2148-2197. Today: `if _sysid_prog is not
   None: <bootstrap|row|break> ; continue`. Becomes: `phase in {TAPE, ARREST, HANDBACK}` emit
   their own command and skip the policy block; POLICY falls through to :2280. The tape-end
   `break` (:2171-2174) becomes the ARREST transition.
2. **Gate bookkeeping must run in ALL phases.** The active-gate update + seeker reset
   (:2199-2209) currently sits AFTER the sysid `continue`, so today's player never tracks
   gates. Move it above the phase dispatch (it already only needs `gi_now`, computed at :2127).
   Banked-gate count during TAPE = `gi_now`; this is also what `--truncate-at-gate` anchors to.
3. **Perception must keep ticking in TAPE+ARREST** (warm seeker/builder at handback; ARREST
   needs slot-0). That is the perception column :2211-2283 (nav.update :2220, fresh-frame gate
   lever :2239-2258, builder.update :2280). Loop-rate conflict: the champion loop ran 26.4 Hz
   effective because SYNC detect costs 16-31 ms on fresh frames (ego_timing.jsonl), while TAPE
   wants an honest 40 Hz. Resolution: `--video-async-detect` ON (worker thread, :2015-2038,
   :2215) + nav.update every tick (stale-tick cost ~0.4-1.4 ms); if still over budget, run the
   lever at reduced cadence during TAPE only (ARREST/POLICY full cadence).
4. **Command emission** stays single-site: `client.send_command(ControlCommand(BODY_RATE, ...))`
   :2312-2317 (TAPE already emits via :2180-2181; unify). Assist (:2296) and floor (:2305) apply
   in POLICY phase only (the tape already CONTAINS the recorded assist ticks — champion k=0-2).
5. **Arrestor class** in `rl/replay_arrestor.py`, pure-python state machine mirroring
   `EgoTakeoffAssist` (:768): ctor(thresholds), `step(nav_state, diag, obs, tape_row|None) ->
   (rate_frd, collective_g, phase)`, permanent-latch transitions, injectable clock — unit-tested
   like tests/test_ego_takeoff_assist.py, no sim needed.
6. **Replay flags are part of the contract:** tape rows are RAW [-1,1] actions consumed at
   :2057-2063 + :2170-2197; the printed replay command from `tape_extract.py` pins `--rate 40
   --ego-rate-scale <recorded, 1.2> --no-virtual-flip --sysid-climb-s 0 --sysid-settle-s 0`.
   The bootstrap-zeroing keeps the recorded takeoff (assist included) as tape content; with
   climb/settle 0 exactly one hover row (1 g, zero rates) precedes row 0 on the pad — harmless
   at rest.

## 6. Rollout order + open risks

- **Probe order:** replay-only fidelity probe first (full tape, NO arrestor — measure how many
  gates re-bank; this validates determinism + timing-resample). Then arrestor at
  `--truncate-at-gate 1` (tape-end 8.3 m/s incl. margin ticks), then ratchet upward. Gate-4/5
  truncations start at 11.6-15.2 m/s — only attempt after the brake distance is measured.
- **Risks:** (a) log rounding (4 dp rates / 5 dp collective) + 25 ms ZOH quantization vs
  determinism — divergence growth is THE probe measurement; (b) replay loop must actually hold
  40 Hz (perception cadence, §5.3); (c) rel_up hold quality during high-speed brake (PnP range
  error grows with range/blur) — bounded by the drift abort; (d) sim-side nondeterminism
  unknowns (spawn jitter, physics reseed) — if the probe shows early divergence, the ratchet
  premise itself is falsified cheaply, which is the point of the probe.
