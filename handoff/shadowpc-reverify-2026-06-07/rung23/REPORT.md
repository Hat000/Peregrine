# Rung 2/3 — gate-0 + FULL COURSE. LIVE, ShadowPC 2026-06-07

## ✅ VERDICT: VQ1 FLOOR BANKED — valid, sim-recognized 6/6 finish, ≈35.3 s.

After rung 1's cycle was accepted (delay-driven, position-harmless), the moving course was verified.
The faithful config (`kp_alt 3.0/kd_alt 1.75`, raw vz) + a small vertical offset **threaded all 6 gates
and the sim recognized the finish** (GUI ground-truth; the recording's `finished=False` was a snapshot
race — force-disarm fired at the instant of finish, before the flag propagated). A slow valid finish is
the VQ1 floor. Time ≈35.3 s (sim-official, GUI readout) — slow, but **valid > fast-invalid**.

**Envelope note:** flown with the conservative mid-air aborts OFF (`--max-tilt-deg`/`--geofence-m`/
`--max-climb-m` dropped), only the 10/60 s cap + force-disarm + hard-collision/non-finite sanity. The
`--max-tilt-deg 60` guard had FALSE-ABORTED rung 2 mid-transient (drone tilts ~64° in the initial lunge,
then recovers) → the hands-off disarm caused the "crazy spin" the GUI saw. With the controller kept
engaged, it flies. (See the rung-2 progression below.)

## Rung-2 progression (gate 0) — the false-abort story, then the fix
| run | aborts | alt-offset | result |
|---|---|---|---|
| `192550_vq1` | tilt 60° | 0 | **ABORT** @64° tilt, 1 s in (initial lunge) → hands-off → "crazy spin" (GUI) |
| `193244_gate0_nobounds` | none | 0 | threaded g0 but **closest 0.74 m** (right at the ~0.75 half-opening — the "top clip" the GUI saw); disarmed AT the gate (mission FINISH) |
| `194047_gate0_lower` | none | −0.4 | g0 **in-plane 0.11 m** (down −0.11, dead-centred); stayed engaged past the gate |

→ Two fixes confirmed: (a) **stay engaged past the gate** (don't disarm at the pass) removes the
hands-off tumble; (b) **−0.4 m vertical offset centres gate 0** (0.74 m marginal → 0.11 m). The offset
worked — an earlier 1 Hz-trace read that "it didn't" was too coarse; the in-plane miss confirms it.

## Rung-3 OBSERVATION — full course (`194615_course_60s`, FINISHED 6/6, 33.9 s cmd-log)
**Per-gate in-plane miss** (offset in the opening plane; PASS if < ~0.75 m inner half-opening):
| gate | centre (x,y,z) | in-plane miss | note |
|---|---|---|---|
| g0 | (−23.3, −0.4, −1.4) | **0.37 m** (right +0.37, down −0.00) | vertically dead-centre; lateral = initial-bank residual |
| g1 | (−46.9, −2.5, +3.7) | **0.05 m** | |
| g2 | (−74.6, +1.2, +12.3) | **0.06 m** | |
| g3 | (−111.5, −5.1, +23.2) | **0.03 m** | |
| g4 | (−135.5, −0.8, +24.0) | **0.10 m** | |
| g5 | (−159.2, −4.4, +24.6) | **≈0.71 m** (sim-confirmed pass) | marginal; cmd-log truncated at finish-disarm so the geometric crossing wasn't fully recorded |

g1–g4 are excellent (0.03–0.10 m). g0 (0.37 m) carries the initial-bank residual; g5 (≈0.71 m) is the
marginal gate. Course = ~159 m horizontal + ~26 m descent (g0 +1.4 m → g5 −24.6 m). All passed; **no clips** (GUI).

**GUI ground-truth (teammate, relayed):** *"The sim DID recognize the finish, time 35.3 … we only needed
to finish."* · *"the initial big bank … this time it got worse, even though we didn't change anything —
that's weird."* · *"during descents the drone drops aggressively, and during the drop it wobbles a little,
no clips."* · *"the whole race we could hear the drone throttle up and down constantly to maintain height."*

## INTERPRETATION — three findings (kept separate from OBSERVATION)

### 1. ⚠️ Initial hover→forward transient: NON-DETERMINISTIC + severe (the #1 issue)
Same config (−0.4), wildly different transient run-to-run:
| run | peak bank | peak lateral | peak climb | peak cmd body-rate |
|---|---|---|---|---|
| `gate0_lower` (clean) | 18.2° | 0.91 m | 1.36 m | **8.00 rad/s (saturated)** |
| `course_60s` (bad) | **54.3°** | **5.99 m** | **3.35 m** | **8.00 rad/s (saturated)** |

**Root-cause hypothesis:** at the TAKEOFF→RUN switch the attitude→rate command **saturates the 8 rad/s
`max_body_rate` clamp in EVERY run** (leveling the −17.8° rest pitch + pitching forward for the max-accel
lunge, all at once). The saturated, lightly-damped (`kp_att 10`/`kd_att 0.15`) response is **sensitive to
the exact tick-timing/state at that instant** — which varies with sim/loop nondeterminism — so the bank
ranges 18°→54° / lateral 0.9→6.0 m. A 6 m swing at the start is a real gate-0/obstacle-miss risk.
**Fix (offline, Commander):** ramp the speed/accel target at the start (don't demand `max_accel 12` from a
standstill), and/or soften the attitude command (lower `kp_att`, raise `kd_att`) / rate-limit the initial
attitude step so it doesn't slam the 8 rad/s clamp. NOT to be re-tuned live.

### 2. ⚠️ Aggressive descents — max sink **9.3 m/s**
The course descends in steps; the drone dives hard between gate altitudes (peak 9.3 m/s), with minor
wobble (GUI), no clips, all gates passed. A smoothness/time issue — the racing-line + speed-profile is the
VQ2 differentiator, not the VQ1 floor.

### 3. 📌 The alt-loop relay is ALWAYS-ON, not static-only (refines the rung-1 framing)
Cruise throttle (12 s..end−3 s): mean 0.354, std **0.257**, **47% at the 0.6 ceiling / 36% at the 0.05
floor** — the bang-bang relay runs through the MOVING descent too (the audible cycling the GUI heard all
race), not just at a static hover. It is **validity-harmless** (gate misses 0.03–0.37 m prove the position
tracks fine), but the Commander's "static-only" note should read **"always-on, validity-harmless."** The
post-VQ1 clean-up (gain-schedule / RL) still applies.

## Recommendation
- **VQ1 floor is banked.** Next, in priority order for a *reliable* (not just one-shot) finish:
  1. **The non-deterministic start transient** (#1) — the biggest risk to repeatability.
  2. Descent smoothness + the always-on relay (#2/#3) — the clean-hover/gain-schedule deferral now also
     covers dynamic flight.
  3. Time (≈35 s) — racing-line/speed = the VQ2 rank target.
- These are OFFLINE jobs. The live ladder did its job: gate-0 centring + the full thread are verified.

## Artifacts (this dir)
- `course_60s_{extract.json,commands.jsonl,meta.json}` — the full-course finish (6/6).
- `gate0_nobounds_*`, `gate0_lower_*` — the rung-2 progression (false-abort → engaged → centred).
- `../scratch/gate_analyze.py` — per-gate in-plane-miss + transient analyzer (regenerates every number above).
- Full `mavlink.tlog`/`video.bin` stay on ShadowPC (gitignored; move via the adroit-connector if needed).

**Process:** GUI ground-truth obtained before each interpretation; force-disarm on exit; `memory/` (repo) not edited.
