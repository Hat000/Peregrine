# Task 3 — §7 altitude balloon: SIM auto-thrust vs OUR controller (LIVE, ShadowPC 2026-06-06)

**Question:** during a CTBR run the drone climbs away (z→−8 m). Is it the **SIM** (an auto-thrust
overriding our collective when it dips, per memo §7) or **OUR** controller (the vertical
alt-PD/vz-damping/tilt-comp law)? **Decisive test:** hold level + step the collective DOWN through
(and below) the hypothesized trigger band (~0.22–0.25), watching the parser-independent
`ACTUATOR_OUTPUT_STATUS` witness — do the motors **track** our command (we own thrust) or **stay
pinned high / climb** (sim auto-thrust)?

**Instrument:** `scripts/rate_sysid.py --mode hover` (holds level via damped body-rate, steps
collective, respects countdown, abort-guards, force-disarm) + added live `ACTUATOR_OUTPUT_STATUS`
logging. Two live races, GUI-corroborated (ACRO confirmed both).

---

# OBSERVATIONS (raw)

## Run 1 — `20260605_234151_thrust_step_ground1` (kp_hold 1.0; reached 0.25)
| phase | cmd_thr | motors (mean) | vz (NED +=down) | behavior |
|---|---|---|---|---|
| init_level | 0.260 | **0.260** | 0→+1.06 | descend (tilted −18°→−4°) |
| thr_0.27 | 0.270 | **0.270** | +1.07→+0.40 | descent decelerating |
| thr_0.25 | 0.250 | **0.251** | +0.39→+1.46 | descent re-accelerating → ABORT (18 m −X drift) |
Motors tracked command exactly; drone descended + drifted −X (resting-tilt forward dive); aborted before reaching the low band.
**GUI (teammate):** ACRO; tracks straight forward, leaves pad, slowly descends while moving forward, flew **under** gate 0.

## Run 2 — `20260606_154420_thrust_step_fwd1` (kp_hold 2.0; reached 0.21, BELOW the hypothesized trigger)
| phase | cmd_thr | motors (mean) | mot range | vz mean (NED +=down) | behavior |
|---|---|---|---|---|---|
| init_level | 0.270 | 0.27 (after 0.05→0.27 spin-up) | 0.05–0.27 | **−0.05** | hold / tiny climb (peak −0.21 m/s) |
| thr_0.25 | 0.250 | **0.251** | 0.25–0.27 | +0.50 | sink |
| thr_0.23 | 0.230 | **0.231** | 0.23–0.25 | +2.57 | sink (faster) |
| thr_0.21 | 0.210 | **0.212** | 0.21–0.23 | +4.72 | sink (fastest) → ABORT (8 m descent) |

- **Motors track commanded collective to ±0.002** at every level (0.250→0.251, 0.230→0.231, 0.210→0.212).
- **Max motor mean over the whole run = 0.27 = max commanded thrust** — motors NEVER exceeded our command.
- **vz is monotonic in collective:** 0.27→hold, 0.25→+0.5, 0.23→+2.6, 0.21→+4.7 (more thrust = less sink). Hover ≈ 0.26.
- **Peak climb the entire run = −0.21 m/s (0.12 m above start);** otherwise pure descent to 8 m down. NO climb at low collective.
- Pitch leveled to ~0°; **no roll, no yaw, perfect heading** (GUI + telemetry).

**GUI (teammate, given BEFORE the data):** ACRO; "very similar to run 1… a little more thrust, almost
an **arrest of a descent**, **no lurch, no upward motion the entire time**, just slowing the fall for
a bit;" relinquished control (→ freefall = post-disarm coast) before gate 0; "no roll, no yaw, perfect
heading to gate 0." — **matches the telemetry exactly** (the "arrest/little-more-thrust" = the 0.27
init phase; the only −vz was −0.21 m/s).

---

# INTERPRETATION

## VERDICT: **OUR controller — NOT a sim auto-thrust. We own thrust in CTBR.**
The discriminator is unambiguous and reproduced across 2 GUI-corroborated runs: as we commanded the
collective **down** through and **below** the hypothesized trigger band (0.27→0.25→0.23→**0.21**), the
`ACTUATOR_OUTPUT_STATUS` motors **tracked our command exactly** and the drone **descended faster** —
it never climbed, never pinned the motors high, never showed an override. **The sim does not engage an
auto-thrust on collective dips in CTBR.**

- **Trigger collective: NONE.** Probed to 0.21 (below the memo's hypothesized ~0.22–0.25); no engagement.
- Memo §7's mechanism ("collective dips → sim auto-thrust takes over → climbs") is **REFUTED for CTBR.**
  (That broken auto-thrust is real for the ANGLE-mode *velocity-setpoint* path — see the velocity-fork
  REPORT — but CTBR's `SET_ATTITUDE_TARGET` body-rate + explicit collective does NOT trigger it.)
- Therefore the §7 balloon (climb-away during a run) is **OUR vertical control law commanding too much
  collective** — alt-PD / vz-damping / tilt-comp driving thrust up — and the motors faithfully obey.

## Cross-checks / instrument discipline
- Parser-independent witness (ACTUATOR_OUTPUT_STATUS) + world vz (fixed velocity) + GUI, all agree.
- Reproduced: Run 1 (→0.25) and Run 2 (→0.21) both show motors-track + descend + no-balloon.
- Honest limits: the level-hold has no horizontal position hold, so both runs flew forward (resting
  −18° tilt) and aborted on a bound (drift / 8 m descent) rather than a clean stationary hover. This
  does **not** affect the verdict (the actuator-vs-command relationship is read every tick regardless
  of horizontal motion), but it did cap how long each run lasted.

## Implication (stated, NOT acted on — control-design deferred per protocol)
Because **we own thrust in CTBR**, the offline twin (`CtbrPlant`, which models clean rotor physics with
no auto-thrust) is **faithful for CTBR vertical dynamics** — so the balloon can be diagnosed + fixed
**offline on the twin** (it is our controller). The memo's "next lever" — *floor collective just above
the sim's auto-thrust trigger (~0.25)* — is **moot**: there is no sim trigger to floor above.
