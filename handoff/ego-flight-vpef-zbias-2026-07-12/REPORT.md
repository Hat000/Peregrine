# EGO vpef + vertical-bias probe (2026-07-12) — FIRST gate passes, and a corrected gate-0 diagnosis

Follow-up to `handoff/ego-flight-vpef-2026-07-12/`. Same two ckpts (vpefwh2 PRIMARY, vpeffs0),
same branch `claude/ego-deploy-2026-07-09` @ `b6e9b5c`, same `--ego-yaw-clamp 0.7 + vq2_ego_lean +
negreal42 TRT engine`, one deliberate change: a **temporary −0.5 m vertical bias on the perceived
gate**, injected into the ego policy's obs to test whether the gate-0 failure was vertical.

## The experiment (exact, temporary, now reverted)

One local line in `src/racer/ego_obs.py::rel_pos_body_frd_from_gatepose` (the ego builder's *only*
caller — the estimator/nav uses a separate `localization._apply_camera_vert_offset`, untouched):

```python
voff = frames.BORESIGHT.vert_offset_m
voff = voff + 0.5   # TEMP: gate reads 0.5 m lower on body-vertical (FRD +Z down), policy-obs only
```

So the policy saw every gate **0.5 m lower** than PnP reported. Scoped to the obs the policy
consumes — nothing in the estimator changed. **The edit was reverted after these two flights**
(nothing committed to code); this bundle documents it for reproducibility.

## Headline

**Both policies passed gate 0 — the first ego gate passes on record** (`gates=1` on both, vs
`gates=0` on every prior ego flight). The bias got them through, then the flights failed at the
**gate-0 → next-gate transition**. Two things came out of it, one of which **corrects the prior
report's diagnosis**.

## CORRECTION: gate 0 did NOT fail from "over-climb"

The prior report (`ego-flight-vpef-2026-07-12/REPORT.md`) called the gate-0 failure an
"over-climbing approach into the gate structure." **That was wrong** (operator-corrected). The real
gate-0 problem is **perception loss from an over-aggressive head-down attitude**, and the telemetry
backs it cleanly:

The camera is mounted **+20° up** from body-forward, so camera-elevation = body-pitch + 20°. On the
run to gate 0 the policy slams nose-down to **−50…−52°** to thrust forward, which points the camera
**~30° below the horizon** — and at that look-angle it **loses the gate**:

| approach phase (vpefwh2) | body pitch | camera elev | `pose_seen` / conf |
|---|---|---|---|
| level-ish start | −18° | +2° | 1 / 1.00 |
| **hard head-down** | **−48…−51°** | **−28…−31°** | **0 / 1.00→0.43** (lost, decaying ~0.25 s) |
| eases pitch back | −44°→−22° | −24°→−2° | 1 / 1.00 (re-acquired) |
| levels to pass | −0.6° | +20° | 1 / 1.00 → **PASS** |

vpeffs0 shows the identical signature (loses the gate at pitch −52°/cam −32°, regains it easing
back). **Operator, live:** *"during forward flight to gate 0 we point the camera very far down — so
much that we have trouble seeing not just the next gate but even the current one. If the policy were
less aggressive with the head-down thrust-forward, we'd have seen more and had better success."* The
telemetry is that statement in numbers: the gate-0 leg is a **lose-gate / regain-gate oscillation
driven by pitch**, not an altitude overshoot. **The real lever for gate 0 is a less aggressive
head-down forward attitude** (keep the gate in the camera), not the vertical.

(Why the −0.5 m bias nonetheless got them through gate 0 is not established — the approach pitch
profile is basically unchanged with the bias on. Its value here was **exposing the next wall**, not
proving a vertical root cause. Do not read the bias as a fix.)

## The next wall: map-free course-following (horiz sector ≡ 0, no gate-ID track)

After gate 0 both drones lost the *course*. The `coarse_sector` **horizontal** channel — the "which
way does the course turn next" signal — is **hard-pinned to 0 for every tick of both flights**
(`sector_mode=auto` cannot compute the turn without a map; verified `unique=[0.0]`). Trained with
the real turn sign, deployed with 0, the policy behaves as if **every next gate is dead ahead**.
That produces exactly the two operator-observed crashes:

- **vpefwh2** → *operator:* "after gate 0 it turned **up and to the left, where there was nothing** —
  the detector never saw a gate there." Telemetry: one fleeting detection at **+18 m left, 27 m out**
  (conf 1.0 for ~2 ticks) right after the pass, then **conf = 0.00 for ~4 s straight**, thrust
  collapses to ~0, and it drifts/twirls into the **environment** (collision id 1002). It chased a
  ghost, then flew blind into a wall.
- **vpeffs0** → *operator:* "it didn't hit gate 1 or 2 — it **skipped both** (they were off to the
  right) and flew into **gate 3**, which was dead ahead; after some twirling it 'discovered' gate 3
  and drove into it." Telemetry: the bearing sweeps left→right→center while the **forward distance
  falls monotonically 22 → 1.8 m** into one straight-ahead gate (collision id 1001). With no turn
  signal it never went for the correct right-turn gates.

This is the between-gate **gate-association / course-following** problem, not altitude and not
thrust. Real fixes are architectural: **populate the horizontal sector** (a coarse course/turn model
or a light map) and/or add a **gate-ID association+tracking layer** so slot0 carries the *correct*
next gate rather than the most visible one.

## Per-flight facts

| | vpefwh2 + zbias | vpeffs0 + zbias |
|---|---|---|
| Gate 0 | ✅ **PASSED @ 1.97 s** | ✅ **PASSED @ 1.92 s** |
| Final | CRASH into **environment** (1002 ×3), gates=1, 200 ticks/8.2 s | CRASH into a **gate** (1001 ×1, =gate 3 per operator), gates=1, 80 ticks/4.5 s |
| Post-gate-0 | ghost-chase + 4 s blackout + thrust cut → drift into wall | straight-ahead lock onto gate 3, skipped the right-turn gates |
| Handover | rates @ 0.217 s | rates @ 0.220 s |
| Loop rate | 24.0 Hz (27% over budget) | 17.2 Hz (59% over budget) — heavily dt-OOD |
| Spin | net heading +183° (post-gate-0 flail); no sustained corkscrew | net heading +27° (tight); no corkscrew |
| Attitude | finite/sane, roll to +1.28 rad in the flail | finite/sane, roll ±0.9 rad |

Both loops choked (24.0 / 17.2 Hz) — the between-gate behavior is dt-OOD-confounded; the gate-0
pass and the camera-down signature are robust to it.

## Bundle

`handoff/ego-flight-vpef-zbias-2026-07-12/{vpefwh2,vpeffs0}/` — each: `ego_obs.jsonl`,
`ego_timing.jsonl`, `mavlink.tlog` (185 Hz IMU + collisions), `meta.json`, raw `console.log`.
Video local-only (`data/runs/20260712_064812_..._f1`, `..._065011_..._f1`); flight-only 4× clips
were rendered for review on request.

## MEMORY-DELTA

- **vpef + temp −0.5 m perceived-gate z-bias (2026-07-12): BOTH ckpts PASSED gate 0 — first ego gate
  passes ever** (gates=1 vs gates=0 unbiased). Bias was one local line in
  `rel_pos_body_frd_from_gatepose` (policy-obs only, estimator untouched), reverted after.
- **CORRECTION to the vpef report: gate 0 did NOT fail from over-climb.** Root cause (operator +
  telemetry) = **perception loss from over-aggressive head-down**: camera is +20° mount, policy
  pitches to −50° → camera looks ~−30°, **loses the gate** (pose_seen→0) until it eases pitch. Gate-0
  leg is a pitch-driven lose/regain-gate oscillation. Real lever = **less aggressive head-down
  forward attitude**, not vertical.
- **Next wall = map-free course-following.** `coarse_sector` HORIZ is **pinned 0** in deploy
  (`sector_mode=auto`, no map) — policy has no turn-direction signal, flies as if every next gate is
  straight ahead. → vpefwh2 chased a ghost detection up-left into the env; vpeffs0 skipped the
  right-turn gates 1–2 and drove into straight-ahead gate 3. Needs horiz-sector populated (course/turn
  model or map) and/or a gate-ID association+tracking layer.
- Loop 24.0 / 17.2 Hz (both choked, dt-OOD); gate-0 + camera-down findings robust to it.
