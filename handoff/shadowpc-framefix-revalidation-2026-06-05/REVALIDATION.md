# Frame-fix re-validation — REVALIDATION (Phases 1–4)

**Date:** 2026-06-05 · **Mode:** OFFLINE replay of recorded saga runs (no live sim, no flight, no
controller change, no re-tuning). **Fix under test:** `c3b5a8e` (rotate ODOMETRY body-FRD twist →
world NED via `frames.world_vec_from_body_quat`). **Pre-registration:** `UNDERSTANDING.md`.
**Script:** `revalidate.py` (Phases 1–3); selection in `scratch/select_probe.py`; raw stdout in
`scratch/phase{1,2,3}_results.txt`. All numbers below are computed from the recorded RAW MAVLink
messages — no pre-fix checkout (the OLD pass neutralizes the rotation to identity in the production
`_handle`, so only the velocity frame differs between OLD and NEW).

**Runs used:** `20260604_025347_gate0_course1` (PRIMARY — clean, dense, FINISHED, yaw locked −180°);
`20260604_120918_gate0_roll1` (§5-magnitude lateral at −180°; sparse ~10–16 Hz);
`20260604_121555_gate0_roll2` (ADD #2 cross-attitude, real-motion segment only).
`20260604_145110_gate0_given1` was **discarded** — frozen telemetry (see Observation F).

---

# OBSERVATIONS (raw numbers)

## A. Frame enums (ADD #1) — definitive, not inferred
Every ODOMETRY message in every run, **100% constant**:
- `ODOMETRY.frame_id       = 1  (MAV_FRAME_LOCAL_NED)`  ← the **world** frame the *pose* (x/y/z) lives in
- `ODOMETRY.child_frame_id = 8  (MAV_FRAME_BODY_NED)`   ← the **body** frame the *twist* (vx/vy/vz) lives in

The sim **declares** the velocity in a body child-frame distinct from the world pose frame. (It uses
the deprecated `BODY_NED`=8 rather than `BODY_FRD`=12 named in the fix comment; both are body-fixed,
z-down — the exact convention is validated empirically in C.)

## B. The bug is present in the saga data — raw disagrees, rotated agrees (Phase 1 / Check 1)
At the −180° course heading, RAW body-velocity-stored-as-world vs LPN world truth, and the same body
twist ROTATED→world via the ODOMETRY quaternion vs truth (vec-RMS, m/s):

| run | RAW (bug) | ROTATED (fix) | vx sign-flipped | vy sign-flipped |
|---|---|---|---|---|
| course1 | **7.489** | **0.0032** | 482/482 (100%) | 40/61 (66%) |
| roll1   | 5.854 | 0.0045 | 669/669 (100%) | 332/332 (100%) |
| roll2   | 5.489 | 0.0050 | 941/941 (100%) | 608/608 (100%) |

Concrete rows (`body → truth → rotated`):
- course1 yaw −176°: vx **+4.57 → −4.62 → −4.62**
- roll1 yaw −174°: vy **−2.16 → +1.92 → +1.91** (the §5-class cross-track magnitude)

**Truth self-consistency** (LPN world velocity vs robust central-difference d(pos)/dt, moving samples):
course1 median **0.096** / RMS 0.277; roll1 median **0.084** / RMS 0.243. This ~0.09 m/s floor is
**recv-clock jitter + stale/quantized LPN position**, NOT a velocity error — the independent
rotated-ODO==LPN agreement (0.003–0.005) is the tight triangulation (two sensors agree to mm/s).

## C. Rotation validated OFF-LEVEL, all 3 axes, on REAL motion (ADD #2)
Filtered to real motion (LPN velocity corroborated by d(pos)/dt — rejects frozen telemetry).
rotated-vs-truth RMS (m/s):

| run | real-motion roll / pitch range | `|roll|>8°` | `|pitch|>8°` | `|roll|>8° & |pitch|>5°` |
|---|---|---|---|---|
| roll2 | [−11°,+12°] / [−17°,+10°] | 0.0084 (n=345) | 0.0132 (n=156) | **0.0137 (n=52, max 0.055)** |
| roll1 | [−4°,+10°] / [−17°,0°]    | 0.0059 (n=85)  | 0.0111 (n=36)  | — (not co-exercised) |
| course1 | [−2°,0°] / [−17°,+2°]   | —              | 0.0052 (n=163) | — |

At the simultaneous-roll+pitch samples the RAW (unrotated) error is **4.59 m/s** vs **0.0137 m/s**
rotated. **No attitude-dependent growth** (0.008 / 0.013 / 0.014 are all tiny and comparable) ⇒ the
full-quaternion rotation and body convention are correct on all 3 axes within the real flight
envelope (~±12° roll, ~±20° pitch). The 53° roll that appeared in selection was frozen garbage (F),
not real flight, so the rotation is *not* claimed validated beyond ~12° roll from this data.

## D. DECISIVE replay — does the fix make the KF track truth? (Phase 2 / Check 3)
Production `MavlinkClient._handle` + `Navigator` replayed twice over the recorded stream; identical
except `velocity_ned` (OLD = raw body interleaved with world LPN, last-writer-wins; NEW = rotated).
Compared to LPN truth over the −180° moving segment:

**course1 (clean, dense, primary):**
| | OLD (buggy mix) | NEW (rotated fix) |
|---|---|---|
| KF-vel error vec-RMS | **2.410 m/s** | **0.051 m/s** |
| forward vx: truth / KF | −3.658 / −1.357 = **0.37×** | −3.658 / −3.660 = **1.00×** |
| forward vx: *ingested measurement* | **−0.753 = 0.21× truth** | −3.656 = 1.00× |

**roll1 (§5-magnitude lateral; sparse):**
| | OLD | NEW |
|---|---|---|
| KF-vel error vec-RMS | 1.848 | 0.346 |
| forward vx: truth / KF | −2.779 / −1.109 = 0.40× | −2.779 / −2.780 = **1.00×** (axis RMS 0.003) |
| lateral vy: truth / KF | −0.474 / −0.305 = 0.64× | −0.474 / −0.581 = 1.22× |
| lateral vy: ingested meas | −0.176 = 0.37× truth | −0.476 = 1.00× truth |

- The §5 "reported ≈ ¼ of truth" reproduces **only under OLD**: the mixed measurement is **0.21×**
  truth (course1 forward) — matching §5's −0.22/−0.92 ≈ 0.24.
- NEW tracks truth with **no residual lag** on clean data (course1 0.051 m/s; forward 1.00×).
- roll1's NEW lateral residual (0.346) is **10–16 Hz sparsity** smoothing of a fast vy transient,
  **not** frame lag — proven by the *perfect forward recovery* (0.003 RMS, 1.00×) on the same run.

## E. Data flow — what the controller actually consumed (Phase 3 / Check 0)
Code facts: KF `navigator.py:281‑285` → `update_velocity(ds.velocity_ned, std=0.10)`; workaround
`fly_vq1.py:355‑358` → `gs = client.state; rawv = gs.velocity_ned; vel = [rawv[0], rawv[1], KF_vz]`.
**Both read `ds.velocity_ned` — the interleaved mix.** What the "raw given horizontal velocity"
workaround fed the lateral loop at −180°:

| run | forward fed/truth (sign-flipped %) | lateral fed/truth (sign-flipped %) |
|---|---|---|
| course1 | −0.753 / −3.658 = **0.21×** (40%) | −0.220 / −0.384 = **0.57×** (25%) |
| roll1   | −0.687 / −2.773 = 0.25× (38%)     | −0.172 / −0.439 = 0.39× (37%) |

(course1's lateral fed value is literally **−0.220** — i.e. §5's "−0.22" was this mixed-field value.)
The "pristine raw given velocity" the lateral fix switched TO was the **same corrupted mix** — at
−180° even more cancelled than the KF output.

## F. Recording-cleanliness findings (bonus; for "fit a twin to")
- **`given1` is frozen telemetry**: velocity pinned at exactly **13.13 m/s** and roll at **53.4°** for
  the first 970 s (only the last 51 s is real, at rest), despite meta `final_state=FINISHED`. Its 73k
  "attitude-varied" samples were one frozen pose — discarded.
- **`roll2` is partly frozen** (global d(pos)/dt error median 25.5 m/s); only its early real-motion
  banking segment (selected by the corroboration filter) is usable.
- recv-time clock has **jitter + sub-ms burst duplicates** (given1: 1992; "deduped" before
  differencing); given1 also had an **epoch reset (`reset_counter=5`) + a 61 m position teleport**.
- LPN **position field is stale/quantized between updates** (median |Δpos|=0 per 96 Hz step), so naive
  d(pos)/dt is unreliable; velocity fields are fine.
- **Clean dense runs (course1) ARE twin-quality on velocity**: two independent world-velocity sensors
  (LPN velocity, rotated-ODO twist) agree to ~0.003 m/s.

---

# INTERPRETATION

## Pre-registered outcome: **CONFIRMED** (outcome 1 of `UNDERSTANDING.md` §6)
With the fix, the re-run KF velocity **tracks truth** (course1 vec-RMS 0.051 m/s, forward 1.00×; no
residual lag), the raw ODOMETRY velocity **was** body-frame and **sign-flipped at −180°** (100% of
moving-axis samples, all three runs), and the rotation **validates across pitch and roll** (≤0.014 m/s
to ±12°/±20°). The §5 "KF velocity lags 4×" signature (reported ≈ ¼ of truth) reproduces **only under
the OLD mixed velocity** (measurement 0.21× truth) and **vanishes under the fix**. The "Partial"
outcome (residual real lag on clean velocity) is **rejected** — the only NEW residual is data-rate
sparsity (roll1 lateral), and the bug-affected forward axis recovers perfectly. "Refuted" is rejected
by the frame enums + 100% sign-flips.

## Saga findings — explained / retired vs still real

**RETIRE / recast (now explained by the frame bug):**
- **§5 "KF velocity LAGS the truth ~4×."** It was **not a KF lag** — it was the frame mix: at −180°
  `velocity_ned` interleaved a sign-flipped body velocity into the KF measurement (and the raw given
  field), cancelling to ~0.2–0.4× of truth. The fixed KF tracks truth to 0.05 m/s. Retire the "4× KF
  lag" framing.
- **§5 "control on the raw GIVEN horizontal velocity (KF lags)."** Premise **invalid**: the raw given
  velocity (`ds.velocity_ned`) was the *same* corrupted mix (0.21–0.57× truth at −180°, 25–40%
  sign-flipped), not pristine. The lateral-velocity workaround was **fighting a phantom**. With the
  frame fix, `ds.velocity_ned` is frame-consistent, so the per-axis raw-vs-KF distinction in fly_vq1
  is moot for the *frame* problem (a control-policy choice, no longer a bug workaround).
- **§5 weak cross-track damping / lateral-velocity part of the oscillation.** The corrupted lateral
  velocity (weak + intermittently wrong-signed) is a real mechanism for weak/erratic cross-track
  damping; explained by the same fix.

**STILL REAL / unaffected by this fix (do NOT retire):**
- **§1 ODOMETRY-quaternion ROLL inversion** — a *separate* attitude-feedback sign bug (positive-
  feedback lateral loop). Independent of the velocity frame; the lateral oscillation had (at least)
  this second real cause. Keep `odo_att_sign / odo_rate_sign / rate-sign` roll fix.
- **§7 altitude balloon / sim auto-thrust** — separate vertical-thrust mechanism, OPEN. Untouched.
- **§2 gate centre, §3 gate frame, §4 unreliable live map, §6 detector false-pass** — unrelated to
  velocity frame; still real.
- **The fix `c3b5a8e` itself is correct** — validated independently here on all three axes (B, C, D).

## Practical follow-ups (not done here — offline, no changes made)
1. Update `project_ctbr_control_sysid.md` §5: replace "KF velocity lags 4×" with "ODOMETRY body-frame
   velocity bug (fixed `c3b5a8e`); KF now tracks truth (0.05 m/s)"; recast the raw-given-velocity note.
2. Re-examine whether the lateral oscillation still needs the per-axis raw-velocity control now that
   `velocity_ned` is frame-consistent (likely the §1 roll fix was the real lateral cure).
3. Recordings for twin-fitting: filter frozen/garbage runs (given1, roll2 tail) and the recv-clock
   bursts; prefer clean dense runs (course1) and trust the velocity fields, not naive d(pos)/dt.

**Net:** the fix is correct and **explains the §5 saga pain** (KF "lag" + the raw-velocity workaround
premise). §1 (roll) and §7 (altitude) remain real and separate.
