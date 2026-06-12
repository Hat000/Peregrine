# MANIFEST — debug_obs_17runs.zip

**Purpose:** Laptop joint translational refit (collective-vs-airspeed thrust lapse + drag)
from 2026-06-12 ShadowPC inc6 live sessions. No new flights needed — all 17 recordings
captured today under `stage1_inc6_actor.pth` (md5 `8fb8855e07d4fd01045e7b2ecbc5acd3`).
All runs instrumented with `--debug-obs ON`; each directory contains `debug_obs.jsonl` +
`meta.json`. Raw `mavlink.tlog` and `video.*` are NOT included (~51–129 MB/run) — not
needed for translational refit.

**Commit at time of packaging:** c5f8bcd (post roll-convention fix bcc93f9).

## Contents

Each zip entry path: `{run_name}/debug_obs.jsonl` and `{run_name}/meta.json`.

| # | Run name | Start mode | Outcome | Key data | Why included |
|---|----------|-----------|---------|----------|--------------|
| 1 | `20260612_034852_inc6_standing_f1` | standing | SPIN_ABORT (74 s) | 0 gates; ~2200 ticks | Pre-fix failure: roll-mirror spin → OOD; covers initial accel phase |
| 2 | `20260612_035014_inc6_standing_f2` | standing | SPIN_ABORT (20 s) | 0 gates; ~600 ticks | Short early abort; accel phase only |
| 3 | `20260612_035048_inc6_standing_f3` | standing | CRASH (27 s) | 0 gates; ~800 ticks | Crash into env; accel + some cruise |
| 4 | `20260612_035131_inc6_standing_f4` | standing | SPIN_ABORT (20 s) | 0 gates; ~600 ticks | Early abort; accel phase |
| 5 | `20260612_035205_inc6_standing_f5` | standing | TIMEOUT (120 s) | 0 gates; ~3600 ticks | Full 120 s run; widest speed range in failure set |
| 6 | `20260612_035415_inc6_standing_f6` | standing | SPIN_ABORT (28 s) | 0 gates; ~840 ticks | Accel + early cruise |
| 7 | `20260612_035458_inc6_standing_f7` | standing | TIMEOUT (120 s) | 0 gates; ~3600 ticks | Full 120 s; widest speed range |
| 8 | `20260612_035708_inc6_standing_f8` | standing | TIMEOUT (120 s) | 0 gates; ~3600 ticks | Full 120 s; widest speed range |
| 9 | `20260612_035918_inc6_standing_f9` | standing | TIMEOUT (120 s) | 0 gates; ~3600 ticks | Full 120 s; widest speed range |
| 10 | `20260612_040128_inc6_standing_f10` | standing | SPIN_ABORT (23 s) | 0 gates; ~690 ticks | Accel phase; last pre-fix standing run |
| 11 | `20260612_040440_inc6_bridge_f1` | bridge | TIMEOUT (120 s) | 1 gate; ~3600 ticks | Post-gate-0 wander; best long-duration bridge run |
| 12 | `20260612_040748_inc6_bridge_f2` | bridge | TIMEOUT (120 s) | 1 gate; ~3600 ticks | Same failure mode; large-radius orbit |
| 13 | `20260612_041004_inc6_bridge_f3` | bridge | SPIN_ABORT | 1 gate | Short post-gate-0; spin accumulation |
| 14 | `20260612_041035_inc6_bridge_f4` | bridge | SPIN_ABORT | 1 gate | Short; same as f3 |
| 15 | `20260612_041113_inc6_bridge_f5` | bridge | STALLED | 1 gate | Stall after gate-0 pass; different terminal |
| 16 | `20260612_044219_inc6_rollfix_f1` | standing | TIMEOUT (post-fix) | 0 gates; ~3600 ticks | **POST-FIX (bcc93f9)** — no spin, smooth flight, ~5 m +y gate-0 miss; primary thrust-lapse datum |
| 17 | `20260612_044438_inc6_rollfix_f2` | standing | TIMEOUT (post-fix) | 0 gates; ~3600 ticks | **POST-FIX (bcc93f9)** — same residual; second independent lapse measurement |

## Not included

- `20260612_040236_inc6_bridge_f1` — aborted UDP run (sim crash mid-session); not instrumented to completion
- `20260612_034154_mixer_probe2`, `…034440…`, `…034640…` — mixer probe (open-loop): separate refit target, not translational dynamics
- All `mavlink.tlog` / `video.*` files (~51–129 MB per run) — not needed for translational refit

## Refit target

From diag §6 (`diag_thrust_lapse.py`):
- **Thrust lapse:** smooth-tick ratio `F_measured / K(collective)` = **0.74–0.88 at 3–12 m/s**
  (n≈1700 ticks), ≈1.0 at 12–18 m/s, noisy 0.4–1.3 above. Axial-speed structure not in model.
- **Drag:** collective map fit at near-zero airspeed; needs drag-vs-airspeed from cruise phases.
- Refit runs 16–17 (rollfix) are the cleanest lapse signal: correct conventions, smooth flight,
  known initial conditions. Runs 1–10 (pre-fix) cover the same 3–12 m/s regime but with
  roll-mirror contamination — use carefully (rate channel forensics still valid per diag §6).
- Runs 11–15 (bridge) enter at ~5 m/s and reach cruise; useful for drag at higher airspeed.

## Column reference (debug_obs.jsonl)

Line 0 is a JSON `type=header` with `obs_labels`.
Subsequent lines are per-tick records with fields:
- `t_mono`, `sim_time_ns` — wall and sim timestamps
- `pos_ned`, `vel_ned` — NED position/velocity (pristine from simulator)
- `q_raw_wxyz` — raw ODOMETRY quaternion (true attitude, AS-IS after bcc93f9)
- `w_raw` — raw ODOMETRY angular rate (true rate after [+1,−1,+1] sign correction)
- `obs` — policy observation vector (labels in header)
- `rate_frd`, `collective` — wired commands (FRD frame; collective in hover-normalized units)
- `normed_thrust`, `actor_mean`, `tanh`, `act_rescaled` — policy internals
- `k`, `gate_index`, `odo_age_ms`, `reset_counter`, `n_coll` — bookkeeping
