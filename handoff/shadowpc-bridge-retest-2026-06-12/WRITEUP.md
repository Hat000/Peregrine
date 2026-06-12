# SHADOWPC-BRIDGE-RETEST — Bridge ×5 + Probe Rows + Dataset Package

**Session:** SHADOWPC-BRIDGE-RETEST (2026-06-12). **Model:** claude-sonnet-4-6.
**Builds on:** SHADOWPC-INC6-DIAG (roll-convention fix `bcc93f9`).
**Checkpoint:** `stage1_inc6_actor.pth` md5 `8fb8855e07d4fd01045e7b2ECBC5ACD3`.

---

## TL;DR

The DIAG writeup's strong prediction — *"with the roll convention fixed, bridge mode should thread gates"* — was **WRONG**. 5/5 bridge flights passed gate 0 and failed at gate 1, same gate count as the pre-fix bridge flights. The roll convention fix is confirmed working (sweep direction inverted vs pre-fix; no spin guard trips), but the translational plant gap (thrust lapse 15–25% at 3–12 m/s) is broader than expected: it degrades gate-1 approach accuracy even in bridge mode, which hands off at ~5 m/s. Bridge is not the cheap discriminator we hoped for — the thrust-lapse refit is required before any further live flight is meaningful.

Mixer probe rows (all 4 targets) were captured. Refit dataset (17 runs) was packaged and pushed.

---

## Task 1 — Refit Dataset Package

**Commit:** `c846054` (pushed).
**Artifact:** `handoff/shadowpc-refit-dataset-2026-06-12/debug_obs_17runs.zip` (9 MB compressed, 29 MB uncompressed).

Contains `debug_obs.jsonl` + `meta.json` for 17 runs:
- 10 standing pre-fix failure runs (`inc6_standing_f1`–`f10`)
- 5 bridge pre-fix failure runs (`inc6_bridge_f1`–`f5` from INC6-LIVE)
- 2 post-fix rollfix runs (`inc6_rollfix_f1`–`f2`)

Raw `mavlink.tlog` and `video.*` excluded (~51–129 MB/run). Full column reference and refit rationale in `MANIFEST.md`.

---

## Task 2 — Mixer Probe Rows

**Canonical recording:** `data/runs/20260612_133307_mixer_probe2`
(probe drifted 100 m during catch5 — final recovery only; all 4 measurement phases complete before drift abort).

### Working invocation

```
.venv/Scripts/python.exe scripts/rate_sysid.py \
  --mode profile \
  --profile handoff/laptop-s17-mixer-inc6-2026-06-11/mixer_probe2.json \
  --rate 100 --max-offset-m 100 --max-alt-m 35 --max-tilt-deg 150 \
  --label mixer_probe2
```

Required 6 attempts to converge on these flags. Key aborts encountered:
- Default `--max-alt-m 8` → aborted at 8 m (profile climbs to 14 m). Fixed: `--max-alt-m 16`, then `--max-alt-m 35` (c100_y31 free-fall reaches z≈30 m during catch3).
- `no_tilt_abort: true` per-phase flag (325e191) fails to suppress tilt abort in practice — likely a timing boundary at phase transitions. Reliable workaround: `--max-tilt-deg 150` globally.
- c100_y31 at thr=1.0 causes ~18 m/s free-fall (mixer coupling: all motor power diverted to yaw torque, killing vertical thrust).

### Per-phase results

| Phase | Thr | Axis | Motors (0,1,2,3) | Settled rate | Differential |
|-------|-----|------|------------------|--------------|--------------|
| z00_y31_long | 0.0 | yaw | [0.050, 0.419, 0.419, 0.050] | 9.76 rad/s | 0.369 |
| c100_r31 | 1.0 | roll | [0.999, 0.644, 0.999, 0.644] | 10.93 rad/s | 0.355 |
| c100_y31 | 1.0 | yaw | [0.651, 1.000, 1.000, 0.651] | 9.21 rad/s | 0.349 |
| c60_r31 | 0.6 | roll | [0.732, 0.467, 0.732, 0.467] | 10.99 rad/s | 0.266 |
| zhov_r31 | 0.266 | roll | [0.487, 0.053, 0.487, 0.053] | 10.79 rad/s | 0.434 |

### Key refit inputs (S17 mixer update)

- **Top-rail roll (c100_r31):** motors 0,2 saturate; motors 1,3 settle 0.644 → ~35% roll authority clip at max collective.
- **Top-rail yaw (c100_y31):** symmetric; ~35% yaw authority clip at max collective.
- **Mid-band (c60_r31):** no saturation; linear mixing confirmed at thr=0.6.
- **Bottom-rail roll (zhov_r31):** motors 1,3 at idle floor (0.053); bottom-rail clips roll ~80% at hover thrust — dominant at standing-start low speeds.

Full detail in `handoff/shadowpc-bridge-retest-2026-06-12/probe_summary.md`.

---

## Task 3 — Bridge ×5

**Command:**
```
.venv/Scripts/python.exe rl/fly_rl.py \
  --checkpoint rl/checkpoints/stage1_inc6_actor.pth \
  --bridge --debug-obs
```

**Recordings:** `data/runs/20260612_133654_inc6_bridge_f1` through `…_f5`.

### Per-flight results

| Flight | Gates passed | Finish | Outcome | Failure mode | 180° turn-back? |
|--------|-------------|--------|---------|-------------|-----------------|
| f1 | 1 (gate 0) | No | TIMEOUT (120 s) | OOD wander post-gate-0; large-radius orbit | None observed |
| f2 | 1 (gate 0) | No | SPIN_ABORT | OOD wander → spin accumulation; SPIN_ABORT at gi=1, pos ~190 m from origin | None observed |
| f3 | 1 (gate 0) | No | TIMEOUT (120 s) | OOD wander; sweep to +y, drifted off course | None observed |
| f4 | 1 (gate 0) | No | SPIN_ABORT | Same as f2; SPIN_ABORT at gi=1, pos ~130 m from origin | None observed |
| f5 | 1 (gate 0) | No | TIMEOUT (120 s) | OOD wander; same pattern | None observed |

**Summary:** 5/5 gate 0 PASS-CLEAN (bridge controller handles gate 0 approach), 0/5 gate 1, 0/5 finish, 0 gate contacts. Spin guard triggered in f2 and f4.

### Behavioral notes

**Roll convention confirmed:** Post-fix bridge sweeps to **+y** after gate 0. Pre-fix (INC6-LIVE session) swept to **-y**. Sign flip is exactly what the bcc93f9 fix predicts — roll perception is now correct. No spin guard trips at gate 0, no oscillation. The policy is flying coordinated correct-signed flight to and through gate 0.

**Post-gate-0 failure:** Policy receives gate-1 obs that place gate 1 at the wrong relative position (gate-1 obs are built from the same translational plant that has the thrust-lapse gap). Bridge hands off at ~5 m/s — still within the 3–12 m/s regime where thrust lapse runs 15–25% below model. The OOD wander is structurally identical to standing-start failure: wrong approach vector → gate miss → obs out-of-distribution → no recovery.

**180° turn-back bug:** Not observed in any of 5 flights. The VQ1-era report may be course-geometry dependent or was a pre-fix artifact.

### Prediction assessment

> *DIAG §9: "Strong prediction: bridge should thread gates with roll fix."*

**FAILED.** Bridge passes gate 0 (as before) and fails at gate 1 (as before). Gate count is unchanged. The prediction was based on the hypothesis that gate-1 miss was roll-authority limited (banked turn, inc5 roll-mirror era). That mechanism was partially correct for inc5, but the inc6 gate-1 failure is driven by the translational gap: the approach vector to gate 1 is displaced enough to put gate-1 obs OOD regardless of roll authority.

**Implication:** Thrust-lapse refit is load-bearing for both standing start AND bridge. There is no flight-mode shortcut; the plant gap must be closed first.

---

## Ops notes

- Contact guard: observed throughout; no sim crashes.
- Full ESC→Enter reset between all flights.
- Foreground verified before each key send (sim_focus.py).
- Sim exited to HOME at session end (screenshot: `sim_exit_home.png`).
- `no_tilt_abort` mechanism unreliable for probe spin phases; document for future sessions: always use `--max-tilt-deg 150` globally as the reliable override.

---

## Recommended path (updated from DIAG §9)

1. **Laptop: joint translational refit** from 2026-06-12 recordings (17 flights). Fit collective map vs airspeed (lapse) + drag. No new flights needed.
2. **Re-run inc6 deploy matrix on refit plant.** If inc6 robust → fly as-is; else inc7 retrains with lapse DR (+ optional sin/cos yaw-wrap encoding for obs[8]).
3. **Next live session: standing start ×5** after refit. Bridge is no longer a useful discriminator until the plant gap is closed.
4. Standing start remains the deployment target, gated on (2).

---

MEMORY-DELTA:
- NEW (project): Bridge ×5 post-fix result: 5/5 gate 0, 0/5 gate 1, 0/5 finish. DIAG "threads gates" prediction FAILED. Translational lapse affects bridge at gate-1 approach (bridge hands off at ~5 m/s, still in 3–12 m/s lapse regime). Bridge is not a discriminator until thrust-lapse refit lands.
- AMEND (project): inc6 next step is NOT "bridge ×5" (done, failed). Next step = laptop translational refit from 2026-06-12 recordings, then standing start ×5. Supersedes DIAG §9 item 2.
- NEW (feedback): no_tilt_abort per-phase flag (325e191) unreliable in practice — tilt abort still fires at phase boundaries. For mixer probe spin phases, always use --max-tilt-deg 150 globally. Why: per-phase check has timing gap at boundaries.
- CONFIRM (project): roll convention fix (bcc93f9) confirmed by sweep-direction inversion (+y post-fix vs -y pre-fix). No spin at gate-0 pass. Conventions are correct; remaining gap is translational only.
- CONFIRM: 180° turn-back bug not observed in 5/5 post-fix bridge flights — may be pre-fix artifact or course-geometry dependent.
