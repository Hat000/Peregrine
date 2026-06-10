# Perception characterization — ShadowPC (sim host), 2026-06-08

Repo @ `016bdbd` (pulled origin/main first). venv: `.venv\Scripts\python.exe`.
Two new scripts exercised: `scripts/race_outcome.py` (Part 1) + `scripts/characterize_perception.py` (Part 2).

## Session dir used
`data/runs/20260607_194615_course_60s` — the canonical VQ1 **6/6 full-course finish** (sim-official
≈35.3 s per the rung23 GUI ground-truth). `meta.json`: `max_gates=6`, `final_state=FINISHED`,
`gate_index=6`, `collisions=0`, 1159 FPV frames + `mavlink.tlog`. Analyzed per-gate in
`handoff/shadowpc-reverify-2026-06-07/rung23/REPORT.md`. (The three later `*_vq1` runs — `200505`,
`200650`, `200906` — are equivalent 6/6 finishes and score IDENTICALLY on the oracle; see Part 1.)

---

## Part 1 — race_outcome.py  ✅ ran clean (with an important caveat)

```
.venv\Scripts\python.exe scripts\race_outcome.py data\runs\20260607_194615_course_60s ^
    --json handoff\perception-char-2026-06-08\race_outcome_course_60s.json
```
Full result dict (`race_outcome_course_60s.json`):
```json
{
  "started": true, "finished": false, "started_t": 1780861642.4060328,
  "finish_t": null, "recognized_time_ns": null,
  "gates_passed": 5, "max_active_gate_index": 5,
  "passes": [
    {"gate": 0, "t": 1780861652.169296,  "contact": false, "threat_level": 0, "verdict": "PASS-CLEAN"},
    {"gate": 1, "t": 1780861657.176792,  "contact": false, "threat_level": 0, "verdict": "PASS-CLEAN"},
    {"gate": 2, "t": 1780861662.686324,  "contact": false, "threat_level": 0, "verdict": "PASS-CLEAN"},
    {"gate": 3, "t": 1780861670.451057,  "contact": false, "threat_level": 0, "verdict": "PASS-CLEAN"},
    {"gate": 4, "t": 1780861675.2094388, "contact": false, "threat_level": 0, "verdict": "PASS-CLEAN"}
  ],
  "n_pass_clean": 5, "n_pass_contact": 0,
  "n_gate_collisions": 0, "n_gate_collisions_no_pass": 0, "unmatched_gate_hits": [],
  "n_env_collisions": 0, "env_collisions": [],
  "had_pre_race_residue": false, "clean_finish": false,
  "session": "data\\runs\\20260607_194615_course_60s"
}
```

### ⚠️ Oracle reports 5, NOT 6, finished=False — the recorder-truncation gotcha (not a bug, not a bad pick)
The oracle counts a pass on the `RACE_STATUS.active_gate_index` increment G→G+1. The recording shows
0→1→2→3→4→5: **gates 0–4 passed CLEAN, 0 contact, 0 collisions**, gate 5 active at the end. The pass of
gate 5 (the 6th gate) is the increment 5→6, which **coincides with the finish** — and `fly_vq1`'s
force-disarm fires at that instant, ending the recording **before** that final `RACE_STATUS`/finish flag
hits the tlog. So `race_finish_time_ns = -1` / `finished=False` in every one of these recordings
(`memory/race-outcome-recording-gotchas.md`: "recorder truncates before sim finish"). Timing
corroborates: `started_t`→gate-4 = **32.8 s**; the unrecorded gate-5/finish at ≈35.3 s (GUI) is ~2.5 s
later — exactly where it should be.

**All four 6/6 recordings score the same** (`race_outcome_allruns_stdout.txt`):

| run | started | finished | gates_passed | clean/contact | note |
|---|---|---|---|---|---|
| `194615_course_60s` | True | False | 5 | 5 / 0 | canonical; `had_pre_race_residue=False` |
| `200505_vq1` | True | False | 5 | 5 / 0 | "discarded leading RACE_STATUS residue from a PRIOR race" |
| `200650_vq1` | True | False | 5 | 5 / 0 | same prior-race residue discarded |
| `200906_vq1` | True | False | 5 | 5 / 0 | clean leading epoch |

**For the commander:** oracle AUTHORITATIVELY confirms **5 clean gate passes, 0 contact, 0 collisions**
on the recorded telemetry; the true **6/6 finish is GUI-confirmed only** (35.3 s) — the 6th pass is
structurally unrecordable in the current force-disarm-at-finish flow. Two repeat runs exercised + correctly
handled the "prior-race RACE_STATUS residue" gotcha.
👉 Suggest a `fly_vq1` fix: hold ~0.5–1 s past the finish flag (or capture the final `RACE_STATUS`
before disarm) so a recording self-certifies 6/6 + the official time.

---

## Part 2 — characterize_perception.py  ✅ ran (detector installed this session)

`characterize_perception.py` takes `--bundle DIR` (a `frames.json` PNG bundle), not a session dir, and
needs the YOLO detector. `.venv` had **no ultralytics/torch** (the prior `data/runs/_ultralytics_install.log`
never finished). Installed this session (user-approved): `ultralytics-8.4.62 + torch-2.12.0` (CPU) — see
`pip_install_ultralytics.txt`. Then built a full-course bundle (the shipped `export_frame_bundle.py` is
gate-0-only; wrote `scratch/export_course_bundle.py`, torch-free) and ran the chain.

### How it was run (faithful, on the same session)
Per-gate bundles from `course_60s` (race-window frames: `speed>1 m/s`, `range<=26 m`, sampled evenly in
time; 60 frames each), then `characterize_perception.py` per gate. **Why per-gate:** the script's
"modal-gate, |fix|<3 m" robust subset + TAIL are designed for a **single-gate** bundle. On a multi-gate
course it lumps every fix to a *different* gate into "TAIL" (so a full-course run reported a meaningless
90% tail; the per-frame chain was fine). Commands in `commands.txt`; raw per-gate output in
`characterize_pergate_stdout.txt`; full-course run kept in `characterize_course_60s_stdout.txt` for the record.

### Headline: the perception-noise model (pooled 6 per-gate runs, N=312 solved fixes)
Magnitude (|fix|), not gate-identity, is the error metric — a good fix to the centred **next** gate is
still accurate, so "wrong-gate" overcounts. (`aggregate_perception_stdout.txt`.)

**NOISE FLOOR — per-axis bias + std** (world N/E/D fix error vs the given pose):

| subset | N | bias N | std N | bias E | std E | bias D | std D | \|fix\| p50 / p90 |
|---|---|---|---|---|---|---|---|---|
| all good fixes (\|fix\|<3 m) | 169 (54%) | −0.39 | 0.82 | +0.03 | 0.56 | −0.30 | 0.39 | 0.89 / 1.81 |
| **KF-ACCEPTED** (\|fix\|<3 m & maha<χ²) | 135 (43%) | **−0.42** | **0.73** | **+0.06** | **0.47** | **−0.28** | **0.29** | **0.82 / 1.66** |
| correct-gate only (assoc==nearest) | 90 (29%) | −0.17 | 0.77 | +0.04 | 0.41 | −0.26 | 0.28 | 0.62 / 1.35 |

→ **Noise floor ≈ 0.8 m typical (p50), ~1.7 m p90; per-axis σ ≈ 0.7/0.5/0.3 m (N/E/D); small biases
(N −0.4 m, D −0.3 m, E ~0).** D (vertical) is the tightest axis. Flat across range (no growth 0→24 m):

```
 band     n  biasN biasE biasD |fix|p50 |fix|p90   (correct-gate good fixes)
 0-5m    14  -0.63 +0.01 -0.28    0.72     0.98
 5-10m   27  +0.13 -0.08 -0.22    0.60     1.75
 10-15m  36  -0.18 +0.07 -0.31    0.53     1.29
 15-24m  13  -0.24 +0.26 -0.17    0.77     1.51
```
The script's own per-gate **robust-subset (modal gate, |fix|<3 m) noise floor** agrees: clean |fix| p50
**0.43–1.14 m**, p90 0.98–1.72 m across the 6 gates. Per-axis linear fits give an angular/calibration
slope of **≈ −3.8°…+4.1°** (the ~3.6°/range yaw bias Task 2 flagged, now through the full PnP chain).

### Flip / wrong-gate TAIL rate — and why it's mostly defended in-loop
- **Raw catastrophic rate (|fix|≥3 m): 46 %** (143/312) under *naive nearest-centre association* on the
  receding 6-gate course (87 wrong-gate locks + 56 correct-gate frontal-PnP depth flips; tail |fix| p50
  16 m, max 139 m). This is the hazard, NOT the in-loop rate.
- **The KF covariance gate kills it:** catastrophic fixes carry a huge Mahalanobis — **97 %** of |fix|≥3 m
  fixes exceed χ²₃,₀.₉₉₉=16.27 and would be **rejected** by the filter's gate. **Residual leak = 1.6 %**
  (5/312) of solved fixes are bad AND slip the gate → *that* is the bad-fix rate the RL twin must model.
- **Cost / caveat:** the analytic PnP covariance is a touch tight — it also rejects **20 %** of GOOD
  (|fix|<3 m) fixes (their maha>16.27). Recommend modestly **inflating the PnP fix covariance** (beyond the
  existing `P3P_FIX_COV_INFLATION`) so the gate keeps catching the 97 % bad without dropping good fixes.

### One-line RL-twin perception model (VQ1-given-pose regime)
Per-axis Gaussian world-fix noise **σ≈[0.73, 0.47, 0.29] m (N,E,D)** + bias **[−0.4, +0.06, −0.28] m**,
range-independent to ~24 m, + small ≈±3° angular term; detector assoc rate ~85–95 % in 5–15 m; plus a
**~1.6 % residual catastrophic-fix leak** after a χ²₀.₉₉₉ Mahalanobis gate (raw 46 % pre-gate). Detector
drop/assoc by range in `characterize_pergate_stdout.txt`.

---

## Files
- `race_outcome_course_60s.json` / `race_outcome_stdout.txt` — Part 1 result + console.
- `race_outcome_allruns_stdout.txt` — all four 6/6 runs scored.
- `pip_install_ultralytics.txt` — detector install log (Successfully installed torch-2.12.0, ultralytics-8.4.62).
- `characterize_pergate_stdout.txt` + `characterize_g{0..5}.json` — the 6 per-gate runs (raw).
- `aggregate_perception_stdout.txt` — the pooled noise model (the headline numbers above).
- `characterize_course_60s_stdout.txt` + `characterize_course_60s.json` — the full-course run (kept for
  the record; its 90% TAIL is the modal-gate artifact, see above).
- `course_bundle/` and `pg/course_g{0..5}/` — the frame bundles (PNGs + frames.json).
- `scratch/export_course_bundle.py` — torch-free multi-gate / per-gate bundle exporter.
- `scratch/aggregate_perception.py` — the proper magnitude-based pooled decomposition.
- `commands.txt` — every command run, verbatim.

### Notes for the commander
1. Part 1: VQ1 6/6 is real (GUI 35.3 s) but recordings only self-certify **5 clean / 0 contact** — fix
   `fly_vq1`'s disarm timing to capture the 6th pass.
2. Part 2: perception noise floor is **sub-metre (~0.8 m p50)** and **range-flat to 24 m**; the scary 46 %
   raw flip/wrong-gate tail is **97 % rejectable by the KF χ² gate** (leak ~1.6 %), but the PnP covariance
   is slightly tight (drops 20 % of good fixes) — **inflate it**.
3. The detector now lives in `.venv` on ShadowPC (torch CPU). Re-run Part 2 anytime via `commands.txt`.
