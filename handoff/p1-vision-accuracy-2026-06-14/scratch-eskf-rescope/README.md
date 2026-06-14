# scratch-eskf-rescope — IMU-bias ESKF re-scope (P1-ESKF-RESCOPE)

DESIGN-ONLY supporting evidence for `../ESKF_RESCOPE_DESIGN.md`. No `src/` edits. Pure numpy/scipy
(py3.13, numpy 2.2 / scipy 1.17).

**Dependency:** these sims import `../scratch-eskf/eskf_geometry.py` (the realised-distribution geometry +
real track_map loader), which lives on branch `origin/p1-eskf-design`. To reproduce from this branch alone:
`git checkout origin/p1-eskf-design -- handoff/p1-vision-accuracy-2026-06-14/scratch-eskf/` first.

Run:
```
python accelbias_realistic.py        # TRUSTWORTHY: continuous trajectory + constant-rate fixes -> the RATE verdict
python accel_bias_observability.py   # geometry-generator study (gyro-dead + illustration; superseded on RATE)
python validate_accelbias.py         # instrument-validation checks (dense-fix / many-lap / maneuver limits)
```

Files:
- `accelbias_realistic.py` (+ `_console.txt`, `_results.json`) — **the authoritative accel-bias instrument**:
  continuous race trajectory, CONSTANT realised fix rate, 90 Hz sub-stepped predict (production Q). Reports
  2-axis (Y,Z) sigma vs laps, the 3-axis-vs-2-axis NEES contrast (along-track X overconfidence), laps-to-budget,
  drift floor, gyro-dead.
- `accel_bias_observability.py` (+ artifacts) — geometry-generator study. Confirms gyro-DEAD and the H_β
  re-target; its accel-bias RATE numbers are over-optimistic (artificial inter-gate gaps) -> use the realistic one.
- `validate_accelbias.py` — sanity checks that the cov instrument behaves per observability theory.

Headline: GYRO = DEAD (not built). ACCEL = LIVE 2-axis body lateral(Y)+vertical(Z) (margin in-plane axes),
NEES 1.97/2 honest, under budget (g·sin0.6°=0.103 m/s²) in ~2 laps @fr0.07; along-track(X) DROPPED
(unobservable, NEES 18). Boresight = OFFLINE bake (calib-v2). Watchdog = read-only monitor.
