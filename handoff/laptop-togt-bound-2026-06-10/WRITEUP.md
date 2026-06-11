# Time-optimal bound for the VQ1 course (LAPTOP-TOGT-BOUND, 2026-06-10/11)

**TL;DR — the ceiling is ~4.3 s; our banked VQ1 finish (35.3 s) is 8× off it.**
On our measured plant (T/W 3.765, sustained body-rate authority 11 rad/s r/p, linear drag
0.21/s), the time-optimal lap — standing start on the pad, finish at the gate-5 plane —
is **4.27 s** with gate crossings confined to the inscribed circle of the opening
(Euclidean in-plane miss < 0.75 m, the validity rule exactly as we measure it), or
**4.13 s** if the full 1.5 m square opening counts (the optimum clips gate corners at
~1.06 m Euclidean). **THRUST is the binding constraint** (collective saturated ~84 % of
the lap; halving thrust costs +2.3 s) — **body rates barely bind** (the old wrong
7.85 rad/s model costs only +0.05 s; unbounded rates buy only −0.04 s). The committed
reference line (`rl/reference_line_vq1.json`, margined + centred variant) is **4.55 s**;
tracked through the faithful twin with the existing geometric controller it flies a
**valid 6/6 at ~8.3 s** (the line time-dilated 1.85×) — the ideal-vs-tracked gap is the
controller/architecture cost, not the plant's.

**Mid-session plant correction (twin-falsify, 2026-06-11, §8):** linear drag and the
3.765 g ceiling were both falsified while this ran (real: quadratic drag c2≈0.052/m +
convex collective to ~8 g). An exploratory re-plan with the corrected aero lands at
**4.71 s** — the two corrections nearly cancel, so the **~4.3–4.7 s ceiling is robust**
to the plant-model revision (authoritative re-run queued post-S16).

Pipeline + env: `scripts/togt/` (README has build steps + hard-won environment facts).
Per-case configs/outputs: `cases/<case>/` here; combined table: `results_table.md`.

## 1. The numbers

Lap = time from standing start (rest, spawn pad z-up `[0,0,-0.02]`) to the **gate-5
plane crossing**. Every planned trajectory continues to a virtual endpoint 25 m past
gate 5 (vel 0), so terminal constraints cannot slow the finish; total durations in the
raw files are therefore larger than the lap.

| case | T/W | ω_xy (rad/s) | gate region | TOGT init (s) | **refined lap (s)** | max in-plane miss (m) | vmax (m/s) | collective sat | ω_xy sat |
|---|---|---|---|---|---|---|---|---|---|
| **bound_circle** | 3.765 | 11 | inscribed circle r=0.75 | 5.36 | **4.27** | 0.78 | 52.4 | 84 % | 11 % |
| bound_nominal | 3.765 | 11 | full 1.5 m square | 5.07 | 4.13 | 1.06 (corner) | 55.2 | 84 % | 9 % |
| bound_nodrag | 3.765 | 11 | square, drag 0 | 5.07 | 4.05 | 1.06 | 55.7 | 84 % | 12 % |
| bound_free | 3.765 | 11 | square shrunk 0.6 + refine ball 0.30 | 5.80 | 4.43 | 0.35 | 52.6 | 85 % | 11 % |
| **ref_circle** (exported) | 3.765 | 11 | ball r=0.40 + tol 0.10 | 5.81 | **4.55** | 0.14 | 51.1 | 83 % | 11 % |
| ref_margin | 3.765 | 11 | square shrunk 0.7 + tol 0.10 | 5.49 | 4.31 | 0.61 | 53.7 | 82 % | 10 % |
| sens_omega785 | 3.765 | **7.85** | square | 5.09 | 4.18 | 1.06 | 55.2 | 84 % | 13 % |
| sens_omega_unbounded | 3.765 | **25** | square | 4.94 | 4.09 | 1.06 | 55.2 | 87 % | 4 % |
| sens_thr75 | **2.824** | 11 | square | 5.89 | 4.90 | 1.06 | 47.0 | 84 % | 7 % |
| sens_thr50 | **1.883** | 11 | square | 7.25 | 6.42 | 1.07 | 37.6 | 89 % | 5 % |

All refines: IPOPT `Solve_Succeeded`, dt = 0.02 s nodes, drag 0.21/s in the dynamics
(except `bound_nodrag`). "Sat" = fraction of nodes at ≥99 % of the bound.

**What binds: thrust, overwhelmingly.**
- Thrust 100 %→75 %→50 % ⇒ lap 4.13→4.90→6.42 s (+19 %, +55 %). Collective rides the
  3.765 g ceiling ~84 % of the lap.
- Rates 7.85→11→25 rad/s ⇒ lap 4.18→4.13→4.09 s (±1 %). The course's wide (1.5 m)
  gates and gentle geometry never demand sustained fast rotation; ω_xy touches its bound
  only ~10 % of nodes (attitude flips between thrust directions).
  ⇒ The super-rate map discovery (7.85→11 rad/s real authority) is worth almost nothing
  on THIS course's lap time; its value is control-fidelity (the S1.2 divergence) and
  insurance for tighter VQ2 tracks.
- Drag (0.21/s linear) costs 0.085 s on the lap (4.05→4.13). Smaller than a flat-out
  estimate suggests: the 26 m descent is a gravity assist and the optimizer re-shapes.
- Speeds: vmax ≈ 52 m/s (188 km/h), gate-5 crossing at ~42 m/s. Tilt peaks ~170°
  (momentary near-inversion during thrust-vector flips — the characterize-sweep says the
  sim's attitude dynamics are clean through inversion; no anomaly).

**Context for the 35.3 s:** the VQ1 stack's ReactivePlanner cruises at 5 m/s by
configuration — the 8× gap is by design of the conservative floor, not a mystery. The
practical near-term target from this work: the twin-tracked 8.3 s (§4) with the EXISTING
controller, and whatever an RL/MPCC tracker can recover toward ~4.6 s.

## 2. Plant → planner mapping (and what is NOT modeled)

Mass-normalized (1 kg); TOGT's g = 9.8066 (vs our 9.80665 — irrelevant).

| plant fact (source) | TOGT/refine parameter |
|---|---|
| collective ceiling: normalized 1.0 → g/hover = 9.80665/0.2656 = 36.92 m/s² = 3.765 g (twin/sysid) | per-rotor `thrust_max` = 3.765·9.8066/4 = 9.2305 N (refine bound = ×4 collective) |
| collective floor ~0 | `thrust_min` 0 (refine); TOGT init `minThr` 0.1 N (0.05 fails — flatness singularity; bias immeasurable) |
| sustained body-rate authority: super-rate map g(π)·π ≈ 11.05–11.22 rad/s r/p measured holds (characterize-sweep §1) | `omega_max` xy = 11.0 (state bound in refine; flatness constraint in init) |
| yaw: level-attitude plateau ~7.4 rad/s (maneuver-dependent caveat) | `omega_max` z = 7.0 |
| linear drag 0.21/s world-frame (twin `linear_drag` 0.2111) | `linear_drag: 0.21` patched into the refine dynamics exactly (`- drag·v`); TOGT init is drag-free (warm start only) |
| inertia/allocation | tiny inertia [0.001, 0.001, 0.0017] + arm 0.15/beta 45/torCoeff 0.05 so the rotor-differential torque envelope never binds ⇒ the per-rotor model reduces to collective + ω bounds = our CTBR plant class |

**Not modeled by the planner** (checked downstream by the twin replay):
inner-loop lag τ≈0.019 s, slew ≈260 rad/s² r/p (the refined references momentarily
demand up to ~700 rad/s²), command latency, the super-rate map's small-signal shape
(only its ceiling enters, as `omega_max`), aero beyond linear drag at 50 m/s airspeed
(UNMEASURED on the real sim — the sweep probed near hover; biggest physics unknown here),
gate-frame collision geometry (the in-plane validity radius is the clearance proxy).

## 3. Validity geometry: corners vs the inscribed circle

With the full square opening as the constraint region, the optimum crosses EVERY gate at
a corner (in-plane components ~(±0.73, ±0.74) m, Euclidean ~1.06 m): the corners
straighten the descending zig-zag. Our validity rule as measured everywhere in this repo
(`twin_fly_course.gate_plane_miss`, the VQ1 0.37 m record) is **Euclidean in-plane miss
< 0.75 m** = the inscribed circle; whether the sim's race_outcome accepts corner passes
is UNVERIFIED (every VQ1 pass was well inside the circle). So:
- **bound_circle (4.27 s) is THE defensible bound** — ball-shaped crossing regions
  (radius 0.75) enforce the rule exactly as we measure it.
- bound_nominal (4.13 s) is the if-corners-count variant — 0.14 s faster; cheap live
  probe someday: deliberately thread one gate at a corner and see what race_outcome says.
- (Refine ball-tol overshoot is ≤3 cm — soft-constraint leakage + node interpolation;
  the numbers above are measured on the final trajectory, not trusted from the solver.)

## 4. The committed reference line + twin reality check

**`rl/reference_line_vq1.json`** (64 KB, schema `peregrine.reference_line.v1`, loader =
`src/racer/reference_line.py`): the `ref_circle` solution — crossings confined to a
0.40 m ball + 0.10 m tol ⇒ actual misses ≤0.14 m, leaving a ~0.6 m tracking budget
against the 0.75 m rule. Lap 4.551 s, NED world / FRD body, 368 samples @ ~62 Hz
effective, fields t/pos/vel/acc/yaw/quat/omega/thrust_norm + per-gate crossing states.
Gate crossings: t = 1.33, 1.90, 2.49, 3.37, 3.97, 4.55 s — all ≥41 m/s after gate 1.

**Twin replay** (`scripts/twin_track_reference.py`): faithful PHYSICS twin
(`twin_fit.faithful_config(super_rate=True)`: gain map s=0.30, slew [260,260,80],
τ=0.019, drag 0.2111, hover 0.2656) with canonical telemetry (the report-sign quirks and
their controller compensations cancel exactly; live runs the full circus), tracked by the
EXISTING geometric controller (`Controller` BODY_RATE non-decoupled: accel-ff + PD →
tilt+collective → rotvec rate law, ff_gain 2.5, race clamps: tilt 80°, ω 11.5).

- **Pure time-indexed tracking at 1× fails by design**: the reference is
  thrust-saturated ~84 % of the lap — zero headroom to recover any lag, so error
  compounds (25 m rmse). This is an architecture statement, not a tuning one.
- **Time-dilated tracking** (same geometry, velocity ff ×1/k, accel ×1/k²): first VALID
  6/6 at **k = 1.85 ⇒ twin-tracked lap 8.31–8.50 s** across command latency 0/20/40 ms
  (misses ≤0.52 m, worst at gates 1–2 where the line drops 8.6 m between gates).
- A 54-combo gain sweep (kp_pos ≤10, kd_vel ≤7, kp_att ≤16) finds NOTHING valid at
  k ≤ 1.7: the frontier is structural — attitude-loop bandwidth + the τ=19 ms lag +
  saturation — not gains.
- **Interpretation:** plant ceiling 4.27 s; this line flown by this controller class,
  8.3 s. The 1.8× gap is the recoverable space for a real tracking architecture
  (per-segment retiming / MPCC / the RL policy, which trains on exactly this twin).
  Even the dilated 8.3 s is 4.2× faster than the banked VQ1 finish.

## 5. Caveats (ranked)

1. **The plant aero model was falsified mid-session** (twin-falsify 2026-06-11): real
   drag is quadratic body-frame (c2≈0.052/m, measured to 7.6 m/s, extrapolated above)
   and the collective curve is convex to ~8 g — both replace the linear 0.21/s + 3.765 g
   used by the main table. §8 brackets the corrected bound at **4.71 s** (isotropic-v²
   extrapolation); the corrections nearly cancel, but the per-case numbers above are
   linear-model figures. Authoritative re-run after S16 lands the measured
   CandidatePlant. (The 20 km/h "cap" in old memory was the VQ1 planner's cruise
   setting, not a plant limit.)
2. **Corner validity unverified** (§3) — bracketed: 4.13 (corners) vs 4.27 (circle).
3. Planner ignores inner-loop lag/slew (§2) — bracketed by the twin replay: the
   gap to 8.3 s is real for today's controller; the bound stands as a plant property.
4. Refine constraints enforced at 0.02 s nodes (RK4 between); soft leakage ≤3 cm /
   ≤0.2 % thrust observed. Lap discretization error ~±10 ms.
5. Start modeled as level at rest at the spawn point (PVAJ pinned; the 17.8° pad tilt
   and arm/launch-ramp time are below the 10 ms noise floor at this scale — the live
   stack's 0.6 s launch ramp is a CONTROLLER artifact the RL/tracker owns, not plant).
6. Yaw treated with the 7.0 rad/s level-cap bound and FORWARD heading in the init; the
   refine lets yaw float within ±7 — yaw never binds (≤7.0 touched at single nodes).

## 6. Reproduce

WSL Ubuntu, one-time setup + build: `scripts/togt/README.md` (TOGT-Planner @ HEAD
6707d3a + local Eigen 3.4 + rapidjson include + `test_peregrine.cpp` gtest driver;
casadi venv). Then from the repo root (Windows side):

```
.venv\Scripts\python.exe scripts\togt\gen_cases.py            # case dirs (LF!)
wsl -d Ubuntu -- bash /mnt/c/.../scripts/togt/run_cases.sh /mnt/c/.../cases   # plan+refine
.venv\Scripts\python.exe scripts\togt\analyze.py              # the table
.venv\Scripts\python.exe scripts\togt\export_reference.py --case .../ref_circle
.venv\Scripts\python.exe scripts\twin_track_reference.py      # twin reality check
```

Environment landmines (CRLF→segfault, MSYS path mangling, WSL tmpfs, the pre-main CLI
crash, minThr 0.05, RaceParams absolute paths): `scripts/togt/README.md` §"hard-won".

## 7. What this unlocks / suggested next

- **RL progress reward**: `racer.reference_line.ReferenceLine.progress(pos)` gives
  arc-length progress along a near-optimal line (vs the current gate-to-gate shaping).
- **Decomposed plan+track**: the line is the explicit plan; the 8.3→4.6 s gap is the
  tracker's to close (MPCC or RL-as-tracker with the line in the observation).
- **VQ2 expectations**: with thrust the binding constraint, lap time scales ~with
  1/√(T/W); any future plant re-measurement should re-run `sens_thr*` first.
- Cheap live probes when convenient: (a) one deliberate corner pass → does race_outcome
  accept it (closes the 4.13/4.27 bracket); (b) a full-throttle straight-line speed run
  → drag/thrust validity at 40+ m/s (caveat #1, the big one).

## 8. Post-scriptum: the twin-falsify correction (banked mid-session, 2026-06-11)

While this session ran, the ShadowPC twin-falsify campaign (commit 9684ae1, memory
fc6194b) falsified two inputs of §2: real drag is **quadratic body-frame**
(c2 ≈ 0.052/m: 0.042 nose-first … 0.076 climb; measured to 7.6 m/s, arena-limited) and
the collective map is **convex** — full stick reaches **~8 g**, not the linear-model
3.765 g. The banked note says "the TOGT bound is conservative; re-run after S16" — that
reasoning counted only the thrust correction. Both together, the exploratory case
**`expl_corrected_aero`** (T/W 8.0, isotropic quadratic drag 0.052·|v|·v in the refine
dynamics, circle gates, otherwise nominal):

| case | model | refined lap (s) | vmax (m/s) | collective sat |
|---|---|---|---|---|
| bound_circle | linear drag 0.21/s, 3.765 g | 4.27 | 52.4 | 84 % |
| expl_corrected_aero | quad drag 0.052/m, 8 g | **4.71** | 39.3 | 91 % |

The corrections **nearly cancel — net +0.44 s**: v² drag walls the top speed at
~39 m/s (drag accel ≈ 79 m/s² ≈ the full 8 g there) long before the extra thrust pays.
Thrust/drag remains the binding constraint axis (collective 91 % saturated); rates still
don't bind. Standing of this number: **exploratory** — it extrapolates v² far beyond the
7.6 m/s measurement and flattens the direction-dependence to isotropic; the
post-S16 re-run with the measured CandidatePlant (drag knots + collective knot table) is
the authoritative correction, and a single high-speed coast probe (§7) would pin the
extrapolation. Consequences meanwhile: (a) the **headline ceiling ~4.3–4.7 s is robust**
to the plant revision — the strategic picture (8× headroom, thrust-bound) is unchanged;
(b) the committed reference line (4.55 s geometry) sits inside the corrected envelope's
speeds below ~35 m/s for most of the lap, but its >40 m/s tail segments are NOT flyable
under v² drag — regenerate the line at the S16 re-run before using it for fine
time-targets (its geometry + progress() use is unaffected).
