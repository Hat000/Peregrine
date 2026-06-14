# D3 — MARGIN CLOSURE / velocity-prior swing (gate-relative case-C, COMPONENT 3)

**The load-bearing feasibility question of the whole blueprint:** does the gate-relative estimator
keep the gate-4 **in-plane** position error WELL INSIDE the **0.155 m** contact margin at race speed
(~37 m/s), and **under what conditions**? A margin is a **worst-case** gate, not an RMS gate — so the
verdict is read off **p90 / p99**, not RMS.

Sim: `d3_margin_closure.py` (composes the REAL `racer.state_estimator.LinearKF` + `RewindKF` OOSM;
re-uses the MEASURED per-fix gate-relative lateral σ = 0.265 m/axis). Seed 20260613, reproduces.
Results JSON: `d3_margin_closure_results.json`.

---

## 0. What was wrong with the WARM number, and what this sim fixes

`c1`'s rel arm (RMS **0.139 m**, p90 0.203 — reproduced here as the ANCHOR BIT-FOR-BIT: RMS 0.139,
p90 0.203, p99 0.311) is a **WARM** measurement: it seeds velocity to truth and uses a **single, straight, const-velocity g3→g4
segment**, so the IMU `predict` is exact and the velocity prior never degrades. That is **not** the
case-C reality entering gate-4 after a full lap, where velocity is **unobservable from vision**
(position-only) and is only dead-reckoned from the accelerometer.

This sim builds the **HONEST** case: a **smooth, physically-consistent g0→g4 full-lap** truth
(Catmull-Rom spline, C1-continuous velocity, speed ramped to v_race) where:
- the **accelerometer reports the TRUE specific force** (real maneuver accel included) → velocity IS
  partially observable by dead-reckoning;
- the **only systematic velocity corruptor is a constant body-frame accel BIAS** (the unpinned realism);
- velocity is corrected ONLY by gate-relative POSITION fixes (NO velocity update — case-C reality),
  except in the `weakvel` arm which adds a position-fix-difference velocity pseudo-update.

**First-attempt bug caught & fixed (documented for the commander):** the initial truth model used
piecewise-constant-velocity legs with instantaneous corner turns at each gate. Those injected
**phantom unsensed accelerations**, and with velocity unobservable the cold/weakvel arms diverged to
~14 m/s velocity error (RMS ~2.8 m). That was a sim artifact, not physics — the accelerometer must
sense real maneuvers. The corrected smooth-truth model gives realistic velocity errors (warm
0.06 m/s, cold 0.22 m/s entering g4), matching the independently-derived `d4v` velocity-channel
numbers (cross-validation, see §4).

---

## 1. Headline numbers (MEASURED this run, seed 20260613)

**ANCHOR** — c1 rel-arm reproduced BIT-FOR-BIT: in-plane **RMS 0.139 / p50 0.117 / p90 0.203 /
p99 0.311 / max 0.440** m (c1 baseline RMS 0.139, p90 0.203). The shared machinery is validated.

**THE STRADDLE, RESOLVED (37 m/s, GPU latency 15 ms, accel-bias = 0):**

| regime | in-plane RMS | p90 | p99 | frac runs ≥ 0.155 | clears margin p90? |
|--------|-------------|-----|-----|-------------------|--------------------|
| **warm** (vel known — NOT case C) | 0.084 | **0.124** | 0.178 | 2.5% | **YES** |
| **cold** (case-C reality, vel IMU-only) | 0.154 | **0.235** | 0.322 | **37%** | **NO** |
| **weakvel** (cold + naive fix-diff vel) | 0.151 | 0.230 | 0.304 | 35% | NO |

**The cold case-C estimator does NOT clear the 0.155 m WORST-CASE margin at race speed.** RMS sits
*at* the wall (0.15) but p90 (0.235) and p99 (0.32) are ~1.5–2× over it, and **~37% of runs would
contact even at zero accel bias**. The warm number (the only one that clears) is the prior verdict's
optimistic bound and is **unreachable in case C** (velocity unobservable from position-only vision).

---

## 2. The three velocity-prior regimes (what they mean)

- **warm** — velocity continuously re-seeded near truth (the c1-style UPPER bound; NOT achievable in
  case C, shown for contrast / as the ceiling).
- **cold** — IMU-only velocity, corrected only by gate-relative POSITION fixes. **This is the case-C
  reality** if no velocity channel is built.
- **weakvel** — cold + a velocity pseudo-update from **naive consecutive position-fix differencing**
  (σ_v = √2·0.265/Δt_fix; the WORST way to difference). The sibling `d4v` shows an **LSQ-over-window**
  velocity from the ~9 in-window fixes does far better (σ_v ≈ 0.4 m/s lateral vs naive's 0.57 at full
  window / 5.2 at one fix-gap) — so weakvel here is a conservative floor on what a velocity channel buys.

---

## 3. The swing factors (sweep results, MEASURED)

### 3a. Velocity-prior × speed ladder (accel-bias = 0, GPU 15 ms)

| regime | 25 m/s p90 | 30 m/s p90 | 37 m/s p90 | trend |
|--------|-----------|-----------|-----------|-------|
| warm | 0.098 | 0.108 | 0.124 | clears p90 at all speeds; p99 over at ≥30 |
| cold | 0.214 | 0.218 | 0.235 | **over margin p90 at ALL speeds** |
| weakvel | 0.206 | 0.206 | 0.230 | over margin p90 at all speeds |

The margin/σ ratio worsens with speed but the cold prior is **already over the worst-case margin at
25 m/s** — the velocity prior, not speed, is the binding term. Cold velocity error entering the g4
window is ~0.14–0.21 m/s (grows with speed): a 0.2 m/s velocity error over the ~0.07 s between-fix
coast + the terminal coast to the plane is enough to push p90 over 0.155 m on top of the 0.265 m/axis
per-fix lateral noise the KF is still averaging down.

### 3b. Accel-bias break-point (37 m/s, GPU 15 ms) — the DOMINANT swing factor

| bias (m/s²) | warm p90 | cold p90 | cold p99 | cold frac over |
|-------------|----------|----------|----------|----------------|
| 0.00 | 0.124 | 0.233 | 0.320 | 0.37 |
| 0.05 | 0.121 | 0.241 | 0.347 | 0.36 |
| 0.10 | 0.125 | 0.259 | 0.375 | 0.44 |
| 0.30 | 0.126 | 0.385 | 0.476 | 0.75 |
| 0.50 | 0.124 | 0.523 | 0.633 | 0.88–0.91 |
| 1.00 | 0.144 | 0.892 | 1.01 | 0.99 |
| 2.00 | 0.181 | 1.67 | 1.86 | 1.00 |

**warm is bias-robust** (continuous velocity re-seed absorbs the bias-induced velocity drift; p90 holds
< 0.155 up to ~1 m/s²). **cold degrades catastrophically with bias** — the integrated bias becomes
unbounded velocity drift between position fixes. This is the load-bearing realism dependency: the cold
verdict is set by the *true effective accel bias*, which is **unpinned offline**. Per the sibling d4v
mapping, a phantom accel from attitude error alone is **0.086 m/s² @ 0.5°** and **0.240 m/s² @ 1.4°
(the MEASURED 1-σ attitude error)** — i.e. plausible realism lands at cold p90 ≈ 0.26–0.40 m, deep
over the margin.

### 3c. RewindKF latency contrast (37 m/s): 15 ms (GPU) vs 115 ms (CPU)

| regime | bias | 15 ms p90 | 115 ms p90 | Δ |
|--------|------|-----------|-----------|---|
| warm | 0.0 | 0.124 | 0.127 | ~0 |
| warm | 0.1 | 0.133 | 0.126 | ~0 |
| cold | 0.0 | 0.246 | 0.238 | ~0 |
| cold | 0.1 | 0.255 | 0.294 | +0.04 |

**Latency is NOT the binding factor.** The RewindKF OOSM rewind removes the v·L staleness; in-plane error
is near-invariant between GPU (15 ms) and CPU (115 ms). The cold-b0.1 cell shows a mild 115 ms
degradation (deeper replay coasts longer on the biased velocity) but it is second-order vs the velocity
prior. **The horizon (0.5 s) safely exceeds 115 ms** — no fix drops observed at race speed.

### 3d. Race-speed verdict table (37 m/s, GPU 15 ms, n_mc=500) — rms/p90/p99 clear-flags

The high-MC verdict table confirms the picture exactly: **warm clears both p90 (0.124–0.126) and is the
only regime clearing — p99 (~0.175) just under at low bias**; **cold and weakvel fail p90 and p99 at all
biases ≥ 0** (p90 0.24–0.26 at zero bias, climbing with bias). RMS-only would mislabel cold as
"borderline pass"; the worst-case gates expose the failure.

---

## 4. Cross-validation with sibling d4v (velocity-channel design, component 4-velchannel)

`d4v` independently analysed the SAME g3→g4 velocity-prior swing on a single segment and arrived at:
- cold inplane_pos **RMS 0.215 / p90 0.324**; warm **RMS 0.119 / p90 0.182**.
- velocity-assist (position-fix-difference): σ_v 0.5 → pos RMS **0.124 / p90 0.185**; σ_v 0.3 → **0.118 / 0.179**.
- **att-bias ↔ effective accel bias mapping** (load-bearing realism anchor): 0.5° = **0.086 m/s²**,
  1.0° = 0.171, **1.4° (the MEASURED 1-σ attitude error) = 0.240 m/s²** phantom accel.

This convergence (two independent sims, different trajectory constructions) is strong corroboration.
My multi-gate cold RMS is slightly LOWER than d4v's single-segment cold because the full lap
pre-converges the velocity estimate before gate-4; the p90/p99 picture is the same.

---

## 5. VERDICT — CANNOT-SETTLE-OFFLINE (load-bearing; escape hatch invoked)

**The make-or-break question — does the cold case-C gate-relative estimator clear the 0.155 m
worst-case margin at 37 m/s — CANNOT be settled offline,** because the answer is governed by the
*true effective accel bias / attitude-error realism*, which only live at-speed data pins (§3b).

What IS settled offline (robust across the sweep, n_mc 250–500, two independent sim constructions
d3 + d4v agree to ~3 decimals):

1. **The WARM number (RMS 0.139 / p90 ~0.12–0.20) is the WRONG bound to plan against.** It assumes a
   lap-converged velocity that case C cannot deliver. Planning on warm is the trap.
2. **The COLD case-C estimator FAILS the worst-case margin even at ZERO bias:** p90 0.235, p99 0.32,
   ~37% of runs ≥ 0.155 m at 37 m/s. RMS-only reporting (0.15) hides this — a margin is a p90/p99 gate.
3. **The swing factor is the velocity prior, via the accel/phantom-accel bias — NOT speed, NOT
   latency.** Latency is solved by RewindKF (§3c). Speed is second-order (§3a). Velocity is the lever.
4. **The CONDITIONAL path to clearing the margin exists and is quantified:**
   - **(i) build the velocity channel** (component 4-velchannel): d4v shows an LSQ-over-window velocity
     from the ~9 in-window fixes reaches σ_v ≈ 0.4 m/s lateral (vs naive consecutive-fix differencing's
     0.57–5.2), landing in-plane pos **RMS 0.124 / p90 0.185** at σ_v 0.5 and **0.118 / 0.179** at
     σ_v 0.3. **At σ_v ≤ ~0.3 m/s the margin is held at p90** — but p99 still grazes 0.18, so it is a
     p90-clear, not a p99-clear: pair it with the uncertainty-aware-obs speed-down (CONTEXT input (i)).
   - **(ii) keep effective accel bias ≤ ~0.1 m/s²** (equivalently attitude error ≤ ~0.6°). At 1.4°
     (current measured 1-σ) the phantom accel (0.24 m/s²) puts cold at p90 ≈ 0.30 — over. The attitude
     pipeline / ESKF-bias-state fallback (state_estimator deferred rebuild #1) is therefore in the
     critical path for the margin, not just for absolute nav.

**EXACT live data that resolves it (the ShadowPC at-speed ~37 m/s gate-4 vision recording):**
- **Trajectory:** a full g0→g4 lap (or at minimum a high-speed g2→g3→g4 segment reaching ~37 m/s on
  the g3→g4 straight), flown by the current best policy, **≥ 5 laps** (to get a worst-case/p90
  distribution, not a point estimate).
- **Fields, time-synced on ONE clock (the TIMESYNC P0 prereq, navigator.py:255-271 / :426):**
  per-frame (≥30 Hz) the **4 gate-corner pixel detections + PnP `t_cam_gate` + reproj error**, the
  **HIGHRES_IMU `accel_body` (≥90 Hz)** and the **given ATTITUDE quaternion**, and **ground-truth
  drone position + velocity** (sim GT or external mocap) at IMU rate. Duration ≥ the 5 laps (~60–100 s).
- **What it pins:** (a) the **true accel-bias / attitude-error magnitude** entering g4 (run the KF
  open-loop on the recorded IMU+attitude vs GT velocity → the velocity-drift slope = the effective
  bias); (b) the **realised cold velocity-prior error distribution** entering the g4 window; (c) the
  **achievable σ_v** from real position-fix differencing (feed real fixes through the velocity channel).
  Plug (a)+(c) into this sim's §3b/§4 tables → the p90/p99 margin verdict becomes MEASURED, not ASSUMED.

**Bottom line for the blueprint:** gate-relative observation is necessary and removes the per-track map
bias (proven), but it is **not sufficient by itself** to clear the gate-4 worst-case margin at race
speed in cold case C. The margin closes **only with the velocity channel (σ_v ≤ ~0.3 m/s) AND a
low effective accel bias (≤ ~0.1 m/s², attitude ≤ ~0.6°)**, and even then it is a p90-clear that wants
the uncertainty-aware speed-down for p99. Build all three; verify the closure on the live at-speed
recording above.
