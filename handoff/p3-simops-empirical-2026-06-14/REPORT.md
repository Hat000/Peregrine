# P3 SIM-OPS/EMPIRICAL — REPORT (2026-06-14)

**Session:** Peregrine P3 SIM-OPS/EMPIRICAL · **Operator:** opus-4.8 (MAX) · **Machine:** ShadowPC
**Branch (worktree):** `claude/hardcore-lehmann-1de77c` · **Reports to:** Fengyou → Overall Commander
**Constraint honored:** READ + offline analysis only. NO sim flight, NO src/racer or rl/ edits, NO estimator/RL change.

---

## TASK 1 — BORESIGHT LOCK (offline bearing/range-resolve path) — DONE, verdict = **INCONCLUSIVE → live static test required**

### Objective
Settle whether the L3 at-speed gate-2/gate-4 vertical bias (`rel_vert` ≈ −0.215 m, common-mode) is:
- **(a) track_map offset δ_map** → constant METRES vs range → cancels in the +L gate-relative obs → **margin CLOSES**; or
- **(b) camera-boresight / perception pitch bias ε_vert** → constant ANGLE δθ → `rel_vert ≈ −R·tan δθ` → does NOT cancel → **margin OPEN**.

The L3 report (commit `ee62f00`) chose (a) from the *metric* common-mode (g2 −0.211 ≡ g4 −0.215). The δ_map discriminator (`d7c592e`) chose (b) from a DIRECT δ_map measurement (true-opening-z ≈ map-z) that is "mildly self-fulfilling". This task runs an **independent discriminator the L3 report never ran**: `rel_vert` vs `true_range` per fix (range-resolve), purely over the existing `shadow_gate{2,4}_rows.json` — no sim, no detector re-run.

### Method
`handoff/p3-simops-empirical-2026-06-14/analysis/boresight_bearing_resolve.py` (stdlib only). Loads accepted fixes
(g2 N=234, g4 N=81; `abs_vert ≡ rel_vert` byte-exact → one vertical-error per fix). OLS `rel_vert = a + b·range`
with 4000-sample bootstrap CIs; angular-vs-metric constancy; range-binned means; cross-gate metric-ratio test;
bearing coverage. Physics: MAP ⇒ b≈0, intercept≈−0.21; BORESIGHT(0.56°) ⇒ b≈−tan(0.56°)=−0.0098 m/m, intercept≈0.

### Result (CONFLICTED — the headline)
| subset | N | range span | slope b (m/m) | 95% CI | intercept (m) | reading |
|---|---|---|---|---|---|---|
| **all-accepted pooled** | 315 | 4.5–25.9 | **+0.0011** | [−0.0067, +0.0090] | −0.235 | flat → **MAP-like**; CI **excludes** 0.56° boresight |
| gate-2 all (max leverage) | 234 | 4.5–25.9 | +0.0013 | [−0.0071, +0.0099] | −0.236 | flat; excludes boresight |
| gate-4 all (narrow) | 81 | 20.1–24.0 | +0.0064 | [−0.020, +0.032] | −0.357 | underpowered (4 m span ≪ noise) |
| **clean 4-corner pooled** | 172 | 18.6–25.2 | **−0.0427** | [−0.068, −0.019] | **+0.71** | steep but **2.4°** (≠0.56°), nonphysical intercept |

- **Bearing coverage = the structural blocker:** **ZERO accepted fixes below 30° bearing** at g2 AND g4. Every fix
  is a 30–45° oblique gate-(G−1)-approach sighting (inc7's 66° crab never points head-on at the active gate).
  **There is no head-on data to bearing-extrapolate to ε(bearing→0).**
- **Cross-gate metric ratio:** observed g4/g2 = 1.015 vs boresight prediction (range ratio) 1.106 → favors MAP.
- **Range-binned (all):** −0.194 (<19 m) → −0.109 (19–21) → −0.166 (21–23) → −0.394 (23–25) → −0.225 (25+):
  **non-monotonic** — not a clean linear trend; the clean-subset slope is driven by a 23–25 m band artifact.
- **Control:** `rel_cross` ≈ 0 at both gates, all subsets (−0.04…+0.04) → no gross attitude yaw/roll error (the
  bias is vertical-only — consistent with EITHER a camera-pitch boresight OR a vertical map offset; doesn't decide).

### Verdict — INCONCLUSIVE, and it does NOT cleanly confirm OR refute either prior session
- The wide-range data (best leverage, g2 to 4.5 m) is **flat / map-constant** and its CI **excludes** the 0.56°
  boresight slope — leaning **case (a)/CLOSE**, in tension with the δ_map discriminator's confident case (b).
- The clean narrow-band subset shows a **steep negative slope** but at **2.4°** with a **positive intercept** —
  physically inconsistent with a simple 0.56° boresight; an extrapolation/PnP-depth artifact, not a clean signal.
- Root cause of the inconclusiveness: **(1) no head-on (bearing<30°) coverage; (2) narrow range span at gate-4
  (4 m ≪ per-fix σ 0.10 m); (3) ~35% noisy 3-corner P3P fixes + a 23–25 m depth-coupled band artifact.**
- This neither rescues the L3 case-(a) nor confirms the discriminator's case-(b). It demonstrates the **at-speed
  oblique fix data is the wrong instrument** for the δ_map-vs-ε decomposition. **The DIRECT live static head-on
  measurement is the only airtight arbiter** (exactly as the mission flagged: the vertical evidence is self-fulfilling).

### What IS pinned (independent of a/b)
- Effective in-plane fix-vs-GT bias at the gate-4 band: **lateral ≈ 0** (−0.04 m), **vertical −0.215 m** (≈0.56° @ 22 m),
  common-mode g2≡g4. Single-fix σ lat 0.19 / vert 0.10 m (< modeled 0.265). Relinnov accepts 100% of offered fixes.
- **The decomposition of that −0.215 m into δ_map (cancels) vs ε_vert (binds) is STILL OPEN.** P1's boresight
  calibration (sign + magnitude) cannot be pinned from offline data alone.

---

## NEXT — airtight live test (DESIGNED; pending Fengyou authorization)

A **STATIC, head-on, LEVEL** vision fix at gate-2 AND gate-4 (bearing ≈ 0, ~18–22 m range) directly isolates
ε_vert from δ_map: at bearing 0, level, static (no crab, no motion blur, known GT pose), `rel_vert = δ_map_vert − ε_vert`
with the geometry controlled. Run at ≥2 ranges per gate → the range-slope is then clean (constant-metre=map vs
constant-angle=boresight) AND the head-on intercept is the pure boresight.

**Feasibility flag for Fengyou:** this is **NOT the proven inc7 harness** — it needs a position/attitude **hold**
flight (SET_POSITION_TARGET_LOCAL_NED + level attitude-quat to enter ANGLE mode), which the first-contact notes
list as **UNTESTED** ("RE-TEST position/velocity in ANGLE mode"). It is MEASUREMENT (static hover 18–22 m back from
the gate, low contact risk), but it is a new control path outside the "orchestrate the proven harness" lane.
→ **I am holding for your authorization on the approach before building/dispatching the worker** (see up-report).

---

## Deliverables (`handoff/p3-simops-empirical-2026-06-14/`)
- `analysis/boresight_bearing_resolve.py` — the range-slope discriminator (stdlib; `--clean`, `--json`).
- `analysis/boresight_resolve_all.json` — per-gate slope/intercept/CI machine output.
- `REPORT.md` — this file.

## MEMORY-DELTA (≤10 lines)
- **BORESIGHT offline range-resolve = INCONCLUSIVE** (P3, 2026-06-14). Independent `rel_vert`-vs-`true_range` slope
  over L3 `shadow_gate{2,4}_rows.json`: all-accepted pooled slope **+0.001 m/m CI[−0.007,+0.009]** (flat, MAP-like,
  CI EXCLUDES the 0.56° boresight −0.0098); but clean 4-corner subset slope **−0.043** (steep, implies 2.4°, +0.71 m
  intercept = artifact). Non-monotonic; 23–25 m band artifact. CONFLICTED → cannot decompose δ_map vs ε_vert offline.
- **Root cause = ZERO head-on coverage:** 0 accepted fixes <30° bearing at g2 AND g4 (inc7 66° crab → only oblique
  30–45° gate-(G−1) sightings). Bearing-extrapolation to ε(0) impossible from this data. lateral control ≈0 (no gross attitude err).
- **Does NOT confirm/refute the δ_map discriminator (case b):** wide-range data mildly tensions it toward map/CLOSE,
  but is PnP/3-corner-confounded. The −0.215 m bias is PINNED; its δ_map-vs-ε_vert split is STILL OPEN.
- **Airtight arbiter = LIVE STATIC HEAD-ON fix at g2 & g4 (≥2 ranges)** — needs an UNTESTED position/attitude-hold
  control path (not the inc7 harness); held for Fengyou authorization before dispatch.
