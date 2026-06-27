---
name: project-togt-time-optimal-bound
description: "VQ1 time-optimal lap bound (frictionless ~4.3-4.7 s; twin-tracked ~8.3 s realistic), what binds, reference line, twin-tracked gap, TOGT pipeline + env landmines"
metadata:
  type: project
---

> 🚩 **CURRENT-FRAMING NOTE (banked to repo 2026-06-27):** this file's headline "~4.3–4.7 s bound" is
> the FRICTIONLESS/thrust-bound optimum and is now framed by `MEMORY.md` doctrine as the
> **rate-infeasible TOGT optimum** — the realistic upright-feasible lap is **~8 s** (measured quad-drag
> caps upright racing at ~27–30 m/s). The ~8 s figure traces directly to THIS file's **twin-tracked 8.3 s**
> (k=1.85) result, which stands. Read the bound below as the analytic ceiling, not the achievable target.
> → [[index-rl-training]] (speed-ladder / ~8 s doctrine) · [[project-rl-increment-history]].

**VQ1 time-optimal bound (LAPTOP-TOGT-BOUND 2026-06-10/11; commits 35f451d+c4a8134+00d7cfe; source of truth `handoff/laptop-togt-bound-2026-06-10/WRITEUP.md`):**

- **THE BOUND ≈ 4.27 s** (crossings in the inscribed circle = Euclidean miss <0.75, the validity rule as we measure it) / **4.13 s if gate corners count** (square region — the optimum clips every corner at ~1.06 m Euclidean; sim acceptance of corner passes UNVERIFIED — cheap live probe). VQ1's 35.3 s is **8× off the ceiling** (by design: 5 m/s cruise planner).
- **THRUST binds, rates don't**: collective saturated ~84% of the lap; 50%/75% thrust → 6.42/4.90 s. The old-wrong 7.85 rad/s rate model costs only +0.05 s; unbounded rates buy −0.04 s ⇒ the super-rate map's value is control fidelity, NOT lap time on THIS course. Linear drag costs 0.085 s. vmax ~52 m/s.
- **🚩 Plant falsified mid-session (TWIN-FALSIFY): corrected-aero bracket = 4.71 s** (quad drag 0.052·|v|·v isotropic + 8 g convex collective, `expl_corrected_aero`): v² drag walls speed at ~39 m/s and EATS the 2× thrust — corrections nearly cancel; the memory note "bound is conservative" was thrust-only reasoning. **Ceiling ~4.3–4.7 s robust; authoritative re-run queued post-S16** (quad drag exploration extrapolates v² beyond the 7.6 m/s measurement).
- **Reference line committed: `rl/reference_line_vq1.json`** (schema peregrine.reference_line.v1, NED/FRD; loader+arc-progress `src/racer/reference_line.py` → RL progress reward + plan/track). ref_circle solution: lap 4.551 s, misses ≤0.14 m (0.6 m tracking budget). 🚩 Its >40 m/s tail segments are NOT flyable under v² drag — regenerate at S16 before fine time-targets (geometry/progress() use unaffected).
- **Twin reality check** (`scripts/twin_track_reference.py`, faithful physics + canonical telemetry, existing geometric Controller, race clamps): pure 1× time-indexed tracking diverges BY DESIGN (84% thrust-saturated ref = zero headroom); time-dilated line first flies valid 6/6 at **k=1.85 ⇒ twin-tracked ≈8.3 s** (latency 0–40 ms; gain sweep to 4× stiffer finds nothing ≤1.7 — the frontier is structural: τ=19 ms lag + attitude bandwidth). The 4.55→8.3 s gap = the tracking architecture's to close (MPCC / RL-as-tracker); 8.3 s is still 4.2× faster than VQ1.
- **Pipeline (reusable)**: `scripts/togt/` — gen_cases→run_cases.sh (WSL)→analyze→export_reference; TOGT-Planner planTOGT (warm start + crossing points) → vendored CasADi multiple-shooting refine (drag terms added) = the bound. Re-run cost ~minutes/case. For new tracks or post-S16: edit course json / quad yaml knots.
- **Env landmines** (cost hours; `scripts/togt/README.md` §hard-won): CRLF → TOGT's hand-rolled YAML parser reads `0.05\r` garbage → segfault (gen writes LF; `.gitattributes` guards); the standalone planner CLI segfaults pre-main (use the gtest driver `test_peregrine.cpp`); `RacePlanner::plan()` two-phase is broken upstream (planTOGT only); RaceParams needs ABSOLUTE paths; `minThr` 0.05 → L-BFGS fails (use 0.1); init brittle off-nominal → speedGuess/dynCC retry ladder. **Driving WSL from Windows: git-bash MSYS mangles `/mnt/...` args + env assignments and corrupts long inline commands — invoke wsl from PowerShell, logic in script FILES, logs to persistent paths (WSL /tmp is tmpfs, instances recycle).**

Related: [[project-phase2-rl-vision-decisions]] (S16 aero integration gates the re-run; RL progress reward consumer), [[project-ctbr-control-sysid]] (plant numbers fed in).
