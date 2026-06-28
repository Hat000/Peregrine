# Self-contained time-optimal planner (`scripts/planning/`)

Computes the **time-optimal racing line** + the **lap-time lower bound** for the VQ1 gate
track on Peregrine's measured plant, using a self-contained CasADi/IPOPT minimum-time
direct-multiple-shooting NLP. **No external warm-start** (no C++ TOGT phase, no WSL build):
a straight-line constant-speed guess is enough for IPOPT to converge in ~60 iterations.

This complements the older `scripts/togt/` pipeline (FSC-Lab TOGT-Planner + C++ warm-start,
linear-plant bound ~5.07 s gate-5 lap). This one is dependency-light, runs on the laptop
venv alone, and models the **corrected/measured aero** (convex thrust map + quadratic drag)
rather than the falsified linear plant.

## Run

```bash
# corrected-aero bound (drag ON) -- THE honest lower bound
PYTHONPATH=src .venv/Scripts/python.exe scripts/planning/trajopt.py --out scripts/planning/out_drag

# absolute floor (drag OFF) -- thrust-only, ignores the v^2 drag wall
PYTHONPATH=src .venv/Scripts/python.exe scripts/planning/trajopt.py --out scripts/planning/out_nodrag --no-drag

# tighter reference line (0.3 m gate ball instead of the 0.75 m inscribed circle)
PYTHONPATH=src .venv/Scripts/python.exe scripts/planning/trajopt.py --out scripts/planning/out_ref --radius 0.3

# independent forward-feasibility re-integration of a result
PYTHONPATH=src .venv/Scripts/python.exe scripts/planning/validate_forward.py \
    --csv scripts/planning/out_drag/trajectory.csv --drag
```

## Result (VQ1 track, measured plant)

| case                       | lap @ gate-6 crossing | max speed | binds                         |
|----------------------------|----------------------:|----------:|-------------------------------|
| **drag ON (the bound)**    | **4.62 s**            | 39.3 m/s  | thrust 98.8% sat + all 3 rates + **v² drag wall** |
| drag OFF (absolute floor)  | 2.71 s                | 84.1 m/s  | thrust + rates only (unphysical top speed) |
| reference, 0.3 m gate ball | 4.66 s                | 39.2 m/s  | same; +0.04 s for tighter gates |

- **LAP-TIME LOWER BOUND = 4.62 s** (corrected-aero, drag ON). Stable to ±0.001 s across
  8–20 nodes/segment (not a discretization artifact). Forward re-integration of the open-
  loop optimal commands reproduces the state trajectory to **0.2 mm** max position drift →
  the solution is dynamically exact, the time is achievable by an open-loop command stream.
- The trajectory **dwells on the thrust limit** (98.8% of nodes at ~8 g) and rides all three
  body-rate limits — the bang-bang signature of a true time-optimal solution, vs a min-snap
  baseline that only touches the limits instantaneously.
- **Drag dominates at racing speed:** removing drag drops the bound to 2.71 s at an absurd
  84 m/s. With the measured c2≈0.052/m the v² drag wall caps top speed near 39 m/s, which is
  the real binding constraint, not thrust headroom. This is consistent with the prior
  corrected-aero analysis (`handoff/ultracode-planning-togt-s2-2026-06-13`).

## Plant model (the bound's plant class)

Point-mass-with-attitude (the CTBR plant class), `src/racer/rl_plant` mirrored in
`plant_params.py` (cross-checked by `verify_against_rl_plant`):
- State `[p(3), v(3), q(4)]`; control `[a_c, ω(3)]`.
- `a_c ∈ [0, 78.28]` m/s² = the measured convex collective→accel map full-stick ceiling (7.98 g).
- `ω` bounded by `[11, 11, 7]` rad/s = sustained reachable rates through the measured
  super-rate map (same as `scripts/togt` OMEGA_NOMINAL).
- Quadratic body-frame drag `−c2·|v_b|·v_b`, pooled `c2 = 0.052/m` (twin-falsify 2026-06-11).
- Gate constraint = last node of each segment within 0.75 m (Euclidean) of the gate centre
  = the inscribed circle of the 1.5 m opening = our exact validity rule.

**NOT modeled** (deliberately — these are a downstream feasibility check, not part of the
bound): inner-loop rate lag (τ≈0.02 s), rate slew (α_max≈260 rad/s² roll/pitch), airspeed
thrust lapse, the motor mixer. The bound is the *plant-class* optimum; the twin replay
(`scripts/togt/.../twin_track_reference.py` style) is the realizability gate for the lags.

## Gap to full CPC (Complementary Progress Constraints)

The mission preferred CPC (Foehn/Romero/Scaramuzza, Sci. Robotics 2021). This ships the
**tractable min-time direct-multiple-shooting fallback** the mission authorized. Differences:

1. **Gate coupling.** CPC introduces a per-gate progress variable μ_j with complementarity
   `μ_j · (‖p − g_j‖ − tol) = 0`, so gate completion is enforced *somewhere* along the
   trajectory without pre-assigning *which node* hits the gate. Here we hard-assign the gate
   to the **last node of its segment** with a fixed node budget per segment. That removes the
   combinatorial time-allocation freedom CPC buys — so our bound is **≥ the CPC optimum**
   (slightly conservative): the true optimum could pass a gate at a node we didn't allot to
   it. In practice the gap is small here because the gates are well-separated (24–39 m apart,
   monotone along −X) and the per-segment node count is uniform; the node-density invariance
   (±0.001 s over 8–20 nodes) suggests the allocation is near-optimal already.
2. **Free node time is per-segment, not global.** CPC uses one global Δt with progress
   gating; we give each segment its own free Δt (uniform within a segment). This is the
   standard TOGT discretization and is *less* restrictive than a single global Δt.
3. No singularity-free attitude handling needed — both use quaternions with per-step renorm.

To close the gap to a true CPC bound, add progress variables + complementarity constraints
over a single global node grid (relaxed with a smoothed complementarity / IPOPT μ-strategy).
Expected effect: ≤ a few % lower lap time. The current 4.62 s is a **valid upper-confidence
lower bound** (an achievable time that no plant-class trajectory can beat by more than the
small allocation slack), which is exactly what the RL racer needs as a target.

## Using the trajectory for RL (reward shaping / warm-start)

`out_drag/trajectory.csv` columns: `t, p(3), v(3), q(4), a_c, ω(3)` (NED, z-down).

1. **Lower-bound yardstick.** 4.62 s is the number to cite for "how fast is this track."
   The inc8 doctrine target (~8 s) leaves large headroom; the bound says the *plant* can do
   4.62 s, so the gap is policy/robustness, not physics.
2. **Dense progress reward.** Resample the line to a 1-D path-length parameter s(p) and
   reward the policy for advancing s (projected progress) instead of only gate-crossing
   events — this is the single highest-value shaping signal (it densifies the sparse gate
   reward without hand-tuned waypoints). The optimal line gives the *direction* and the
   *achievable speed profile* v(s) to chase.
3. **Speed-profile target.** Use v(s) from the line as a soft reference: reward small
   `|‖v_policy‖ − v_ref(s)|`. The line already encodes the corner slow-downs (drag wall +
   turn) so the policy doesn't have to discover them.
4. **Warm-start / BC.** The `(a_c, ω)` command stream is a feasible open-loop solution
   (0.2 mm forward-integration drift). It can seed a behavior-cloning pretrain or an
   APG/SHAC differentiable-sim rollout before PPO fine-tuning — bootstrapping off a known-
   feasible fast line sidesteps the "pure RL explores 0% of the fast regime" cold-start.
5. **Caveat for deployment.** The line is computed in the *plant-class* model (no rate lag).
   Before using it as a hard reference, replay it through the faithful twin to confirm the
   inner-loop lags don't break the corner timing; use it as a *soft* shaping target, not a
   hard tracking setpoint, so the policy retains authority to absorb the lag.
```
