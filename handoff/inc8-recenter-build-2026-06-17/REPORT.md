# inc8 RE-TRAIN BUILD — restore through-approach centering (course-flight fix)

**Branch:** `p2-inc8-recenter` (off `main` @ b69b5f1)
**Date:** 2026-06-17
**Goal:** restore the through-approach centering the inc8 lineage dropped, so the policy FLIES the course
(lineage success_rate ≡ 0, dies at gate-0), + make course-completion a PRIMARY GO metric.

---

## 1. The reward diff — what R1-to-centre was, what was lost, what I restored

### What R1-to-centre WAS (inc7 — flies 3/3)
`rl/peregrine_racing.py:364`, in `compute_reward_terms`:
```python
progress = prev_d2g - curr_d2g          # d2g = ||pos - gate_CENTER||  (Euclidean, weight rw_progress=10.0)
reward += w.progress * progress
```
A dense per-meter pull toward the **gate-center point** across the whole approach. Its gradient points
radially at the gate center, so it has **two** components: a forward-closing component AND a
**lateral-restoring** component. The lateral component strengthens as the drone approaches (when close,
most of the remaining distance-to-center is lateral) → it actively centers the drone on the gate-center
line all the way in. (Documented inc7 bias: "radial-to-center shaping fights wide racing lines" — the
reason inc8 tried to replace it.)

### What inc8 did (never flew a lap)
`rl/peregrine_racing_inc8.py`: R1-to-centre is **zeroed** —
```python
w_frozen = replace(self.rw, progress=0.0)   # R1-to-centre OFF; R1' replaces it
```
and replaced by two things that do **not** supply a lateral restoring force:
- **arc-Γ progress** (`r1p = R8.arc_progress_reward(s_curr, s_prev, rw_progress)`): rewards advancing
  **along** the reference line Γ. Its gradient is Γ's tangent → **along-track only**. Lateral drift off Γ
  barely changes the projected arc-length `s`, so there is **no penalty for drifting off the line**.
- **near-gate `centering_reward`** (`cr`): ramps up near the crossing (range→0) but penalises
  `err_ip = |KF_pos − truth|_inplane` — the **estimator error** (σ_p0 accuracy), **not the drone's
  physical offset** from the gate center. It does not pull the flight path toward center, and only acts
  near the gate.

Net: nothing pulls the drone back onto the gate-center line through the approach → it follows the racing
arc + points the camera (look-at) but under-centers and drifts ~5 m → misses gate-0. Confirmed
confound-free (Adroit): in-training `success_rate ≡ 0` across **all** runs incl. the S2-seed2 parent and
warm-start; `l_episode ~1.5 s` (dies at gate-0).

### What I restored (the EFFECT, not blindly)
A NEW gated term `rl/inc8_reward.through_centering_reward` + the env wiring:
```python
# gate frame: +x = crossing axis (plane at x=0); y,z in-plane → centre line is y=z=0
inplane = hypot(y, z)                                   # cross-track distance to the gate-centre line
through_centering = rw_through_centering * (prev_inplane - curr_inplane)   # DELTA form
```
This is **R1-to-centre's lateral/centering component, isolated** (delta/potential form, same as
R1-to-centre's `prev_d2g − curr_d2g`), pulling toward the **gate-center line across the WHOLE approach**
(not near-gate-only). Both `prev_inplane`/`curr_inplane` are measured against the **same post-advance
gate** `tg_new` (mirrors R1-to-centre's "no spike at passage").

**Why cross-track-only (decoupled from forward progress), not the full Euclidean R1-to-centre:**
- It is **orthogonal to arc-Γ** (arc-Γ = along-track progress; this = cross-track offset) → it adds
  exactly the missing lateral restoring force **without double-counting forward progress** and without
  re-introducing the M-2 corner-cut at full strength (the very thing inc8 replaced R1-to-centre to avoid).
- On the gate-0 approach (straight, from the standing pad) arc-Γ's tangent ≈ the radial-to-center forward
  direction, so the **only** thing arc-Γ lacks vs R1-to-centre is the lateral term — which is exactly what
  this adds. Non-redundant.
- Telescoping potential form → dense gradient toward center, sums ~0 over a closed approach, cannot
  dominate the racing objective.

It **composes** with: arc-Γ (orthogonal), the existing near-gate σ_p0 centering (kept intact, OFF by
default), and the look-at primitive (camera, not flight path). 20-dim obs unchanged.

**Implementation:** `rl/inc8_reward.py` (new `through_centering` weight field default 0.0 + new
`through_centering_reward` fn); `rl/peregrine_racing_inc8.py` (reuses the already-computed
`rel_prev`/`rel_curr` gate-frame positions gathered at `tg_new`, adds `tc` to the reward the same way
`fb`/`cr` are added, logs `inc8_through_centering`).

### Escape-hatch (arc-Γ vs centering) — NOT triggered
They do not irreconcilably fight. arc-Γ rewards along-track motion; the new term penalises cross-track
drift — an orthogonal decomposition, no double-count. Any near-gate tension (if Γ crosses slightly
off-center) is mild and bounded: the cross-track delta → 0 at the center (which is inside the opening →
still a valid pass), and inc7 proves this exact "pull toward gate center" signal is compatible with
completing the course. The full-Euclidean exact-inc7 form is the documented alternative if cross-track
under-centers (a one-line change: use `prev_d2g`/`curr_d2g`), but it overlaps arc-Γ on forward progress
and re-introduces M-2; cross-track is the cleaner, single-variable restore.

---

## 2. OFF byte-identity proof

`rw_through_centering = 0.0` (the default) → **current inc8 byte-identical** for the meaningful contract
(reward stream, RNG draw sequence, event set, obs):
- **Behavioral (laptop, bit-exact):** `tests/test_inc8_reward.py::test_through_centering_disabled_is_exact_zero`
  — `through_centering_reward(prev, curr, 0.0)` is `torch.equal` to `torch.zeros_like(prev)` (not allclose;
  bit-exact). So the env's `reward + tc` adds the **exact** zero tensor → reward stream unchanged. This is
  the identical pattern to the already-baseline `fix_bonus`/`centering` default-0 terms (which current inc8
  already adds as zero tensors).
- The extra gather/hypot draw **no RNG** and mutate no state → RNG sequence + terminations identical.
- Only `loss_components`/TB gains one scalar `inc8_through_centering` (= 0.0 when OFF) — a logging artifact,
  not part of the reward/RNG/event-set byte-identity contract (mirrors how `inc8_fix_bonus`/`inc8_centering`
  are always logged at 0.0 by default).
- The inc8=OFF→inc7 structural guard (`tests/test_inc8_off_identity.py`) is untouched and still green.

**ON (sign/shape sane):**
- `test_through_centering_sign_and_shape`: `rw·(prev−curr)` → **positive** when moving toward the center
  line (curr<prev), **negative** when drifting off, **zero** when holding; linear in the weight.
- `test_through_centering_inplane_distance_semantics`: recenter (0.5→0) and drift (0→0.5) are equal and
  opposite (telescoping/potential symmetry).

**Tests run (laptop, `.venv`):**
- `tests/test_inc8_reward.py` — **16 passed** (13 prior + 3 new).
- `tests/ -k inc8` — **97 passed**.
- `tests/test_inc8_off_identity.py` — 3 passed.
- `scripts/green_gate.py` — see §5.

---

## 3. The sbatch — `rl/peregrine_inc8_recenter.sbatch`

Modeled on `peregrine_inc8_s0.sbatch`/`_warmstart.sbatch`. **3 FRESH seeds** (no warm-start — the lineage
never flew, so there is nothing good to warm-start from). Key knobs:
- `+env.rw_through_centering=${RW_TC:-10.0}` — the restore (the **only** change vs S0). Default 10.0 =
  inc7's `rw_progress` / the arc-Γ weight, applied to the cross-track delta only (gentler than inc7's
  full-Euclidean 10.0). Sweepable via `RW_TC=`.
- `++env.lookat_g_yaw=-3.0 ++env.lookat_g_pitch=3.0` (DOUBLE-plus; both empirical, opposite signs).
- `+env.lookat_warmup_updates=${LOOKAT_WARMUP:-200}` — gain-warmup ON (fresh seeds need the ramp to
  survive the fresh-policy value_loss spike that killed 2/3 S3 cold seeds @ step100).
- per-seed explicit `hydra.run.dir` (the Hydra dir-collision footgun fix).
- `rw_centering` stays **OFF** (default 0) — single-variable change; it is the S3-cliff driver and only
  meaningful once laps complete. Layer it in a follow-up once flight is recovered.
- COMMON (DR map-ON, contact-true geometry, BSR3, arm A, band-pass perception, look-at) = verbatim S0.
- AUP block verbatim (SLURM-only / no-internet compute nodes / /scratch / accurate --mem / checkquota).

### GO-criterion fix (the lesson — course-completion is PRIMARY)
- The TB trace (`rl/inc8_tb_trace.py`) now surfaces **`metrics/success_rate`** and
  **`metrics/n_passed_gates`** as columns from step 0, plus a machine-greppable
  `[inc8-tb-trace] FLIGHTCHECK success_rate_max=… n_passed_gates_max=…` summary line.
- Header EARLY-STOP guidance: if `success_rate` stays ~0 AND `n_passed_gates` stays ~0 by ~step 800-1000,
  the restore did NOT recover flight → STOP (scancel) + report; do not burn the full run.
- **Automatic gate:** the script trains the FIRST seed, greps its FLIGHTCHECK line, and if
  `success_rate_max < 0.005 AND n_passed_gates_max < 0.1` (still dying at gate-0) it **STOPS** before
  burning the remaining seeds. (Passing gates but no full lap yet ⇒ FLEW ⇒ continues — that's progress.)
- Only IF laps complete do pointing-retention + σ_p0 become meaningful (σ_p0 via `contact_true_eval.py`
  on the PHYSICAL miss, not the estimator-floored TB `inc8_estim_err_inplane_m`).

---

## 4. Exact Adroit launch command

PRE-SYNC the edited files to the login node first (the Adroit `peregrine_repo` is a file copy, no
`git pull`): `rl/inc8_reward.py`, `rl/peregrine_racing_inc8.py`, `rl/inc8_tb_trace.py`,
`rl/peregrine_inc8_recenter.sbatch`. Then, from the Adroit **login** node:

```bash
# one job, 3 fresh seeds (auto flight-check gate on the 1st seed):
sbatch --export=ALL,RUNTAG=rc1 /scratch/network/fl3689/peregrine_repo/rl/peregrine_inc8_recenter.sbatch
```
Or 3 separate jobs (more robust; launch seed 0 first, read its FLIGHTCHECK, then 1,2 only if it flew):
```bash
sbatch --export=ALL,SEEDS=0,RUNTAG=rc1_s0 /scratch/network/fl3689/peregrine_repo/rl/peregrine_inc8_recenter.sbatch
# if seed 0 flew:
for s in 1 2; do sbatch --export=ALL,SEEDS=$s,RUNTAG=rc1_s$s \
  /scratch/network/fl3689/peregrine_repo/rl/peregrine_inc8_recenter.sbatch; done
```
Weight sweep if 10.0 over/under-centers: add `RW_TC=<w>` to `--export` (e.g. `RW_TC=6.0`).

**🚩 Adroit/SLURM launch is Fengyou's call — this build does NOT launch.**

---

## 5. Files changed
- `rl/inc8_reward.py` — `Inc8RewardWeights.through_centering` field (default 0.0) + `through_centering_reward()`.
- `rl/peregrine_racing_inc8.py` — compute `prev_ip`/`curr_ip` (reusing `rel_prev`/`rel_curr` @ `tg_new`),
  add `tc` to the reward, log `inc8_through_centering` (metric_vec 16→17).
- `rl/inc8_tb_trace.py` — `success_rate`/`n_passed_gates` columns + FLIGHTCHECK summary line.
- `tests/test_inc8_reward.py` — 3 new tests (OFF exact-zero, sign/shape, in-plane semantics).
- `rl/peregrine_inc8_recenter.sbatch` — NEW (3 fresh seeds, restore ON, course-completion GO + auto gate).

**green_gate: GREEN** — full suite **940 passed** (baseline 933); load-bearing invariants all GREEN
(OFF==inc7 AST parity, +L sign-faithfulness, VQ1 constants guard); "safe to report this branch up."
