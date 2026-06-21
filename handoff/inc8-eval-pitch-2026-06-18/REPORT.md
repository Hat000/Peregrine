> 🚩 **SUPERSEDED 2026-06-19 (511e85c) re: the σ_p0 bar.** Any "σ_p0 ≲ 0.08 / NO-GO vs 0.08 / near-field-gate-estimator pivot" conclusion in this report is OVERTURNED — the 0.08 bar was a `margin_envelope.py` double-count of the drone (real gate clearance 0.75−r ≈ 0.37–0.47 m). The REAL bar is **σ_p0_lat ≲ 0.15 (p99 ≲ 0.45)**; measured 0.15–0.20 = MARGINAL-PASSING, NOT NO-GO. Gate-4 is a REACH/PASS-RATE problem; RL stays the tool (the "pivot off RL" is RETRACTED). The engineering + measurements below STAND; only the bar and its GO/NO-GO verdict are corrected. → MEMORY.md:8 + memory/project_rl_increment_history.md §inc8-2026-06-19.
# inc8 σ_p0 — eval look-at injection: trace, fix, and the real 2-axis blocker

**Session:** p2-eval-pitch (branch `p2-eval-pitch`, off main) · 2026-06-18 · Opus 4.8

## ANSWER UP FRONT
- **Does rc1 fly with pitch (2-axis look-at) in the eval? — NO.** 0/200 reach gate-4, all 3 seeds.
- **Faithful 2-axis σ_p0_lat: UNMEASURABLE** (reach 0). Best-available faithful number is the
  **yaw-only** probe, **seed0 σ_p0_lat = 0.177 m → NO-GO** vs 0.08 (seeds 1/2 don't fly the eval at all).
- **The prompt's hypothesis is FALSIFIED.** The body-Y (PITCH) injection was **never** broken —
  it is bit-faithful to training. The real injection bug is on **body-Z (YAW)** (found + fixed).
  2-axis still doesn't fly, but **not because of the injection** — because the aggressive
  g_pitch=3 look-at + the eval/train **obs-fidelity gap (#37)** destabilize the rc1 policy.

This is the escape-hatch outcome in spirit: a real divergence was **located and fixed** with a clean
trace, the hypothesized divergence was **disproven**, and the true blocker (obs-fidelity, not the
injection) is identified with evidence.

---

## 1. t_cam comparison (the leading hypothesis test) — FALSE
Matched approach state, gate in band (range 18.0 m, in-image). `handoff/.../trace_bodyrate.py`:

| source | t_cam (camera frame) |
|---|---|
| numpy `fix_surrogate.geometry` (eval) | `[0.488801, 4.106878, 17.533244]` |
| torch `batched_geometry` (training) | `[0.488801, 4.106878, 17.533244]` |
| **\|Δ\| per component** | **`[0, 0, 0]`** |

t_cam is **bit-identical**. The geometry (`t_cam = R_camera_from_body @ R_wb.T @ lever`, same baked 20°
matrix both sides) does not diverge. The "two `_last_geom` builds disagree on Y" hypothesis is dead.

## 2. The real divergence: YAW, not pitch — file:line + mechanism
**Mechanism.** `policy_step` (fly_rl.py:544) maps the policy's FLU rates to FRD via the **live wire
map** `_ACT_FLU_TO_FRD = [1,-1,1]` — because the eval plant runs the **live** `rate_sign`
`_RATE_SIGN_LIVE = [1,1,1]` (offline_rollout.py:64), *not* the trained-world `[1,1,-1]`. The eval's
look-at block reconstructed the FLU action from that FRD using the **training adapter** `_FLIP = [1,-1,-1]`
(diffaero's FLU↔FRD map). `_FLIP` and `_ACT_FLU_TO_FRD` agree on roll/pitch but **differ on yaw**, so
the injected look-at's **yaw** realized rate landed with a flipped sign vs training. (Both maps are
involutory, so with `dlook==0` the executed rate is unchanged — inc7/look-at-OFF untouched.)

**Site:** `rl/contact_true_eval.py:358,360` (the `flu = rate_frd * _FLIP` / `rate_frd = flu * _FLIP`).

**Matched-state realized-omega trace** (`trace_bodyrate.py`; rl_plant is the bit-parity reference for
diffaero, run with training rate_sign `[1,1,-1]` vs eval `[1,1,1]` + the respective injection maps):

```
YAW-ONLY  (a)TRAINING [0.815, 1.052, 3.208]
          (b)EVAL-CUR(_FLIP)      Δ(b-a)=[0, 0, 0.096]   <- yaw only
          (c)EVAL-FIX(_ACT_FLU)   Δ(c-a)=[0, 0, 0]
2-AXIS    (a)TRAINING [0.815,-0.200, 3.208]
          (b)EVAL-CUR(_FLIP)      Δ(b-a)=[0, 0, 0.059]   <- yaw only; PITCH bit-identical (Δ=0)
          (c)EVAL-FIX(_ACT_FLU)   Δ(c-a)=[0, 0, 0]
```

The divergence is **entirely on the yaw axis**; **pitch (body-Y) was already bit-identical to training**
even in the buggy eval. The fix zeroes the divergence on **all** axes.

## 3. The fix (diff)
`rl/contact_true_eval.py` — use the wire map `policy_step` actually used, not the training adapter:
```diff
-from fly_rl import (GateMap, N_GATES, _FLIP, _GATE_POS_ZUP, ...)
+from fly_rl import (GateMap, N_GATES, _FLIP, _ACT_FLU_TO_FRD, _GATE_POS_ZUP, ...)
...
-                flu = rate_frd * _FLIP                  # real FRD -> real FLU
+                flu = rate_frd * _ACT_FLU_TO_FRD        # FRD -> FLU (invert policy_step wire map)
                 flu = np.clip(flu + dlook, _ACT_MIN[1:4], _ACT_MAX[1:4])
-                rate_frd = flu * _FLIP                  # real FLU -> real FRD
+                rate_frd = flu * _ACT_FLU_TO_FRD        # FLU -> FRD (re-apply wire map)
```
Plus a corrected convention comment block. Does **not** touch the policy-output rate mapping, the
`_rate_sign` alias, or the pinned `lookat_correction` primitive (escape-hatch respected).

## 4. Byte-identical / invariants
- **inc7 (obs_dim 17, no look-at): BYTE-IDENTICAL** — `diff before_inc7.txt after_inc7.txt` = empty
  (`INC7_BYTE_IDENTICAL_PASS`). (inc7 never enters the look-at block; guaranteed by construction.)
- **Primitive pin** `tests/test_inc8_lookat.py`: **17 passed** (primitive untouched).
- **green_gate: GREEN** — OFF==inc7 AST parity, +L sign-faithfulness, VQ1 constants guard, 947≥933
  sentinel, 54 diff-scoped tests passed.
- ⚠️ **inc8 YAW-ONLY is NOT byte-identical** — and *cannot be*: the trace proves YAW was the buggy
  axis, so the faithful fix necessarily changes yaw-only's executed yaw rate. The task's
  "yaw-only byte-identical" requirement was predicated on the (now-disproven) assumption that yaw was
  the good axis. **Flight/σ are essentially unchanged** (seed0 yaw-only σ_p0_lat 0.177→0.177, reach
  ~0.67 both) because the yaw look-at has negligible flight effect (fix-rate≈0 either way). **Recommend
  KEEP the fix** — it is the correct, faithful code and a prerequisite for any future 2-axis measurement.

## 5. Re-fly with pitch — still doesn't fly, and WHY (the real blocker)
g_pitch magnitude sweep (rc1 seed0, trainreset, 5 seeds each; `diag_sweep.py`):
```
lookat-OFF / g_pitch=0 : reach_g4 3/5      g_pitch=0.5 : 2/5
g_pitch=1.0 : 0/5      g_pitch=1.5/2.0/3.0 : 0/5
```
Monotonic collapse with g_pitch; **both pitch signs break** (consistent — pitch sign is not the issue).
- **Not the injection:** the pitch injection is bit-faithful (§2). The fix corrects yaw; 2-axis still 0.
- **Not just the rest-launch:** with a launch-grace (look-at off for the first 25 steps, drone past
  gate-0/1 *with velocity*), enabling 2-axis still kills it at the next transition (furthest=2;
  `diag_launchgrace.py`). 2-axis is destabilizing **with velocity too**.
- **Mechanism:** the look-at pitch saturates the rate rail (`dlook_pitch = g_pitch·u_y`, |u_y|→~0.9 when
  the gate is off the 20°-up camera axis — every segment of the descending course), and the rc1 policy
  cannot compensate on the **estimator-emulated** obs. The policy is finely tuned to its **nominal**
  training obs/cold-start (a low-noise KF crashes even yaw-only). This is the **#37 emul-fidelity gap**,
  amplified by the aggressive g_pitch=3 look-at. rc1 flew 2-axis in *training* (success ~0.5) but cannot
  on the numpy eval obs.

## 6. Faithful σ_p0 (all 3 seeds, n=200, trainreset)
| seed | --lookat | reach | σ_p0_lat | σ_p0_vert | lat_p99 | fix-rate | verdict |
|---|---|---|---|---|---|---|---|
| 0 | yaw (faithful) | 133/200 | **0.177** | 0.159 | 0.370 | 0.000 | **NO-GO** (>0.08) |
| 0 | auto (2-axis) | **0/200** | — | — | — | — | UNMEASURABLE |
| 1 | yaw | 0/200 | — | — | — | — | no-fly (generalization gap) |
| 2 | yaw | 0/200 | — | — | — | — | no-fly (generalization gap) |

- Seeds 1/2 don't fly the eval **even with look-at OFF** (0/50) — pre-existing rc1 generalization gap,
  not this fix.
- yaw-only is **pessimistic** (no elevation pointing → fix-rate 0 → no terminal centering tightening);
  it is the best-available *faithful* number. **NO-GO** at 0.177 (≫0.08).
- 2-axis (which *would* add elevation fixes and could tighten σ) is unmeasurable until obs-fidelity (#37)
  lets rc1 fly 2-axis on emulated obs.

## 7. green_gate
```
GREEN  OFF==inc7 AST parity / +L sign-faithfulness / VQ1 constants guard
GREEN  collected 947 >= baseline 933
GREEN  diff subset: 54 passed
 GREEN -- safe to report this branch up.
```

## Artifacts (`handoff/inc8-eval-pitch-2026-06-18/`)
`trace_bodyrate.py` (t_cam + realized-omega trace) · `diag_episode.py` (single-episode) ·
`diag_sweep.py` (g_pitch sweep) · `diag_startgate.py` (per-start-gate) · `diag_launchgrace.py`
(launch-grace) · `diag_lownoise.py` (obs-fidelity isolation) · `before/after_*.txt`, `sig_*.txt` (runs).

---

## MEMORY-DELTA (commander banks; I did not touch memory/)
- 🚩 σ_p0 eval "pitch breaks the plant" diagnosis is WRONG. Matched-state trace: t_cam numpy==torch
  (Δ=0); the look-at PITCH (body-Y) injection was ALWAYS bit-faithful to training. The real bug was
  body-Z (YAW): contact_true_eval reconstructed FLU via `_FLIP=[1,-1,-1]` (train adapter) instead of
  `_ACT_FLU_TO_FRD=[1,-1,1]` (the wire map policy_step used; eval plant runs live rate_sign [1,1,1]).
- 🚩 FIX merged on p2-eval-pitch (contact_true_eval.py:358,360 → `_ACT_FLU_TO_FRD`). inc7 byte-id;
  primitive 17/17; green_gate GREEN (947). yaw-only NOT byte-id by necessity (yaw WAS the bug) but
  flight/σ unchanged. The prior `handoff/inc8-eval-lookat-2026-06-18/REPORT.md` mis-stated
  `_RATE_SIGN_LIVE=[1,1,-1]` — it is `[1,1,1]`; that report's "pitch" diagnosis is superseded.
- 🚩 2-axis STILL doesn't fly the eval (0/200 all seeds) — NOT the injection. The aggressive g_pitch=3
  look-at saturates the rate rail (gate off the 20°-up camera axis on the descending course) and rc1
  can't compensate on emulated obs (monotonic in g_pitch; both signs; dies with velocity, not just
  rest-launch; low-noise KF crashes even yaw-only). = the #37 obs-fidelity gap amplified by look-at.
- 🚩 FAITHFUL σ_p0 (n=200, trainreset): 2-axis UNMEASURABLE; yaw-only seed0 = 0.177 lat / 0.159 vert /
  p99 0.370 → NO-GO vs 0.08. Seeds 1/2 = 0/200 even look-at-OFF (rc1 generalization gap stands).
- 🚩 NEXT for a faithful 2-axis σ_p0: close obs-fidelity (#37 — port eval to the torch emul obs, or a
  real-detector run) and/or reduce g_pitch to a deployable magnitude (≤0.5 partly flies). The eval
  injection is no longer the blocker.
