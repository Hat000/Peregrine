# P4-C05 fix — per-gate-yaw gate frame threaded through the deploy/eval obs builders

**Session:** LAPTOP-P4-C05-GATE-YAW-OBS · **Date:** 2026-06-13 · **For:** Fengyou
**Model:** opus-4.8 · **Base:** main @ d404d7b (code @ 6876f44)

---

## 1. What was wrong (the CONFIRMED bug, from the substrate audit §2.1)

`rl/fly_rl.py`'s observation builders (`obs_from_zup` / `build_obs`) hardcoded the gate frame to
**yaw = π**: a fixed `_R_W2G = diag(-1,-1,1)` rotation plus precomputed lookahead tables
`_GATE_REL_POS` / `_GATE_YAW_REL` (= 0 everywhere). The TRAIN side
(`peregrine_racing.get_observations`) instead rotates **per gate** by `get_gate_rotmat_w2g(gate_yaw)`
(rows `[c,s,0; -s,c,0; 0,0,1]`) and builds per-gate `rel_tables`.

On **VQ1 (every gate yaw = π)** the hardcoded specialization equals the per-gate rotation, so the
bug is invisible — every internal-consistency check passes. On **any non-π / VQ2 / `course_mode=random`
course** the deploy seam keeps applying yaw = π while the policy was trained on the per-gate frame,
silently corrupting the **consumed** gate-relative obs (`pos_g`, `vel_g`, `rpy_g_y`, next-gate
lookahead) by **up to 4.22 m**. The same hardcode was mirrored in `rl/offline_rollout.py` (obs +
scoring) and `rl/contact_true_eval.py` (the inc8 selection metric).

This is the foundation for the gate-relative estimator rebuild (the binding VQ2 workstream) and a
prerequisite for any non-π inc8 training: the per-gate gate frame must be correct before anything
downstream of it can be trusted.

---

## 2. The fix

The gate frame now follows the **runtime per-gate yaw**, threaded as an explicit, optional
`gate_map` so the default (VQ1) path is untouched and bit-exact.

### `rl/fly_rl.py`
- New `GateMap` NamedTuple (`gate_pos`, `gate_yaw`, `gate_rel_pos`, `gate_yaw_rel`) + `make_gate_map(pos, yaw)`.
- `_gate_rotmat_w2g(yaw)` — numpy `[c,s,0; -s,c,0; 0,0,1]`, EXACTLY diffaero `get_gate_rotmat_w2g`
  / `world_to_gateframe`.
- `_rel_tables_np(pos, yaw)` — numpy mirror of `peregrine_racing.rel_tables` (the `[i-1]` wrap, the
  wrapped yaw delta).
- `obs_from_zup(..., gate_map=None)` and `build_obs(..., gate_map=None)`:
  - **`gate_map=None`** → the LEGACY VQ1 path: the *exact* `_R_W2G` diag + `_GATE_REL_POS` /
    `_GATE_YAW_REL`, same constant objects and arithmetic as before → **bit-identical** to the pre-fix
    builder (verified 0.0; see §3).
  - **`gate_map` supplied** → the YAW-AWARE path: per-gate `_gate_rotmat_w2g(gate_yaw[tg])` + the
    map's lookahead tables, reproducing `peregrine_racing.get_observations` exactly on any course.
- **Loud guards** (the "fail loudly on a non-π course" requirement):
  - `assert_gate_map_allpi(gate_yaw, where=...)` — raises if any gate yaw deviates from π beyond
    `1e-4` (the real VQ1 datum `3.141592569` is π to ~8.4e-8, well inside).
  - `_assert_vq1_constants_consistent()` runs **at import**: the hardcoded constants must equal the
    general per-gate construction on VQ1 — catches a half-migrated edit (e.g. bumping `_GATE_POS_ZUP`
    to a non-π course while leaving the hardcoded gate frame).
  - `_assert_live_course_is_vq1(track_gates)` runs **at deploy time** in `_fly_armed` before the RL
    loop: if the sim broadcasts a course whose gate count/positions differ from the hardcoded VQ1
    course (the loop builds obs with `gate_map=None`), it aborts loudly (gates are ~24 m apart, the
    2 m tolerance never false-fires on VQ1). No-op + warn when no track map has arrived (legacy
    behaviour preserved — the live-confirmed 5/5 standing path is unchanged).

### `rl/offline_rollout.py`
- `obs_from_truth(..., gate_map=None)` threads the map into `obs_from_zup`.
- `gate_event` / `frame_strike_other_gates` gain optional `gate_pos_zup` / `gate_yaw` (default VQ1 =
  bit-exact); `_w2g_for(gate, gate_yaw)` selects exact-diag vs per-gate rotation. (Audit-flagged
  `_R_W2G` hardcodes in the scoring path.)

### `rl/contact_true_eval.py` (the inc8 selection metric)
- `_score_gate(..., gate_yaw=None)` — per-gate world→gate rotation (default = exact-diag VQ1).
- `run_episode(..., gate_yaw=None)` — builds a yaw-aware obs `gate_map` AND scores per-gate, so the
  policy and the metric see the *same* gate frame; `gate_d_offset_probe(..., gate_yaw=None)` threads
  it through the re-score. VQ1 (`gate_yaw=None`) stays bit-exact.

No caller signature broke: every new parameter is optional and defaults to the VQ1 behaviour
(`replay_obs.py`, `tilt_segment_analysis.py`, the other suites call the builders unchanged).

---

## 3. Evidence

### 3a. BIT-EXACT VQ1 invariant (mandatory) — the fix changes NOTHING on the π-course

Default `obs_from_zup` (gate_map=None) vs the **literal pre-fix arithmetic** over 2000 random
tilted states, both `virtual_flip` modes:

```
VQ1 default vs old reference   max|diff| = 0.0        (exactly bit-identical)
```

(`tests/test_confirmed_p4_c05.py::test_p4_c05_bitexact_vq1_default_unchanged`, and a direct
500-sample check during development.) The yaw-aware path applied to VQ1 (`make_gate_map` with yaw=π)
differs from the exact-diag default by only the `sin(π)≈1.2e-16` epsilon (≤2.4e-7 in float32) — it
is only ever used for non-π, and there it matches the train side (below).

### 3b. NON-π correctness — OLD corrupts / NEW correct

Non-π course (gate-2 yaw = 3.257 rad = 186.6°, plus two bent gates — the prior-pass datum),
representative tilted state, target gate 2:

```
SHIPPED yaw-aware (gate_map)  vs train ground truth  max|diff| = 0.0   (bit-exact, all 17 dims)
DEFAULT (hardcoded-π)         vs train ground truth  max|diff| = 4.22 m on consumed fields

 idx label        train      OLD(π)     NEW(gm)   |OLD-train|
   0 pos_gx        24.7410    24.5938    24.7410     0.1473
   1 pos_gy        -0.1594     2.7000    -0.1594     2.8594
   3 vel_gx       -11.6886   -12.0000   -11.6886     0.3114
   4 vel_gy         3.3730     2.0000     3.3730     1.3730
   8 rpy_g_y       -0.2903    -0.1745    -0.2903     0.1158
  13 nxt_relx      35.9251    36.9000    35.9251     0.9749
  14 nxt_rely     -10.5209    -6.3000   -10.5209     4.2209   <== the audit's 4.22 m
  16 nxt_relyaw    -0.5658     0.0000    -0.5658     0.5658
```

The NEW (yaw-aware) deploy obs equals the obs the policy was trained on, **bit-exactly**; the OLD
default diverges by exactly the audit's 4.22 m. `train_obs` here is the independent train-side
re-derivation via the public `peregrine_racing.world_to_gateframe` / `rel_tables`.

### 3c. Regression suite

- `tests/test_confirmed_p4_c05.py` extended **5 → 10** tests (all pass): the original LENS1 (real
  recordings reproduce logged obs) / LENS2a (all-π) / LENS2b (external invariant) — with
  `corrected_obs_builder` now driving the **real shipped** `obs_from_zup(gate_map=...)` instead of a
  reimplementation — plus new: bit-exact-VQ1-default, shipped-yaw-aware-matches-train-nonπ,
  build_obs-threads-gate_map, make_gate_map==train rel_tables, and the loud all-π guard fires.
  The LENS2b negative control (default path diverges by 4.22 m on non-π) is **retained by design**:
  it documents that the default builder is VQ1-only and a non-π course must pass an explicit gate_map.
- Full suite (from repo root): **692 passed** (was 687; +5 new p4_c05 tests).
  `test_train_deploy_obs_elementwise`, `test_contact_true_eval`, `test_offline_rollout_events`,
  `test_navigator` included and green. (NB: pytest MUST be run from the repo root — `test_navigator`
  loads a saved track-map JSON via a repo-root-relative path; running from `rl/` FileNotFounds it,
  unrelated to this change.)

---

## 4. Escape-hatch note

Per the brief, the per-gate-yaw gate frame is foundational for the gate-relative estimator rebuild
(not throwaway): both need exactly this `get_gate_rotmat_w2g(gate_yaw[tg])` frame. The fix is the
minimal, backward-compatible substrate for it — VQ1 untouched, non-π now correct, the hardcoded path
guarded loud. No entanglement that would make it throwaway was hit.

---

## 5. Files touched

- `rl/fly_rl.py` — GateMap + helpers + guards; `obs_from_zup`/`build_obs` gate_map; deploy guard.
- `rl/offline_rollout.py` — `obs_from_truth` gate_map; yaw-aware `gate_event`/`frame_strike_other_gates`.
- `rl/contact_true_eval.py` — yaw-aware `_score_gate`/`run_episode`/`gate_d_offset_probe`.
- `tests/test_confirmed_p4_c05.py` — extended to 10 tests (fix-landed + bit-exact + guard).
</content>
</invoke>

---

## MEMORY-DELTA (≤8 lines)

- P4-C05 FIXED on main: `fly_rl.obs_from_zup`/`build_obs` now take opt-in `gate_map` (per-gate yaw via `make_gate_map`), matching `peregrine_racing.get_observations` exactly; default `gate_map=None` = the hardcoded yaw=π VQ1 path, BIT-EXACT vs pre-fix (0.0).
- Non-π verified: shipped yaw-aware path == train ground truth bit-exact; old default diverges 4.22 m on `nxt_rely` (the audit datum).
- Same fix in `offline_rollout.py` (obs + `gate_event`/`frame_strike` scoring) and `contact_true_eval.py` (`_score_gate`/`run_episode`/`gate_d_offset_probe`) → inc8 selection metric correct on non-π.
- Loud guards: import-time `_assert_vq1_constants_consistent`; deploy-time `_assert_live_course_is_vq1`; public `assert_gate_map_allpi`. Hardcoded path now fails loud on a non-π course.
- Foundation for the gate-relative estimator rebuild + non-π inc8 (NOT throwaway — same `get_gate_rotmat_w2g(gate_yaw[tg])` frame).
- Tests: `tests/test_confirmed_p4_c05.py` 5→10 (bit-exact-VQ1, shipped-yaw-aware==train, build_obs threads gate_map, make_gate_map==rel_tables, guard fires). Suite 687→692 green (run pytest from REPO ROOT — `test_navigator` needs root-relative paths).
