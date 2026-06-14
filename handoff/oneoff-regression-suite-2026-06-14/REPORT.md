# Regression Suite Promotion — oneoff-regression-suite-2026-06-14

**Branch:** `oneoff-regsuite` (cc3a25a, pushed)  
**Worktree:** `.claude/worktrees/zen-cray-935c85` (off main 742dc43)  
**Date:** 2026-06-14

---

## Summary

Promoted the **3 remaining external-invariant regression scripts** from
`handoff/ultracode-substrate-audit-2026-06-13/regression_suite/` into `tests/`.

### State at session start

The regression_suite directory has **11 test files**. Of these, **8 were already properly
promoted** to `tests/` in a prior session (`laptop-promote-regression-suite-2026-06-13`).
Those 8 files:

| File | Status |
|---|---|
| `test_confirmed_cr2_01.py` | Already in tests/ (updated, on main) |
| `test_confirmed_cr4_01.py` | Already in tests/ (updated, on main) |
| `test_confirmed_cr4_03.py` | Already in tests/ (updated, on main) |
| `test_confirmed_p4_c05.py` | Already in tests/ (fix-landed version, on main) |
| `test_frame_force_vs_fd_mirror_canary.py` | Already in tests/ (updated, on main) |
| `test_mavlink_velocity_single_rotation.py` | Already in tests/ (updated, on main) |
| `test_train_deploy_obs_elementwise.py` | Already in tests/ (updated, on main) |
| `test_twin_diffaero_extreme_parity.py` | Already in tests/ (updated, on main) |

### Slug collision resolution

`regression_suite/test_confirmed_cr4_03.py` → collides with existing
`tests/test_confirmed_cr4_03.py`. The tests/ version is the PROMOTED/UPDATED copy
(correct `tests/_audit_io.py` path + skipif mark). The regression_suite original
(old `_SCRATCH` path) was NOT re-promoted — it stays in handoff as the audit
artifact. No rename required: the collision was already resolved by the prior session.

### Files promoted this session (3)

| File | What it guards | pytest functions |
|---|---|---|
| `tests/test_confirmed_cr1_01.py` | **CR1-01**: yaw command→wire-rate SIGN FLIP (realized-physics corr(−w_raw[:,2], act[:,3])) | 4 |
| `tests/test_confirmed_cr3_02.py` | **CR3-02**: stale DERIVED deploy channels in refit dataset (recipe-free action/obs lens) | 1 |
| `tests/test_confirmed_p3_c06.py` | **P3-C06**: recording quantization grids in fly_rl.py debug_obs (grid + round-trip) | 2 |

**Total new test functions: 7**

### Path fixes applied (identical pattern to the 8 already promoted)

Each file changed:
```python
# OLD (regression_suite original — points to handoff scratch dir)
_SCRATCH = Path(__file__).resolve().parents[1] / "scratch"
if str(_SCRATCH) not in sys.path:
    sys.path.insert(0, str(_SCRATCH))
import _audit_io as A

# NEW (tests/ version — depth-independent)
import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
import _audit_io as A

pytestmark = pytest.mark.skipif(
    not (A.dataset_present("postfix") and A.dataset_present("refit")),
    reason="ShadowPC audit recordings (postfix+refit extracted) not present",
)
```

`ROOT = A.ROOT` (test_confirmed_cr3_02.py) was retained as-is since it already
derives from `A.ROOT` (depth-independent via `_audit_io._find_root`).

---

## Verification

### Escape hatch: not triggered

All 7 new test functions either PASS or SKIP (datasets not present). None failed.
No external invariant regression detected on current main.

### pytest collection (3 new files)

```
tests/test_confirmed_cr1_01.py::test_corrected_era_yaw_sign_is_positive
tests/test_confirmed_cr1_01.py::test_buggy_era_yaw_sign_is_negative_negative_control
tests/test_confirmed_cr1_01.py::test_clean_sign_flip_between_eras
tests/test_confirmed_cr1_01.py::test_east_canary_positive_control_substrate_discriminates
tests/test_confirmed_cr3_02.py::test_confirmed_cr3_02
tests/test_confirmed_p3_c06.py::test_recorded_fields_lie_exactly_on_declared_grid
tests/test_confirmed_p3_c06.py::test_action_pipeline_round_trip_closes_on_the_round4_grid

7 tests collected in 2.76s
```

### Full suite (from worktree root, pyproject.toml testpaths=["tests"])

```
703 passed, 42 skipped in 342.96s (0:05:42)   ← exit code 0
```

**Prior baseline (main, 742dc43):** 703 passed, 35 skipped = 738 total  
**After promotion:** 703 passed, 42 skipped = 745 total  
**Delta:** +7 skipped (all 7 new tests — datasets absent; skipif guard working correctly)  
**Regressions:** 0

---

## Git

- Branch: `oneoff-regsuite`
- Commit: `cc3a25a` — "tests: promote 3 remaining regression-suite invariants (CR1-01, CR3-02, P3-C06)"
- Pushed to `origin/oneoff-regsuite`
- `git add` scope: `tests/test_confirmed_cr1_01.py tests/test_confirmed_cr3_02.py tests/test_confirmed_p3_c06.py` ONLY
- Not merged to main (commander is merge gate)

---

## MEMORY-DELTA

```
REGRESSION-SUITE PROMOTION COMPLETE (oneoff-regression-suite-2026-06-14, cc3a25a):
- 8 of 11 scripts were already in tests/ (prior session laptop-promote-regression-suite-2026-06-13).
- This session: 3 remaining promoted (CR1-01, CR3-02, P3-C06) → now ALL 11 in tests/.
- Slug collision test_confirmed_cr4_03.py: resolved in prior session; no rename needed.
- Full suite: 703 passed, 42 skipped (7 new tests skip cleanly; datasets absent). 0 regressions.
- Branch oneoff-regsuite pushed (cc3a25a). Awaiting commander merge.
```
