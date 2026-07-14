"""Tests for the STALE COARSE-MAP fix (Bug 1) + the ckpt-promote path resolution fix (Bug 2).

Bug 1 (rl/peregrine_racing_ego.py): reset_idx RE-RANDOMIZES the per-env course geometry every episode
(base reset_idx -> _assign_courses -> in-place gate_pos[env_idx]/spawn_pos[env_idx]) and rebuilds the
racing line, but self._coarse_map -- a DERIVED copy from build_coarse_map, NOT a view of gate_pos -- was
built ONCE in __init__ and never rebuilt. So after episode 1 every reset env carried the INITIAL
geometry's turn hints (wrong acquisition/anticipation prior). The fix rebuilds the reset rows in
reset_idx with the IDENTICAL thresholds the __init__ build used.

The full PeregrineRacingEgo env needs diffaero (cluster-only), so -- like tests/test_ego_course_wiring.py
and tests/test_ego_obs_env.py -- these pin the PURE, laptop-testable pieces:
  * build_coarse_map (pure torch): a geometry change DOES flip sectors; an unchanged geometry is idempotent.
  * the EXACT reset assignment semantics (self._coarse_map[env_idx] = full_map[env_idx]): only reset rows
    change; non-reset rows stay byte-identical (so the all-N rebuild + gather-rows pattern is correct).
  * a SIZING sweep quantifying how many sectors actually flip across trackA-distribution resamples.

Bug 2 (rl/peregrine_train_ego.py::_find_runner_checkpoints): the post-train auto-promotion could not find
the runner's checkpoints/ dir because it re-resolved via logger.logdir, which pointed at a DIFFERENT root
than where best_npg/ (and its sibling checkpoints/) actually lived (observed: best_npg under
.../diffaero_repo/outputs/train/<date>/<time>/ vs the logdir guess .../diffaero/outputs/train/<runname>/).
The fix resolves checkpoints/ as a SIBLING of the known-good best_dir first. peregrine_train_ego imports
diffaero at module top, so we ast-extract the pure _find_runner_checkpoints function and exercise it on a
temp filesystem that reproduces the root mismatch.

Run from repo ROOT: .venv\\Scripts\\python.exe -m pytest tests/test_ego_coarse_map_reset.py -q
"""
import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rl"))

import peregrine_racing_ego as C                                          # noqa: E402  (pure, no diffaero)

torch = pytest.importorskip("torch")
from peregrine_course import sample_courses                              # noqa: E402


DT = torch.float32


# ================================================================================================
# helpers: explicit little Z-up courses with a KNOWN turn direction at gate 0.
# ================================================================================================
def _course_turn(sign: float):
    """One env, 2 gates, spawn at origin. Straight to gate0 (+x), then bend +y (left, sign>0) or -y
    (right, sign<0) to gate1 -> horiz[gate0] is deterministically +1 (left) or -1 (right)."""
    gate_pos = torch.tensor([[[10.0, 0.0, 0.0],
                              [20.0, 5.0 * sign, 0.0]]], dtype=DT)        # (1,2,3)
    spawn_pos = torch.zeros(1, 3, dtype=DT)
    return gate_pos, spawn_pos


# ================================================================================================
# (A) build_coarse_map is geometry-sensitive (flips) and geometry-deterministic (idempotent).
# ================================================================================================
def test_build_reflects_turn_direction():
    gpL, sp = _course_turn(+1.0)                                          # left turn at gate 0
    gpR, _ = _course_turn(-1.0)                                           # right turn at gate 0
    mL = C.build_coarse_map(gpL, sp)                                      # (1,2,2) long {-1,0,1}
    mR = C.build_coarse_map(gpR, sp)
    assert mL.dtype == torch.long and mR.dtype == torch.long
    assert mL.shape == (1, 2, 2)
    # gate-0 horiz sector must be the SIGN of the turn, and must FLIP between the two geometries.
    assert mL[0, 0, 0].item() == 1, mL
    assert mR[0, 0, 0].item() == -1, mR
    assert mL[0, 0, 0].item() != mR[0, 0, 0].item()


def test_build_idempotent_on_unchanged_geometry():
    gp, sp = _course_turn(+1.0)
    m1 = C.build_coarse_map(gp, sp)
    m2 = C.build_coarse_map(gp, sp)
    assert torch.equal(m1, m2)                                            # same geometry -> byte-identical


def test_thresholds_are_threaded():
    """The reset rebuild must use the SAME deadband as the __init__ build. A large deadband collapses a
    real turn to centre (0); a tiny one keeps its sign -- so the threshold argument is load-bearing and
    the fix threads it identically to __init__ (default 0.20)."""
    gp, sp = _course_turn(+1.0)                                           # turn ~0.46 rad at gate 0
    tight = C.build_coarse_map(gp, sp, horiz_thresh_rad=0.20)             # 0.46 > 0.20 -> keeps +1
    wide = C.build_coarse_map(gp, sp, horiz_thresh_rad=1.00)             # 0.46 < 1.00 -> collapses to 0
    assert tight[0, 0, 0].item() == 1
    assert wide[0, 0, 0].item() == 0


# ================================================================================================
# (B) the EXACT reset assignment semantics: self._coarse_map[env_idx] = full_map[env_idx].
#     Only reset envs' rows change; non-reset envs (unchanged geometry) stay byte-identical.
# ================================================================================================
def _mixed_batch():
    """4 envs, 2 gates: envs {0,2} turn LEFT, envs {1,3} turn RIGHT initially."""
    signs = [+1.0, -1.0, +1.0, -1.0]
    gate_pos = torch.stack([_course_turn(s)[0][0] for s in signs], dim=0)  # (4,2,3)
    spawn_pos = torch.zeros(4, 3, dtype=DT)
    return gate_pos, spawn_pos


def test_reset_assignment_updates_only_reset_rows():
    gate_pos, spawn_pos = _mixed_batch()
    coarse0 = C.build_coarse_map(gate_pos, spawn_pos)                     # the __init__ map
    baseline = coarse0.clone()

    # ---- simulate reset_idx for env_idx=[1,3]: their geometry FLIPS (right -> left), in place ----
    env_idx = torch.tensor([1, 3], dtype=torch.long)
    gate_pos[1] = _course_turn(+1.0)[0][0]                                # now LEFT
    gate_pos[3] = _course_turn(+1.0)[0][0]                                # now LEFT

    # ---- the fix: build full-N from CURRENT geometry, assign only the reset rows ----
    full_map = C.build_coarse_map(gate_pos, spawn_pos)
    coarse0[env_idx] = full_map[env_idx]

    # reset rows now reflect the NEW (left) geometry: gate-0 horiz flipped -1 -> +1.
    assert coarse0[1, 0, 0].item() == 1
    assert coarse0[3, 0, 0].item() == 1
    assert baseline[1, 0, 0].item() == -1 and baseline[3, 0, 0].item() == -1   # were right before
    # non-reset rows (envs 0,2) are BYTE-UNTOUCHED (idempotent: geometry unchanged).
    assert torch.equal(coarse0[0], baseline[0])
    assert torch.equal(coarse0[2], baseline[2])


def test_full_rebuild_matches_gather_rows():
    """Rebuilding all-N then gathering the reset rows == rebuilding all-N wholesale, because
    build_coarse_map is deterministic in the geometry and non-reset envs' geometry is unchanged."""
    gate_pos, spawn_pos = _mixed_batch()
    coarse0 = C.build_coarse_map(gate_pos, spawn_pos)
    env_idx = torch.tensor([0, 2], dtype=torch.long)
    gate_pos[0] = _course_turn(-1.0)[0][0]                                # env0 left -> right
    gate_pos[2] = _course_turn(-1.0)[0][0]
    full = C.build_coarse_map(gate_pos, spawn_pos)
    gathered = coarse0.clone()
    gathered[env_idx] = full[env_idx]
    assert torch.equal(gathered, full)                                   # only envs 0,2 changed -> equal


# ================================================================================================
# (C) Bug 2: _find_runner_checkpoints resolves checkpoints/ as a sibling of the real best_dir even when
#     logger.logdir points at a DIFFERENT root (the observed path-root mismatch).
# ================================================================================================
def _load_find_runner_checkpoints():
    """AST-extract the pure _find_runner_checkpoints from rl/peregrine_train_ego.py (the module imports
    diffaero at top level, cluster-only), exec it with os available, and return the callable."""
    src = (ROOT / "rl" / "peregrine_train_ego.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == "_find_runner_checkpoints"), None)
    assert fn is not None, "could not find _find_runner_checkpoints in peregrine_train_ego.py"
    ns: dict = {}
    import os as _os
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<extract>", "exec"), {"os": _os}, ns)
    return ns["_find_runner_checkpoints"]


class _Logger:
    def __init__(self, logdir):
        self.logdir = logdir


def test_promote_finds_checkpoints_via_best_dir_across_root_mismatch(tmp_path):
    find = _load_find_runner_checkpoints()
    # REAL save root: best_npg/ and checkpoints/ are siblings (as the runner writes them).
    real = tmp_path / "diffaero_repo" / "outputs" / "train" / "date" / "time"
    (real / "best_npg").mkdir(parents=True)
    (real / "checkpoints").mkdir()
    best_dir = str(real / "best_npg")
    # logger.logdir points at a DIFFERENT root that has NO checkpoints/ (the observed mismatch).
    wrong = tmp_path / "diffaero" / "outputs" / "train" / "runname"
    wrong.mkdir(parents=True)
    logger = _Logger(str(wrong))

    # WITHOUT best_dir (the OLD behaviour) the logdir guess cannot find it -> None == the bug.
    assert find(logger, best_dir=None) is None
    # WITH best_dir (the fix) it resolves checkpoints/ as best_dir's sibling.
    got = find(logger, best_dir=best_dir)
    assert got is not None
    assert Path(got) == real / "checkpoints"


def test_promote_falls_back_to_logdir_when_best_dir_sibling_absent(tmp_path):
    find = _load_find_runner_checkpoints()
    # No checkpoints/ next to best_npg/, but logger.logdir has one as a SIBLING (parent-of-logdir path).
    run = tmp_path / "run"
    (run / "best_npg").mkdir(parents=True)
    (run / "logs").mkdir()                       # logger.logdir; checkpoints is its sibling under run/
    (run / "checkpoints").mkdir()
    logger = _Logger(str(run / "logs"))
    got = find(logger, best_dir=str(run / "best_npg" / "does_not_matter"))
    # best_dir sibling (best_npg/../checkpoints == run/checkpoints) actually exists here, so it is found.
    assert got is not None and Path(got) == run / "checkpoints"


def test_promote_returns_none_when_nothing_exists(tmp_path):
    find = _load_find_runner_checkpoints()
    logger = _Logger(str(tmp_path / "empty" / "logs"))
    (tmp_path / "empty" / "logs").mkdir(parents=True)
    assert find(logger, best_dir=str(tmp_path / "empty" / "best_npg")) is None


# ================================================================================================
# (D) SIZING: how material is the stale map? Count sector flips across trackA-distribution resamples.
# ================================================================================================
_TRACKA_OV = dict(n_gates=8, spawn_dist_m=(8.0, 15.0), seg_len_m=(10.0, 20.0),
                  yaw_jitter_rad=0.25, gates_above_spawn_m=0.5, spawn_below_g0_m=(0.5, 6.0),
                  spawn_heading=0.0)


def _draw(n, seed):
    g = torch.Generator().manual_seed(seed)
    return sample_courses(n, device="cpu", generator=g, **_TRACKA_OV)


def test_stale_map_is_material(capsys):
    """Two INDEPENDENT resamples of the trackA course distribution (what reset_idx does each episode):
    quantify how many of the 8 per-gate sectors differ. This is the magnitude of Bug 1 -- how wrong the
    un-rebuilt map is after episode 1. Asserts the effect is LARGE (not a rounding artefact)."""
    N = 4000
    a, b = _draw(N, 1), _draw(N, 2)
    ma = C.build_coarse_map(a["gate_pos"], a["spawn_pos"])                # (N,8,2)
    mb = C.build_coarse_map(b["gate_pos"], b["spawn_pos"])
    horiz = (ma[..., 0] != mb[..., 0]).float().sum(dim=1)                 # per-course # differing horiz
    vert = (ma[..., 1] != mb[..., 1]).float().sum(dim=1)
    anysec = (ma != mb).any(dim=-1).float().sum(dim=1)                    # per-course # gates w/ ANY change
    mh, mv, ma_ = horiz.mean().item(), vert.mean().item(), anysec.mean().item()
    with capsys.disabled():
        print(f"\n[coarse-map sizing] trackA n_gates=8, {N} independent resample pairs:")
        print(f"    horiz sectors changed / 8 : mean {mh:.2f}  ({mh/8:.1%})")
        print(f"    vert  sectors changed / 8 : mean {mv:.2f}  ({mv/8:.1%})")
        print(f"    gates w/ ANY change  / 8  : mean {ma_:.2f}  ({ma_/8:.1%})")
    # materiality: the horiz turn hint is wrong on ~half the gates per resample -- assert it is clearly large.
    assert mh > 2.0, f"expected a large horiz flip count, got {mh}"
    assert ma_ > 2.0, f"expected many gates to change, got {ma_}"
