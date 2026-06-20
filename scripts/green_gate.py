#!/usr/bin/env python3
"""green_gate.py -- the diff-scoped GREEN/RED merge-safety gate for the 2-day burn.

A worker runs this in its worktree BEFORE reporting its branch up. It prints a single
GREEN/RED verdict plus exactly which check failed, so a regression in a load-bearing
invariant can never slip silently into a competition submission.

This is a LOCAL gate, NOT cloud CI: torch + diffaero + the detector stack are too heavy
for a hosted runner, and ~35 tests hard-skip / error-collect without them. The gate runs
against the project ``.venv`` (Python 3.13 + torch CPU + scipy + ultralytics).

WHAT IT CHECKS (in order; first RED stops with a clear reason):

  0. STACK GUARD (escape-hatch enforcement). torch + numpy + scipy must import. If torch
     is absent the load-bearing invariants CANNOT be enforced: ``tests/test_obs_sign_faithfulness.py``
     does ``pytest.importorskip("torch")`` so it silently *vanishes* from collection, and
     ~5 modules hard-error at collection (measured: 884 -> 690 collected). So a torch-less
     run is RED, not a pass-with-skips. Run inside the .venv.

  1. BYTE-IDENTICAL / SIGN INVARIANTS (always, regardless of the diff). A regression here is
     a silently corrupted submission:
       a. OFF==inc7 AST parity      tests/test_inc8_off_identity.py   (PeregrineRacingInc8 OFF == inc7)
       b. +L obs sign-faithfulness  tests/test_obs_sign_faithfulness.py (~4.77e-7; -L control breaks ~24 m)
       c. VQ1 import-time guard      rl/fly_rl.py::_assert_vq1_constants_consistent (runs at import)
     The +L test is additionally asserted to have actually RUN (not skipped) -- the importorskip trap.

  2. TEST-COUNT SENTINEL. ``pytest --collect-only`` count must be >= a baseline (default 1091,
     the honest full-stack count measured 2026-06-20; 723/884/933/947 are stale lineage). Catches
     silent test loss (a deleted module, a broken collection) that a passing subset would hide.

  3. DIFF-SCOPED PYTEST. Files changed vs ``main`` (merge-base) are mapped to the test modules
     that import them (by parsing each test's imports -- auto-adapts as the burn lands new files),
     and only those run. ``--full`` runs the entire suite instead. The invariants in #1 ALWAYS run.

EXIT CODE: 0 == GREEN, 1 == RED. Designed to be the last command a worker runs before reporting up.

USAGE:
    python scripts/green_gate.py                 # diff-scoped: invariants + sentinel + changed-file tests
    python scripts/green_gate.py --full          # run the whole suite instead of the diff subset
    python scripts/green_gate.py --baseline 884  # override the test-count floor
    python scripts/green_gate.py --base origin/main   # diff against a different base ref
    python scripts/green_gate.py --list          # show the planned checks + selected tests, run nothing
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO / "tests"
SRC_DIR = REPO / "src"
RL_DIR = REPO / "rl"

# The three load-bearing invariants. (a)+(b) are pytest node ids; (c) is exercised by a
# dedicated import probe (the assert runs at ``import fly_rl`` and is also re-invoked).
INV_OFF_IDENTITY = "tests/test_inc8_off_identity.py"
INV_OBS_SIGN = "tests/test_obs_sign_faithfulness.py"
INV_OBS_SIGN_NODE = "tests/test_obs_sign_faithfulness.py::test_obs_pos_g_is_plus_L_seen_and_minus_L_breaks"

# Honest full-stack collection count (measured 2026-06-20 in .venv on main, cleanup-audit re-confirm:
# `pytest --collect-only -q` = 1091). Lineage (history, do NOT restore): 723 stale burn-survey;
# 884 stale-base worker; 933 was the 2026-06-17 count; 947 was a stale MEMORY quote; 1077 in fe31d62
# was a runtime PASS-count, NOT a collect count. Override with --baseline.
DEFAULT_BASELINE = 1091

# Paths whose changes never need pytest (docs/notes/state) -- but the invariants still ALWAYS run.
# NOTE: data formats that CODE loads (e.g. rl/reference_line_inc8.json, a fixture) are deliberately
# NOT in the skip list: a changed code-adjacent data file widens to the FULL suite (conservative),
# because a silently corrupted reference line / fixture is exactly the regression class this gate guards.
_NONCODE_PREFIXES = ("memory/", "handoff/", "docs/")
_NONCODE_SUFFIXES = (".md", ".txt", ".pdf", ".png", ".jpg", ".jpeg", ".lock")


# --------------------------------------------------------------------------- output helpers
class C:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YEL = "\033[93m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    END = "\033[0m"


def _color_ok() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


_USE_COLOR = _color_ok()


def _c(s: str, col: str) -> str:
    return f"{col}{s}{C.END}" if _USE_COLOR else s


def hdr(msg: str) -> None:
    print(f"\n{_c('=== ' + msg + ' ===', C.BOLD)}")


def info(msg: str) -> None:
    print(f"  {msg}")


def dim(msg: str) -> None:
    print(_c(f"  {msg}", C.DIM))


# --------------------------------------------------------------------------- git / diff
def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=str(REPO), capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout


def changed_files(base: str) -> list[str]:
    """Files changed vs the merge-base of ``base`` and HEAD, INCLUDING uncommitted (staged +
    unstaged) edits and untracked files. The merge-base is used so a stale-base worktree (branched
    off an old main) does not flag every file main moved on as 'changed' -- only the worker's own
    edits. POSIX-slash relative paths."""
    try:
        merge_base = _git("merge-base", base, "HEAD").strip()
    except RuntimeError:
        merge_base = base  # base may be a bare commit with no shared history; diff against it directly
    files: set[str] = set()
    # committed-on-branch + working-tree (committed and uncommitted) vs the fork point
    for line in _git("diff", "--name-only", merge_base, "--").splitlines():
        if line.strip():
            files.add(line.strip())
    # untracked (new, not yet added) files
    for line in _git("ls-files", "--others", "--exclude-standard").splitlines():
        if line.strip():
            files.add(line.strip())
    return sorted(files)


# --------------------------------------------------------------------------- import-graph mapping
def _module_names_for(path: str) -> set[str]:
    """The import name(s) a changed source file is known by, so we can find its importers.
    src/racer/foo/bar.py -> {'racer.foo.bar', 'racer.foo', 'bar'};  rl/baz.py -> {'baz'}."""
    p = (REPO / path).resolve()
    names: set[str] = set()
    try:
        rel = p.relative_to(SRC_DIR)  # src/racer/... -> dotted package path
        parts = list(rel.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            names.add(".".join(parts))
            names.add(parts[-1])
            # also the parent package (editing a submodule can break a package-level importer)
            if len(parts) > 1:
                names.add(".".join(parts[:-1]))
    except ValueError:
        pass
    try:
        rel = p.relative_to(RL_DIR)  # rl/foo.py is a FLAT module 'foo' on sys.path
        if rel.suffix == ".py":
            names.add(rel.with_suffix("").as_posix().replace("/", "."))
            names.add(rel.stem)
    except ValueError:
        pass
    return {n for n in names if n}


def _test_imports(test_path: Path) -> set[str]:
    """All imported names in a test module (top-level dotted roots + leaf names), from a static
    AST parse -- robust to the ``sys.path.insert`` + ``from fly_rl import ...`` pattern these tests use."""
    try:
        tree = ast.parse(test_path.read_text(encoding="utf-8", errors="ignore"))
    except (SyntaxError, OSError):
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.name)
                names.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.add(node.module.split(".")[0])
    return names


def map_changed_to_tests(files: list[str]) -> tuple[set[str], list[str], list[str]]:
    """Map changed files -> the test node files to run.

    Returns (selected_tests, code_changes_unmapped, reasons).
    - a changed tests/test_*.py maps to ITSELF
    - a changed source module maps to every test whose imports reference it
    - a changed code file with NO importer is reported as 'unmapped' (caller widens to --full)
    """
    selected: set[str] = set()
    reasons: list[str] = []
    unmapped: list[str] = []

    # Pre-index every test module's imports once.
    test_files = sorted(TESTS_DIR.glob("test_*.py"))
    test_import_index = {tf: _test_imports(tf) for tf in test_files}

    for f in files:
        fp = f.replace("\\", "/")
        # changed test file -> run it directly (skip if it was DELETED -- the sentinel catches that;
        # handing pytest a missing path would error and mask the real verdict).
        if fp.startswith("tests/") and Path(fp).name.startswith("test_") and fp.endswith(".py"):
            if (REPO / fp).exists():
                selected.add(fp)
                reasons.append(f"{fp}  (changed test -> itself)")
            else:
                reasons.append(f"{fp}  (test DELETED -> sentinel will catch the count drop)")
            continue
        # pure docs/notes/binaries: never needs pytest (invariants still run separately)
        if fp.startswith(_NONCODE_PREFIXES) or fp.endswith(_NONCODE_SUFFIXES):
            continue
        if not fp.endswith(".py"):
            # a non-Python file under a code dir (config / fixture / reference JSON / .sbatch / .toml):
            # not import-mappable, so widen to the FULL suite rather than silently skip it.
            unmapped.append(fp)
            reasons.append(f"{fp}  (code-adjacent data, not import-mappable -> widen to --full)")
            continue
        # source/code module -> tests that import it
        mods = _module_names_for(fp)
        if not mods:
            unmapped.append(fp)
            reasons.append(f"{fp}  (code change, no module mapping -> widen to --full)")
            continue
        importers = [tf for tf, imps in test_import_index.items() if imps & mods]
        if importers:
            for tf in importers:
                rel = tf.relative_to(REPO).as_posix()
                selected.add(rel)
            reasons.append(f"{fp}  (module {sorted(mods)} -> {len(importers)} test module(s))")
        else:
            unmapped.append(fp)
            reasons.append(f"{fp}  (module {sorted(mods)}, NO importer found -> widen to --full)")

    return selected, unmapped, reasons


# --------------------------------------------------------------------------- checks
def check_stack() -> tuple[bool, str]:
    """Escape-hatch enforcement: the load-bearing invariants are only meaningful with the full
    stack. torch absent -> the +L test importorskips away and ~5 modules error at collection
    (measured 884 -> 690). So torch-less is RED, never a green-with-skips."""
    missing = []
    for mod in ("numpy", "scipy", "torch"):
        try:
            __import__(mod)
        except Exception as e:  # noqa: BLE001
            missing.append(f"{mod} ({type(e).__name__})")
    if missing:
        return False, (
            f"required stack not importable: {', '.join(missing)}. The load-bearing invariants "
            f"(notably the +L sign test, which importorskips on missing torch) CANNOT be enforced "
            f"here. Run inside the project .venv (Python 3.13 + torch CPU): "
            f"`{REPO / '.venv' / 'Scripts' / 'python.exe'} scripts/green_gate.py`.")
    import torch  # noqa: E402
    return True, f"numpy/scipy/torch OK (torch {torch.__version__})"


def check_vq1_import_guard() -> tuple[bool, str]:
    """Invariant (c): the import-time VQ1-constants guard. Importing fly_rl runs
    _assert_vq1_constants_consistent() at module scope (rl/fly_rl.py); we import it in a clean
    subprocess (so the heavy import can't pollute this process) and re-invoke the assert explicitly."""
    probe = (
        "import sys; from pathlib import Path; r=Path(r'%s');"
        "sys.path[:0]=[str(r/'rl'),str(r/'src'),str(r/'scripts')];"
        "import fly_rl; fly_rl._assert_vq1_constants_consistent();"
        "print('VQ1_GUARD_OK')" % str(REPO)
    )
    out = subprocess.run([sys.executable, "-c", probe], cwd=str(REPO),
                         capture_output=True, text=True)
    if out.returncode == 0 and "VQ1_GUARD_OK" in out.stdout:
        return True, "fly_rl imported clean; _assert_vq1_constants_consistent() re-ran OK"
    tail = (out.stderr or out.stdout).strip().splitlines()
    return False, "VQ1 import-time guard FAILED: " + (tail[-1] if tail else "(no output)")


def _run_pytest(node_ids: list[str], *, extra: list[str] | None = None) -> tuple[int, str]:
    cmd = [sys.executable, "-m", "pytest", *(extra or []), *node_ids]
    out = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    return out.returncode, out.stdout + out.stderr


_COUNT_RE = re.compile(r"(\d+)\s+tests?\s+collected")
_SUMMARY_RE = re.compile(r"(\d+)\s+(passed|failed|error|errors|skipped|deselected|xfailed|xpassed)")


def check_test_count(baseline: int) -> tuple[bool, str, int]:
    """Sentinel: collected count must be >= baseline. Catches silent test loss / broken collection."""
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests"],
                         cwd=str(REPO), capture_output=True, text=True)
    text = out.stdout + out.stderr
    if "error" in text.lower() and "collected" not in text.lower():
        tail = text.strip().splitlines()[-3:]
        return False, "collection errored:\n      " + "\n      ".join(tail), 0
    m = _COUNT_RE.search(text)
    if not m:
        return False, "could not parse collected count from pytest --collect-only", 0
    count = int(m.group(1))
    # a nonzero collection-error count is itself a silent-loss signal
    errs = re.search(r"(\d+)\s+errors?\b", text)
    n_err = int(errs.group(1)) if errs else 0
    if n_err:
        return False, f"{count} collected but {n_err} collection error(s) -- modules failing to import", count
    if count < baseline:
        return False, (f"collected {count} < baseline {baseline} -- tests went MISSING "
                       f"(silent test loss). If you legitimately removed tests, justify it and "
                       f"lower --baseline deliberately."), count
    return True, f"collected {count} >= baseline {baseline}", count


def run_invariants() -> list[tuple[str, bool, str]]:
    """Always-run load-bearing invariants. Returns [(name, ok, detail)].

    Each pytest invariant is run in its OWN process so its exit code is an unambiguous
    verdict (no parsing a combined run), and ``-rs`` makes any skip explicit -- a skip of
    the +L test (the torch importorskip trap) is treated as RED, not a pass."""
    results: list[tuple[str, bool, str]] = []

    # (a) OFF==inc7 AST parity
    rc, text = _run_pytest([INV_OFF_IDENTITY], extra=["-q", "-rs", "--no-header"])
    summ = _last_summary(text)
    ok = (rc == 0) and ("passed" in summ) and ("failed" not in summ) and ("error" not in summ.lower())
    results.append(("OFF==inc7 AST parity (test_inc8_off_identity.py)", ok,
                    summ if ok else _fail_excerpt(text)))

    # (b) +L sign -- MUST actually run (a skip == torch missing == the silent-skip trap).
    rc, text = _run_pytest([INV_OBS_SIGN_NODE], extra=["-q", "-rs", "--no-header"])
    summ = _last_summary(text)
    skipped = (rc == 5) or ("skipped" in summ and "passed" not in summ) or ("no tests ran" in summ)
    if skipped:
        results.append(("+L sign-faithfulness (test_obs_sign_faithfulness.py)", False,
                        "the +L sign test was SKIPPED / did not run (torch importorskip) -- the "
                        "invariant did NOT execute. This is the silent-skip trap the gate exists "
                        "to catch. Run inside the .venv (torch present)."))
    else:
        ok = (rc == 0) and ("passed" in summ) and ("failed" not in summ)
        results.append(("+L sign-faithfulness (test_obs_sign_faithfulness.py)", ok,
                        summ if ok else _fail_excerpt(text)))

    # (c) VQ1 import-time guard
    ok, detail = check_vq1_import_guard()
    results.append(("VQ1 import-time constants guard (fly_rl::_assert_vq1_constants_consistent)", ok, detail))
    return results


def _last_summary(text: str) -> str:
    for line in reversed(text.strip().splitlines()):
        if _SUMMARY_RE.search(line) or "passed" in line or "no tests ran" in line:
            return line.strip().strip("=").strip()
    return text.strip().splitlines()[-1] if text.strip() else "(no output)"


def _fail_excerpt(text: str) -> str:
    lines = text.strip().splitlines()
    fails = [ln for ln in lines if re.search(r"FAILED|ERROR|assert|Error", ln)]
    summary = _last_summary(text)
    head = "\n      ".join(fails[-6:]) if fails else ""
    return (head + ("\n      " if head else "") + summary).strip()


# --------------------------------------------------------------------------- driver
def main() -> int:
    ap = argparse.ArgumentParser(description="Diff-scoped GREEN/RED merge-safety gate (Peregrine burn).")
    ap.add_argument("--full", action="store_true", help="run the whole suite, not just the diff subset")
    ap.add_argument("--baseline", type=int, default=DEFAULT_BASELINE,
                    help=f"test-count floor (default {DEFAULT_BASELINE}, the honest full-stack count)")
    ap.add_argument("--base", default="main", help="git ref to diff against (default: main)")
    ap.add_argument("--list", action="store_true", help="print the plan + selected tests and exit (run nothing)")
    args = ap.parse_args()

    print(_c("GREEN-GATE", C.BOLD) + f"  repo={REPO.name}  base={args.base}  baseline={args.baseline}"
          + ("  [FULL]" if args.full else ""))

    reds: list[str] = []

    # 0. STACK GUARD ------------------------------------------------------------------
    hdr("0. stack guard (escape-hatch enforcement)")
    ok, detail = check_stack()
    (info if ok else lambda m: print(_c("  RED  " + m, C.RED)))(detail)
    if not ok:
        # cannot meaningfully run anything else; report and stop.
        print(_verdict(False, ["stack guard: " + detail]))
        return 1

    # Plan the diff scope (needed for --list and for the subset run) ------------------
    selected, unmapped, reasons = (set(), [], [])
    if not args.full:
        files = changed_files(args.base)
        selected, unmapped, reasons = map_changed_to_tests(files)
        hdr("diff scope")
        if not files:
            info("no changes vs base -> invariants + sentinel only")
        else:
            info(f"{len(files)} changed path(s) vs {args.base} (merge-base)")
            for r in reasons:
                dim(r)
            if unmapped:
                info(_c(f"{len(unmapped)} code change(s) with no test mapping -> widening to FULL suite", C.YEL))

    if args.list:
        hdr("planned checks (no run)")
        info("0. stack guard: PASS")
        info("1. invariants (ALWAYS): OFF==inc7 AST, +L sign, VQ1 import guard")
        info(f"2. test-count sentinel: collected >= {args.baseline}")
        if args.full or unmapped:
            info("3. diff-scoped pytest: FULL suite")
        elif selected:
            info(f"3. diff-scoped pytest: {len(selected)} module(s):")
            for s in sorted(selected):
                dim(s)
        else:
            info("3. diff-scoped pytest: (no code changes mapped) -- none")
        return 0

    # 1. INVARIANTS (always) ----------------------------------------------------------
    hdr("1. load-bearing invariants (always run)")
    for name, iok, idetail in run_invariants():
        if iok:
            print(_c("  GREEN", C.GREEN) + f"  {name}")
            dim(f"       {idetail}")
        else:
            print(_c("  RED  ", C.RED) + f"  {name}")
            for ln in idetail.splitlines():
                print(_c("       " + ln, C.RED))
            reds.append(f"invariant: {name}")

    # 2. TEST-COUNT SENTINEL ----------------------------------------------------------
    hdr("2. test-count sentinel")
    sok, sdetail, _count = check_test_count(args.baseline)
    if sok:
        print(_c("  GREEN", C.GREEN) + f"  {sdetail}")
    else:
        print(_c("  RED  ", C.RED) + f"  {sdetail}")
        reds.append("test-count sentinel: " + sdetail.splitlines()[0])

    # 3. DIFF-SCOPED PYTEST -----------------------------------------------------------
    hdr("3. diff-scoped pytest")
    run_full = args.full or bool(unmapped)
    if run_full:
        info("running FULL suite" + ("" if args.full else " (an unmapped code change forced it)"))
        rc, text = _run_pytest(["tests"], extra=["-q", "--no-header"])
        line = _last_summary(text)
        if rc == 0:
            print(_c("  GREEN", C.GREEN) + f"  full suite: {line}")
        else:
            print(_c("  RED  ", C.RED) + f"  full suite: {line}")
            for ln in _fail_excerpt(text).splitlines():
                print(_c("       " + ln, C.RED))
            reds.append("full-suite pytest: " + line)
    elif selected:
        nodes = sorted(selected)
        info(f"running {len(nodes)} diff-mapped module(s)")
        rc, text = _run_pytest(nodes, extra=["-q", "--no-header"])
        line = _last_summary(text)
        if rc == 0:
            print(_c("  GREEN", C.GREEN) + f"  diff subset: {line}")
        else:
            print(_c("  RED  ", C.RED) + f"  diff subset: {line}")
            for ln in _fail_excerpt(text).splitlines():
                print(_c("       " + ln, C.RED))
            reds.append("diff-subset pytest: " + line)
    else:
        info("no code changes mapped to tests -> nothing to run here (invariants + sentinel covered it)")

    print(_verdict(not reds, reds))
    return 0 if not reds else 1


def _verdict(green: bool, reds: list[str]) -> str:
    if green:
        return "\n" + _c(_c(" GREEN -- safe to report this branch up. ", C.BOLD), C.GREEN)
    body = "\n".join("   - " + r for r in reds)
    return "\n" + _c(_c(f" RED -- {len(reds)} check(s) failed; DO NOT report up: ", C.BOLD), C.RED) + "\n" + _c(body, C.RED)


if __name__ == "__main__":
    raise SystemExit(main())
