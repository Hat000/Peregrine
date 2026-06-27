# handoff/ archive note (2026-06-27)

During the dormancy-exit cleanup, the **binary raw-evidence** in `handoff/` was removed from the
working tree to slim it from **93 MB → 20 MB**. The cited text reports (`*.md`, `*.py`, `*.json`,
`*.yaml`, `*.txt`) are all KEPT — every `handoff/.../REPORT.md` reference in `memory/` and in source
comments still resolves.

**What was removed from the tree (110 files):** `*.png`, `*.zip`, `*.tfevents`, `*.jsonl` — preview
renders, TensorBoard event files, a debug-obs zip, and raw telemetry/command extracts. None were cited
by path in code or memory.

**Nothing is lost:**
- Full fidelity backup: `Anduril-archive/handoff-full-2026-06-27.tar.gz` (sibling of the repo, 38 MB,
  1331 entries) — the complete pre-cleanup `handoff/` tree.
- The removed files also remain in **git history** (recover any with `git log --all -- <path>` then
  `git show <commit>:<path>`), since history was NOT rewritten.

To restore everything to the tree: `tar xzf ../../Anduril-archive/handoff-full-2026-06-27.tar.gz` from
the repo root.
