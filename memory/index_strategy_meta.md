# Strategy & Meta Sub-Index
Mid-level index for competition rules, VQ1/VQ2 mechanics, Track-A strategy, user directives (detail), budget policy, and project feedback. Deep detail in topic files below.

## VQ1/VQ2 mechanics
- **VQ1 = PASS/FAIL** (spec §8, 8-min cap). **VQ2 = fastest-valid.**
- Submission = Python stack, run unattended, **FULLY AUTONOMOUS** (§7: human interaction = DQ). Unlimited attempts, May→~mid/late July.
- **Adroit summer access CONFIRMED.**
- 🚩 **GATE CONTACT = INVALID RUN** (zero-contact is THE validity rule).

## Track-A strategy + MVP
- **Track A = VQ1 floor + VQ2 baseline.** MVP = min-snap line + slow VALID finish > fast invalid.
- **Overfit GEOMETRY, randomize APPEARANCE.** NEVER fine-tune detector on clean VQ1 frames.
- **Planning + speed = VQ2 differentiator; vision = bounded threshold. RL = COMMITTED VQ2 path.**

## User directives (full detail)
- speed > gate-in-view; vision must be excellent; offline processing LEGAL; SLURM only on Adroit.
- Re-derive every verdict from data; tune offline, fly to verify; bounded actuation, user owns GUI/risk.
- Every new-session prompt carries MODEL (with VERSION) + EFFORT.
- Concurrent laptop sessions must use separate git worktrees or stagger suite runs.

## Budget policy (rev 5, 2026-06-13)
- WEEKLY POOL sole watch; push limits to MAX.
- **opus-4.8 ≈ 0.5 × fable cost — opus IS the affordable design tier.** PUSH models UP; do NOT throttle.
- **opus-4.8 = ALL correctness/judgment-dense work** (commander, env/reward, plant, vision math, S2 arch, selection-metric/eval code) at high/max. **Ultracode on opus** for hardest fan-out/verify.
- **sonnet-4.6 = low-stakes mechanizable only** (banking, flight ops, git, routine reruns). **haiku-4.5:** trivial ops.
- 🚩 **FABLE REVOKED 2026-06-12 (US-gov, PENDING DISPUTE).** opus-4.8 is top tier until resolved.

## Organizer contact status
- 🚩 **ORGANIZER NON-RESPONSE PIVOT:** 3 emails UNANSWERED — build the robust superset, do NOT gate engineering on organizer answers. Keep ONE nudge ~weekly.
- Q①+Q⑤+Q-A+Q-B+Q-C+Q-D still useful but NOT blocking. Q① (VQ2 streams pose?); Q⑤ (eval-HW GPU?); Q-A (race-start-vs-launch ordering); Q-B (per-msg-type telemetry reliability); Q-C (submission entrypoint contract); Q-D (§7 sim-control prohibition).

## Banking protocol (rev 4 — thin-index structure)
- Workers end with **MEMORY-DELTA** (≤10 lines); commander triages; sonnet banking agent banks IN BATCHES.
- **Route detail to the topic file + its domain sub-index; touch MEMORY.md ONLY for NOW / footgun / directive changes.**
- Banking agents `git add` ONLY their own specific paths (never `git add -A` / `git add .`).
- Mirror `memory/` ↔ `~/.claude`.

## Prompt-emission directive
- When amending any worker prompt, ALWAYS re-emit COMPLETE prompt — never a splice/delta.
- Every prompt carries SESSION + MODEL(version) + EFFORT + canary + MEMORY-DELTA requirement + escape hatch + report path.
- 🚩 **EFFORT field is AUTHORITATIVE + the options are `high | extra | max | ultracode` (2026-06-14, Fengyou).** ULTRACODE IS ONE OF THE EFFORT LEVELS, not a separate session type — NEVER say "ultracode session" in prose while the header says high/extra/max (that conflict confused Fengyou on the inc8 harness-fix prompt). Write `ultracode` IN THE EFFORT FIELD ITSELF, and ONLY when the work genuinely needs multi-agent fan-out/adversarial-verify (broad audit, design portfolio like A×5/B×3/C×1, big migration). Small well-scoped fixes (e.g. a 4-item harness fix, one careful edit + trivia) = `high`/`extra`/`max`, NOT ultracode. The structural model "Fengyou pastes each pathway into an ultracode session" applies to BIG pathways; do not auto-ultracode every prompt.

## Canary protocol (rev 2, 2026-06-14)
- 🚩 OVERALL COMMANDER persistent session addresses **Fengyou** by name in EVERY message (missing name = context degradation → rotate). **Sub-commanders + workers do NOT need the canary** (Fengyou, 2026-06-14).

## Git/env
- `main` CANONICAL. 692/692 tests green (P4-C05 f50b9b4 adds 5 new tests). `.venv` Python 3.13.
- `*.pt` + `data/runs` gitignored. Memory mirrored `memory/` ↔ `~/.claude`; ShadowPC via `handoff/`.
- 🚩 **BANKING CONCURRENCY:** banking agents must `git add` only their OWN specific paths (never `git add -A`/`git add .`).
- 🚩 **OPS LESSON (pytest-from-root):** always run `pytest` from the REPO ROOT — `test_navigator` loads a saved track-map JSON via root-relative path and FileNotFounds from `rl/`. The earlier P4-C05 "hang" was BENIGN: a 3rd backgrounded full-suite run hit its 15-min timeout at ~90% under load and was killed without a summary line; the polling loop then spun waiting for a summary that never came — NOT a code/test hang.
- 🚩 Guards are DATA-DEPENDENT (skip on clean clone/CI/Adroit).
- 🚩 **BRANCH-COLLISION FOOTGUN (2026-06-14, observed):** a laptop WORKER session running in the SHARED main repo dir (`C:\Users\Fengy\Downloads\Projects\Anduril`) checked out its feature branch THERE → the commander's next memory commit landed on the WORKER's branch, not main (SSOT contamination). FIX applied: `git branch -f main <commit>` (main not checked out elsewhere → safe) + `git checkout main` (clean tree) + push; worker branch left intact. **PREVENTION: (a) overnight/concurrent LAPTOP worker sessions MUST run in their OWN worktree/clone, NOT the main repo dir (Adroit/remote workers are fine); (b) the COMMANDER must `git branch --show-current` == main BEFORE every memory commit** (the working dir can be silently switched by a co-located session).
- 🚩 **WORKTREES (convention, 2026-06-14) — §worktrees:** the harness auto-provides ONE isolated worktree per spawned session (sub-commander, worker, AND each Agent-tool subagent) → non-interference is already satisfied. **Workers MAY make their own worktrees freely** (rev 2, 2026-06-14 — Fengyou's call: isolation > clutter; enforcing no-add is needless friction; per-worker worktrees prevent conflicts). The overall commander OWNS consolidation — merge every worker branch into main at the gate, then prune MERGED branches/worktrees. CODE-workers report their branch name up (auto-named claude/<slug>); commander keeps the branch↔role ledger + merges + prunes branches on merge. ANALYSIS-workers deliver report→handoff/ + MEMORY-DELTA (no branch merge). The `.claude/worktrees/*` pool is HARNESS-managed and RECYCLES — never manually prune it; only the commander removes confirmed-DONE manual `../Anduril-wt-*` worktrees + runs `git worktree prune`. Supersedes the older "concurrent sessions must use separate worktrees" framing (still true, but the harness now does it automatically — don't double it).
- 🚩 **fix_surrogate MERGED to main (ffb2c74, 2026-06-14):** code (rl/fix_surrogate.py + tests) + calibrated checkpoints (handoff/fix-surrogate-2026-06-14/models/) only; branch memory excluded (already banked). 15 tests green; full-suite green-by-construction (base..main divergence = memory commits only); from-root full-count reconfirm pending.
- 🚩 **Branch cleanup (2026-06-14):** 2 stale local-only branches+worktrees pruned (claude/inspiring-tereshkova-9bd0f7, claude/magical-sutherland-546d7e; 0 commits ahead of main). 3 real-artifact branches KEPT on origin (claude/blissful-kalam-75f52b L3, worktree-agent-a88bc717008e79cd9 δ_map, claude/charming-jemison-b111c5 margin-env) — handoff artifacts only, findings banked, NOT merged.

## Topic file pointers
- [[project-master-plan]] — architecture, Track-A strategy, risks R1–R11, build sequence (LIVING SSOT).
- [[reference-competition-materials]] — spec facts, rules/FAQ, VQ1 mechanics, open unknowns.
- [[project-ai-grand-prix]] — competition overview.
- [[project-hardware-constraint]] — laptop (dev) · ShadowPC (sim) · Adroit (GPU training).
- [[reference-adroit-princeton]] — Slurm/GPU; summer access confirmed.
- [[reference-prior-art]] — drone-racing projects + libraries + 8-paper ledger.
- [[feedback-walking-skeleton-no-vq1-crutches]] — VQ1 must be full VQ2 stack under-tuned.
- [[project-red-team-pass-2]] — historical triage; kept to avoid re-litigation.
- [[feedback-commander-orchestrate-not-execute]] — commander must NOT run tasks/compute (even "quick" offline sims); hand a worker prompt or use sanctioned orchestration. (2026-06-13)
- [[feedback-worker-prompt-copy-paste-box]] — emit every worker prompt (AND every sub-commander relay) in ONE fenced code block for frictionless copy-paste. (2026-06-13; extended 2026-06-14)
- [[feedback-sim-no-physical-risk]] — sim flights carry no physical risk; commander authorizes non-POC measurement flights directly; C6/POC hold = sequencing gate, not risk. (2026-06-14)
