---
name: feedback-worker-prompt-copy-paste-box
description: Every worker prompt must be emitted inside ONE fenced code block for frictionless copy-paste to worker machines.
metadata:
  type: feedback
  date: 2026-06-13
  triggered_by: Commander emitted worker prompts with inline markdown styling that breaks when copy-pasted raw; Fengyou directive to use a single fenced block.
---

## The rule

**Emit every worker prompt inside ONE fenced code block (``` or similar), no inline markdown styling that needs rendering.**

## Why

Fengyou copy-pastes prompts straight to worker machines (ShadowPC, Adroit terminal, ultracode launcher). A single fenced block is one click to copy, zero reformatting needed. Inline markdown (bold, italics, headers inside the prompt body) renders incorrectly as literal `**` `##` etc. when pasted into a terminal or plain-text launcher.

## Required prompt contents (all in the one box)

Every worker prompt must include:
1. `SESSION:` — session name/ID.
2. `MODEL:` — model name with version (e.g. `claude-opus-4.8`, `claude-sonnet-4.6`).
3. `EFFORT:` — effort level (low/medium/high/max/ultracode).
4. Cold-start context — enough for the worker to act without this conversation (file paths, key facts, the question to answer).
5. Task list — explicit numbered tasks.
6. Constraints — what NOT to do (no src edits, no SLURM, no live sim, etc.).
7. MEMORY-DELTA requirement — worker must end with ≤10-line MEMORY-DELTA block.
8. Escape hatch — what to do if the worker hits a blocker (e.g. "if this fails after 2 retries, STOP and report").
9. Report path — where to write the result file (under `handoff/` or similar).

## Cross-references

- [[index-strategy-meta]] — pointer registered here.
- [[feedback-commander-orchestrate-not-execute]] — WHY the commander is emitting prompts instead of running the work.
