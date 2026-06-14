---
name: feedback-commander-orchestrate-not-execute
description: Commander must NOT run tasks or compute — even "quick" offline sims. The deliverable is strategic calls + worker prompts, not executed results.
metadata:
  type: feedback
  date: 2026-06-13
  triggered_by: Commander ran d3_margin_closure.py + margin_xcheck.py via Bash directly; Fengyou corrected → killed + re-issued as worker prompt.
---

## The rule

**The commander must NOT run tasks or compute — even a quick offline laptop sim via Bash.** "Getting the load-bearing number myself" is still execution, even if it feels cheap. Hand a worker prompt, or use sanctioned orchestration (ultracode / background banking agents).

## Why

The commander seat is judgment-dense and output-LEAN. The deliverable is:
1. Strategic calls (which path to take, which numbers to trust, what the verdict means).
2. Transportable worker prompts (complete, self-contained, copy-pasteable).

Running the work yourself couples it to your session window and is not the deliverable. It also creates an unreviewed artifact with no MEMORY-DELTA, no escape hatch, no structured output — exactly the things the worker-prompt contract exists to enforce.

## How to apply

- **ONLY hands-on actions allowed for the commander:** READ (file reading, result inspection, git ops Fengyou explicitly requests).
- When a load-bearing number needs computing: emit a complete worker prompt (see [[feedback-worker-prompt-copy-paste-box]] for format).
- Sanctioned orchestration paths: ultracode (hardest fan-out/verify), background banking agents (memory), Fengyou-approved Bash for trivial git ops.
- If the urge to "just run it quickly" arises: that is the footgun. Stop. Emit the prompt.

## Cross-references

- [[index-strategy-meta]] — pointer registered here.
- [[feedback-worker-prompt-copy-paste-box]] — the required format for the worker prompts you emit instead.
