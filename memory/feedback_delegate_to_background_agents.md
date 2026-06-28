---
name: feedback-delegate-to-background-agents
description: Commander delegates implementation to BACKGROUND agents rather than doing it in-context; assigns models deliberately per task
metadata:
  type: feedback
---

Fengyou, 2026-06-28: "Use MORE background agents instead of doing things yourself. Less commander doing work, more commander commanding agents — this way you have more control over what AI model does what." (Said while the commander was doing too much implementation/debugging in its own main-loop context.)

**Why:** (1) **Parallel throughput** — background agents run concurrently while the commander orchestrates, instead of serializing work through one context. (2) **Deliberate model assignment** — each delegated task gets the right tier (opus for correctness/judgment-dense math; sonnet for mechanical/banking/flight-ops), giving explicit cost+capability control per unit of work. (3) **Lean commander context** — reserves the main loop for synthesis, decisions, triage, and memory, not implementation.

**How to apply:** DEFAULT to spawning a background `Agent` for any build / research / debug / multi-step task — do NOT do it inline. Reserve the commander's own tool-use for ORCHESTRATION: launching agents, triaging + committing their staged output, banking memory, and synthesis. State the model choice explicitly per agent. Run independent workstreams as PARALLEL background agents (e.g. the two offline-exact pillars distill-pipeline + cpc-trajopt launched together). Only do trivial/fast things (a single grep, a one-line status check, a commit) inline. Budget-aware, NOT reckless: more agents is fine, but match the model tier to the task — opus where correctness binds, sonnet where it doesn't. Links: [[feedback-worker-prompt-copy-paste-box]] · [[index-strategy-meta]].
