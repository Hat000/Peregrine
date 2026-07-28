---
name: feedback-delegate-builds-to-agents
description: Fengyou wants multi-file build/implementation work delegated to subagents/Workflows, not carried inline in the commander thread.
metadata:
  type: feedback
---

Fengyou, 2026-07-12 (after I built the coarse-map + pitch-fence deploy changes INLINE across ego_obs.py / fly_rl.py / two test files / a config): "btw in the future for these builds use agents."

**Why:** the commander thread is for judgment, diagnosis, and orchestration -- not for carrying a multi-file edit->test->debug loop itself. Inline builds burn the commander's context on mechanical work and forgo parallelism + adversarial verification. (Case in point: a self-inflicted `@torch.no_grad()` decorator-theft bug slipped in -- a helper inserted between the decorator and `def policy_step` silently stole it, breaking the real-ckpt forward; a focused build+verify agent would have caught it before I did.)

**How to apply:** for any implementation task beyond a trivial edit -- new features, multi-file changes, test authoring -- spawn agent(s) or a Workflow to do the edit + run the tests + self-verify, and keep the commander thread for the decision, the review of their result, and the relay to Fengyou. Ultracode is ON: default to Workflow/Agent for builds. Conversational answers, single-fact lookups, and load-bearing sign/convention verification the commander must personally own stay inline. Relates to [[feedback-worker-prompt-copy-paste-box]] and the COMMANDER MODEL delegation split.
