---
name: feedback-checkpoint-transfer
description: Workflow rule — small RL policy checkpoints go in git so ShadowPC can pull instead of manual file transfer
metadata:
  type: feedback
---

For small RL policy checkpoints (≤a few MB), commit to the repo so all machines get them via `git pull`. Do NOT use manual base64 + file copy workflows for files that are small enough for git.

**Why:** Manual transfers (base64 → decode → copy to ShadowPC) are error-prone and slow. The first S1.2 actor.pth (155 KB) required a multi-step laptop→ShadowPC manual copy. Small checkpoints belong in git.

**How to apply:** Before training, either (a) unignore the specific checkpoint path in .gitignore (e.g., `!rl/checkpoints/stage1_inc1_actor.pth`) or (b) add a dedicated `rl/checkpoints/` directory that is NOT covered by the `*.pt` gitignore rule. After training, pull the checkpoint from Adroit via the connector, commit it, push — ShadowPC just runs `git pull`. Reserve the base64/manual-copy workflow for large files (YOLO weights, data/runs recordings) that legitimately belong gitignored.
