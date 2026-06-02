---
name: reference-adroit-princeton
description: "Princeton's Adroit research cluster — Linux + NVIDIA GPU, used by this project for VQ2 ML training (gate detector)."
metadata: 
  node_type: memory
  type: reference
  originSessionId: 381822b6-b523-4990-a2cf-3739f8593be8
---

**Adroit** is Princeton's introductory research-computing cluster. User has access via Princeton .edu account.

- Linux (Rocky/RHEL family), Slurm job scheduler
- NVIDIA GPU nodes (commonly V100 and A100 partitions; exact availability varies by job queue)
- Free for Princeton affiliates (degree-granting students, faculty, staff)
- SSH access: `ssh <netid>@adroit.princeton.edu`; may require Princeton VPN
- Storage: `/home` (small quota) + `/scratch` (large, transient)
- Princeton OIT docs: search "Princeton Research Computing Adroit"

**Role in this project (VQ2 ML training only):**
- Cannot host the sim — Linux blocks it (sim is Windows-only). Sim host = ShadowPC (Azure + TensorDock both retired).
- Used for: training the YOLO-pose gate detector (detector v2 shipped 2026-06-02).
- Workflow: collect frames on the sim host (ShadowPC) during sim runs → upload to Adroit `/scratch/network/fl3689` → train via Slurm → pull weights (via the adroit-connector) → deploy on the sim host. Detail: [[project-detector-training-pipeline]].
- All training is offline / asynchronous — no real-time control loop touches Adroit.

**Access caveats to confirm:**
- Some Princeton accounts lose Adroit access during summer recess if not enrolled in a course or research project that requires it. User should verify with Research Computing before relying on it.
- Slurm queues can be backed up during peak term times; less likely in summer.

Related: [[project-ai-grand-prix]], [[project-hardware-constraint]]
