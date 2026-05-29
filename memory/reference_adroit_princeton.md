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
- Cannot host the DCL simulator — Linux blocks it per spec sec 5.1.
- Used for: training a YOLO-style gate detector for VQ2's harder course.
- Workflow: collect frames on the Azure Windows VM during sim runs → upload to Adroit `/scratch` → train via Slurm jobs → download model weights → deploy on Azure VM for inference.
- All training is offline / asynchronous — no real-time control loop touches Adroit.

**Access caveats to confirm:**
- Some Princeton accounts lose Adroit access during summer recess if not enrolled in a course or research project that requires it. User should verify with Research Computing before relying on it.
- Slurm queues can be backed up during peak term times; less likely in summer.

Related: [[project-ai-grand-prix]], [[project-hardware-constraint]]
