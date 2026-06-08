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

**Ops learned driving the DiffAero RL bake-off (2026-06-08) — hard to re-derive:**
- Login node `adroit5`; **conda via `module load anaconda3/2024.10`** (versions 2018.12–2025.12; NO default
  → must specify; in a non-login shell first `source /etc/profile.d/modules.sh`). Envs: `yolo` (detector),
  `diffaero` (RL substrate, py3.11 + torch cu121, ~6.4 GB in `~/.conda/envs`). cudatoolkit modules 11.8–13.2.
- **`gpu` partition has A100 (4/node: adroit-h11g1..3, `Gres=gpu:nvidia_a100`) AND V100-PCIE-32GB nodes**;
  `--gres=gpu:1` works. SLURM = `/usr/bin/{sbatch,sinfo,srun,scancel}`.
- **🚩 COMPUTE NODES HAVE NO INTERNET** (login DOES) → run all `pip`/`git` on the LOGIN node (nice it for
  heavy builds), run compute via SLURM. A job needing the network fails with DNS errors.
- **adroit-connector ops:** run it via its OWN venv (`adroit-connector/.venv`, has paramiko). The SSH key is
  **passphrase-encrypted**, so only **`serve`** works (it caches the key after ONE user passphrase entry +
  one Duo, then `x "cmd"` reuses it). `run`/`submit`/`upload` each re-prompt the passphrase via getpass →
  **unusable from a non-interactive shell** → drive everything through `serve` + `x`. **🚩 the daemon
  idle-drops often** (keepalive 15s insufficient; each drop needs the user to approve a reconnect Duo in the
  serve window) → keep `x` commands quick; for long work submit `sbatch` jobs (async, survive daemon drops)
  + poll the output file. **PowerShell→native-exe quoting mangles double-quoted args that contain spaces**
  (e.g. `-o "%N %G"`, `grep -o "a b"`) → ship complex/multiline commands as a **base64'd script file**
  (`echo <b64> | base64 -d > f`) and run that, instead of inlining.

Related: [[project-ai-grand-prix]], [[project-hardware-constraint]], [[project-phase2-rl-vision-decisions]]
