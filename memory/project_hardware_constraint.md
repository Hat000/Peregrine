---
name: project-hardware-constraint
description: "Three-machine architecture: laptop (dev), Azure Windows GPU VM (sim host, decision pending Free Trial vs Student+Upgrade), Adroit Princeton (VQ2 ML training)."
metadata: 
  node_type: memory
  type: project
  originSessionId: 381822b6-b523-4990-a2cf-3739f8593be8
---

**Setup this summer (three-machine plan):**

| Machine | Role | Specs |
|---|---|---|
| HP Envy x360 15-fe1xxx (user's laptop) | Code editor, unit tests, git, replay/viz | Core Ultra 7 155U, 31.4 GB RAM, Intel iGPU only, Win 11 Home, USB4 |
| Azure NV12ads A10 v5 (cloud) | DCL simulator host + autonomy stack runtime | 12 vCPU, 110 GB RAM, **8 GB A10 partition**, Win 11 Pro, ~$1.10/hr |
| Adroit (Princeton) | VQ2 detector training (offline only) | Linux, V100/A100 partitions, free, Slurm |

**Why this combination:**
- Spec requires Windows + ≥8 GB VRAM for the sim; no way around Azure (or equivalent paid Windows GPU host).
- User has Adroit access, which is free and good for ML training — perfect for the VQ2 YOLO detector.
- Adroit cannot host the sim (Linux per spec sec 5.1) but excels at the part the Azure VM is worst at (long-running training).
- Local laptop handles everything that doesn't need a GPU: editing, unit tests, MAVLink/JPEG/frames code, recorded-frame replays.

**Why NV12ads A10 v5 specifically (Azure):**
- Smallest Azure SKU that delivers ≥8 GB VRAM on a modern (Ampere) GPU partition.
- NV6ads A10 v5 has only 4 GB VRAM — below spec.
- NV4as_v4 has 8 GB VRAM but uses older M60-class hardware substantially below the RTX 2060 Super reference; ~$15 savings over the project not worth the spec risk.
- Smallest GPU VM in Azure period is 4 vCPUs (NV4as_v4 or NC4as_T4_v3), so the 3-vCPU Student cap blocks *any* GPU work; upgrade-to-PAYG is unavoidable for GPU use.
- **NOTE (2026-05-29): cost is no longer a constraint (user directive).** NV12ads is the FLOOR, not a ceiling. Size UP to NV18ads (12 GB) or NV36ads (full A10, 24 GB) if it buys sim FPS or headroom for parallel sim instances + on-real-sim optimization (CMA-ES/RL/sweeps). Choose VM by performance, not price.

**Windows-GPU-host decision (RESOLVED 2026-05-29; sim drops 2026-05-30):**
- **✅ PROVISIONED 2026-05-29 — TensorDock RTX 3090 is the sim host.** 1× RTX 3090 (24 GB), 12 vCPU, 32 GiB RAM, 100 GiB disk, **Windows 10 Pro N**, Delaware US, dedicated public IP, RDP on port 3390. Instant, pay-as-you-go, **no GPU-quota ticket** (the Azure friction). Driver 576.80 / CUDA 12.9. Full stack validated there: real python.org **Python 3.13.13** + Git 2.54 (Store-python stub avoided via `py -3.13`), repo on branch `red-team-tier-a`, `pip install -e ".[dev]"` → **159 tests green**. **24 GB VRAM ⇒ the sim-render-vs-YOLO-inference GPU-contention worry is MOOT** (was the reason to size up the Azure A10). This is now the primary sim host; Azure Free-Trial→PAYG stays a durable headless fallback. **GOTCHA (recorded so it doesn't recur): Windows N editions lack Media Foundation, so `import cv2` fails with "DLL load failed" until BOTH the Media Feature Pack (`Add-WindowsCapability -Online -Name Media.MediaFeaturePack*` — needs an ELEVATED PowerShell + a reboot) AND the VC++ 2015-2022 redist (`winget install Microsoft.VCRedist.2015+.x64`) are installed.** The sim may also need the Media Feature Pack present. Security: public-IP RDP — rotate the admin password + restrict source IP if possible. NOT pushed to `main`: the `red-team-tier-a` branch (red-team fixes + first-contact toolkit + runbook) was pushed to origin so the box could pull it; `main` is untouched pending review.
- **Azure for Students ($100) = DEAD END for GPU** — Student tier hard-caps 3 vCPU and locks N-series GPU quota to 0 with NO increase possible. Cannot host the sim. (User-verified.)
- **Azure Free Trial ($200) = the only viable Azure path, ~1-day friction:** GPU quota defaults to 0 → must convert Free Trial to Pay-As-You-Go ($200 stays valid 30 days) → open a support ticket for an N-series quota increase (NVadsA10 v5 / NCasT4_v3 / NVv4) → wait for Microsoft approval (hours–1–2 business days). This is the DURABLE headless/scriptable long-term host (parallel sim instances, VQ2 optimization). FILE THE QUOTA TICKET NOW.
- **Paperspace = OUT:** DigitalOcean deprecated Windows templates for users who joined after 2024-07-01 ⇒ new accounts are Linux-only. (Earlier suggestion retracted.)
- **AWS Educate = OUT:** Starter/lab accounts exclude EC2 GPU instances; a normal AWS account hits the same new-account G-instance quota-ticket wait as Azure. No hyperscaler is "instant" for GPU on a fresh account.
- **INSTANT Windows + NVIDIA GPU, no quota (the ASAP path for first contact) = cloud-PC providers:** AirGPU (hourly ~$0.75/hr+, pay-as-you-go, RDP; add persistent storage), Vagon (NVIDIA, persistent files, engineer-targeted), or Shadow PC (full persistent Windows PC ~$34–55/mo, possible signup queue). Pick a ≥8GB-VRAM NVIDIA tier (gaming RTX 3080/4080 or workstation A4000/A5000 all clear the RTX 2060 Super reference). Run BOTH the sim and the Python client on this one box (localhost MAVLink/UDP → in <50ms budget). Confirm ToS allows non-gaming/compute (AirGPU/Vagon fine).
- **RECOMMENDED PLAY:** AirGPU/Vagon for tomorrow's first contact (instant, no quota) + Azure Free Trial→PAYG→N-series quota ticket in parallel as the durable headless host.

**GPU contention (flagged 2026-05-28 review):** the DCL sim RENDERS on the GPU and the YOLO detector INFERS on the GPU — they share one A10 partition on the same Windows VM (real-time inference must co-locate with the sim; Adroit is Linux/offline-only). With cost not a constraint, **size up to the full A10 (NV36ads, 24 GB)** so neither starves. Also: run the autonomy/ML stack on **Python 3.12** (no CUDA PyTorch wheels for 3.14 yet) on both Azure VM and Adroit. The **Elodin surrogate** (Apache-2.0, Linux) can run on Adroit or laptop-WSL for sim-independent dev/RL before the official sim releases — see [[reference-competition-materials]].

**Critical operational discipline:**
- Always **Stop+Deallocate** the Azure VM via portal when done (not just "shutdown" inside Windows).
- Cost is NOT a constraint (user directive 2026-05-29) — spend when it buys performance. Still **Stop+Deallocate idle VMs** (idle ≠ performance). Any budget alert is for awareness only, not a cap.
- Install **NVIDIA GPU Driver Extension** on the VM after creation — Windows images don't ship with GRID drivers active.
- For Adroit: respect Slurm queue etiquette; Slurm jobs only, no long-running interactive sessions.

**Data flow between machines:**
- Code: laptop → git push → pull on Azure VM and Adroit. Standard.
- Sim frames for training: Azure VM captures during runs → upload to Adroit `/scratch` (rsync or scp).
- Trained model artifacts: Adroit → download to Azure VM (or stage via laptop).
- Run logs: Azure VM → laptop for analysis/replay.

Related: [[project-ai-grand-prix]], [[reference-adroit-princeton]]
