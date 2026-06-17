# DATA-STAGING — MANIFEST (2026-06-17)

**Session:** DATA-STAGING — pull burn artifacts into the local checkout
**Checkout:** `C:\Users\Fengy\Downloads\Projects\Anduril` (main checkout — where the spike/replay/sweeps run)
**Reachability this session:** Adroit ✅ (via the running `serve` daemon, no new Duo — base64 pull = data transfer, no compute). ShadowPC ❌ (no SSH alias, separate machine).
**Binaries NOT git-added** (per directive — only this manifest is committed). `rl/checkpoints/*.pth/.json` left in the working tree for the spike.

---

## LANDED (verified)

| # | Artifact | Source | Local path | Size | sha256 (head) | Verified |
|---|----------|--------|------------|------|---------------|----------|
| 1 | **inc8-best actor** (S2 seed-2) | Adroit `/scratch/network/fl3689/diffaero_repo/outputs/train/s2full/seed2/periodic/actor.pth` | `rl/checkpoints/stage1_inc8_actor.pth` | 161,879 | `3bdf328638f80972` | ✅ sha matched remote; loads `{actor_mean,actor_logstd}`, first layer **(256, 20)** = 20-dim, logstd (1,4) |
| 1 | **inc8-best critic** (S2 seed-2) | Adroit `…/s2full/seed2/periodic/critic.pth` | `rl/checkpoints/stage1_inc8_critic.pth` | 160,042 | `12ce98604656bee2` | ✅ sha matched remote; loads, first layer **(256, 20)** = 20-dim |
| 1 | **inc8-best sidecar** | Adroit `…/s2full/seed2/periodic/actor.json` | `rl/checkpoints/stage1_inc8_actor.json` | 91 | `124c466b4141111a` | ✅ **`obs_dim:20, inc8:true, r5_arm:A`** (the required flags) |
| 3 | **inc7 actor** (case-A baseline) | already local (Jun 12 inc7-staging); == Adroit inc7 | `rl/checkpoints/stage1_inc7_actor.pth` | 158,807 | `edb86d499689a65b` | ✅ loads `{actor_mean,actor_logstd}`, first layer **(256, 17)** = 17-dim case-A |
| 3 | **inc7 sidecar** | already local | `rl/checkpoints/stage1_inc7_actor.json` | 48 | `2dba9d1bb21bd484` | ✅ `{act_max_thrust:3.765, act_max_rate:3.14}` (pre-inc8 sidecar; no obs_dim/inc8 by design — inc7 is 17-dim case-A) |

**Spike-critical status:** the spike needs **#1 + #2**. **#1 is fully staged & verified.** #2 (one flight bundle) is the remaining spike blocker → see STILL MISSING (S2). #1 is the exact checkpoint the warm-start job 3276071 is fine-tuning from RIGHT NOW.

**Naming note:** the inc8 checkpoint is staged under the repo convention `rl/checkpoints/stage1_inc8_actor.pth` (+ matching `.json` sidecar that `load_actor` reads via `Path.with_suffix('.json')`, + `stage1_inc8_critic.pth`). Provenance = S2 seed-2 `/periodic` (jobs 3275300-02). Point the spike at `--ckpt rl/checkpoints/stage1_inc8_actor.pth`.

---

## STILL MISSING (paste-and-go for Fengyou)

Everything below is ShadowPC-resident (the VQ2/photoreal detector + at-speed flight recordings live on ShadowPC; confirmed NOT on Adroit `/scratch` and NOT in git — `*.pt` is gitignored and `0b99157` is unpushed). Placeholders: `<SHADOWPC>` = ShadowPC host/IP (or run the copy locally on ShadowPC); `<SHADOW_REPO>` = the Peregrine repo path on ShadowPC (e.g. `C:\Users\...\Peregrine`).

### S2 — one recorded flight bundle  🚩 SPIKE-CRITICAL (unblocks the vertical spike together with #1)
Run **on ShadowPC** to find a candidate, then copy one bundle into `data/runs/`:
```powershell
# 1. ON ShadowPC — list candidate bundles (at-speed / L3 / gate-0 head-on):
Get-ChildItem "<SHADOW_REPO>\data\runs" -Directory |
  Where-Object { $_.Name -match '_atspd_|_l3atspeed_|head.?on|gate0' } |
  Sort-Object LastWriteTime -Desc | Select-Object Name,LastWriteTime
# 2. copy ONE of them to the laptop checkout (pick the newest at-speed bundle <NAME>):
#    a) if laptop can scp to ShadowPC:
scp -r "<SHADOWPC>:<SHADOW_REPO>/data/runs/<NAME>" "C:/Users/Fengy/Downloads/Projects/Anduril/data/runs/"
#    b) or, ON ShadowPC, push to the laptop / a shared drive, then drop into data/runs/.
```
Verify after it lands (no `scripts/verify_bundle.py` in this repo → use file census):
```bash
ls -la "data/runs/<NAME>" && find "data/runs/<NAME>" -type f | wc -l && du -sh "data/runs/<NAME>"
```

### S4 — VQ2 Round-1 detector best.pt  (champion; ~19.8 MB YOLO11s)
Not in git (data-only `origin/vq2-data`; `0b99157` unpushed) and **not on Adroit** (Adroit only has the older `curriculum_11s_v3` = local `models/gate_yolo11s_curriculum_v3.pt`). It is on ShadowPC (the photoreal YOLO-pose training output).
```powershell
# ON ShadowPC — locate the Round-1 champion (newest best.pt under the yolo runs):
Get-ChildItem "<SHADOW_REPO>" -Recurse -Filter best.pt -ErrorAction SilentlyContinue |
  Sort-Object LastWriteTime -Desc | Select-Object FullName,LastWriteTime,Length
# copy it into models/ with a clear name:
scp "<SHADOWPC>:<path-to-Round1>/best.pt" "C:/Users/Fengy/Downloads/Projects/Anduril/models/gate_yolo11s_vq2_round1.pt"
```
Verify: `.venv/Scripts/python.exe -c "from ultralytics import YOLO; m=YOLO('models/gate_yolo11s_vq2_round1.pt'); print(m.task, m.model.names)"` (expect pose / 8-kpt).

### S5 — rest of the corpus
**Flight recordings (ShadowPC → `data/runs/`):** same mechanism as S2, all matches:
```powershell
# ON ShadowPC — full at-speed/L3/gate-0 corpus census, then bulk-copy:
Get-ChildItem "<SHADOW_REPO>\data\runs" -Directory |
  Where-Object { $_.Name -match '_atspd_|_l3atspeed_|head.?on|gate0' } | Select Name,LastWriteTime
# bulk pull (laptop):
scp -r "<SHADOWPC>:<SHADOW_REPO>/data/runs/*_l3atspeed_*"  "C:/Users/Fengy/Downloads/Projects/Anduril/data/runs/"
scp -r "<SHADOWPC>:<SHADOW_REPO>/data/runs/*_atspd_*"      "C:/Users/Fengy/Downloads/Projects/Anduril/data/runs/"
# gate-0 head-on (N~13k frames) + the 240-frame real-multigate eval set: copy their dirs the same way.
```
**Synthetic VQ2 detector dataset (git-retrievable, NOT spike-critical):** the only git-accessible corpus piece — `origin/vq2-data` holds `DATASET.md` + `handoff/vq2-blender-render-2026-06-15/sets/{allhue,vq1red}` (6,003 files, synthetic Blender renders; NOT the real-multigate eval frames — those are on ShadowPC). Retrieve on demand:
```bash
git checkout origin/vq2-data -- handoff/vq2-blender-render-2026-06-15/sets DATASET.md
# (large; only if the detector retrain/eval needs it locally — not needed for the vertical spike.)
```

---

## NOTES / GOTCHAS
- **Adroit pull mechanism:** `serve` daemon already authenticated (PID 78468); `adroit-connector/pull_file.py` does sha-verified base64 chunk transfer (login-node `cat|base64`, trivial I/O — NOT compute, AUP-safe). 🚩 Git Bash mangles Unix remote paths (`/scratch/...` → `C:/Program Files/Git/scratch/...`) — prefix `MSYS_NO_PATHCONV=1` when invoking `pull_file.py`.
- **inc8 source is static/safe to copy:** `s2full/seed2/periodic` is the finished S2 run; the live warm-start job (3276071) writes to separate `inc8_warmstart_seed*_ws1` dirs — no read/write contention.
- **inc7 already present** — no Adroit pull was needed; verified it loads as 17-dim case-A.

---

## MEMORY-DELTA (text only — do NOT commit memory/)
```
DATA-STAGING (2026-06-17, local main checkout): inc8-best + inc7 staged & verified; rest is ShadowPC-resident.
- LANDED: inc8 S2-seed2 -> rl/checkpoints/stage1_inc8_{actor,critic}.pth + stage1_inc8_actor.json (sidecar
  obs_dim:20 inc8:true r5_arm:A; actor/critic both 20-dim; sha-verified vs Adroit s2full/seed2/periodic).
  This == the warm-start (job 3276071) source checkpoint. inc7 case-A already local (stage1_inc7_actor.pth,
  17-dim, verified) — no pull needed.
- STILL MISSING (all ShadowPC; NOT on Adroit /scratch, NOT in git, *.pt gitignored, 0b99157 unpushed):
  (#2 SPIKE-CRITICAL) one at-speed/L3/gate-0 flight bundle; (#4) VQ2 Round-1 detector best.pt (champion,
  ~19.8MB YOLO11s — Adroit only has older curriculum_11s_v3); (#5) full recording corpus. Paste-and-go
  ShadowPC PowerShell+scp commands in handoff/data-staging-2026-06-17/MANIFEST.md.
- origin/vq2-data = synthetic Blender detector dataset ONLY (allhue/vq1red, 6003 files); NO weights, NO
  real-multigate eval frames (those are on ShadowPC). git-retrievable on demand (command in manifest).
- Adroit pull = pull_file.py via serve daemon (no new Duo); MSYS_NO_PATHCONV=1 to stop Git Bash mangling
  /scratch paths. Binaries NOT git-added (directive); only this manifest committed.
```
