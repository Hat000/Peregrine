# Detector training pipeline (Adroit) + weighted-PnP — 2026-05-31

How we train the gate detector on Princeton **Adroit** autonomously, the tooling that makes it
work, the mistakes that cost time, and the perception decisions made tonight. Companion to the
MEMORY.md "DETECTOR TRAINED" entry.

## The autonomous Adroit pipeline (runbook)
Goal: drive the cluster from the laptop with **one human login**, never touching other users' files.

- **Access** = `adroit-connector` (`C:\Users\Fengy\Downloads\Projects\Adroit\adroit-connector`),
  a paramiko SSH tool. `adroit.py serve` authenticates ONCE (key passphrase via getpass +
  Duo push) and holds the transport, listening on `127.0.0.1` behind a per-session token;
  `adroit.py x "<cmd>"` forwards a command to it (no new Duo). One login → many commands.
  Windows OpenSSH can't ControlMaster-mux, so we multiplex here.
- **Ship source** = `cluster/push_dir.py` — tar+gzip the INCLUDE list (`src/racer`,
  `scripts/gen_synthetic_dataset.py`, the sbatches), base64, stream in ≤20 KB chunks via `x`
  (each fits a Windows cmdline), decode+untar on the cluster. No git creds on the shared box.
- **Train** = `sbatch cluster/yolo_train.sbatch` → generates the dataset then `yolo pose train`
  on ONE idle MIG A100 slice (`--gres=gpu:3g.20gb:1`), time-capped. conda env `yolo` (py3.11 +
  torch cu121 + ultralytics + albumentations + opencv-headless), set up by `cluster/yolo_setup.sh`.
- **Watch** = `watch_job.py <jobid> <results.csv>` polls squeue + the per-epoch mAP, notifies on finish.
- **Pull weights** = `pull_file.py <remote> <local> --sha256 <hex>` — snapshot best.pt on the
  cluster, sha256 it, pull in sha-verified 12 MiB chunks, decode+verify locally.
- **Footprint**: everything in `/home/fl3689` + `/scratch/network/fl3689/peregrine`; compute via
  SLURM only; **never touch other users' files** (user directive — be a good citizen).

## adroit-connector improvements made tonight (all uncommitted)
- `run_remote`: drained channel at **4 KB/recv into `out += chunk`** → a 78 MB (→104 MB base64)
  pull crawled at ~300 KB/s and **wedged the single-threaded daemon for minutes** (it buffers the
  whole response before replying). Fixed → **256 KB recv into a `bytearray`**. Applies on next `serve` restart.
- `pull_file.py` (NEW): sha256-verified **chunked** base64 pull. A single ~100 MB framed response
  blows the 300 s socket timeout; chunking bounds each round-trip. Hardened the size/sha parse with
  labeled `echo SIZE=/SHA=` + regex (the daemon merges stderr, so a stray banner shifted positional parsing).
- `watch_job.py` (NEW): SLURM poller → notify-on-finish (the prior inline watcher had a broken mAP field).
- getpass passphrase (no plaintext; asked at runtime — user directive).

## Mistakes / lessons (the transfer saga + the swap hypothesis)
1. **MSYS path-mangling (cost the most):** a bare `/scratch/...` argument passed to a connector CLI
   *through the Bash tool (Git Bash)* gets rewritten to `C:\Program Files\Git\scratch\...` before
   Python sees it. **Run connector CLIs via PowerShell.** (Paths *inside* a quoted `x "...."` string
   are safe — only bare leading-slash args are mangled.)
2. **Daemon wedge:** the slow 4 KB recv + full-buffer-before-reply meant a big pull blocked the
   accept loop; a `serve` restart both frees it and loads the recv fix.
3. **The corner-swap hypothesis was largely WRONG.** Eyeballing one bad L1 frame suggested identity
   swaps; the *distribution* (`eval_detector`) + per-corner pattern (`diagnose_tail`) showed swaps are
   ~1% (high-roll only) and the real tail is one-corner-off from clipping/clutter. **Diagnose with data
   before prescribing a fix.**
4. **Don't read conclusions off ~9 demo frames** — N=200/level distributions changed the whole picture
   (median 2 px vs the 19 px outlier I'd anchored on).

## Detector / data decisions
- **Curriculum**: geometry maxed at every level; only APPEARANCE ramps L1 clean → L2 shading/clutter →
  L3 chaos. Trained MIXED (one level per image, `[1,2,2,3,3]`) NOT staged → no catastrophic forgetting.
- **Augmentation**: de-stacked to one primary effect/img (`albumentations OneOf`) — ultralytics also
  augments on top, so stacking ~4 effects buried the gate (user caught this).
- **Multi-gate** (`render_scene`): real frames show 2–3 gates; single-gate training is a domain gap +
  misses instance separation. Near→far painter's order; `_in_ring` flags a far corner occluded by a
  nearer gate's ring. Also = clutter-hardening for the `tail_3` (green-rectangle-grabs-a-corner) failure.
- **Targeted oversampling** (`--hard`): `high_roll_prob`/`edge_prob` (0.3) bias `sample_gate_pose` toward
  the swap-prone (26–49° roll) + near-edge-clip configs that dominate the tail. Defaults 0.0 keep the
  legacy RNG draw order bit-identical.
- **OFF-FRAME keypoints REJECTED**: ultralytics `Instances.clip()` (RandomPerspective/Mosaic/LetterBox,
  every training image) zeros visibility + clamps any out-of-frame keypoint. Can't train YOLO to predict
  off-frame corners → that reasoning belongs in the PnP solver.

## Weighted PnP (`gate_pose._refine_pose`, default-on)
Replaces "hard-drop a low-confidence corner → P3P" with a **robust confidence-weighted Gauss-Newton**:
each corner weighted by `1/σ_i²` (σ_i = `WEIGHTED_SIGMA_PX`/clip(conf_i)) × an **annealed Tukey biweight**
on its residual (redescending → a confident-but-wrong corner → ZERO weight, fully rejected; convex Huber
only down-weighted and failed the gross-outlier test). Init from IPPE/P3P (keeps the planar-ambiguity
disambiguation), `cv2.projectPoints` Jacobian, `_refine_ok` reverts to IPPE on divergence. The per-corner
weights also give an **analytic pose covariance** (Fisher info, `[t,rvec]` layout) for the KF — confidence-
aware, replaces the slow Monte-Carlo path. Constants `WEIGHTED_SIGMA_PX=1.5`, `TUKEY_C_HI/LO_PX=40/8`,
`CONF_FLOOR=0.1` — tunable at sim contact. **Pairs with the deferred adapter-relax** (lower the detector's
`kpt_conf_thresh=0.5` so marginal corners reach the now-robust solver; cutoff needs the real conf distribution).

## Status / NEXT (tomorrow, with the user)
- v1 `models/gate_yolo11s_curriculum.pt` (mAP50-95 0.991). v2 `curriculum_11s_v2` done (~0.986, harder val).
  v3 `curriculum_11s_v3` (`--hard`) running overnight.
- **Fixed-set eval v1-vs-v2-vs-v3** (`eval_detector` corner-err = data/detector lever; `diagnose_tail`;
  pose-err now reflects weighted PnP) → pick the tail-winner → then `racer/navigator.py` integration.
- 195 tests green; all Anduril + adroit-connector changes uncommitted.
