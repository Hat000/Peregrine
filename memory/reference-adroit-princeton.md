---
name: reference-adroit-princeton
description: "Adroit connector (adroit.py) protocol — code-grounded daemon facts, wedge mechanisms, and the safe command discipline. Supersedes three wrong 2026-07-18 theories (stdin-wedge / NFS-hang / background-sandbox)."
metadata: 
  node_type: memory
  type: reference
  originSessionId: cd451162-1e4a-450b-987c-7c21ee7a5592
---

# Adroit connector protocol (code-grounded 2026-07-18, from adroit.py source)

**Architecture facts (read the source, don't theorize):** `serve` holds ONE paramiko transport; the serve loop is **STRICTLY SERIAL** (accept → recv → `run_remote` → reply → print `ran:`). `run_remote` is **UNBOUNDED** — one hung remote command blocks every later call forever. **`ran:` prints AFTER completion** (a silent window = a command still executing, not "never arrived"). On a dropped transport the NEXT command triggers reconnect which **BLOCKS ON A DUO PUSH** — if unapproved, the loop wedges there (this, not NFS/stdin, caused the 2026-07-18 outage). Windows `SO_REUSEADDR` lets an old and new serve **double-bind port 8765** — clients then hit either daemon at random. Client `x` socket timeout = 300 s.

**Safe discipline:**
1. `x "__ping__"` = loop-health probe (answered inside the daemon, NO SSH touch) — use before any diagnosis.
2. `command_history.log` (next to adroit.py, LOCAL) = ground truth of every COMPLETED command with output — read it before resending anything (double-execution hazard).
3. ONE short command per call; wrap every /scratch-touching command in `timeout N`.
4. PS quoting: prefer double-quoted PS arg with NO `$`/backticks (bash single-quotes inside are fine); PS expands `$?`/`$vars` in double quotes and mangles them — a mangled command usually ERRORS fast (see history log), it does not hang.
5. Client timeouts ≠ daemon state: a timed-out client's command may STILL be executing/queued remotely.
6. Exactly ONE serve window at a time; kill old ones before restarting (double-bind).
7. Wedge triage order: `__ping__` → history-log tail → ask Fengyou what the serve window shows (`ran:` lines + any "approve Duo" prompt) → only then restart.

**AUP (standing, bake into every SLURM worker prompt):** SLURM only (never compute on login node); compute nodes have NO internet (pre-stage deps); output → /scratch; accurate --mem; zero-GPU-util jobs killed at 2 h; run `checkquota` routinely.
