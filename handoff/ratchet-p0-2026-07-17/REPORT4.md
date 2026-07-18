# Ratchet-P0.3 settled-GO policy reproducibility — N=5: gates {2,2,1,2,2}; gate-0 pass time reproduces the champion within 60 ms; the wall is the gate-2 descend; champion's 5 was a tail event

2026-07-18 (ShadowPC, policy flights on `claude/ego-deploy-2026-07-09` head — command path
identical to the champion commit). N=5 champion-recipe policy flights, settled-GO protocol
(operator counted ≥3 s between race reset and GO on every run; all five GOs were fresh —
`pos_off=0.00`, no late-joins; the late-join guard correctly REFUSED one leftover race).

## Recipe recovery — meta.json was NOT the full story (lesson pinned)

The relay's recipe pins were correct but incomplete, and `meta.json` doesn't record the
detector or assist level. The committed **panel launch log**
(`tools/pilot_panel_logs/p1784060588_67.log` @ 28404fa, timestamped 9 s before the champion
recorder's t0) is the authoritative record; it corrected THREE things:

1. **Detector = TRT M-engine**, not red_glow: `--seeker-detector yolo --seeker-weights
   C:/Users/Shadow/Peregrine/models/vq2_partial_m_2026-07-06_fp16_384x640.engine` (fp16
   384x640, cuda:0, ~4 s prewarm). The ego path hard-errors without explicit weights.
2. **`--ego-assist-thrust 1.3`** (default is 1.10; champion banner shows floor 1.3 g).
3. **The map the champion flew ≠ the map committed at the same commit.** Its log echoes
   `[[0,0],[-1,1],[0,-1],[0,0],[1,-1],[0,0],[-1,0]]`; the `configs/vq2_coarse_map.json` blob at
   28404fa has rows 3/4 swapped (edited between flight and commit), and today's working map
   differs further. Flights used the log-echo map via a side file (`--ego-coarse-map`), leaving
   the working tree untouched.

Bonus mystery closed: **the champion was itself a LATE-JOIN** (`to_go=-2.75 s` in its log) — the
"2.83 s post-GO silent delay" from REPORT2/3 was the drone resting on the block through an
unattended race start while the detector prewarmed. The settled-GO protocol reproduces that rest
condition with a fresh GO.

Full command (repeated 5×, only the label changed):
`PYTHONPATH=src python rl/fly_rl.py --endpoint udp:127.0.0.1:14550 --ego-ckpt ckpts/vpeffs0_actor.pth --seeker-detector yolo --seeker-weights <M-engine> --rate 40 --ego-rate-scale 1.2 --virtual-flip --ego-slot1 --ego-sector-mode map --ego-coarse-map <champion-log-echo-map> --ego-pitch-clamp 1.0 --ego-yaw-clamp 0.7 --ego-det-hold 0.2 --ego-stale-horizon 0.5 --ego-assist-thrust 1.3 --max-seconds 120 --wait-seconds 1800 --flights 1 --label ratchet_p03_rN`

## Results

| run | session | gates | end | collisions | gate-0 pass | gate-1 pass | note |
|---|---|---|---|---|---|---|---|
| r1 | `023752` | 2 | STALLED | 1 | 2.942 s | 5.452 s | hunted gate 2 too HIGH (pilot visual), glancing contact, no abort |
| r2 | `023849` | 2 | CRASH | 4 | 2.964 s | 5.465 s | died on g2 leg |
| r3 | `024314` | 1 | CRASH | 2 | 2.975 s | — | died on g1 leg |
| r4 | `024533` | 2 | CRASH | 1 | 2.981 s | 5.476 s | died just after g1 pass |
| r5 | `024621` | 2 | CRASH | 2 | 3.000 s | 5.468 s | died at/just after g1 pass |
| champion | `20260714_202317` | 5 | CRASH | 1 | 2.945 s | 5.168 s | the tape source |

(times relative to each run's first policy tick, same convention as the champion analysis)

## Adjudication — P(bank | settled)

- **Gates-passed distribution: {2, 2, 1, 2, 2}.** P(≥1)=5/5, P(≥2)=4/5, **P(≥3)=0/5**. The
  champion's 5-gate run is a tail event, not the settled-launch mode.
- **Zero-contact validity: 0/5 attempts were contact-free** (4 terminal crashes, 1 stall with a
  glancing gate contact). As flown — no arrest available — no attempt would have banked clean.
  The gates-before-first-terminal-event counts above are the raw material the arrestor converts
  into bankable segments.
- **The settled opening REPRODUCES the champion**, quantitatively: gate-0 pass at
  **2.942–3.000 s across all five runs, champion 2.945 s inside that 58 ms envelope**. The
  settled-GO protocol delivers a near-deterministic first leg.
- **Divergence starts on the g0→g1 leg**: the five runs are *self*-consistent to 24 ms at
  gate 1 (5.452–5.476 s) but ~0.30 s SLOWER than the champion's 5.168 s. The champion flew an
  atypically fast g0→g1 leg and kept threading; the settled population flies a slower,
  extremely repeatable one.
- **The wall is gate 2** (the descend leg, map row [0,-1]): 4/5 died there or immediately after
  g1; the survivor overflew HIGH hunting it. Failure mode is consistent (too high / clipping),
  not scattered.

## Implications for the bank strategy

1. **Bank unit that exists today: gate 0 (arguably g0+g1).** Opening reproducibility is
   58 ms-tight; g1 is 4/5 with a 24 ms-tight pass time. An arrestor targeting post-g0 or
   post-g1 arrest converts these into reliably banked segments.
2. **Gate 2 is the next engineering target** — consistent too-high descend failure. Whether
   that's map row, z-bias, or policy is the laptop side's call; not touched here per DO-NOTs.
3. The champion's full-lap line is not the thing to chase; its g0→g1 speed was already
   atypical by tick ~120. Segment-wise banking + per-segment correction is the strategy the
   numbers support.

## Ops notes

- One launch aborted pre-flight (ego path requires explicit `--seeker-weights`; no session).
- One leftover-race refusal handled by the pilot's own guard ("late race not joinable —
  waiting") — the fresh-GO ritual held for all five counted runs.
- Panel restarted for Fengyou after the probe (working-tree map, ego-deploy code).
- Committed per run (panel pattern): `ego_obs.jsonl`, `ego_timing.jsonl`, `meta.json`,
  `video_index.jsonl`.

## MEMORY-DELTA (≤10 lines)

1. P0.3 2026-07-18, N=5 settled-GO champion-recipe policy flights: gates **{2,2,1,2,2}** —
   P(≥2)=4/5, P(≥3)=0/5; champion's 5-gate run is a TAIL event. Gate-0 pass time reproduces
   across runs AND matches the champion within 58 ms; gate-1 self-consistent to 24 ms but
   0.3 s slower than champion; **wall = gate-2 descend (too high, consistent)**.
2. Champion recipe GROUND TRUTH lives in the committed panel launch log, not meta.json:
   TRT M-engine detector + `--ego-assist-thrust 1.3` + a map that differs from the same-commit
   configs blob (rows 3/4). The champion was itself a late-join (to_go −2.75 s) — the "2.83 s
   warmup" was block-rest during detector prewarm; settled-GO reproduces it.
3. Bankable unit today = g0 (g0+g1 at 4/5); arrestor should arrest post-g0/post-g1; zero-contact
   attempts were 0/5 as flown, so arrest-before-contact is what converts these into banked gates.
