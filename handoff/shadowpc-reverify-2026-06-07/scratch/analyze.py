"""Rung-1 limit-cycle analyzer for the 2026-06-07 re-VERIFY (raw-vz alt-loop fix).

Loads a fly_vq1 extract.json (schema=faithful) and reports, for BOTH the whole run and the
settle window (last 5 s), the true-vz cycle (mean/std/rms/|max|, zero-cross + FFT period) and the
commanded-thrust duty against the clips -- the apples-to-apples comparison vs the old live cycle
(settle std 0.23, |max| 0.61, period ~0.16 s, thrust 0.05<->0.40 ~67% hi) and the twin prediction
(raw vz: vz_rms <0.05 at <=10 ms transport, mild bob ~0.2 at 30 ms).

Usage: .venv\\Scripts\\python handoff/shadowpc-reverify-2026-06-07/scratch/analyze.py <run_dir_or_extract.json>
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np


def _cycle(t, v):
    v = np.asarray(v, float); t = np.asarray(t, float)
    v0 = v - np.mean(v)
    sv = np.sign(v0)
    cr = np.where(sv[1:] * sv[:-1] < 0)[0]
    if len(cr) > 1:
        xs = [t[i] + (v0[i] / (v0[i] - v0[i + 1])) * (t[i + 1] - t[i]) for i in cr]
        per_zc = 2.0 * float(np.mean(np.diff(xs)))
    else:
        per_zc = float("nan")
    dt = float(np.median(np.diff(t)))
    tu = np.arange(t[0], t[-1], dt); vu = np.interp(tu, t, v) - np.mean(v)
    fr = np.fft.rfftfreq(len(vu), dt); sp = np.abs(np.fft.rfft(vu))
    fpk = fr[1:][np.argmax(sp[1:])] if len(fr) > 2 else float("nan")
    per_fft = (1.0 / fpk) if fpk and fpk == fpk and fpk > 0 else float("nan")
    return dict(mean=float(np.mean(v)), std=float(np.std(v)), rms=float(np.sqrt(np.mean(v**2))),
                absmax=float(np.max(np.abs(v))), per_zc=per_zc, per_fft=per_fft,
                vmin=float(np.min(v)), vmax=float(np.max(v)), n=len(v), dur=float(t[-1] - t[0]),
                rate=1.0 / dt if dt else float("nan"))


def main() -> int:
    p = Path(sys.argv[1])
    ext = p / "extract.json" if p.is_dir() else p
    j = json.loads(ext.read_text())
    print(f"=== {ext} ===")
    print("top-level keys:", list(j.keys()))
    for k, vv in j.items():
        if isinstance(vv, dict) and "series" in vv:
            s = vv["series"]
            print(f"  {k}: series {len(s)} x {len(s[0]) if s else 0}  labels={vv.get('labels', vv.get('cols','?'))}")
    if "summary" in j:
        print("summary:", json.dumps(j["summary"]))

    lt, lalt, lvz = (np.array(x) for x in j["true_lpn"]["series"])
    ct = np.array(j["commanded"]["series"][0]); cthr = np.array(j["commanded"]["series"][3])

    def report(tag, tm, vm, tcm):
        c = _cycle(tm, vm)
        thr = cthr[tcm]
        print(f"\n[{tag}]  n={c['n']} dur={c['dur']:.1f}s @~{c['rate']:.0f}Hz")
        print(f"  vz: mean {c['mean']:+.3f}  std {c['std']:.3f}  rms {c['rms']:.3f}  |max| {c['absmax']:.3f}  "
              f"range [{c['vmin']:+.3f},{c['vmax']:+.3f}]")
        print(f"  period: zc {c['per_zc']*1000:.0f} ms  fft {c['per_fft']*1000:.0f} ms  "
              f"({(1.0/c['per_fft']) if c['per_fft']==c['per_fft'] else float('nan'):.1f} Hz)")
        print(f"  thrust: mean {np.mean(thr):.3f}  duty hi(>=0.45) {np.mean(thr>=0.45-1e-3)*100:.0f}%  "
              f"lo(<=0.05) {np.mean(thr<=0.05+1e-3)*100:.0f}%  range [{thr.min():.3f},{thr.max():.3f}]")

    report("WHOLE RUN", lt, lvz, np.ones_like(ct, bool))
    m = lt >= lt[-1] - 5.0; mc = ct >= ct[-1] - 5.0
    report("SETTLE last 5s", lt[m], lvz[m], mc)
    # also a tighter settle excluding the first 6s (takeoff+initial) to match the old 'last 5-7s'
    m2 = lt >= lt[0] + 6.0; mc2 = ct >= ct[0] + 6.0
    if np.sum(m2) > 50:
        report("POST-6s (steady)", lt[m2], lvz[m2], mc2)

    print(f"\n  alt: mean {np.mean(lalt[m]):.3f}  std {np.std(lalt[m]):.4f}  "
          f"slope {np.polyfit(lt[m]-lt[m][0], lalt[m], 1)[0]:+.4f} m/s (settle)")
    print(f"  alt range whole: [{lalt.min():.3f},{lalt.max():.3f}]  (max climb-rate during takeoff: "
          f"vz_min {lvz.min():+.3f} / vz_max {lvz.max():+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
