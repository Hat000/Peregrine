"""Evaluate EVERY save_period checkpoint of a run on the standardized good-fix metrics + Wilson CI.
Deploy-checkpoint selection is by GOOD-FIX across snapshots, never mAP (F-INFRA-1 / brief sec 5).

  python eval_run.py <run_dir_or_weights.pt> [more ...]

Accepts a run dir (globs weights/*.pt), a weights dir, or individual .pt files.
"""
import sys
from pathlib import Path

# Import the PARENT-dir eval_goodfix.py (NOT the work/ copy): only the parent copy is at the depth
# where its _ROOT = parents[2] resolves to the repo root (so src/ + scripts/ land on sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# precision-loss checkpoints pickle cluster/vq2_precision_loss.PrecisionKeypointLoss into the model
# criterion -> torch.load needs the module importable. Harmless for normal checkpoints.
sys.path.insert(0, "C:/Users/Shadow/Peregrine/cluster")
import eval_goodfix as E  # noqa: E402  (does the src/scripts sys.path setup + heavy imports)
from wilson import wilson  # noqa: E402


def collect(args):
    out = []
    for a in args:
        p = Path(a)
        if p.is_file() and p.suffix == ".pt":
            out.append(p)
        elif (p / "weights").is_dir():
            out += sorted((p / "weights").glob("*.pt"))
        elif p.is_dir():
            out += sorted(p.glob("*.pt"))
    # de-dup, stable
    seen, uniq = set(), []
    for p in out:
        if p.resolve() not in seen:
            seen.add(p.resolve()); uniq.append(p)
    return uniq


def main():
    weights = collect(sys.argv[1:])
    if not weights:
        print("no checkpoints found"); return 1
    print(f"{'checkpoint':34}{'<0.5m (oracle)':>22}{'<1m':>8}{'<2m':>8}{'course/204':>22}")
    for w in weights:
        e = E.task2_goodfix(str(w)); N = len(e)
        n05 = int((e < 0.5).sum()); n1 = int((e < 1.0).sum()); n2 = int((e < 2.0).sum())
        v, M = E.course_validfix(str(w))
        _, lo, hi = wilson(n05, N); _, clo, chi = wilson(v, M)
        tag = f"{w.parent.parent.name}/{w.name}"
        print(f"{tag:34}"
              f"{f'{n05}/{N} [{100*lo:.0f},{100*hi:.0f}]':>22}"
              f"{f'{100*n1/N:.0f}%':>8}{f'{100*n2/N:.0f}%':>8}"
              f"{f'{v}/{M} [{100*clo:.0f},{100*chi:.0f}]':>22}", flush=True)
    print("\nCHAMPION: 32/40 [65,90] | 95% | 98% | 155/204 [70,81]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
