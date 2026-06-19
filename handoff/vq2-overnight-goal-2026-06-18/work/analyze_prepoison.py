"""Diagnose the champion's real (prepoison) training manifest: which sets, how many images exist."""
import os
from collections import Counter

ROOT = r"C:/Users/Shadow/Peregrine-vq2data/handoff/vq2-blender-render-2026-06-15/sets"


def setname(p):
    p = p.replace("\\", "/")
    if "/sets/" in p:
        return p.split("/sets/")[-1].split("/")[0]
    return "?"


def analyze(manifest):
    by_set = Counter(); exist_by_set = Counter(); miss_by_set = Counter()
    n = 0
    for ln in open(os.path.join(ROOT, manifest), encoding="utf-8"):
        p = ln.strip()
        if not p:
            continue
        n += 1
        s = setname(p)
        by_set[s] += 1
        if os.path.exists(p.replace("\\", "/")):
            exist_by_set[s] += 1
        else:
            miss_by_set[s] += 1
    print(f"\n=== {manifest}: {n} images ===")
    print(f"{'set':10}{'total':>8}{'exist':>8}{'missing':>9}")
    for s in sorted(by_set):
        print(f"{s:10}{by_set[s]:>8}{exist_by_set[s]:>8}{miss_by_set[s]:>9}")
    print(f"TOTAL exist={sum(exist_by_set.values())} missing={sum(miss_by_set.values())}")


for m in ["train.txt.prepoison.bak", "val.txt.prepoison.bak"]:
    analyze(m)
