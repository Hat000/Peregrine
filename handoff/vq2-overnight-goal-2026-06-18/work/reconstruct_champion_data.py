"""Reconstruct the Round-1 champion's EXACT training manifest.

The champion (runs/pose/runs/vq2_pose_8kp, args.yaml data=.../sets/data.yaml) trained on train.txt/val.txt,
backed up as train.txt.prepoison.bak (3587) / val.txt.prepoison.bak (399). FOUR of its sets -- lr1, hv1,
hv2, hv3 (720 train + 80 val imgs) -- were LATER moved to .../_quarantine_nonphotoreal/ and excluded from
every "clean" reproduction, which is why data_base2000 (allhue+vq1red only, 1800) never reproduced 80%.

This rewrites the quarantined sets' paths into _quarantine_nonphotoreal/ and verifies EVERY image+label
exists, producing champion-faithful manifests for the corrected sanity gate.
"""
import os

SETS = r"C:/Users/Shadow/Peregrine-vq2data/handoff/vq2-blender-render-2026-06-15/sets"
RENDER = r"C:/Users/Shadow/Peregrine-vq2data/handoff/vq2-blender-render-2026-06-15"
QUAR = {"lr1", "hv1", "hv2", "hv3"}
OUT = r"C:/Users/Shadow/Peregrine/handoff/vq2-overnight-goal-2026-06-18/work"


def label_for(img):
    return img.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"


def rewrite(line):
    p = line.strip().replace("\\", "/")
    if not p:
        return None
    setn = p.split("/sets/")[-1].split("/")[0] if "/sets/" in p else None
    if setn in QUAR:
        # .../vq2-blender-render-2026-06-15/sets/lr1/...  ->  .../_quarantine_nonphotoreal/lr1/...
        p = p.replace("/sets/" + setn + "/", "/_quarantine_nonphotoreal/" + setn + "/")
    return p


def build(src, dst):
    n = miss_img = miss_lbl = 0
    out_lines = []
    for line in open(os.path.join(SETS, src), encoding="utf-8"):
        p = rewrite(line)
        if p is None:
            continue
        n += 1
        if not os.path.exists(p):
            miss_img += 1; continue
        if not os.path.exists(label_for(p)):
            miss_lbl += 1; continue
        out_lines.append(p)
    with open(os.path.join(OUT, dst), "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"{src} -> {dst}: {n} referenced, {len(out_lines)} written, "
          f"missing_img={miss_img} missing_lbl={miss_lbl}")
    return len(out_lines)


nt = build("train.txt.prepoison.bak", "champion_train.txt")
nv = build("val.txt.prepoison.bak", "champion_val.txt")
print(f"\nCHAMPION RECONSTRUCTION: train={nt} val={nv} (expected 3587 / 399)")
