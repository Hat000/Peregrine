"""Definitive label-drift check: for every image in train_base2000.txt, compare the CURRENT label to the
pristine sets_base2k_orig label (CRLF-insensitive) and count empties (backgrounds)."""
import os

SETS = r"C:/Users/Shadow/Peregrine-vq2data/handoff/vq2-blender-render-2026-06-15/sets"
ORIG = r"C:/Users/Shadow/Peregrine-vq2data/handoff/vq2-blender-render-2026-06-15/sets_base2k_orig"


def lbl_path(img, root):
    p = img.replace("\\", "/")
    # .../sets/<set>/images/train/x.png -> <root>/<set>/labels/train/x.txt
    after = p.split("/sets/")[-1]
    return os.path.join(root, after.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt")


def norm(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return "\n".join(ln.strip() for ln in f if ln.strip())


cur_empty = orig_empty = differ = missing_cur = missing_orig = same = 0
differ_examples = []
n = 0
for line in open(os.path.join(SETS, "train_base2000.txt"), encoding="utf-8"):
    img = line.strip()
    if not img:
        continue
    n += 1
    cl = lbl_path(img, SETS); ol = lbl_path(img, ORIG)
    cur = norm(cl); orig = norm(ol)
    if cur is None:
        missing_cur += 1
    elif cur == "":
        cur_empty += 1
    if orig is None:
        missing_orig += 1
    elif orig == "":
        orig_empty += 1
    if cur != orig:
        differ += 1
        if len(differ_examples) < 5:
            differ_examples.append((os.path.basename(cl),
                                    "EMPTY" if cur == "" else ("MISSING" if cur is None else f"{len(cur.splitlines())}L"),
                                    "EMPTY" if orig == "" else ("MISSING" if orig is None else f"{len(orig.splitlines())}L")))
    else:
        same += 1

print(f"train_base2000 images: {n}")
print(f"identical (cur==orig): {same}")
print(f"DIFFER: {differ}   (cur_missing={missing_cur} orig_missing={missing_orig})")
print(f"current empty/background labels: {cur_empty}   | orig empty/background: {orig_empty}")
print("differ examples (file, current, orig):")
for e in differ_examples:
    print("  ", e)
