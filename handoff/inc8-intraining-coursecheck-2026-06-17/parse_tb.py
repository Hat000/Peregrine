#!/usr/bin/env python3
"""Decode base64-pulled TB event files, md5-verify, parse scalars (pure python, no TF)."""
import base64, hashlib, struct, sys
from pathlib import Path

HERE = Path(__file__).parent
TB = HERE / "tb"
EXPECT = {
    "ws_seed0": (1208081, "3be86fe44c0e811c7b039ce723da5348"),
    "ws_seed1": (1208081, "a93b11a3a10e745009879df44a1683af"),
    "ws_seed2": (1208081, "de20faf6ca35ddd9b7a29ccf1fb92563"),
    "s2full_seed2_yawprim": (None, "505526ec619af6c24b2208e21aeb2c2d"),
}

def read_varint(b, i):
    shift = 0; res = 0
    while True:
        byte = b[i]; i += 1
        res |= (byte & 0x7F) << shift
        if not (byte & 0x80): break
        shift += 7
    return res, i

def parse_fields(b):
    """Yield (field_no, wire_type, payload) where payload is int (varint/fixed) or bytes (len-delim)."""
    i = 0; n = len(b)
    while i < n:
        key, i = read_varint(b, i)
        fn = key >> 3; wt = key & 7
        if wt == 0:
            val, i = read_varint(b, i); yield fn, wt, val
        elif wt == 1:
            yield fn, wt, b[i:i+8]; i += 8
        elif wt == 2:
            ln, i = read_varint(b, i); yield fn, wt, b[i:i+ln]; i += ln
        elif wt == 5:
            yield fn, wt, b[i:i+4]; i += 4
        else:
            raise ValueError(f"bad wire type {wt}")

def parse_event(b):
    step = None; summary = None
    for fn, wt, val in parse_fields(b):
        if fn == 2 and wt == 0: step = val
        elif fn == 5 and wt == 2: summary = val
    out = []
    if summary is not None:
        for fn, wt, val in parse_fields(summary):
            if fn == 1 and wt == 2:  # Summary.Value
                tag = None; sv = None; tensor_val = None
                for vfn, vwt, vval in parse_fields(val):
                    if vfn == 1 and vwt == 2: tag = vval.decode("utf-8", "replace")
                    elif vfn == 2 and vwt == 5: sv = struct.unpack("<f", vval)[0]
                    elif vfn == 8 and vwt == 2:  # TensorProto fallback
                        for tfn, twt, tval in parse_fields(val if False else vval):
                            if tfn == 4 and twt == 2 and len(tval) >= 4:  # float_val packed
                                tensor_val = struct.unpack("<f", tval[:4])[0]
                v = sv if sv is not None else tensor_val
                if tag is not None and v is not None:
                    out.append((step, tag, v))
    return out

def iter_records(data):
    i = 0; n = len(data)
    while i + 12 <= n:
        ln = struct.unpack("<Q", data[i:i+8])[0]; i += 12  # skip len-crc
        if i + ln + 4 > n: break
        rec = data[i:i+ln]; i += ln + 4  # skip data-crc
        yield rec

def main():
    for name, (esize, emd5) in EXPECT.items():
        b64path = TB / f"{name}.b64"
        raw = base64.b64decode("".join(b64path.read_text().split()))
        md5 = hashlib.md5(raw).hexdigest()
        ok_md5 = (md5 == emd5); ok_sz = (esize is None or len(raw) == esize)
        (TB / f"{name}.tfevents").write_bytes(raw)
        print(f"\n========== {name} ==========")
        print(f"  decoded {len(raw)} bytes  md5={md5}  md5_ok={ok_md5}  size_ok={ok_sz}")
        if not ok_md5:
            print("  !! MD5 MISMATCH — skipping parse"); continue
        scal = {}
        for rec in iter_records(raw):
            try:
                for step, tag, v in parse_event(rec):
                    scal.setdefault(tag, []).append((step, v))
            except Exception:
                pass
        tags = sorted(scal.keys())
        # find target tags
        targ = [t for t in tags if "success_rate" in t.lower() or "n_passed_gates" in t.lower()]
        print(f"  total scalar tags: {len(tags)}")
        print(f"  TARGET tags: {targ}")
        for t in targ:
            series = scal[t]
            steps = [s for s, _ in series]; vals = [v for _, v in series]
            mx = max(vals); mn = min(vals)
            # trajectory: sample ~12 points evenly
            k = len(series)
            idxs = sorted(set([0] + [round(j*(k-1)/11) for j in range(12)] + [k-1])) if k > 1 else [0]
            traj = ", ".join(f"s{steps[j]}={vals[j]:.4f}" for j in idxs)
            argmax = steps[vals.index(mx)]
            print(f"\n  --- {t} ---  ({k} points, step {steps[0]}..{steps[-1]})")
            print(f"      MIN={mn:.5f}  MAX={mx:.5f} (@step {argmax})  LAST={vals[-1]:.5f}")
            print(f"      ever>0.01? {'YES' if mx>0.01 else 'NO'}   ever>0.05? {'YES' if mx>0.05 else 'NO'}")
            print(f"      traj: {traj}")
        # also list any course-completion-ish tags for context
        ctx = [t for t in tags if any(k in t.lower() for k in ("finish","collision","miss","survive","l_episode","pointing","fix_rate","reward"))]
        print(f"\n  context tags present: {ctx[:25]}")

if __name__ == "__main__":
    main()
