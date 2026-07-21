"""Read scalars out of a TensorBoard event file with NO tensorboard dependency.

The old scratchpad/tb_parse.py was stashed into _deprecated/ but the v1.7/v1.8 launcher
checklists still cite it, so this is a standalone replacement. It walks the TFRecord
framing and the protobuf wire format directly -- no schema, no compiled protos:

  TFRecord  : uint64 len | uint32 crc(len) | payload[len] | uint32 crc(payload)
  Event     : field 2 = step (varint), field 5 = Summary (length-delimited)
  Summary   : field 1 = repeated Value (length-delimited)
  Value     : field 1 = tag (string), field 2 = simple_value (float32)

CRCs are not verified (they are masked CRC32C, and a truncated tail is the only realistic
corruption here) -- a short final record is simply dropped.

Usage:
    py tb_scalars.py <event_file> [tag_substring ...]
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path


def read_varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = val = 0
    while i < len(buf):
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if not b & 0x80:
            return val, i
        shift += 7
    raise ValueError("truncated varint")


def walk_fields(buf: bytes):
    """Yield (field_number, wire_type, value) where value is bytes | int | float."""
    i = 0
    while i < len(buf):
        key, i = read_varint(buf, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = read_varint(buf, i)
            yield fn, wt, v
        elif wt == 1:
            yield fn, wt, struct.unpack_from("<d", buf, i)[0]
            i += 8
        elif wt == 2:
            ln, i = read_varint(buf, i)
            yield fn, wt, buf[i:i + ln]
            i += ln
        elif wt == 5:
            yield fn, wt, struct.unpack_from("<f", buf, i)[0]
            i += 4
        else:
            raise ValueError(f"unsupported wire type {wt}")


def parse(path: Path):
    raw = path.read_bytes()
    series: dict[str, list[tuple[int, float]]] = {}
    i = 0
    while i + 12 <= len(raw):
        (ln,) = struct.unpack_from("<Q", raw, i)
        start = i + 12
        end = start + ln
        if end + 4 > len(raw):
            break                                   # truncated tail
        payload = raw[start:end]
        i = end + 4
        step = 0
        summaries = []
        try:
            for fn, _wt, v in walk_fields(payload):
                if fn == 2 and isinstance(v, int):
                    step = v
                elif fn == 5 and isinstance(v, bytes):
                    summaries.append(v)
        except ValueError:
            continue
        for s in summaries:
            try:
                for fn, _wt, v in walk_fields(s):
                    if fn != 1 or not isinstance(v, bytes):
                        continue
                    tag, val = None, None
                    for vfn, vwt, vv in walk_fields(v):
                        if vfn == 1 and vwt == 2:
                            tag = vv.decode("utf-8", "replace")
                        elif vfn == 2 and vwt == 5:
                            val = float(vv)
                    if tag is not None and val is not None:
                        series.setdefault(tag, []).append((step, val))
            except ValueError:
                continue
    return series


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1])
    wanted = [a.lower() for a in sys.argv[2:]]
    series = parse(path)

    tags = sorted(t for t in series
                  if not wanted or any(w in t.lower() for w in wanted))
    if not tags:
        print(f"no matching tags. {len(series)} tags present, e.g. "
              f"{sorted(series)[:8]}")
        return 1

    print(f"{len(series)} tags in file; showing {len(tags)}")
    print(f"  {'tag':<40}{'n':>4}{'first':>12}{'last':>12}{'min':>12}{'max':>12}")
    print("  " + "-" * 92)
    for t in tags:
        pts = sorted(series[t])
        vals = [v for _s, v in pts]
        print(f"  {t:<40}{len(pts):>4}{vals[0]:>12.4f}{vals[-1]:>12.4f}"
              f"{min(vals):>12.4f}{max(vals):>12.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
