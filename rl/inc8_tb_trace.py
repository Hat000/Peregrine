"""inc8 smoke signal-trace reader (compute-node safe).

The inc8 env logs its signal metrics to **TensorBoard only** (`env_loss/inc8_*`, `env_loss/total_reward`,
`agent_loss/*`, `metrics/*`) -- they never reach stdout, so the smoke sbatch's old stdout-grep found
nothing. This helper parses the TB event file and prints the TRAJECTORY (not just first/last) every
~50-100 updates of the metrics the smoke gate reads:

    total_reward · inc8_pointing_rate · inc8_terminal_pointing · inc8_fix_rate · entropy · value_loss

Usage (invoked by rl/peregrine_inc8_smoke.sbatch on the COMPUTE node):
    python inc8_tb_trace.py <RUN_DIR> [--stride-updates N]

It searches <RUN_DIR> recursively for the event file with the most scalar steps (the real training log,
not a stub), so it is robust to diffaero's decorated TB logdir living a level below the hydra run dir.

DEPENDENCIES: only `tensorboard`'s event_accumulator (already in the diffaero env, since diffaero logs
to TB) + the stdlib. The tensorboard import is DEFERRED into main() so this module imports clean on a
machine without tensorboard (e.g. the dev laptop, for an AST/import syntax check).
"""
import os
import sys

# (display label, ordered tag-suffix candidates -- first match wins). Suffix = last '/'-segment of the
# scalar tag. Exact-suffix match is tried before substring so e.g. inc8_pointing_rate never shadows
# inc8_terminal_pointing.
COLUMNS = [
    ("total_reward",      ["total_reward"]),
    ("pointing_rate",     ["inc8_pointing_rate", "pointing_rate"]),
    ("terminal_pointing", ["inc8_terminal_pointing", "terminal_pointing"]),
    ("lockband_pointing", ["inc8_lockband_pointing", "lockband_pointing"]),
    ("fix_rate",          ["inc8_fix_rate", "fix_rate"]),
    ("band_az_abs_deg",   ["inc8_band_az_abs_deg", "band_az_abs_deg"]),   # look-at sign/efficacy (S0)
    ("estim_err_ip_m",    ["inc8_estim_err_inplane_m", "estim_err_inplane_m"]),  # sigma_p0 proxy (S1)
    ("centering",         ["inc8_centering", "centering"]),               # dense centering reward (S1)
    ("entropy",           ["entropy_loss", "entropy"]),
    ("value_loss",        ["critic_loss", "value_loss"]),
]


def _find_event_files(run_dir):
    out = []
    for root, _dirs, files in os.walk(run_dir):
        for fn in files:
            if "tfevents" in fn:
                out.append(os.path.join(root, fn))
    return out


def _resolve_tag(available, candidates):
    """Pick the best scalar tag for a column. Exact last-segment match first, then substring."""
    by_suffix = {t.split("/")[-1]: t for t in available}
    for cand in candidates:
        if cand in by_suffix:
            return by_suffix[cand]
    for cand in candidates:
        for t in available:
            if cand in t.split("/")[-1]:
                return t
    return None


def _value_at(series, step):
    """series = {step: value}; exact step else nearest by |Δstep|."""
    if step in series:
        return series[step]
    if not series:
        return None
    nearest = min(series, key=lambda s: abs(s - step))
    return series[nearest]


def main(argv):
    run_dir = argv[1] if len(argv) > 1 else "."
    stride_updates = 50
    if "--stride-updates" in argv:
        try:
            stride_updates = int(argv[argv.index("--stride-updates") + 1])
        except (ValueError, IndexError):
            pass

    # Deferred import: keeps this module import-clean where tensorboard is absent (dev laptop).
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as exc:  # noqa: BLE001
        print("[inc8-tb-trace] tensorboard.event_accumulator unavailable (%s) -- cannot parse TB "
              "events; the run still trained, this is a reporting-only failure." % exc)
        return 0

    event_files = _find_event_files(run_dir)
    if not event_files:
        print("[inc8-tb-trace] no tfevents file found under %s -- nothing to trace." % run_dir)
        return 0

    # Load every candidate event file; keep the one exposing the most scalar steps (the real log).
    best = None  # (n_steps, path, {tag: {step: value}}, [scalar_tags])
    for path in event_files:
        try:
            acc = EventAccumulator(path, size_guidance={"scalars": 0})  # 0 = load all
            acc.Reload()
            tags = acc.Tags().get("scalars", [])
            if not tags:
                continue
            series = {}
            max_steps = 0
            for t in tags:
                pts = {ev.step: ev.value for ev in acc.Scalars(t)}
                series[t] = pts
                max_steps = max(max_steps, len(pts))
            if best is None or max_steps > best[0]:
                best = (max_steps, path, series, tags)
        except Exception as exc:  # noqa: BLE001
            print("[inc8-tb-trace] skipped unreadable event file %s (%s)" % (path, exc))

    if best is None:
        print("[inc8-tb-trace] event file(s) present but no scalars logged under %s." % run_dir)
        return 0

    _n, path, series, tags = best
    resolved = [(label, _resolve_tag(tags, cands)) for label, cands in COLUMNS]

    print("[inc8-tb-trace] event file: %s" % path)
    print("[inc8-tb-trace] resolved tags:")
    for label, tag in resolved:
        print("    %-18s -> %s" % (label, tag if tag else "(MISSING)"))

    # Row index = the steps of the reference column (total_reward), else the longest available series.
    ref_tag = resolved[0][1]
    if ref_tag is None or not series.get(ref_tag):
        ref_tag = max(tags, key=lambda t: len(series[t]))
    ref_steps = sorted(series[ref_tag].keys())
    if not ref_steps:
        print("[inc8-tb-trace] reference series empty -- nothing to print.")
        return 0

    # Select printed steps: first, last, and every >= stride_updates apart.
    selected = [ref_steps[0]]
    for s in ref_steps[1:]:
        if s - selected[-1] >= stride_updates:
            selected.append(s)
    if selected[-1] != ref_steps[-1]:
        selected.append(ref_steps[-1])

    labels = [lbl for lbl, _ in resolved]
    header = "  ".join(["%8s" % "step"] + ["%18s" % lbl for lbl in labels])
    print("\n[inc8-tb-trace] TRAJECTORY (every ~%d updates):" % stride_updates)
    print(header)
    print("-" * len(header))
    for step in selected:
        cells = ["%8d" % step]
        for _label, tag in resolved:
            if tag is None:
                cells.append("%18s" % "-")
            else:
                v = _value_at(series[tag], step)
                cells.append("%18s" % ("%.5f" % v if v is not None else "-"))
        print("  ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
