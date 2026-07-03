"""A32 diagnosis pass 2: robust time base + post-contact attitude trace + vertical channel."""
import json, math
import numpy as np

RUN = r"C:/Users/Shadow/Peregrine/.claude/worktrees/agent-a08b24f288de69e40/data/runs/20260703_172104_rl_s1_f1/nav_estimate.jsonl"
rows = [json.loads(l) for l in open(RUN, encoding="utf-8") if l.strip()]

# robust elapsed time: cumulative positive deltas
raw = np.array([r["sim_time_ns"] for r in rows], dtype=np.int64)
d = np.diff(raw) / 1e9
print("negative-delta ticks:", np.flatnonzero(d < 0), "values:", d[d < 0])
d2 = np.clip(d, 0, 1.0)
t = np.concatenate([[0.0], np.cumsum(d2)])
print(f"ticks={len(rows)} dur={t[-1]:.1f}s mean_rate={len(rows)/t[-1]:.1f}Hz median_dt={np.median(d2)*1000:.0f}ms")

deg = 180 / math.pi
pitch = np.array([r.get("pitch_rad") or 0.0 for r in rows]) * deg
roll = np.array([r.get("roll_rad") or 0.0 for r in rows]) * deg
thrust = np.array([r.get("thrust") if r.get("thrust") is not None else np.nan for r in rows])
contact = np.array([bool(r.get("contact_frozen")) for r in rows])
vz = np.array([r.get("vert_vz_est") if r.get("vert_vz_est") is not None else np.nan for r in rows])
zoff = np.array([r.get("z_off_est") if r.get("z_off_est") is not None else np.nan for r in rows])
vzt = np.array([r.get("vz_t") if r.get("vz_t") is not None else np.nan for r in rows])
regime = [r.get("seeker_regime") for r in rows]

# vertical channel, 0.5 s bins, full flight
print("\nt, pitch, roll, vz_est, z_off, vz_t, thrust, regime")
for lo in np.arange(0, t[-1], 0.5):
    w = (t >= lo) & (t < lo + 0.5)
    if w.sum():
        regs = {regime[i] for i in np.flatnonzero(w)}
        print(f"  t={lo:5.1f} pitch={np.nanmean(pitch[w]):+6.1f} roll={np.nanmean(roll[w]):+7.1f} "
              f"vz={np.nanmean(vz[w]):+5.2f} zoff={np.nanmean(zoff[w]):+6.2f} vzt={np.nanmean(vzt[w]):+5.2f} "
              f"thr={np.nanmean(thrust[w]):.3f} {'/'.join(sorted(regs))}")

# tick-level trace around first contact
i0 = np.flatnonzero(contact)[0]
print(f"\nfirst contact tick idx={i0} t={t[i0]:.2f}")
for i in range(max(0, i0 - 6), min(len(rows), i0 + 40)):
    r = rows[i]
    print(f"  t={t[i]:6.2f} pitch={pitch[i]:+6.1f} roll={roll[i]:+7.1f} thr={thrust[i]:.3f} "
          f"contact={int(contact[i])} vz={vz[i]:+5.2f} reg={regime[i]}")
