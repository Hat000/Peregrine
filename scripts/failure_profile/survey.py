import json, os, glob, collections
SCR = r"C:\Users\Fengy\AppData\Local\Temp\claude\C--Users-Fengy-Downloads-Projects-Anduril--claude-worktrees-peregrine-rl-commander-74b0dd\b85130f9-ac88-4db2-8c68-0e28b966cf80\scratchpad"
ROOTS = [r"C:\Users\Fengy\Downloads\Projects\wt-arrest\data\runs",
         os.path.join(SCR,"new25","data","runs"), os.path.join(SCR,"v2f","data","runs"),
         os.path.join(SCR,"case","data","runs"), os.path.join(SCR,"gate1","data","runs")]
seen={}; dup=0
rows=[]
for R in ROOTS:
    for p in sorted(glob.glob(os.path.join(R,"*"))):
        if not os.path.isdir(p): continue
        n=os.path.basename(p)
        if n in seen: dup+=1; continue
        seen[n]=p
        try: m=json.load(open(os.path.join(p,"meta.json"),encoding="utf-8"))
        except Exception as e: rows.append((n,R,"ERR",str(e)[:40],None,None,None,None)); continue
        rs=m.get("race_status") or {}
        rows.append((n,R,m.get("ego_ckpt"),m.get("final_state"),m.get("collisions"),
                     m.get("duration_s"),rs.get("active_gate_index"),rs.get("finished"),
                     m.get("label"),m.get("ego_gate_z_bias"),m.get("ego_sector_mode"),
                     m.get("ego_obs_coast"),m.get("ego_pitch_clamp")))
print("sessions:",len(rows),"dups skipped:",dup)
c=collections.Counter(r[2] for r in rows)
for k,v in c.most_common(): print(f"  ckpt {k}: {v}")
print("--- final_state ---")
for k,v in collections.Counter(r[3] for r in rows).most_common(): print(f"  {k}: {v}")
print("--- label ---")
for k,v in collections.Counter(r[8] for r in rows).most_common(20): print(f"  {k}: {v}")
print("--- max active_gate_index ---")
for k,v in sorted(collections.Counter(r[6] for r in rows).items(), key=lambda x:(x[0] is None,x[0])): print(f"  gate {k}: {v}")
print("--- finished ---")
for k,v in collections.Counter(r[7] for r in rows).most_common(): print(f"  {k}: {v}")
json.dump({n:p for n,p in seen.items()}, open(os.path.join(SCR,"profile","paths.json"),"w"))
