import json, glob, os, statistics as st, hashlib
from collections import defaultdict
ROOT='/home/xav/code/Trace-experiment0/experiments/recursive_opt/EXP22'
runs=sorted(glob.glob(ROOT+'/runs/strict_*'))+sorted(glob.glob(ROOT+'/artifacts/parallel_trace_*/v9_runtime/runs/strict_*'))
def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
def score(m):
    if not m: return None
    s=m.get('combined_score'); return s if isinstance(s,(int,float)) else None
for r in runs:
    name=r.split('/')[-1][:48]
    curve=jl(r+'/solution_curve.jsonl'); cands=jl(r+'/candidate_history.jsonl'); pol=jl(r+'/policy_history.jsonl')
    if not curve: continue
    print('=== ',name,'| n',len(curve),'final best %.4f'%curve[-1]['best_score'])
    ws=[round(e['feedback']['window_metrics']['combined_score'],3) for e in pol if e.get('feedback')]
    print('  meta window scores fed to proposer:',ws, ' activated:',[e.get('activated') for e in pol])
    # per policy candidate stats
    by=defaultdict(list); pop={}  # id->score
    improvements=[]; parent_rank=[]
    best=None
    for c in cands:
        cd=c.get('candidate'); 
        if not cd: continue
        s=score(cd.get('metrics'))
        pid=c.get('parent_id')
        if pid in pop and pop:
            ranked=sorted(pop.values(),reverse=True)
            parent_rank.append(ranked.index(pop[pid])/max(1,len(ranked)-1))
        if s is not None:
            if best is not None and s>best+0.01: improvements.append(c['iteration'])
            best=s if best is None else max(best,s)
            pop[cd['id']]=s
    # map iteration->active policy from curve
    act={row['iteration']:row['active_policy_hash'][:8] for row in curve}
    for c in cands:
        cd=c.get('candidate'); s=score(cd.get('metrics')) if cd else None
        by[act.get(c['iteration'],'?')].append(s)
    for h,v in by.items():
        vv=[x for x in v if x is not None]
        print('   policy %s: attempts %3d valid %3d  mean %.3f  median %.3f  max %.3f'%(h,len(v),len(vv),st.mean(vv) if vv else float('nan'),st.median(vv) if vv else float('nan'),max(vv) if vv else float('nan')))
    print('  iterations with global-best improvement >0.01:',improvements)
    if parent_rank: print('  parent relative rank (0=best,1=worst) mean %.2f, frac best-parent %.2f'%(st.mean(parent_rank), sum(1 for x in parent_rank if x==0)/len(parent_rank)))
