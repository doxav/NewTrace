import json, glob, os, statistics as st
from collections import defaultdict
ROOT='/home/xav/code/Trace-experiment0/experiments/recursive_opt/EXP22'
runs=sorted(glob.glob(ROOT+'/runs/strict_*'))+sorted(glob.glob(ROOT+'/artifacts/parallel_trace_*/v9_runtime/runs/strict_*'))
out=defaultdict(list)
seen=set()
for r in runs:
    task='prism' if 'prism' in r else 'signal'
    p=r+'/candidate_history.jsonl'
    if not os.path.exists(p): continue
    init=json.load(open(r+'/kernel_result.json'))['initial_score'] if os.path.exists(r+'/kernel_result.json') else None
    pop={}
    for l in open(p):
        c=json.loads(l); cd=c.get('candidate')
        pid=c.get('parent_id')
        if cd is None:
            continue
        key=(cd['id'])
        if key in seen: continue
        seen.add(key)
        s=cd['metrics'].get('combined_score'); s=s if isinstance(s,(int,float)) else None
        ps=pop.get(pid)
        if ps is None and init is not None and pid not in pop: ps=init  # initial program
        best=max(pop.values()) if pop else init
        out[task].append({'parent':ps,'child':s,'best':best})
        if s is not None: pop[cd['id']]=s
for t,rows in out.items():
    v=[r for r in rows if r['parent'] is not None]
    inval=sum(1 for r in v if r['child'] is None)
    d=[r['child']-r['parent'] for r in v if r['child'] is not None]
    print(t,'n',len(v),'invalid',inval,'delta mean %.4f sd %.4f'%(st.mean(d),st.stdev(d)),'P(child>parent) %.2f'%(sum(x>0 for x in d)/len(d)))
    # relation with parent gap to best
    import math
    q=sorted(d); print('  delta quantiles', [round(q[int(i*(len(q)-1))],3) for i in (0,.1,.25,.5,.75,.9,.95,1)])
json.dump(out,open(os.path.dirname(__file__)+'/deltas.json','w'))
