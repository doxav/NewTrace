"""Empirical-replay simulator of EXP22 solution search (no LLM).
child = parent + delta, delta drawn from real EXP22 logs, conditioned on whether
parent was the current best (improving the best is harder). Invalid rate from logs."""
import json, math, random, statistics as st, os
D=json.load(open(os.path.dirname(os.path.abspath(__file__))+'/deltas.json'))
CFG={'prism':dict(init=21.891622105209393, invalid=0.30), 'signal':dict(init=0.49904861783269006, invalid=0.03)}
NB=5
def buckets(task):
    rows=[r for r in D[task] if r['parent'] is not None and r['child'] is not None]
    ps=sorted(r['parent'] for r in rows); edges=[ps[int(i*(len(ps)-1)/NB)] for i in range(1,NB)]
    b={'edges':edges, 'bins':[[] for _ in range(NB)]}
    for r in rows: b['bins'][sum(r['parent']>e for e in edges)].append(r['child']-r['parent'])
    return b
def draw(B,par,rng):
    return par+rng.choice(B['bins'][sum(par>e for e in B['edges'])])
def pick(policy, pop, rng):
    if policy=='uniform': return rng.choice(pop)           # stock EvoX initial strategy
    if policy=='greedy': return max(pop)
    if policy=='topk':   return rng.choice(sorted(pop)[-3:])
    if policy=='fitprop':                                  # like evolved Trace/EvoX policies
        lo=min(pop); w=[(p-lo)+1e-9 for p in pop]; return rng.choices(pop,weights=w)[0]
def run(task, schedule, rng, n=100, B=None):
    """schedule: list of policies; switch to next policy after 10 stagnant iters (EvoX trigger)."""
    c=CFG[task]; pop=[c['init']]; best=c['init']; idx=0; stag=0; last=best
    wstart=best; windows=[]; curve=[]
    for it in range(n):
        pol=schedule[min(idx,len(schedule)-1)]
        par=pick(pol,pop,rng)
        if rng.random()>=c['invalid']:
            ch=draw(B,par,rng); pop.append(ch); best=max(best,ch)
        curve.append(best)
        stag = 0 if best-last>0.01 else stag+1; last=best
        if stag>=10 and it<n-1 and len(schedule)>1:
            stag=0
            windows.append((pol,(best-wstart)*(1+math.log(1+max(0,wstart)))/math.sqrt(10)))
            wstart=best; idx+=1
    return best, windows, curve
def summary(task, R=4000):
    B=buckets(task); rng=random.Random(0); res={}
    print(f'\n##### {task}: parent-score bins edges {[round(e,3) for e in B["edges"]]}, sizes {[len(x) for x in B["bins"]]}, P(improve) per bin {[round(sum(d>0.01*(1 if task=="prism" else 0.01) for d in x)/len(x),2) for x in B["bins"]]}')
    for pol in ('uniform','fitprop','topk','greedy'):
        f=[run(task,[pol],rng,B=B)[0] for _ in range(R)]
        res[pol]=f
        print(f'  fixed {pol:8s} final best mean {st.mean(f):.4f} sd {st.stdev(f):.4f}  p10 {sorted(f)[R//10]:.4f} p90 {sorted(f)[9*R//10]:.4f}')
    u=res['uniform']; sd=st.stdev(u)
    for pol in ('fitprop','topk','greedy'):
        dlt=st.mean(res[pol])-st.mean(u); s=math.sqrt((st.variance(res[pol])+st.variance(u))/2)
        n=math.ceil(2*((1.96+0.84)*s/dlt)**2) if dlt>0 else float('inf')
        # P(single run of pol beats single run of uniform)
        pw=sum(a>b for a,b in zip(res[pol],u))/R
        print(f'  {pol} - uniform: delta {dlt:+.4f} ({dlt/s:+.2f} sd); P(1 run beats 1 run)={pw:.2f}; runs/arm for 80% power: {n}')
    # Meta signal: does the window score identify the better policy?
    # alternate uniform/greedy windows; check whether greedy windows score higher
    wins={'uniform':[], 'greedy':[]}; first=[]; zero=0; tot=0
    for _ in range(R):
        _,w,_=run(task,['uniform','greedy']*6,rng,B=B)
        for i,(p,s) in enumerate(w):
            wins[p].append(s); tot+=1; zero+= s==0
            if i==0: first.append(s)
    print(f'  meta windows per run {tot/R:.1f}; fraction with score 0: {zero/tot:.2f}; first-window mean {st.mean(first):.3f} vs later-window mean {st.mean([s for p in wins for s in wins[p]][:]):.3f}')
    print(f'  window score by policy: uniform {st.mean(wins["uniform"]):.3f}  greedy {st.mean(wins["greedy"]):.3f}')
    # position bias: window score vs window index regardless of policy
    pos={}
    for _ in range(R):
        _,w,_=run(task,['uniform']*12,rng,B=B)
        for i,(p,s) in enumerate(w): pos.setdefault(i,[]).append(s)
    print('  same policy, window score by position:',{i:round(st.mean(v),3) for i,v in sorted(pos.items()) if len(v)>50})
for t in ('prism','signal'): summary(t)
