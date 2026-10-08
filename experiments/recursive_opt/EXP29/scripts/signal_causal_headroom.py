"""EXP29 P0 probe: legitimate (causal) headroom of Signal Processing with hand-written classical filters.

No LLM. Evaluates 34 causal filters with the EXP25 white-box evaluator (stock score, valid score, causal probe).
Run: TRACE_ROOT=$PWD experiments/recursive_opt/EXP22/.venv/bin/python experiments/recursive_opt/EXP29/scripts/signal_causal_headroom.py
"""
import sys, json, re, itertools, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path; HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE.parents[1]/'EXP25'/'signal')); sys.path.insert(0,str(HERE.parents[3]))
import whitebox as W
INIT=W.INITIAL.read_text()
HEAD='def enhanced_filter_with_trend_preservation(x, window_size=20):'
start=INIT.index(HEAD); end=INIT.index('def process_signal')
FILTERS={
'ema': '''
    import numpy as np
    x=np.asarray(x,float); a={a}; f=np.empty_like(x); s=x[0]
    for i,v in enumerate(x): s=a*v+(1-a)*s; f[i]=s
    return f[window_size-1:]
''',
'butter': '''
    import numpy as np
    from scipy.signal import butter, lfilter, lfilter_zi
    x=np.asarray(x,float); b,a=butter({o},{c}); f=lfilter(b,a,x,zi=lfilter_zi(b,a)*x[0])[0]
    return f[window_size-1:]
''',
'abfilt': '''
    import numpy as np
    x=np.asarray(x,float); al={a}; be={b}; lv=x[0]; tr=0.0; f=np.empty_like(x)
    for i,v in enumerate(x):
        p=lv+tr; r=v-p; lv=p+al*r; tr=tr+be*r; f[i]=lv
    return f[window_size-1:]
''',
'endpoly_ema': '''
    import numpy as np
    x=np.asarray(x,float); n=len(x)-window_size+1; t=np.arange(window_size)
    V=np.vander(t,{d}+1); P=np.linalg.pinv(V); e=np.vander([window_size-1],{d}+1)[0]@P
    y=np.array([e@x[i:i+window_size] for i in range(n)]); a={a}; out=np.empty_like(y); s=y[0]
    for i,v in enumerate(y): s=a*v+(1-a)*s; out[i]=s
    return out
''',
}
def prog(kind,**p): return INIT[:start]+HEAD+FILTERS[kind].format(**p)+'\n\n'+INIT[end:]
grid=[('ema',dict(a=a)) for a in (0.05,0.1,0.15,0.2,0.3)]
grid+=[('butter',dict(o=o,c=c)) for o in (1,2,3) for c in (0.03,0.05,0.08,0.12)]
grid+=[('abfilt',dict(a=a,b=b)) for a in (0.1,0.2,0.3) for b in (0.005,0.02,0.05)]
grid+=[('endpoly_ema',dict(d=d,a=a)) for d in (1,2) for a in (0.2,0.4,0.7,1.0)]
def run(g):
    m,_=W.evaluate(prog(g[0],**g[1])); return g, {k:m.get(k) for k in ('combined_score','valid_score','causal_fraction')}
if __name__=='__main__':
    t=time.time(); print('initial', W.evaluate(INIT)[0]['combined_score'])
    with ProcessPoolExecutor(12) as ex: res=list(ex.map(run,grid))
    res.sort(key=lambda r:-(r[1]['valid_score'] or 0))
    for g,m in res: print(g, m)
    json.dump([[g,m] for g,m in res],open(HERE.parent/'results'/'signal_causal_headroom.json','w'),indent=1); print('s',time.time()-t)
