"""Rebuild the exact OptoPrimeV2 meta step from a saved EXP22 observation, capture the prompt (no API)."""
import json, sys, hashlib
from pathlib import Path
RUN=Path(sys.argv[1]); K=int(sys.argv[2])
from opto import trace
from opto.optimizers.optoprime_v2 import OptoPrimeV2
from opto.utils.llm import DummyLLM
sys.path.insert(0, str(Path.cwd()))
from src.observations import encode_observation
CAP={}
def fake(messages=None, **kw):
    CAP['messages']=messages
    return 'x'
obs=json.loads((RUN/f'observation_{K:03d}.json').read_text())
src=(RUN/f'policy_{K-1:03d}.py').read_text() if K>1 else Path('/home/xav/code/evo-compare/repos/skydiscover/skydiscover/optimize/search/evox/database/initial_search_strategy.py').read_text()
digest=lambda s: hashlib.sha256(s.encode()).hexdigest()
obs['active_policy_hash']=digest(src)
@trace.bundle()
def measured_window(policy_source, observation):
    return observation
class M(trace.Module):
    def __init__(s): super().__init__(); s.policy_source=trace.node(src,trainable=True,name='policy_source')
    def forward(s,o): return measured_window(s.policy_source,o)
m=M(); opt=OptoPrimeV2(m.parameters(), llm=DummyLLM(fake), max_tokens=32000, log=False, initial_var_char_limit=100000)
fb=json.dumps(encode_observation(obs))
out=m({'active_policy_hash':obs['active_policy_hash'],'window_metrics':obs['window_metrics']})
opt.zero_feedback(); opt.backward(out, fb)
try: opt.step()
except Exception as e: pass
msgs=CAP['messages']; text='\n'.join(str(x.get('content')) for x in msgs)
print('parameters:',len(opt.parameters),'| optimizer memory_size:',getattr(opt,'memory_size',None))
print('prompt chars:',len(text),'~tokens:',len(text)//4)
print('feedback chars:',len(fb),'share of prompt: %.0f%%'%(100*len(fb)/len(text)))
print('policy source chars:',len(src))
print('window_metrics:',obs['window_metrics'])
# how much of feedback is solution code
code=sum(len(p.get('solution','')) for p in obs['search_stats']['db_stats'].get('previous_programs',[]))+sum(len(p.get('solution','')) for p in obs['policy_history'])
print('raw solution/policy code inside observation: %d chars of %d (%.0f%%)'%(code,len(json.dumps(obs)),100*code/len(json.dumps(obs))))
i=text.find('#Instruction'); print('--- prompt head ---'); print(text[:1800])
