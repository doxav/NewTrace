"""Independent completed P1/control verification; no candidate or live model execution."""
from __future__ import annotations
import collections
import datetime
import hashlib
import json
import math
import random
import socket
import statistics
import time
import zipfile
from pathlib import Path
from typing import Any
from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import generation as G
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import evidence_io as I
from experiments.recursive_opt._shared.optimizer_discovery.investigation16.production import analysis as A
from experiments.recursive_opt._shared.optimizer_discovery.investigation16.production import baseline_control as C

ROOT = Path('/home/xav/code/Trace/experiments/recursive_opt/_shared/optimizer_discovery/investigation16')
P = ROOT / 'production_run'
Q = ROOT / 'production_baseline_control'
SEEDS = [16411,16423,16437,16441,16453,16467]
ARMS = ['A0','I','C','R','W']
started = time.time_ns()
blocked_calls: list[str] = []

def forbidden(*args: Any, **kwargs: Any) -> Any:
    """Reject any candidate, model, key, network or persistence call in this review."""
    blocked_calls.append('forbidden execution boundary')
    raise AssertionError('forbidden execution boundary')

socket.socket.connect = forbidden
socket.create_connection = forbidden
B.evaluate = forbidden
B.propose_point = forbidden
G.make_live_llm = forbidden
G._load_key = forbidden
I.persist = forbidden
original_objective, original_normalization = B.objective, B.normalization
B.objective = forbidden
B.normalization = forbidden
original_read = E.read
hashes: dict[str,str] = {}

def filehash(path: Path) -> str:
    """Identify bytes of preserved source or evidence without executing them."""
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path: Path) -> Any:
    """Track every evidence file read by reused checks before decoding its JSON."""
    physical = path if path.exists() else Path(str(path)+'.gz')
    actual = filehash(physical)
    previous = hashes.setdefault(str(physical), actual)
    assert previous == actual
    return original_read(path)

E.read = read
for name in ['pipeline_complete.json','generation_frozen.json','selections_frozen.json','audit_results.json','analysis_results.json']:
    assert E.exists(P/name), name
print('Reading frozen bundle and checking chronology', flush=True)
bundle = A.read_bundle(P)
primary_saved = read(P/'analysis_results.json')
primary_recomposed = A.summarize(bundle)
assert primary_recomposed == primary_saved
control_freeze, primary_freeze = C.preflight(Q)
assert primary_freeze == bundle['freeze']
assert B.digest(primary_freeze) == '113f2eb03abfbb3f8e80e8ce5946beecd6b039fef28968b96171475a21d57e8e'
assert B.digest(control_freeze) == 'f0aeb8b78d56745c2e54b02462877952bdda15ec4e184fc0346e13d0bb0e288f'
assert primary_saved['outer_seeds'] == SEEDS
control_saved = read(Q/'results.json')
verification = read(Q/'primary_complete_verified.json')
assert control_saved['primary_verification'] == verification
print('Frozen full analysis reproduces exactly; independent arithmetic follows', flush=True)


def equal(a: float, b: float) -> bool:
    """Permit only final floating-point summation differences across independent paths."""
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-14)

def aggregate(rows: list[dict[str,Any]], metric: str) -> float:
    """Rebuild equal-stratum aggregation directly, independent of benchmark.aggregate."""
    groups: dict[str,list[float]] = collections.defaultdict(list)
    for row in rows:
        assert row['valid']
        if metric == 'auc': value = math.fsum(row['metrics']['curve'])/len(row['metrics']['curve'])
        elif metric == 'final_regret': value = row['metrics']['curve'][-1]
        else: value = row['metrics'][metric]
        groups[row['stratum']].append(value)
    assert len(groups) == 6
    return math.fsum(math.fsum(values)/len(values) for values in groups.values())/6

def contrast(values: list[float]) -> dict[str,Any]:
    """Independently reproduce the frozen paired resampling using uniform RNG indices."""
    rng = random.Random(1515)
    n = len(values)
    assert n == 6
    samples = sorted(math.fsum(values[math.floor(rng.random()*n)] for _ in range(n))/n for _ in range(10000))
    interval = []
    for fraction in [0.025,0.975]:
        position = (len(samples)-1)*fraction
        low = math.floor(position)
        interval.append(samples[low]+(samples[math.ceil(position)]-samples[low])*(position-low))
    interpretation = 'positive signal' if interval[1]<0 else 'negative signal' if interval[0]>0 else 'no detectable difference' if all(v==0 for v in values) else 'inconclusive'
    return {'deltas':values,'mean':math.fsum(values)/n,'median':statistics.median(values),'paired_bootstrap_95':interval,'interpretation':interpretation,'replication_unit':'outer_seed'}

def compare_contrast(actual: dict[str,Any], saved: dict[str,Any]) -> float:
    """Keep every contrast and return the largest independent arithmetic discrepancy."""
    assert actual['interpretation']==saved['interpretation'] and actual['replication_unit']==saved['replication_unit']
    left = actual['deltas']+[actual['mean'],actual['median']]+actual['paired_bootstrap_95']
    right = saved['deltas']+[saved['mean'],saved['median']]+saved['paired_bootstrap_95']
    assert all(equal(a,b) for a,b in zip(left,right))
    return max(abs(a-b) for a,b in zip(left,right))

primary_values: dict[str,dict[str,float]] = {}
for outer in SEEDS:
    primary_values[str(outer)] = {}
    for arm in ARMS:
        rows = bundle['audit']['per_seed'][str(outer)][arm]['rows']
        assert len(rows)==24 and all(r['valid'] and len(r['observations'])==32 for r in rows)
        primary_values[str(outer)][arm] = aggregate(rows,'auc')
        assert equal(primary_values[str(outer)][arm],primary_saved['per_seed'][str(outer)][arm]['auc'])
        assert equal(aggregate(rows,'final_regret'),primary_saved['per_seed'][str(outer)][arm]['final_regret'])
primary_contrasts = {}
contrast_max_error = 0.0
for left,right in [('R','I'),('R','C'),('W','R'),('R','A0')]:
    key = f'{left}-{right}'
    result = contrast([primary_values[str(s)][left]-primary_values[str(s)][right] for s in SEEDS])
    contrast_max_error=max(contrast_max_error,compare_contrast(result,primary_saved['contrasts'][key]))
    primary_contrasts[key] = result

selected_records = []
for outer in SEEDS:
    for arm in ARMS[1:]:
        key = f'{outer}/{arm}'
        pool = bundle['pools'][key]
        assert len(pool)==9 and [r['index'] for r in pool]==list(range(-1,8))
        eligibility = []
        for candidate in pool:
            assert B.source_hash(candidate['source'])==candidate['source_sha256']
            assert len(candidate['train'])==48 and len(candidate['validation'])==24
            valid = all(r['valid'] for split in ['train','validation'] for r in candidate[split])
            assert candidate['eligible']==valid
            if valid:
                value = aggregate(candidate['validation'],'auc')
                assert equal(value,candidate['validation_auc'])
                eligibility.append((value,candidate['index'],candidate))
            else: assert candidate['validation_auc'] is None
        chosen = min(eligibility,key=lambda v:(v[0],v[1]))[2]
        saved = bundle['selections'][key]
        assert chosen['index']==saved['index'] and chosen['source']==saved['source']
        assert B.source_hash(saved['source'])==saved['source_sha256']
        if saved['index']>=0:
            response=bundle['slots'][f'{key}/{saved["index"]}']['response']
            assert response['source']==saved['source']
        selected_records.append({'outer':outer,'arm':arm,'index':saved['index'],'source_sha256':saved['source_sha256'],'source_path':f'production_run/raw/{key}/selection.json:source','validation_auc':saved['validation_auc']})
representative = min([s for s in selected_records if s['arm']=='R'],key=lambda s:(s['validation_auc'],SEEDS.index(s['outer'])))
assert representative['outer']==primary_saved['representative']['outer']
assert representative['source_sha256']==primary_saved['representative']['source_sha256']

slot_rows = []
all_ids = set()
for outer in SEEDS:
    for arm in ARMS[1:]:
        known = {primary_freeze['seed_sha256']}
        for index in range(8):
            slot=bundle['slots'][f'{outer}/{arm}/{index}']; response=slot['response']; request=slot['request']
            assert response['completed'] and response['id'] not in all_ids
            all_ids.add(response['id'])
            assert request['parent_sha256'] in known
            assert response['source_sha256']==B.source_hash(response['source'])
            known.add(response['source_sha256'])
            assert response['model']==G.MODEL and request['model']==G.MODEL
            assert len([a for a in slot['attempts'] if a['status']=='completed'])==1
            assert slot['metadata']['id']==response['id']
            slot_rows.append({'outer':outer,'arm':arm,'slot':index,'id':response['id'],'attempts':len(slot['attempts']),'source_status':response['source_status'],'source_sha256':response['source_sha256'],'finish_reason':response['finish_reason'],'completed_ns':response['completed_ns']})
assert len(slot_rows)==len(all_ids)==192 and sum(r['attempts'] for r in slot_rows)==194
attempts=[a for s in bundle['slots'].values() for a in s['attempts']]
assert sum(a['status']=='transport_failure' for a in attempts)==2
assert sum(bool(a.get('possible_remote_completion_or_duplicate_billing')) for a in attempts)==2

barriers={name:read(P/(name+'.json')) for name in ['generation_frozen','selections_frozen','audit_results']}
assert primary_freeze['created_ns']<control_freeze['created_ns']<read(P/'generation_started.json')['wall_ns']
assert max(r['completed_ns'] for r in slot_rows)<barriers['generation_frozen']['completed_ns']
selection_times=[v['selected_ns'] for v in bundle['selections'].values()]
assert barriers['generation_frozen']['completed_ns']<min(selection_times)<=max(selection_times)<barriers['selections_frozen']['completed_ns']
cache_by_split={split:[entry for entry in bundle['cache'].values() if entry['key']['split']==split] for split in ['train','validation','audit']}
assert min(v['computed_clock']['wall_ns'] for v in cache_by_split['validation'])>barriers['generation_frozen']['completed_ns']
assert min(v['computed_clock']['wall_ns'] for v in cache_by_split['audit'])>barriers['selections_frozen']['completed_ns']
assert max(v['computed_clock']['wall_ns'] for v in cache_by_split['audit'])<barriers['audit_results']['completed_ns']
for name,barrier in barriers.items():
    assert verification['hashes'][name]==B.digest(barrier)

numeric=read(P/'numeric_verification/attempt_001.json')
cache_hashes={key:entry['row_sha256'] for key,entry in bundle['cache'].items()}
assert numeric['status']=='PASS' and not numeric['failures'] and numeric['cache_rows_verified']==len(cache_hashes)==14592
assert numeric['frozen_manifest_sha256']==B.digest(primary_freeze)
assert numeric['input_cache_rows_sha256']==B.digest(cache_hashes)
assert filehash(ROOT/'production/verify_numerics.py')==numeric['helper_sha256']
assert {r['cache_key'] for r in numeric['rows']}==set(cache_hashes)
for row in numeric['rows']:
    raw=bundle['cache'][row['cache_key']]['row']
    assert row['verified'] and row['valid']==raw['valid'] and row['recomputed_metrics']==raw['metrics']
    assert row['observations_recomputed']==len(raw['observations'])
assert numeric['integrity_objective_calls']['observations']==sum(len(v['row']['observations']) for v in bundle['cache'].values())

snapshot=read(P/'source_snapshot.json')
print('Main slots, pools, numeric-report binding and chronology pass', flush=True)
print('Snapshot keys',list(snapshot),flush=True)
# Exact archive and source bytes are covered by its frozen metadata.
archive_path=P/'frozen_sources.zip'
archive_hash=filehash(archive_path)
assert archive_hash=='f5120ad690b842f6fd2ca1558d00c5c2e84f723984dbbecd336e60c76fc546eb'
with zipfile.ZipFile(archive_path) as z:
    assert len(z.namelist())==len(set(z.namelist()))==97 and z.testzip() is None
    for name in z.namelist():
        path=Path('/home/xav/code/Trace')/name
        assert path.is_file() and path.read_bytes()==z.read(name)

# Only now permit the exactly counted objective/reference arithmetic of the fixed control.
control_jobs=control_freeze['jobs']
assert len(control_jobs)==144 and control_jobs==C.jobs(primary_freeze)
B._reference_scale.cache_clear()
objective_counts=collections.Counter()
mode='references'

def counted_objective(task: dict[str,Any], point: list[float]) -> float:
    """Count integrity calls separately from already completed scientific work."""
    objective_counts[mode]+=1
    return original_objective(task,point)

B.objective=counted_objective
B.normalization=original_normalization
scales={B.task_identity(task):B.normalization(task) for task in primary_freeze['tasks']['audit']}
assert len(scales)==12 and objective_counts['references']==1536
mode='observations'
control_rows=[]
control_row_receipts=[]
for job in control_jobs:
    saved=read(Q/'raw'/(job['id']+'.json'))
    row=saved['row']
    assert saved['job']==job and B.digest(row)==saved['row_sha256']
    assert saved['primary_audit_sha256']==verification['hashes']['audit_results']
    assert barriers['audit_results']['completed_ns']<saved['started']['wall_ns']<=saved['completed']['wall_ns']
    assert read(Q/'attempts'/(saved['attempt_id']+'.started.json'))=={'job_id':job['id'],'clock':saved['started']}
    assert row['source_sha256']==control_freeze['source_sha256'] and row['task_identity']==B.task_identity(job['task']) and row['local_seed']==job['local_seed']
    assert row['valid'] and row['candidate_valid'] and not row['fallback_used']
    assert row['budget']==row['objective_calls']==len(row['observations'])==32 and row['subprocess_executions']==64
    values=[]
    for observation in row['observations']:
        point=observation['x']
        assert len(point)==job['task']['dimension'] and all(math.isfinite(x) and -5<=x<=5 for x in point)
        value=B.objective(job['task'],point)
        assert value==observation['value']
        values.append(value)
    assert B.metrics(values,scales[row['task_identity']],32)==row['metrics']
    # Independently reconstruct prefix minima, target censoring and mean as well.
    curve=[]
    incumbent=math.inf
    for value in values:
        incumbent=min(incumbent,max(0,value/scales[row['task_identity']]))
        curve.append(incumbent)
    assert curve==row['metrics']['curve']
    target=next((i+1 for i,x in enumerate(curve) if x<=0.01),None)
    assert equal(math.fsum(curve)/32,row['metrics']['auc']) and curve[-1]==row['metrics']['final_regret']
    assert target==row['metrics']['target_evaluations'] and row['metrics']['attained']==(target is not None)
    assert row['metrics']['capped_target_evaluations']==(target if target is not None else 33)
    control_rows.append((job,row))
    control_row_receipts.append({'id':job['id'],'row_sha256':saved['row_sha256'],'started_ns':saved['started']['wall_ns'],'completed_ns':saved['completed']['wall_ns'],'observations_verified':32})
assert objective_counts['observations']==4608
assert len(list((Q/'raw').glob('*/*.json*')))==144
control_values={str(outer):aggregate([row for job,row in control_rows if job['outer']==outer],'auc') for outer in SEEDS}
for outer in SEEDS:
    assert equal(control_values[str(outer)],control_saved['per_seed'][str(outer)]['B2']['auc'])
    for arm in ['A0','R']:
        assert equal(primary_values[str(outer)][arm],control_saved['per_seed'][str(outer)][arm]['auc'])
control_contrasts={}
for left,right in [('B2','A0'),('R','B2')]:
    def value(outer: int, arm: str) -> float:
        """Use the fixed control or preserved primary value without rerunning comparators."""
        return control_values[str(outer)] if arm=='B2' else primary_values[str(outer)][arm]
    key=f'{left}-{right}'
    result=contrast([value(s,left)-value(s,right) for s in SEEDS])
    contrast_max_error=max(contrast_max_error,compare_contrast(result,control_saved['contrasts'][key]))
    control_contrasts[key]=result
assert control_saved['resources']['objective_calls']==4608 and control_saved['resources']['subprocess_executions']==9216
assert control_saved['resources']['recorded_invocation_reference_objective_calls']==1536
assert not control_saved['resources']['infrastructure_errors'] and not control_saved['resources']['uncheckpointed_attempt_ids']
assert not blocked_calls
print('Control: 4608 objective values and 1536 reference values recomputed; all metrics and contrasts pass',flush=True)
for filename, expected in hashes.items():
    assert filehash(Path(filename))==expected,filename
assert filehash(archive_path)==archive_hash
resources=primary_saved['resources']
report={
 'schema':'investigation16.independent_final_review.v1','status':'PASS','started_ns':started,'completed_ns':time.time_ns(),'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'primary_freeze_sha256':B.digest(primary_freeze),'control_freeze_sha256':B.digest(control_freeze),'primary_analysis_sha256':B.digest(primary_saved),'control_results_sha256':B.digest(control_saved),
 'checks':{'frozen_analysis_recomputed_exactly':True,'independent_primary_and_control_contrasts_match':True,'maximum_arithmetic_difference':contrast_max_error,'selected_sources_and_validation_choices_verified':24,'representative_verified':representative,'all192_response_ids_unique':True,'all194_attempts_retained':True,'transport_failures_and_uncertain_remote_billing':2,'all_registered_seeds':SEEDS,'source_archive_entries':97,'source_archive_bytes_match_workspace':True,'source_archive_sha256':archive_hash,'main_numeric_cache_binding_verified':14592,'main_invalid_partial_numeric_rows_retained':numeric['invalid_partial_rows_verified'],'main_numeric_failure_count':0,'control_rows_values_and_metrics_verified':144,'evidence_files_hashed_before_after':len(hashes),'evidence_hash_manifest_digest':B.digest(hashes),'all_read_evidence_bytes_unchanged':True},
 'scope':{'candidate_executions':0,'model_calls':0,'network_calls':0,'key_loads':0,'frozen_or_raw_edits':0,'blocked_execution_calls':blocked_calls,'reused_structural_checks':'A.read_bundle/A.summarize and C.preflight; main paired results additionally recomputed independently from audit rows; no C.run or persist calls'},
 'chronology':{'generation_completed_ns':barriers['generation_frozen']['completed_ns'],'latest_response_completed_ns':max(r['completed_ns'] for r in slot_rows),'first_validation_cache_ns':min(v['computed_clock']['wall_ns'] for v in cache_by_split['validation']),'earliest_selection_ns':min(selection_times),'latest_selection_ns':max(selection_times),'global_selections_completed_ns':barriers['selections_frozen']['completed_ns'],'first_primary_audit_cache_ns':min(v['computed_clock']['wall_ns'] for v in cache_by_split['audit']),'primary_audit_completed_ns':barriers['audit_results']['completed_ns'],'first_control_started_ns':min(r['started_ns'] for r in control_row_receipts),'last_control_completed_ns':max(r['completed_ns'] for r in control_row_receipts)},
 'primary_per_seed_auc':primary_values,'primary_contrasts':primary_contrasts,'primary_arms':{arm:{'auc':primary_saved['arms'][arm]['auc'],'candidate_eligibility':primary_saved['arms'][arm].get('candidate_eligibility'),'source_invalid_fraction':primary_saved['arms'][arm]['generation']['source_invalid_fraction'],'execution_trajectory_invalid_fraction':primary_saved['arms'][arm]['generation']['trajectory_invalid_fraction'],'audit_fallback_fraction':primary_saved['arms'][arm]['deployment']['fallback_fraction'],'known_usage':primary_saved['arms'][arm]['known_usage']['usage']} for arm in ARMS},
 'primary_resources':{key:resources[key] for key in ['allocated_proposal_slots','completed_responses','transport_attempts','transport_failures','possible_remote_completion_or_duplicate_billing_attempts','response_usage','finish_reasons','logical','physical_cache','cache_accesses','cache_hits','cache_misses','physical_cache_rows','cache_rows_without_miss_event','repeated_cache_miss_events','normalization_unique_design_calls','accounting_limits']},
 'selected_sources':selected_records,'all_response_slots':slot_rows,'control_per_seed_auc':control_values,'control_contrasts':control_contrasts,'control_resources':control_saved['resources'],'control_row_verifications':control_row_receipts,
 'integrity_work':{'this_review':{'control_observation_objective_calls':objective_counts['observations'],'reference_objective_calls':objective_counts['references'],'total_objective_calls':sum(objective_counts.values()),'reference_cache_misses':B._reference_scale.cache_info().misses,'reference_cache_hits':B._reference_scale.cache_info().hits},'previous_primary_numeric_verification':numeric['integrity_objective_calls'],'separate_from_scientific_budget':True,'main_observation_values_not_reexecuted_in_this_review':True},
 'timing':{key:primary_saved['timing'][key] for key in ['generation','selection','audit']},
 'limitations':['All P1 and supplementary contrasts are exploratory with six outer seeds sharing one audit panel; bootstrap intervals are fragile and unadjusted for multiplicity.','R-C and W-R have the frozen 5/1 precedence imbalance; neither observed direction nor source-only review proves a provider drift effect.','R/W combine raw trajectory feedback and aggregate training AUC; W-R changes breadth and number of rounds. No isolated memory/depth benefit is established.','Main full numerical verification was not repeated: its PASS is bound to every current cache row/hash and preserved reconstructed metric; control numerical verification is new and separately counted.','No candidate, provider or operating-system sandbox security guarantee is added by this audit.','Physical accounting covers persisted work; uncertain remote billing and incompletely instrumented normalization rebuilds remain unknown.','This result does not amend EXP-15, establish novelty, demonstrate additional recursion depth or measure amortization.'],
 'verification_program_path':'/tmp/investigation16_final_review.py','verification_program_sha256':filehash(Path(__file__)),
}
out=ROOT/'production/INDEPENDENT_FINAL_REVIEW.json'
assert not out.exists()
out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'PASS','main_mean_auc':{a:primary_saved['arms'][a]['auc']['mean'] for a in ARMS},'primary_contrasts':primary_contrasts,'control_mean_auc':math.fsum(control_values.values())/6,'control_contrasts':control_contrasts,'integrity_objective_calls':dict(objective_counts),'input_files_unchanged':len(hashes),'max_error':contrast_max_error},ensure_ascii=False),flush=True)
