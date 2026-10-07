"""EXP28 harness checks (offline): both runners work in mock mode and hide the causal cue."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
ENV = {**os.environ, 'TRACE_ROOT': str(HERE.parents[2])}


def run(script, *args, tmp):
    out = subprocess.run([sys.executable, '-I', str(HERE / 'scripts' / script), *args, '--mock', '--out', str(tmp)],
                         env=ENV, capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads((tmp / 'summary.json').read_text())


@pytest.mark.parametrize('arm', ['vs_stagnation', 'vs_combine'])
def test_trainer_arm_mock(arm, tmp_path):
    summary = run('run_trainer_signal.py', '--arm', arm, '--steps', '8', tmp=tmp_path)
    assert summary['status'] == 'success' and summary['solution_calls'] == 7
    prompts = [json.loads(line) for line in (tmp_path / 'transcripts.jsonl').read_text().splitlines()]
    assert not any('causal' in p['user'] for p in prompts)
    if arm == 'vs_combine':
        assert 'combine' in summary['modes'] and any(p['variation'] for p in prompts)


@pytest.mark.parametrize('arm', ['brief', 'guard'])
def test_coevo_arm_mock(arm, tmp_path):
    summary = run('run_coevo_variation.py', '--arm', arm, '--seed', '42', '--horizon', '12', tmp=tmp_path)
    assert summary['status'] == 'success' and summary['solution_attempts'] == 12
    config = json.loads((tmp_path / 'raw_spec.json').read_text())['levels'][0]['engine']['config']
    assert ('meta_brief' in config) == (arm == 'brief') and ('diverge_guard' in config) == (arm == 'guard')
    assert 'causal_fraction' not in (tmp_path / 'transcripts.jsonl').read_text()
