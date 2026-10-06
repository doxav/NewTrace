"""EXP27 harness checks: prompt variants change only their factor; the evaluator flags look-ahead; the summary counts."""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / 'scripts'))
A = pytest.importorskip('prompt_ablation')

PROMPT = """# Current Solution Information
- Main Metrics: guided_score=0.5000
# Current Solution

## IMPORTANT: YOU MUST FOLLOW THE FOLLOWING IN YOUR GENERATION. DIVERGE-TEXT

## Program Information
guided_score: 0.5000
Score breakdown:
  - combined_score: 0.5000
  - valid_score: 0.5000
  - causal_fraction: 1.0000
  - fallback_signals: 0

```python
import numpy as np
```

## Evaluator Feedback
Valid score 0.5 per signal...

## projection
wrapped with a per-call fallback onto the baseline

# Task
Suggest improvements to the program that will improve its COMBINED_SCORE."""


def test_no_causal_cue_removes_only_causal_fraction():
    out = A.no_causal_cue(PROMPT)
    assert 'causal_fraction' not in out
    assert out == PROMPT.replace('\n  - causal_fraction: 1.0000', '')


def test_stock_like_strips_diagnostics_keeps_label_code_and_uses_stock_task():
    out = A.stock_like(PROMPT)
    for gone in ('causal_fraction', 'valid_score', 'fallback_signals', 'guided_score', 'Evaluator Feedback', 'per-call fallback'):
        assert gone not in out, gone
    assert 'DIVERGE-TEXT' in out and '```python\nimport numpy as np\n```' in out and 'combined_score: 0.5000' in out
    assert 'maintains diversity' in out and out.count('# Task') == 1
    assert A.parent_of(out) == A.parent_of(PROMPT) == 'import numpy as np'


def test_evaluator_flags_lookahead():
    initial = A.W.INITIAL.read_text()
    lookahead = initial + '''

def enhanced_filter_with_trend_preservation(x, window_size=20):
    from scipy.signal import butter, filtfilt
    b, a = butter(2, 0.05)
    return filtfilt(b, a, np.asarray(x, dtype=float))[window_size - 1:]
'''
    assert A.score(initial)['causal_fraction'] == 1.0
    assert A.score(lookahead)['causal_fraction'] < 1.0


def test_summarize_counts(tmp_path):
    parent = {'valid_score': 0.5}
    rows = [{'run': 'r', 'prompt': 0, 'variant': 'T0_recorded', 'applied': True, 'scipy': True, 'parent': parent,
             'child': {'valid_score': 0.6, 'causal_fraction': 0.2}},
            {'run': 'r', 'prompt': 0, 'variant': 'T0_recorded', 'applied': True, 'scipy': False, 'parent': parent,
             'child': {'valid_score': 0.4, 'causal_fraction': 1.0}},
            {'run': 'r', 'prompt': 0, 'variant': 'T0_recorded', 'applied': False, 'parent': parent}]
    (tmp_path / 'completions.jsonl').write_text('\n'.join(map(json.dumps, rows)) + '\n')
    s = A.summarize(tmp_path)['T0_recorded']
    assert (s['n'], s['applied'], s['lookahead'], s['scipy'], s['beats_parent'], s['best_valid']) == (3, 0.667, 0.5, 0.5, 0.5, 0.6)
