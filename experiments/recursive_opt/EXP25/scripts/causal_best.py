"""Best valid score per run among strictly causal candidates (causal_fraction == 1: no look-ahead on any signal)."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'signal'))
import whitebox as W  # noqa: E402

campaign = Path(sys.argv[1])
out = {}
for run in sorted(p for p in campaign.iterdir() if p.is_dir()):
    rows = []
    if (run / 'evaluations.jsonl').exists():
        for line in (run / 'evaluations.jsonl').read_text().splitlines():
            r = json.loads(line)
            rows.append((r.get('valid_score') or 0.0, r.get('causal_fraction') or 0.0))
    else:
        seen = set()
        for line in (run / 'candidate_history.jsonl').read_text().splitlines():
            c = json.loads(line).get('candidate')
            if c and c['solution'] not in seen:
                seen.add(c['solution'])
                m = W.evaluate(c['solution'])[0]
                rows.append((m.get('valid_score', 0.0), m.get('causal_fraction', 0.0)))
    causal = [v for v, c in rows if c == 1.0]
    out[run.name] = {'best_valid_any': round(max(v for v, _ in rows), 4), 'best_valid_causal': round(max(causal), 4) if causal else None,
                     'causal_candidates': f'{len(causal)}/{len(rows)}'}
    print(run.name, out[run.name], flush=True)
(campaign / 'causal_analysis.json').write_text(json.dumps(out, indent=1) + '\n')
