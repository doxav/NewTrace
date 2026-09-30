"""Compile every EXP23 alternative through normalize_spec -> compile_plan and persist the plans."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.control_plane import compile_all

if __name__ == '__main__':
    plans = compile_all(ROOT / 'configs')
    (ROOT / 'configs' / 'index.json').write_text(json.dumps({name: plan for name, plan in plans.items()}, indent=2, sort_keys=True, default=str) + '\n')
    for name, plan in plans.items():
        print(f'{name:45s} engines={plan["engines"]} units={plan["execution_units"]} fingerprint={plan["fingerprint"][:12]}')
