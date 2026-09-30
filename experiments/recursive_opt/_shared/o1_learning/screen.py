import json
import random
import hashlib
from pathlib import Path
from experiments.recursive_opt._shared.o1_learning import study
from experiments.recursive_opt._shared.optimizer_discovery.phase0 import _load_key

_load_key()
variants = study.variants()
random.Random(19019).shuffle(variants)
paths = [
    "experiments/recursive_opt/_shared/o1_learning/study.py",
    "opto/features/recursive_opt/spec.py",
    "opto/features/recursive_opt/traces.py",
    "opto/trace/bundle.py",
    "opto/trainer/loader.py",
    "opto/trainer/sampler.py",
    "opto/trainer/search_template.py",
    "opto/trainer/algorithms/classical_algorithms.py",
    "opto/trace/io/telemetry_session.py",
    "opto/trace/io/sysmonitoring.py",
]
manifest = {
    "id": "EXP-19",
    "stage": "screen",
    "variants": variants,
    "seed": 19021,
    "calls": 4,
    "order": [v["id"] for v in variants],
    "source_hashes": {
        p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths
    },
    "pilot_separate": 19001,
    "test_seeds": [19031, 19043, 19059],
    "target": 0.9,
    "bootstrap": {"seed": 19090, "draws": 10000, "unit": "outer_seed"},
    "model": study.MODEL,
    "concurrency": 1,
    "replay_workers": 8,
}
study.persist(study.ROOT / "manifest.json", manifest)
for config in variants:
    report = study.run(config, 19021, "screen", 4)
    print(
        json.dumps(
            {
                "id": config["id"],
                "valid": report["valid"],
                "calls": report["calls"],
                "curve": report["validation_curve"],
                "hit": report["first_hit"],
                "error": report["error"],
            }
        ),
        flush=True,
    )
