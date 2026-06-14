"""Reviewable regression runner over the recursive_opt examples (no notebooks).

Notebooks stay the demo surface; THIS file is the diff-able, pytest-able truth
of "do the examples still work end-to-end on the bounded real adapter". Run:

    python examples/recursive_opt_review_regression.py

Requires Trace-Bench installed (the examples use the eval-only REAL adapter; no
stubs by design). Each row reports the example, the family, the final score and
the memory summary so regressions show up as one-line diffs.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
import sys
from pathlib import Path
import tempfile
from typing import Any, Dict, Generator, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opto.features.recursive_opt.tracebench import ensure_eval_only_task_adapter

import examples.recursive_opt_example_A_learn_setup as exA
import examples.recursive_opt_example_C_learn_capability as exC


@contextmanager
def _isolated_workdir() -> Generator[Path, None, None]:
    """Run examples in a temporary cwd so their relative mem_* stores are stable."""
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="recursive_opt_review_") as tmp:
        os.chdir(tmp)
        try:
            yield Path(tmp)
        finally:
            os.chdir(previous)


def run_review_regression() -> List[Dict[str, Any]]:
    ensure_eval_only_task_adapter(require=True)
    rows: List[Dict[str, Any]] = []

    with _isolated_workdir():
        best_cfg, mem_a = exA.run_offline("internal:multi_param")
        rows.append({"example": "A", "family": "internal:multi_param",
                     "artifact": str(best_cfg)[:120], "memory": mem_a.summary()})

        impl_c, objs_c, mem_c, _ = exC.learn_capability(
            n_tasks=1,
            candidate_impls=[exC.CANDIDATE_IMPLS[0], exC.CANDIDATE_IMPLS[2]],
        )
        rows.append({"example": "C", "family": "internal:multiobjective_gsm8k",
                     "score": objs_c.get("accuracy", 0.0) - 0.5 * objs_c.get("cost", 0.0),
                     "artifact": str(impl_c)[:120], "memory": mem_c.summary()})
    return rows


def main() -> None:
    for row in run_review_regression():
        print(f"[{row['example']}] family={row['family']} "
              f"score={row.get('score', 'n/a')} episodes={row['memory']['episodes']} "
              f"priors={row['memory']['priors']}")


if __name__ == "__main__":
    main()
