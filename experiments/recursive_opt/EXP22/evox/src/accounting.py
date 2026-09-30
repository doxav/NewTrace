"""Shared, dependency-free accounting for observed OpenRouter attempts."""

from typing import Any


def usage_summary(records: list[dict[str, Any]], solution_limit: int | None = None) -> dict[str, Any]:
    """Aggregate observed HTTP attempts, returned tokens and reported cost."""
    if solution_limit is not None:
        if not isinstance(solution_limit, int) or isinstance(solution_limit, bool) or solution_limit < 1:
            raise ValueError('Solution usage boundary must be a positive integer')
        solutions = 0
        for index, row in enumerate(records):
            solutions += row['role'] == 'solution'
            if solutions == solution_limit:
                records = records[:index + 1]
                break
        if solutions != solution_limit:
            raise ValueError('Recorded solution calls do not reach the requested usage boundary')
    totals: dict[str, Any] = {'total_calls': len(records), 'roles': {}, 'input_tokens': 0, 'output_tokens': 0, 'cached_tokens': 0, 'reported_cost': 0.0, 'calls_missing_cost': 0}
    for row in records:
        role = row['role']
        totals['roles'][role] = totals['roles'].get(role, 0) + 1
        usage = row.get('usage') or {}
        totals['input_tokens'] += usage.get('prompt_tokens', 0)
        totals['output_tokens'] += usage.get('completion_tokens', 0)
        totals['cached_tokens'] += (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0)
        if usage.get('cost') is None:
            totals['calls_missing_cost'] += 1
        else:
            totals['reported_cost'] += usage['cost']
    return totals
