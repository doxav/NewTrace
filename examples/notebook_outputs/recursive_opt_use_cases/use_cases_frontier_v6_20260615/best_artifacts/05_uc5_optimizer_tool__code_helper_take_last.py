def _baseline_take_last(self, n, k):
    hard = [i for i in range(n) if i % 3 == 0]
    picked = hard[:k]
    if len(picked) < k:
        remaining = [i for i in range(n) if i not in set(picked)]
        picked.extend(remaining[: k - len(picked)])
    return picked
