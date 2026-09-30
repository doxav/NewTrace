#!/usr/bin/env bash
# Differential equivalence matrix: stock SkyDiscover EvoX vs recursive_opt coevolution (control plane v2).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
STOCK_PY="$HERE/../../EXP22/.venv/bin/python"
V2_PY=${V2_PY:-python}
TRACE_ROOT=${TRACE_ROOT:-$HOME/code/Trace}
status=0
while read -r name seed horizon meta; do
  out="$HERE/out/$name"; mkdir -p "$out"
  export EQUIV_SEED=$seed EQUIV_META=$meta
  "$STOCK_PY" -I "$HERE/run_stock.py" "$horizon" "$out" > "$out/stock.log" 2>&1 || { echo "$name: stock run failed"; status=1; continue; }
  PYTHONPATH="$TRACE_ROOT" "$V2_PY" "$HERE/run_v2.py" "$horizon" "$out" > "$out/v2.log" 2>&1 || { echo "$name: v2 run failed"; tail -3 "$out/v2.log"; status=1; continue; }
  result=$(python3 "$HERE/compare.py" "$out" | tail -1)
  echo "$name (seed=$seed horizon=$horizon): $result"
  [ "$result" = "EQUIVALENT" ] || status=1
done <<'CASES'
base 1234 60 invalid,greedy_refine,invalid,topk_diverge,late_raise,uniform_label,greedy_refine,topk_diverge,uniform_label
labels 7 100 uniform_label,topk_diverge,greedy_refine,topk_diverge,uniform_label
meta_failure 99 80 invalid,invalid,invalid,greedy_refine,invalid,invalid,invalid,topk_diverge
rollback_early 5 100 late_raise,uniform_label,late_raise,topk_diverge,greedy_refine
long 2024 200 topk_diverge,invalid,uniform_label,greedy_refine,late_raise,topk_diverge
CASES
exit $status
