# EXP26 — evidence locations and campaign commands

Run data are local (`experiments/**/*.jsonl` and run JSON are ignored); aggregates are summarised in
[RESULTS.md](../RESULTS.md).

Per native run (`<campaign>/<arm>_s<seed>/`): `transcripts.jsonl` (every call: role, messages, response — new in
EXP26), `events.jsonl`, `evaluations.jsonl`, `sources/`, `calls.jsonl`, `labels.json`, `report.json`, `summary.json`,
`run_manifest.json` (includes `label_packages`, plan fingerprint, coevolution SHA-256). Stock runs add `labels.json`
with the `fallback` flag. Stock label draws: `<campaign>/stock_labels/{draws.jsonl,labels_s<seed>.json}`.

## Commands (from EXP26, EXP22 venv, `TRACE_ROOT=~/code/Trace`, `OPENROUTER_API_KEY` set)

```sh
C=results/runs_$(date -u +%Y%m%dT%H%M%S); PY=../EXP22/.venv/bin/python
for s in 42 43 44; do $PY -I scripts/gen_stock_labels.py --seed $s --out $C/stock_labels; done   # before any run
# three waves, one per seed, each running all four arms concurrently (PROTOCOL.md scheduling)
for s in 42 43 44; do
  $PY -I scripts/run_signal.py --arm native --seed $s --out $C/native_s$s &
  $PY -I scripts/run_signal.py --arm native_pkg --seed $s --out $C/native_pkg_s$s &
  $PY -I scripts/run_signal.py --arm native_stocklabels --seed $s --labels $C/stock_labels/labels_s$s.json --out $C/native_stocklabels_s$s &
  $PY -I scripts/run_evox_stock.py --seed $s --out $C/evox_stock_s$s &
  wait
done
$PY -I scripts/analyze.py $C
```

[Results](../RESULTS.md)
