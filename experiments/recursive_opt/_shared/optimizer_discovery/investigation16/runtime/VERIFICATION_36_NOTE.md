# Exact command for the earlier 36-test verification

This note transcribes the previously returned execution-tool output. It is not
a newly executed test run and is not an original stdout log file.

```bash
/tmp/phase0-venv/bin/python -m pytest -q \
  artifacts/optimizer_discovery/investigation16/production/test_verify_numerics.py \
  artifacts/optimizer_discovery/investigation16/production/test_analysis.py
```

Original result: `36 passed in 3.24s`.

The invocation did **not** contain `--disable-socket` or `--allow-hosts`.
Its original output was returned by the execution tool for session 35617; no
separate log file was saved. This note must not be described as proof that those
network-isolation flags were used. The tests have not been rerun for this note.
