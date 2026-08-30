# Prompt 18-R3F watchdog-child registry stop

## Classification

The first main-size Trace transport stress attempt stopped before any provider
request. The parent process had registered Experiment-0 components, but the
hard-watchdog process uses Python `spawn`; its fresh interpreter therefore had
no entry for `recursive_experiments.dataset.gsm8k@2`.

The exact child error was:

`ValueError: unregistered dataset ref 'recursive_experiments.dataset.gsm8k@2'`

This is valid infrastructure evidence about watchdog child initialization. It
has no optimizer-efficacy, transport-reliability, or scientific interpretation,
and incurred no model usage.

## Correction boundary

Each isolated watchdog child must initialize the existing Experiment-0
registry before invoking its target. The scientific protocol, runtime control
plane, model, request parameters, data, objective, seeds, budgets, concurrency,
and watchdog deadline remain unchanged. A real spawned-child regression test
must prove the v2 GSM8K dataset is registered without relying on parent memory.

The passing pre-correction live micro and its outputs remain preserved under
`reports/pre_watchdog_child_registry_fix/`. A fresh micro and stress gate are
required under the corrected experiment-source lock before main execution.
