Parallel Trace comparison — PARTIAL_OR_DIAGNOSTIC

Snapshot: 2026-09-27T20:39:38.147816+00:00. Each worker targets 100 solution HTTP attempts.

| Task | Arm | State | Calls / outcomes | Best score | Gain | Relative gain | AUC / 100 | Valid / invalid | Policies deployed / proposed | Reported USD | Calls missing cost |
|---|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| prism | TRACE-RECURSIVE | incomplete | 98 / 98 | 26.439704 | 4.5480817 | 0.20775444 | unmeasured | 79 / 19 | 5 / 6 | 0.31722691 | 1 |
| prism | TRACE-FIXED | complete | 100 / 100 | 26.364386 | 4.4727642 | 0.20431397 | 25.970451 | 70 / 30 | 0 / 0 | 0.10296214 | 0 |
| signal_processing | TRACE-RECURSIVE | incomplete | 52 / 52 | 0.55581322 | 0.056764599 | 0.11374563 | unmeasured | 49 / 3 | 3 / 4 | 0.19483223 | 1 |
| signal_processing | TRACE-FIXED | complete | 100 / 100 | 0.60629172 | 0.1072431 | 0.2148951 | 0.59149633 | 98 / 2 | 0 / 0 | 0.17462593 | 0 |

AUC / 100 is the mean best-so-far score across all 100 attempts; unavailable for incomplete runs. Live usage is a lower bound through the latest recorded outcome. Costs are provider-reported; missing costs are unknown.

| Task | Completed contrast | Final score difference | Gain difference | Relative-gain difference | AUC difference |
|---|---|---:|---:|---:|---:|
| Unmeasured | No completed matched pair | unmeasured | unmeasured | unmeasured | unmeasured |

Positive differences favor TRACE-RECURSIVE for these observed runs only.

| Task | Arm | Native best-solution metrics |
|---|---|---|
| prism | TRACE-RECURSIVE | {"combined_score": 26.4397037976586, "execution_time": 0.008398999338564665, "max_kvpr": 25.519703797658597, "success_rate": 0.92} |
| prism | TRACE-FIXED | {"combined_score": 26.36438629294493, "execution_time": 0.003917979157489279, "max_kvpr": 25.444386292944927, "success_rate": 0.92} |
| signal_processing | TRACE-RECURSIVE | {"accuracy_score": 0.8504266900176463, "avg_error": 0.4427019001688599, "combined_score": 0.5558132165351044, "composite_score": 0.5330397176962121, "correlation": 0.8504266900176463, "efficiency_score": 1.0, "execution_time": 0.0012955665588378906, "false_reversals": 61.2, "lag_error": 0.14557541402102409, "noise_reduction": 0.31189677403503613, "output_length": 91.0, "overall_score": 0.5558132165351044, "responsiveness_score": 0.8729237619459311, "runs_successfully": 1.0, "slope_changes": 76.8, "smoothness_score": 0.2066115702479339, "success_rate": 1.0} |
| signal_processing | TRACE-FIXED | {"accuracy_score": 0.8635641920376097, "avg_error": 0.43263542279208184, "combined_score": 0.606291718262855, "composite_score": 0.5834290425372086, "correlation": 0.8635641920376097, "efficiency_score": 1.0, "execution_time": 0.018358802795410155, "false_reversals": 35.2, "lag_error": 0.12611871461306262, "noise_reduction": 0.33762080116197113, "output_length": 91.0, "overall_score": 0.606291718262855, "responsiveness_score": 0.8880058443426212, "runs_successfully": 1.0, "slope_changes": 40.2, "smoothness_score": 0.3322259136212624, "success_rate": 1.0} |

PRISM’s max_kvpr is inverse mean maximum pressure over successful cases. Interpret it alongside success_rate.

One stochastic run per task and arm cannot establish statistical reliability. Workers run concurrently: shared provider load and local CPU contention can affect latency, cost and timeout outcomes. Equal solution attempts do not equal total compute. Incomplete pairs cannot establish whether meta-optimization helps.

Detailed trajectories, policy events, native metrics, tokens, costs and per-worker evidence locations are in results.json.
