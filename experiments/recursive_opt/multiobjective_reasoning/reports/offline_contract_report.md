# Experiment 0 offline contract report

Overall: **PASS**

- runtime_tree_sha256: `37072c1364a02c277a677bf43ad8132a32a9f233488c80cd2b6bf1a7e344f33e`
- registry_sha256: `f1a94b02c94607c2d22e6f10bce25b10d9b642814915ec01c72765280be26fa4`
- provider/network calls: none (deterministic local provider)

## Assertions

| assertion | result |
|---|---|
| same_normalized_base_spec | pass |
| same_module_ref | pass |
| same_evaluator_ref | pass |
| same_dataset_refs | pass |
| same_initial_artifact | pass |
| same_objective | pass |
| same_role_configuration | pass |
| same_runtime_tree_sha256 | pass |
| same_registry_sha256 | pass |
| one_forward_per_evaluator | pass |
| evaluator_receives_exact_output | pass |
| holdout_inaccessible_during_optimization | pass |
| usage_attributed_once | pass |
| output_persistence_and_resume | pass |
| gepa_does_not_receive_holdout | pass |
| trace_real_candidate | pass |
| gepa_real_candidate_proposed | pass |
| validation_gate_rejects_harmful | pass |
| ungated_harmful_candidate_is_observable | pass |
| candidate_accounting_consistent | pass |
