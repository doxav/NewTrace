# Prompt 18-R3E restarted-main stop decision

## Corrected runner and external gates

The Experiment-0-only correction was committed as
`dc7dc92e19386d3ed2f18bcea83209f08bbc554b`. Required workflow run
`32773567177`, job `97579313398`, completed successfully. The versioned main
lock was then committed in branch HEAD
`0a0757eed317e071286384113ca086e2f1678eb4`; its final branch workflow run
`32773729820`, job `97579836564`, also completed successfully.

The locked control-plane runtime digest remained
`ba4836d9f43cffcd0271086932745b270d75478b5287a7d8100be4928b623cbc`.
The corrected Experiment-0 source digest is
`2df697378ec86089368e804f47a412644ec56ad8f1d54024e3678daedcbe0535`.
The preregistration, P0, parser, data, model profiles, objective, safety
constraint, seeds, budgets, arms, and statistics remained unchanged.

## Restart result before infrastructure failure

The restarted main matrix used the new
`main_after_scientific_vs_infrastructure_gate_fix` namespace and did not reuse
the preserved pre-fix A result.

The first unit, seed 0 / budget 6 / arm A, completed normally:

- status `success`, valid `true`, execution completed `true`;
- accuracy `0.9166666666666666`;
- invalid rate `0.0` and safety passed `true`;
- forward-token ratio `1.0448809323400696`;
- latency `11.991922816048222` seconds/example;
- 48 forward calls, 24 evaluator runs, 12,809 accounted tokens;
- no optimizer calls, proposals, evaluations, empty-text responses, or retries;
- token-price proxy `$0.00146112`; provider monetary cost unavailable;
- every infrastructure check, including exact resume with no new call, passed.

## Blocking infrastructure failure

The second unit, seed 0 / budget 6 / arm B (Trace), did not produce a canonical
result or checkpoint. During a concurrent forward evaluation, LiteLLM reported
non-retryable OpenRouter transport failures:

- `[Errno 104] Connection reset by peer`;
- `Server disconnected without sending a response.`

The confirmed call path was:

`components.py: CompoundReasoningModule._call`
→ recursive-opt `_GuardedRoleClient`
→ `runmode.py`
→ `opto/utils/llm.py`
→ LiteLLM/OpenRouter.

The process then remained blocked in the evaluator thread pool for more than
12 hours. Because the blocked provider threads did not return control to the
budget guard, the frozen per-run `wall_time_s=7200` check could not finalize a
canonical budget-exhausted result. The process was interrupted to prevent new
calls. Only B's raw spec, normalized spec, and resolved plan exist; no B
`result.json` exists. No B metrics, candidate counts, or efficacy conclusions
are inferred from console progress.

This is a genuine infrastructure failure. It is independent of the corrected
scientific-safety classification and is exactly the class of failure on which
the frozen protocol requires an immediate stop.

## Consequences

- Main completion: 1/40 canonical units.
- The complete matrix was not produced and no paired main statistics are valid.
- Quality, efficiency, engine efficacy, and global safety conclusions are not
  available from this incomplete matrix.
- The candidate-trajectory audit was not run because it is sequenced after
  40/40 infrastructure-complete units.
- The completed A result and incomplete B infrastructure evidence are preserved
  and must not be statistically combined with a future restarted matrix.

`BLOCKED_INFRASTRUCTURE`
