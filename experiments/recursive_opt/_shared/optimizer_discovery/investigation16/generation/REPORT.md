# G1 — token-cap intervention and failure-mechanism audit

G1 completed all **24 registered response slots and six-task evaluations**. The
registered reliability rule selects **32,000 completion tokens** for later
diagnostics: 10/12 candidates are eligible at 32,000 versus 8/12 at 8,000; length
finishes are 0/12 versus 1/12. This is an exploratory reliability observation,
not proof of improved task performance or of recursive feedback's usefulness.

Every 32,000-cap response actually stopped below 8,000 completion tokens; the
largest used **6,581**. Consequently, G1 does **not** establish that extra realized
token consumption rescued an otherwise identical response. Generation and routing
are stochastic. Two more usable samples under the higher configured cap can
justify the predeclared engineering choice without proving its causal mechanism.

The analysis reads complete preserved evidence, performs AST inspection and
recomputes metrics from recorded observations. It makes no model calls and does
not execute or repair candidate source. See `analysis.py`, its eight tests, and
`analysis_results.json` for every slot, pair, source hash, diagnostic, and input
content hash. Earlier EXP-15 conclusions and all G1 source remain unchanged.

## Registered comparison and all paired outcomes

Six request-seed blocks (16001–16006), two fixed contexts, two caps, one completed
response per cell. Context I uses the unchanged independent-generation prompt and
seed. Context L uses a fixed previous validation-selected EXP-15 program with
fresh G1 sparse training feedback and the seed as previous attempt. L is a probe
of an iterative **prompt**, not a new iterative search run. Each candidate is
allocated six fresh balanced family/dimension trajectories, each at B=32.

Other settings are identical within each pair: OpenRouter model
`deepseek/deepseek-v4-flash-0731`, temperature 0.6, top_p 1.0, native low reasoning,
timeout 300 seconds, concurrency 1, no cache or empty-response retry. Request seeds
are hints; they are not established deterministic common random numbers.

| Context | Cap | Eligible / responses | Source screen passes | Length finishes | Other ineligible candidates |
|---|---:|---:|---:|---:|---:|
| I | 8,000 | 4/6 | 5/6 | 1 | 1 execution failure |
| I | 32,000 | 5/6 | 5/6 | 0 | 1 protocol rejection |
| L | 8,000 | 4/6 | 4/6 | 0 | 2 protocol rejections |
| L | 32,000 | 5/6 | 5/6 | 0 | 1 protocol rejection |
| Combined | 8,000 | 8/12 | 9/12 | 1 | 3 |
| Combined | 32,000 | 10/12 | 10/12 | 0 | 2 |

The 12 binary paired eligibility differences (32,000 minus 8,000) are
`[1,0,0,0,0,0,0,0,0,1,0,0]`, ordered by block and I then L. There are two gains,
zero losses, eight pairs eligible under both caps, and two under neither. The
sample eligibility difference is +16.7 percentage points. Six blocks and two
fixed contexts do not establish a general reliability effect or near-perfect
validity at the selected cap.

| Block | Context | 8,000 outcome | 32,000 outcome | Eligibility difference |
|---:|---|---|---|---:|
| 16001 | I | missing source / length | eligible | +1 |
| 16001 | L | eligible | eligible | 0 |
| 16002 | I | exception on two trajectories | protocol: `__all__` | 0 |
| 16002 | L | protocol: numpy and private names | protocol: local `dir` | 0 |
| 16003 | I | eligible | eligible | 0 |
| 16003 | L | eligible | eligible | 0 |
| 16004 | I | eligible | eligible | 0 |
| 16004 | L | eligible | eligible | 0 |
| 16005 | I | eligible | eligible | 0 |
| 16005 | L | protocol: `__name__`; incompatible API | eligible | +1 |
| 16006 | I | eligible | eligible after transport retry | 0 |
| 16006 | L | eligible | eligible | 0 |

`ORDER_NOTE.md` documents a scheduling limitation: I always runs 8,000 first and L
always runs 32,000 first. Pooled first/second order is balanced; each context's
cap contrast is confounded with order. Context order alternates between blocks.
Upstream providers also vary. Neither limitation is corrected retrospectively.

## Root mechanisms supported by the preserved failures

1. **One direct truncation event:** 16001/I/8000 exhausted exactly 8,000 completion
   tokens, of which 7,795 were reported reasoning tokens, and produced no usable
   source. This establishes lost generation opportunity in that slot. Its paired
   32,000 response was a different stochastic sample using only 6,581 tokens.
   Earlier EXP-15's 16 truncated responses remain stronger evidence that the
   original cap was sometimes binding; G1 does not reproduce that failure rate.
2. **A source-screen false positive for the stated prohibited operations:**
   16002/L/32000 uses `dir` solely as a local comprehension variable, at line 122:
   `x_quad = [b + dir * max_step for b, dir in zip(best, direction)]`.
   The AST contains no call to the builtin `dir`, disallowed import, private
   attribute, or other forbidden operation. The frozen screen rejects every
   `ast.Name('dir')`, including Store and locally bound Load. Its exact source hash
   is `068c006eaa6ba610ccad85d15245f7786f365789a7dd09ba36d97ed77ebc0fd2`.
   This remains invalid under G1's declared validator; it is not evidence that the
   model attempted introspection. Separate synthetic tests reproduce this lexical
   rejection without repairing or running the candidate.
3. **Another coarse source-name restriction:** 16002/I/32000 assigns
   `__all__ = ["propose"]` and is rejected for that dunder name. Export metadata is
   not itself filesystem or environment access. The frozen gate is intentionally
   broader than prohibited operations described in the generation prompt. Its
   post-gate execution validity is unknown and is not imputed as valid.
4. **Genuine contract noncompliance also remains:** 16002/L/8000 imports numpy and
   uses private names/attributes. 16005/L/8000 supplies two parameters with
   dictionary-shaped bounds/results instead of the required three-parameter API,
   and also includes `__name__`. Both stop normally, well below the cap. Increasing
   tokens cannot be presumed to fix these instruction-following errors.
5. **A source-valid program has conditional runtime failure:** 16002/I/8000
   completes four trajectories, but throws on Rosenbrock/2 after six observations
   and Rosenbrock/4 after sixteen. Recorded observations and failed proposal indices
   are preserved. AST/source inspection identifies `statistics.stdev` at line 154
   while the source imports only `random` and `math`; reaching that fallback would
   raise NameError. Recorded rows expose only typed `exception`, not a traceback,
   so this is a concrete candidate-code explanation consistent with the failures,
   not a separately executed causal reproduction. All six rows remain ineligible
   as a panel, despite the four valid trajectories.

Five of six ineligible candidates finish with `stop`; only one finishes with
`length`. There are no syntax-error outcomes and no recorded candidate timeouts
in this sample. Thus **the token cap is not a sufficient diagnosis**. It also
cannot explain why useful feedback might fail to beat independent generation:
G1 neither changes the sparse feedback nor adds the missing anytime instruction.

The next prompt version can state the complete lexical restrictions symmetrically
across arms before freezing new calls. G1's validator and rows must remain intact.
A later scope-aware screen would require independent tests and a new evaluator
version; counting these original rejected programs as retrospectively eligible
would be invalid. Stronger security confinement is not established by either
version of the source screen.

## Resources, receipt coverage, and interruption

| Resource | 8,000 cap | 32,000 cap | Total |
|---|---:|---:|---:|
| Completed response slots | 12 | 12 | 24 |
| Transport attempts | 12 | 13 | 25 |
| Prompt tokens | 30,024 | 30,024 | 60,048 |
| Completion tokens | 48,953 | 50,781 | 99,734 |
| Reasoning tokens, included in completion | 35,313 | 36,621 | 71,934 |
| Total tokens | 78,977 | 80,805 | 159,782 |
| Reported response cost, USD | 0.010260145 | 0.010471155 | 0.020731300 |
| Logical objective allocation | 2,304 | 2,304 | 4,608 |
| Actual objective calls | 1,686 | 1,920 | 3,606 |
| Unused objective allocation | 618 | 384 | 1,002 |
| Candidate subprocess executions | 3,374 | 3,840 | 7,214 |

Usage and generation receipts cover all 24 completed responses; their cost totals
agree. Reasoning consumes about 72.1% of reported completion tokens and must not
be added a second time. Providers are Sail Research (22), CoreWeave (1), and
Inceptron (1). Actual requests and SDK responses use the exact frozen model ID;
receipts separately report the canonical identifier
`deepseek/deepseek-v4-flash-20260731`. These distinct metadata fields are preserved.

The user suspended the machine during G1. **Timing is contaminated by that
suspension and cannot support an endpoint-latency comparison.** Raw response
durations, monotonic attempt durations, and wall timestamps remain available.
The final slot 16006/I/32000 has a recorded first-attempt transient timeout, then
a successful second attempt with its own response ID and provider receipt. Its
elapsed wall interval is about 9,001 seconds, which is not model processing time.
There are 24 unique completed response IDs, not 25 scientific proposals. The
failed attempt explicitly preserves possible remote completion/duplicate billing;
its usage is unknown. USD 0.020731300 is the total for the 24 known receipts, not
a guarantee of complete account billing. No completed response was replaced.

## Performance and implications

For transparency, the eight pairs eligible under both caps have mean task-AUC
difference **+0.031275** (32,000 minus 8,000; positive is worse). Five differences
are negative and three positive; one large deterioration dominates the mean.
This conditions on both programs passing and omits no failed row from the main
reliability analysis. It is not an overall deployment-policy comparison, and was
not used to select the cap. G1 supplies no evidence that the higher cap improves
task performance, let alone the A2–A1 contrast.

Use 32,000 as the predeclared resource margin for the next matched probes, retain
the same low-reasoning model and strict slot accounting, clarify the source-name
restrictions for every arm, and isolate anytime instruction from actual trace
content. Increasing training examples or changing search strategy requires its
own contrast. No numerical gain for a future recursive run can be projected from
these 24 generation samples.

## Verification

`analysis.py` verified all 24 exact requests against the freeze, completed/model
identity, unique response IDs, raw extraction/source hashes, six task identities
and local seeds per slot, objective allocations, metric recomputation, retry
counts, receipt identity, and frozen source/environment versions. It rejects an
incomplete run before producing a cap comparison. Input canonical JSON hashes
are stored in `analysis_results.json`; raw bytes are not changed.

Commands run with `/tmp/phase0-venv/bin/python`:

```text
-m pytest -q artifacts/optimizer_discovery/investigation16/generation/test_analysis.py
  8 passed
-m black --target-version py313 <analysis.py> <test_analysis.py>
  passed
-m ruff check <analysis.py> <test_analysis.py>
  passed
-c "import runpy; runpy.run_path('artifacts/optimizer_discovery/investigation16/generation/analysis.py', run_name='__main__')"
  {'responses': 24, 'pairs': 12, 'recommended_cap': 32000}
```

The transport-uncertainty test first failed on the missing accounting field, then
passed after that field and the suspension limitation were added. Test data use
the explicit `G1_UNIT_ANALYSIS` namespace and do not execute prospective task
policies. No live calls, raw evidence edits, production changes, or commits were
made by this analysis task.
