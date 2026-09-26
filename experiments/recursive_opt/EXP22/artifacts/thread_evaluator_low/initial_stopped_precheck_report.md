STATUS: STOPPED_PRECHECK

Completed benchmark runs: **none**. Completed evaluator runs: **none**.
LLM calls and paid execution: **zero**. No scores, gains or costs were measured.

**FACT — source identity failure.** The requested Trace branch `recursive_opt`
resolves to `846580defe935195c1f5f39d6336079ef6ff1e10`. The specified checkout
is instead on `codex/exp17-parent-selection-exp18-memory-pareto` at
`7e701b40485b9880ccfbb64c1faaabb22401c294`, 40 commits ahead, with additional
relevant local changes. In particular, `opto/features/recursive_opt/spec.py`
is dirty. The local research audit independently describes these distinct
versions. A clean worktree of current HEAD would not resolve this source mismatch.
The protocol requires the Trace branch to resolve to `recursive_opt`; therefore
S0 failed before dependency setup, evaluator testing or provider requests.

**FACT — stock benchmark provenance.** SkyDiscover is detached at
`3f7a611fe83980970dd14f1d65c49e40de61b7df`. All six PRISM/Signal initial-program,
config and evaluator files match both the supplied SHA-256 values and this HEAD.
The untracked benchmark ZIP files were not used. No evolved program was loaded.

**MEASURED RESULT — engineering checks only.** Five source-gate and evidence-safety
unit tests pass. Lint and byte compilation pass. The recursive secret scan passes.
These checks do not demonstrate evaluator, provider, control-plane or algorithmic parity.

Evidence: [manifest](manifest.json), [source hashes](artifacts/source_hashes.json),
[stop record](artifacts/STOP.json), [secret scan](artifacts/secret_scan.json), and
the immutable audit directory named in the manifest. It includes sanitized
working-tree patches for relevant files and audit-interpreter metadata.

**INFERENCE.** Running the checkout as-is and labeling it the requested branch
would introduce an undocumented implementation difference. The appropriate
outcome is a diagnostic stop, not a negative result about Trace or EvoX.

**LIMITATION.** No benchmark environment, CP-A/CP-B implementation, compiled
Trace specs, transport smoke, golden evaluator checks, pilots, strict arms or
advanced arms exist. Required runtime tests and plots remain unperformed; there
is no measured trajectory to plot. The preregistration describes intended work,
not successful execution. Historical HotpotQA EXP22 artifacts in the Trace
repository refer to a different experiment and supply no results for this one.

Final decisions:

| Question | Answer |
|---|---|
| Does Trace meta-optimization improve over Trace fixed? | Not measured. |
| Does SkyDiscover EvoX improve over its fixed policy? | Not measured. |
| How close are they at equal solution-generation budget? | Not measured. |
| What is each arm's extra total-compute cost? | Not measured; no calls made. |
| Was the advanced phase warranted or run? | No: strict eligibility gates were never reached. |
| Did additional freedom help? | Not tested. |

Required resolution: establish the intended Trace source identity without losing
the existing dirty checkout. If the required branch remains `recursive_opt`, an
explicitly designated isolated execution checkout at its resolved revision is a
possible next setup. Substituting the later HEAD requires an explicit protocol
change. Neither choice was silently made. Restart all remaining gates after
resolving this input; this source audit alone cannot authorize full execution.

Exact verification commands, from `/home/xav/code/Trace-experiment0`:

```bash
python3 experiments/recursive_opt/EXP22/scripts/preflight.py
python3 -m unittest discover -s experiments/recursive_opt/EXP22/tests -v
ruff check experiments/recursive_opt/EXP22/scripts/preflight.py experiments/recursive_opt/EXP22/tests/test_preflight.py
python3 -m compileall -q experiments/recursive_opt/EXP22/scripts experiments/recursive_opt/EXP22/tests
```

Results respectively: exit 2 (`STOPPED_PRECHECK`, expected rejection); five tests
pass; lint passes; compilation passes. Initial lint reported SIM117 in the test
file; it was corrected and the entire targeted suite rerun successfully.
The final preflight rerun refreshes the recursive secret scan after documentation.
An intermediate scan detected a synthetic test credential that Python had constant-folded
into test bytecode. The fixture now constructs it at runtime inside a temporary
directory; bytecode was regenerated and the scan rerun. No real credential was stored.
An additional `git diff --check` found pre-existing trailing whitespace in the
workspace's unrelated dirty `opto/features/recursive_opt/spec.py`; that file was
left untouched. `git diff --check -- experiments/recursive_opt/EXP22` passes
(the new experiment files are untracked, so lint is their substantive style check).
No source checkout was modified, no worktree created, no dependency installed,
and no commit, PR, push or paid request was made.
