# EXP26 — design analysis (2026-10-06)

Why EXP26 is shaped the way it is. Every claim below was checked against EXP25's raw evidence
(`../EXP25/results/runs_20260930T235849/`) or against source; nothing here needed a new run.

## 1. What EXP25 actually shows, re-analysed

| | stock EvoX | Trace (native, 6 runs) | source |
|---|---:|---:|---|
| Distinct candidates importing SciPy | **68–76%** per run | **0–1.2%** (1 program in 6 runs) | `candidate_history.jsonl`; `sources/*.py` |
| DIVERGE label names SciPy | not persisted | **6 / 6** | `report.json:/labels/diverge` |
| REFINE label names SciPy | not persisted | **0 / 6** | `report.json:/labels/refine` |
| DIVERGE label steers "numpy only" | — | 3 / 6 | same |
| Share of iterations using DIVERGE | not logged | **0–24%, median 12%** (one run 0%); REFINE ~85% of labelled ones | `events.jsonl` `label` |
| Label prompt receives package list | yes (`get_available_packages`) | no | source |
| Label prompt mandates a LIBRARIES/TOOLS block | yes | no (only permits) | source |
| Label prompt receives the initial program | **no** (controller omits it) | no | source |
| SciPy importable at evaluation | yes | yes (evaluator imports it; worker unrestricted) | source |
| Best valid score, median of 3 | 0.713 | 0.586 / 0.555 | EXP25 RESULTS |
| Best valid **causal**, median of 3 | 0.537 | 0.532 / 0.532 | EXP25 RESULTS |

These figures come from `scripts/analyze.py` run on EXP25's campaign, which reproduces all nine of EXP25's
published per-run P1 and P2 values exactly (validation of the analyzer itself).

Two consequences shape everything below:

- **Two candidate mechanisms, not one.** SciPy reached the native prompt only through DIVERGE, and
  the learned selection policy chose DIVERGE for a median 12% of iterations. So the gap can come from label
  *content* (M1: REFINE never mentions a library; DIVERGE half-discourages one) or from label
  *selection* (M2: the policy rarely picks the one label that mentions it). EXP25 cannot separate them.
- **The gap exists only on the look-ahead-tolerant score.** Under causality every arm ties. A
  fidelity fix may therefore just reproduce the SciPy look-ahead exploit faster.

## 2. Next action 1 — instrumentation

EXP25's `calls.jsonl` stores metadata only, so whether proposals *tried* SciPy was unknowable until
`sources/` was scanned; stock's generated labels were never persisted at all.

| option | answers "what did the model see and say?" | cost | decision |
|---|---|---|---|
| A. Full transcript of every native call (system, user, response) | yes, completely | ~2–5 MB/run, local only | **chosen** |
| B. Sample every N-th proposal | partially; can miss the decisive one | lower | rejected |
| C. Prompt hashes only | no | minimal | rejected |
| D. Persist stock's generated labels (and whether they fell back to defaults) | yes, for the label itself | 1 file/run | **chosen**, required by §3 |

Stock silently falls back to its default templates when the guide LLM is unavailable or errors;
EXP26 records a `fallback` flag so a stock run with generic labels cannot pass unnoticed.

## 3. Next action 2 — prompt fidelity

| option | what it isolates | weakness | decision |
|---|---|---|---|
| A. Port stock's 641-line generator verbatim | everything | couples the library to SkyDiscover's prompt text; large | rejected |
| B. Compact contract: `label_packages` (package list + LIBRARIES/TOOLS rule in both blocks) | whether the *missing contract* explains the gap | it is our paraphrase, not stock's text | **arm `native_pkg`** (library change landed in `b0b0d18913`) |
| C. Inject labels produced by stock's own generator into the native engine | label **content**, holding engine, operator, proposer and selection constant | one label draw per seed (temperature 0.7) | **arm `native_stocklabels`**: the decisive control |
| D. Force a DIVERGE quota | selection (M2) | overrides the very policy being studied | rejected as a treatment; DIVERGE share is **measured** as a mediator instead |

B against C tells us whether the compact port is good enough to become the default.

## 4. Next action 3 — the target

| option | consequence | decision |
|---|---|---|
| A. Benchmark as scored (look-ahead allowed) | rewards offline smoothers; contradicts the task text ("real-time") | secondary endpoint only |
| B. Causal-guided search | the honest objective, but changes the guide; bundling it with the label change repeats EXP24's attribution problem | **deferred to EXP27** |
| C. Hold EXP25's valid guide, pre-register **both** endpoints | fidelity is tested where the gap exists; causality comes free (`causal_fraction` is measured in every evaluation) | **chosen** |

EXP26 asks one question — *does prompt fidelity close the gap?* — on the objective where the gap
was observed, and reports the causal endpoint alongside it. Changing the guide and the labels
together would leave the result unattributable, exactly as EXP24's bundled O0 change did.

## 5. Arms

All native arms use EXP25's `trace_exp24` configuration unchanged (valid-score guide, per-signal
feedback, fallback projection, Trace proposer); only the label source differs.

| arm | labels | tests |
|---|---|---|
| `native` | native generator (as EXP25) | re-baseline in the same time window |
| `native_pkg` | native generator + stock package list + LIBRARIES/TOOLS rule | compact port (B) |
| `native_stocklabels` | stock's own generator, injected | label content (C) |
| `evox_stock` | stock, unchanged; labels now persisted | reference |

## 6. What each outcome would mean

| observation | interpretation | next |
|---|---|---|
| `native_stocklabels` reaches EvoX's range and its SciPy share rises | label content explains the gap (M1) | compare B with C; adopt B if close |
| `native_stocklabels` stays near `native` with a low DIVERGE share | content is not enough; selection (M2) implicated | experiment on label selection |
| `native_stocklabels` stays near `native` with a high DIVERGE share | the gap lies elsewhere (operator template, cascade) | prompt-level diff of the solution operator |
| every arm ties under causality | EvoX's lead is look-ahead only (confirms EXP25) | EXP27 with a causal guide |
