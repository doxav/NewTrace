# EXP29 — what recursive_opt should discover (planning)

Not run yet, apart from one local, LLM-free probe of Signal's causal headroom.

- [prior_analysis.md](prior_analysis.md), v3 (supersedes v2 `ecb415041c` and v1 `632b5e0426`):
  - recursive_opt (control-plane recursion) vs coevolution (one online engine) vs `VariationSearch` (a trainer, i.e. a target);
  - the EXP00 notebooks re-read;
  - meta targets T1–T7;
  - task order: numeric optimizer programs first, then 4-document QA; Signal kept for smoke tests only;
  - when tracing can matter (worker-side capture, not multi-step agents);
  - programme P0–P4 and its pre-commitments.
- [design_recursion_pieces.md](design_recursion_pieces.md): the two missing pieces (the `child_spec@1` module and declared code hooks), the alternatives eliminated, and decisions D1–D6.
- [scripts/signal_causal_headroom.py](scripts/signal_causal_headroom.py) → [results/signal_causal_headroom.json](results/signal_causal_headroom.json): 34 classical causal filters score 0.467–0.539 (initial 0.499; best LLM causal 0.565).

[All experiments](../README.md) · [EXP00 later equivalents](../EXP00/later_equivalent.md)
