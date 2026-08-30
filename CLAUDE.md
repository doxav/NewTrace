# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Trace (`trace-opt` on PyPI) is a PyTorch-like Python library for end-to-end generative optimization of AI
agents: user code is "traced" into a computation DAG via `node()`/`@bundle`, and optimizers (`OptoPrime`,
`TextGrad`, `OPRO`, and variants) use LLMs to propose updates to parameters marked `trainable=True`, driven
by `.backward()`/`.step()` calls analogous to PyTorch training.

See `@OVERVIEW.md` for the module breakdown (`opto.trace`, `opto.optimizers`, `opto.trainer`, `opto.utils`)
and extension guidelines. See `@docs/multi_objective_scores.md` for multi-objective (vector-score) support.

## Setup

- Python >= 3.10 (CI runs 3.13). Install for development: `pip install -e .`
- Real dependencies live in `setup.py`'s `install_requires` — `pyproject.toml` deps are dynamic and pull
  from there; don't edit deps in `pyproject.toml`.
- Running any optimizer requires an LLM backend. Default backend is LiteLLM; set provider keys as env vars
  (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) and optionally `TRACE_LITELLM_MODEL` to fix the default model.
  Switch backend to legacy AutoGen v0.2 with `TRACE_DEFAULT_LLM_BACKEND=AutoGen` (requires the `[autogen]` extra).

## Build / test / lint

- Fast unit tests (no LLM key needed, this is what CI gates on): `pytest tests/unit_tests/`
- LLM-dependent optimizer tests (need a real or local LLM backend; CI runs these against a local Ollama
  model and treats failures as non-blocking): `pytest tests/llm_optimizers_tests/test_optimizer.py`
- Formatting/linting is enforced via pre-commit, not a Makefile or npm-style script: `pre-commit run --all-files`
  (runs black, ruff --fix, codespell, and nbQA black/ruff over notebooks). There's no committed
  `[tool.black]`/`[tool.ruff]` config, so defaults apply.

## Code style

- Type hints (from `typing`) are used throughout core modules (`opto/trace/`, `opto/optimizers/`).
- Public functions/classes use NumPy-style docstrings (`Parameters`/`Returns`/`Raises`/`Examples` sections).

## Repo/branch conventions

- Current work targets the `experimental` branch, not `main`. Per `CONTRIBUTING.md`: `main` only takes
  bug-fix PRs; feature work branches from and merges back into `experimental`. Branch names follow
  `feature/xxx` or `fix/xxx`.
- New features generally need unit tests under `tests/unit_tests/` or an example script under `examples/`.
- Standalone features under `opto/features/<name>/` get a lighter review, but code there must not be
  imported by anything outside `opto/features/<name>/` — keep those subpackages self-contained.

## Repo layout notes

- `examples/notebook_outputs/`, `notebook_outputs/`, root-level `*.log`, `*.pkl`, and `OUTPUTS_mem.zip` are
  local experiment run byproducts, not tracked project structure — ignore them when reasoning about the
  codebase.
