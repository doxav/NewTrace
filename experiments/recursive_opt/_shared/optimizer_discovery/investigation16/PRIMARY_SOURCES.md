# Primary-source checks for the diagnostic design

Consulted 2026-09-09. These sources motivate hypotheses; they do not prove a local
EXP-15 cause or a future gain.

- [OpenRouter reasoning-token documentation](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
  states that reasoning is billed as output, documents per-model supported effort
  levels, and distinguishes the effort instruction from a specific reasoning-token
  allocation. Do not interpret `low` as a guaranteed 20% thinking cap for DeepSeek.
- [DeepSeek's released 0731 message encoder](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/encoding/encoding_dsv4.py)
  implements low/high/max effort through prompt prefixes in thinking mode; low
  adds no extra effort prefix. This is not a numerical token limiter. Gateway
  acceptance of `reasoning.effort=low` does not independently certify every
  upstream provider's encoder/template implementation.
- [OpenRouter's model page](https://openrouter.ai/deepseek/deepseek-v4-flash-0731)
  and the public `/api/v1/models` endpoint advertise caps far above 8000 and support
  for `max_tokens`/reasoning. Advertised maximums differ between page and API;
  the inquiry therefore tests the modest 32000 cap through actual requests rather
  than relying on a claimed maximum. Provider receipts govern realized cost.
- [FunSearch's original research paper](https://www.nature.com/articles/s41586-023-06924-6)
  uses scored program populations and island diversity. It makes a population
  ablation scientifically motivated; it does not establish that adding islands or
  a second parent to this small, different benchmark will improve performance.

No change of model, no literature-novelty claim, and no transfer of published
FunSearch results into local evidence.

## Mechanism comparison for the final synthesis

The [targeted primary-literature review](research/LITERATURE_MECHANISMS.md) adds
versioned papers and official code for FunSearch, GEPA, Self-Refine and ADAS. It
records authors, dates, exact sections and URLs, separating paper versions from
current code observations. These sources distinguish scored-code selection,
per-instance parent selection, generated critiques and archive context. They do
not supply local efficacy estimates or a universal minimum of six or seven examples.
No method or scientific design is changed in the running P1 from this review.
