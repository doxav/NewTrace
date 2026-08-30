# Local Experiment-0 live configuration

- The frozen live model is `deepseek/deepseek-v4-flash-0731` through OpenRouter.
- `.env` records the local secret-source path as `OPENROUTER_API_KEY_SOURCE`; it is intentionally ignored by Git.
- Before a live run, extract the first `sk-or-v1-...` value from that local source into `OPENROUTER_API_KEY` without printing it.
- Never copy, log, commit, or echo the API key. Keep the model and canonical request parameters frozen.
