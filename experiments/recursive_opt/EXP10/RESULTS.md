# EXP10 — results

Historical; use corrected interpretation in the result entry, not the old registry verdict.

[Detailed report or operational record](../_history/reviews/recursive_opt_etude_2026-09-24.md).

The following row preserves the corrected September 24 audit, including its limits:

| Experiment / data | Test and volume | Observed result | Interpretation |
|---|---|---|---|
| **EXP10 / knobs S** — [brut](results/probe_s_knobs_results.json) | Variation des paramètres de config sur **13 bundles** LLM4AD ; inventaire élargi de 47 tâches | Pas de variation de score dans ce chemin ; 13/13 bundles n’exposent qu’un exemple externe ; 43/47 tâches dans cette situation, 2 chargements en erreur | Le chemin à `inner_steps=0` et ces données n’exercent pas utilement l’ordre/batch d’apprentissage. Ne pas généraliser à un curriculum actif ou à tous les hyperparamètres. |
