# EXP07 — results

Historical; use corrected interpretation in the result entry, not the old registry verdict.

[Detailed report or operational record](../_history/reviews/recursive_opt_etude_2026-09-24.md).

The following row preserves the corrected September 24 audit, including its limits:

| Experiment / data | Test and volume | Observed result | Interpretation |
|---|---|---|---|
| **EXP07 / Probes S,T,T2** — [table](results/probe_t_routing_menu.json) | 9 heuristiques de routage × 4 tâches, contrôles invalides, répétitions ; sensibilité à `max_examples` 2/4/8/16 | `nearest` meilleur **4/4**, corrélations de rang 0,51–1 ; 7/9 scores distincts sur TSP, 9/9 ailleurs | Optimum partagé dans un menu fini. CVRP/OVRP sont presque doublons ; famille effective de 3 tâches. Ce n’est pas une condition nécessaire à toute méta-optimisation. |
