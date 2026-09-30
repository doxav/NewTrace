# Optimizer discovery — EXP-16, 10 septembre 2026

1. Question : une recherche de code utilisant ses évaluations TRAIN améliore-t-elle une politique black-box davantage que huit générations indépendantes ?
2. Artefact commun : `def propose(history, bounds, seed): return x`, sans accès à l'objectif, aux paramètres cachés ou au modèle ; un fichier source portable.
3. Benchmark : Sphere décalée, Quadratique anisotrope et Rosenbrock, dimensions 2/4 ; 24 instances TRAIN, 12 validation, 12 audit, deux seeds locaux et 32 évaluations par trajectoire.
4. Six réplications externes, huit réponses DeepSeek par bras : I indépendant ; C réécrit le parent choisi sur TRAIN ; R ajoute ses traces ; W utilise deux parents sur quatre rondes.
5. AUC normalisée moyenne, plus bas meilleur : seed 0,179329 ; I 0,077260 ; C 0,047588 ; R 0,122517 ; W 0,112582. Toutes les sélections précèdent l'audit.
6. R−I = +0,045257, intervalle bootstrap [+0,003176 ; +0,085126] : signal exploratoire négatif. R−C est également négatif ; W−R inconclusif. Six réplications donnent une incertitude fragile.
7. Un contrôle fixe changeant seulement le premier point vers le milieu du domaine atteint 0,031633 d'AUC, sans LLM ; son regret final reste moins bon que I/C.
8. Les 192 réponses sont conservées, dont 18 candidats inéligibles ; un R retient le seed. Aucun repli pendant l'audit. Coût connu 0,488195 USD et 6 980 225 tokens, hors deux transports à facturation inconnue.
9. Le représentant R choisi sur validation combine Halton, noyau gaussien et amélioration espérée. Aucune nouveauté ni supériorité du feedback n'est démontrée ; le sous-processus n'est pas une sandbox de sécurité du système.
10. Prochaine question : « Can we plug your FunSearch/OpenEvolve search into this exact optimizer.py contract and evaluator, keeping the artifact, tasks and budgets fixed? » Patrick n'a pas à adopter l'infrastructure recursive_opt.

Le [rapport](REPORT.md) conserve les diagnostics, résultats négatifs et limites.
La [source exacte du représentant](production/selected_R/16411_optimizer.py.txt)
porte le SHA-256 `40567fda87a2734f31a680db90e54d5f975a73b7930bc487be821d0cc5c9f243`.

Pour ajouter un moteur, écrire uniquement un adaptateur produisant une source
OptimizerProgramV0 et passer cette source à
`artifacts.optimizer_discovery.benchmark.evaluate(source, host_task, local_seed,
budget=32, deployment=..., seed_source=...)`. Le moteur ne reçoit que les
évaluations TRAIN autorisées. Réutiliser le même écran de validité, les mêmes
allocations, la sélection validation et la barrière globale avant audit ; compter
ses appels internes et réparations dans ses slots. Enregistrer une nouvelle
expérience avec de nouvelles instances et seeds : les panels présents sont vus.
Le manifeste de tâche avec ses paramètres reste côté hôte, hors du workspace
candidat. Le [contrat d'exécution portable](../exp15/PORTABILITY.md) reste inchangé.

La priorité suggérée avant cette intégration est une confirmation séparée de C
contre I, avec le contrôle fixe B2. Son avantage apparent sur I est post hoc et
ne constitue pas un résultat confirmatoire. Ce document n'a pas été envoyé.
