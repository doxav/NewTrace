# Découvrir des programmes d’optimisation : ce que nous pouvons comparer

Nous cherchons à savoir si un modèle qui **réécrit un optimiseur après avoir vu
ses résultats** découvre de meilleurs programmes que le même modèle générant
plusieurs optimiseurs indépendamment. Nous avons un contrat exécutable et des
résultats locaux ; **l’avantage de cette réécriture avec feedback n’est pas établi**.
Ce travail ne démontre ni nouveauté algorithmique, ni bénéfice d’une récursion
plus profonde, ni amortissement sur une application réelle.

L’objet livré est un fichier `optimizer.py` contenant :

```python
def propose(history, bounds, seed):
    # Choisir le prochain point à partir des observations précédentes.
    return x
```

Le programme reçoit les points déjà essayés, leurs valeurs, les bornes et une
graine locale. L’évaluateur externe calcule la valeur de chaque point proposé.
Le programme n’appelle aucun LLM et ne reçoit ni le code de la fonction, ni son
optimum. Un autre moteur de recherche peut donc fournir le même fichier.
[Contrat complet](../OPTIMIZER_PROGRAM_V0.md).

Notre petit benchmark couvre trois difficultés numériques :

| Famille, dimensions 2 et 4 | Difficulté testée |
|---|---|
| Sphere transformée | Trouver un minimum sur une géométrie séparable |
| Quadratique anisotrope | Adapter la recherche à des courbures différentes selon les axes |
| Rosenbrock transformée | Suivre une vallée courbe avec coordonnées couplées |

Chaque programme dispose de **32 évaluations**. Dans les études récentes,
24 instances guident la recherche, 12 servent à choisir le programme final et
12 restent réservées au test final, avec deux graines locales par instance.
La mesure principale est le regret normalisé moyen au cours des 32 appels :
**plus bas est meilleur**. Nous rapportons aussi le regret final. Ce sont des
diagnostics élémentaires, pas encore des problèmes de matrices générales ou de
recherche scientifique à long horizon.

Les résultats répondent à des questions successives, sans être fusionnés :

| Comparaison | Ce que le résultat permet de dire |
|---|---|
| **EXP-15 : seed manuscrit / huit générations indépendantes / huit générations avec feedback**, cinq répétitions | Le feedback améliore le seed : Δ −0,02290, IC95 % [−0,04246 ; −0,00334]. Face à l’indépendant : Δ −0,00507 [−0,03773 ; +0,02455], **inconclusif**. |
| **EXP-16 : feedback détaillé / indépendant / réécriture du parent sans scores explicites**, six répétitions | Le feedback détaillé perd face à l’indépendant : Δ +0,04526 [+0,00318 ; +0,08513]. La meilleure moyenne de la simple réécriture est une observation exploratoire, non confirmée. |
| **EXP-18 : ajout d’une mémoire des essais, choix parmi des parents non dominés, ou les deux**, six répétitions | **Aucun avantage établi** de ces mécanismes. Effet mémoire −0,00411 [−0,02065 ; +0,01491] ; effet de sélection +0,00851 [−0,01502 ; +0,03568]. Il n’y a pas de contrôle indépendant à ce budget. |

Les intervalles sont des bootstraps appariés entre répétitions de génération ;
avec cinq ou six répétitions, ils restent fragiles. EXP-17, destinée à confirmer
la simple réécriture face à l’indépendant, est **suspendue avant le test réservé**
et n’apporte pas encore de résultat. Sources : [EXP-15](../EXP15_REPORT.md),
[EXP-16](../investigation16/REPORT.md), [EXP-18](REPORT.md).

EXP-18 alloue 16 réponses par recherche, quatre variantes et six répétitions :
**384 réponses réelles**, DeepSeek V4 Flash via OpenRouter, plafond commun
32 000 tokens, reasoning low, température 0,6. Les allocations logiques sont
30 240 trajectoires, soit 967 680 évaluations objectives ; le cache et les
échecs expliquent des consommations effectives différentes. 48 programmes sur
384 sont inadmissibles durant l’apprentissage/sélection ; les 864 trajectoires
finales sont valides, sans repli. Un contrôle fixe commençant au centre obtient
déjà AUC 0,04099, contre 0,14429 pour le seed : battre ce seed seul est insuffisant.
[Budgets, validité et limites](REPORT.md).

Un exemple concret est le **programme A2/41**, choisi par validation dans EXP-15,
puis rejoué sur de nouvelles instances dans S1. Il alterne une exploration Halton,
des perturbations du meilleur point et un ajustement quadratique quand assez
d’observations sont disponibles. Sur ce nouveau panel, son AUC vaut **0,07557**,
contre 0,16269 pour le seed ; un programme indépendant atteint 0,08069. Cela montre
un artefact utilisable, sans établir la supériorité de son moteur de découverte.
[Source exacte, SHA256 `1684f91a…66abb7`](../investigation16/selection/sources/1684f91acdc36c0ca6aac70afeb9cc2c4eed7ab847926d5880590e059266abb7.py.gz),
[replay S1](../investigation16/selection/REPORT_S1.md).

La proposition de discussion est : **« Can we plug your FunSearch/OpenEvolve
search into this exact optimizer.py contract and evaluator, keeping the artifact,
tasks and budgets fixed? »** Patrick n’est pas invité à adopter l’infrastructure
recursive_opt. L’entrée réutilisable est [`benchmark.evaluate`](../benchmark.py) ;
une future comparaison demanderait des instances neuves, le même budget total
de propositions — réparations comprises — et une sélection séparée du test final.

L’exécution locale utilise un sous-processus, un timeout et un environnement
assaini. **Ce n’est pas un sandbox OS** et cela n’implémente pas les snapshots ou
le rejeu d’environnement de Projet 1. Le périmètre réutilisable est décrit dans
[PORTABILITY.md](../exp15/PORTABILITY.md). Ce brief n’a pas été envoyé.
