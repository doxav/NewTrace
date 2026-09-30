# B2 : isoler le premier point du seed

**Le remplacement du seul premier point diminue fortement l’AUC moyenne sur les fixtures B1 publiques : −85,66 % en condition centrale et −67,14 % en condition élargie.** La politique restante est exactement celle du seed. Ce résultat révèle un levier important du contrôle de départ sur ce benchmark ; il ne démontre ni un avantage du feedback récursif sur la génération indépendante, ni un gain futur.

Le protocole `B2_PROTOCOL.md` et le gel `b2/freeze.json` ont précédé toute évaluation de cette variante. Les tâches B1 avaient déjà été vues : cette ablation unique est exploratoire. Aucun réglage de P1/P1-E1, aucune source antérieure, aucun programme généré et aucun contrôle B1 n’ont été modifiés.

## Intervention et mesure

La seule substitution est la branche `history` vide : proposer le milieu de chaque borne. La signature, le mélange exploration/perturbation, la décroissance du pas et le RNG `seed + len(history)` restent identiques. Les tests de la frontière subprocess vérifient le rejeu déterministe, le point central avec bornes asymétriques et l’égalité exacte des propositions à histoire non vide identique, dans les dimensions 2 et 4.

Le nouveau premier point modifie ensuite les observations, l’incumbent et les propositions induites. Le contraste est donc l’effet total de l’initialisation sur la trajectoire. Le RNG est recréé à chaque proposition ; il n’existe pas de consommation aléatoire persistante supprimée par le nouveau premier appel.

## Résultats agrégés

Chaque ligne contient 48 trajectoires : 12 tâches × 4 seeds locaux, B=32. Les six strates ont le même poids. AUC et regret final : plus bas = mieux. Les 96 nouvelles trajectoires sont valides, sans timeout, exception ou fallback. Les 96 contrôles seed sont réutilisés depuis B1, sans réexécution.

| Condition | Politique | AUC | Regret final | Atteinte cible | Temps à cible censuré/plafonné |
|---|---|---:|---:|---:|---:|
| central | Seed original | 0.192300 | 0.020946 | 60.42 % | 21.979 |
| central | Premier point central | 0.027569 | 0.006325 | 75.00 % | 17.229 |
| broad | Seed original | 0.167906 | 0.011340 | 70.83 % | 21.146 |
| broad | Premier point central | 0.055169 | 0.008867 | 75.00 % | 18.854 |

La cible est un regret normalisé ≤0,01. Les non-atteintes conservent `target_evaluations = null` (non observé) dans les preuves ; 33 est la convention censurée/plafonnée, pas un temps réellement observé.

| Condition | ΔAUC variante − seed | Variation relative | Paires AUC meilleures / pires | Δregret final |
|---|---:|---:|---:|---:|
| central | -0.164730 | -85.66 % | 37 / 11 | -0.014621 |
| broad | -0.112737 | -67.14 % | 33 / 15 | -0.002473 |

Aucune paire n’est ex æquo. Toutes les pertes sont conservées. La contribution du seul premier terme au delta AUC vaut −0,026389 en central et −0,019123 en broad, soit respectivement 16,02 % et 16,96 % du delta total. Les 31 autres termes expliquent arithmétiquement le reste ; ils mêlent persistance best-so-far du premier point et adaptation de la trajectoire. Cette décomposition ne sépare pas causalement ces deux mécanismes.

## Détail par strate

| Condition | Strate | AUC seed | AUC variante | ΔAUC | Regret final seed | Regret final variante |
|---|---|---:|---:|---:|---:|---:|
| central | quadratic/2 | 0.175568 | 0.011334 | -0.164235 | 0.002145 | 0.005075 |
| central | quadratic/4 | 0.213484 | 0.080052 | -0.133432 | 0.029585 | 0.008772 |
| central | rosenbrock/2 | 0.064232 | 0.009860 | -0.054372 | 0.000410 | 0.000642 |
| central | rosenbrock/4 | 0.245555 | 0.016615 | -0.228940 | 0.007134 | 0.001426 |
| central | sphere/2 | 0.135615 | 0.028013 | -0.107603 | 0.001988 | 0.011839 |
| central | sphere/4 | 0.319344 | 0.019543 | -0.299801 | 0.084415 | 0.010194 |
| broad | quadratic/2 | 0.145542 | 0.019259 | -0.126283 | 0.000489 | 0.000906 |
| broad | quadratic/4 | 0.183302 | 0.101923 | -0.081379 | 0.022515 | 0.009899 |
| broad | rosenbrock/2 | 0.102044 | 0.029983 | -0.072061 | 0.000819 | 0.000652 |
| broad | rosenbrock/4 | 0.166460 | 0.046196 | -0.120265 | 0.002553 | 0.002894 |
| broad | sphere/2 | 0.135035 | 0.055853 | -0.079183 | 0.003066 | 0.003135 |
| broad | sphere/4 | 0.275050 | 0.077799 | -0.197250 | 0.038597 | 0.035715 |

L’AUC moyenne s’améliore dans chacune des six strates des deux conditions. Le regret final se dégrade néanmoins dans trois strates sur six en central et trois sur six en broad. Améliorer l’initialisation anytime ne garantit donc pas une meilleure précision terminale dans chaque famille/dimension.

## Détail par seed local

| Condition | Seed local | AUC seed | AUC variante | ΔAUC | Regret final seed | Regret final variante |
|---|---:|---:|---:|---:|---:|---:|
| central | 16201 | 0.296440 | 0.031898 | -0.264542 | 0.016749 | 0.007420 |
| central | 16202 | 0.126522 | 0.032128 | -0.094393 | 0.018198 | 0.006457 |
| central | 16203 | 0.293248 | 0.024913 | -0.268336 | 0.032224 | 0.006482 |
| central | 16204 | 0.052990 | 0.021339 | -0.031651 | 0.016615 | 0.004940 |
| broad | 16201 | 0.269486 | 0.060819 | -0.208667 | 0.007330 | 0.010020 |
| broad | 16202 | 0.075266 | 0.055579 | -0.019687 | 0.010886 | 0.011305 |
| broad | 16203 | 0.255183 | 0.053263 | -0.201920 | 0.022403 | 0.007113 |
| broad | 16204 | 0.071687 | 0.051014 | -0.020673 | 0.004741 | 0.007030 |

Les quatre agrégats de seed local s’améliorent en AUC dans les deux conditions. En broad, le regret final empire pour trois seeds sur quatre ; l’amélioration terminale moyenne dépend particulièrement du seed 16203. Ces seeds locaux ne sont pas des réplications de génération LLM et les 96 lignes ne sont pas indépendantes.

## Interprétation et limites

B1 comparait un midpoint constant au seed adaptatif et ne pouvait isoler le premier point. B2 isole maintenant cette intervention de code, sous réserve des trajectoires induites. L’hypothèse selon laquelle l’initialisation du seed ne laisserait qu’une marge négligeable est contredite sur ces fixtures. Une grande amélioration par rapport à ce seed peut donc provenir d’une décision initiale très simple, sans mécanisme de feedback récursif.

L’amélioration en broad n’est pas contradictoire : élargir une distribution symétrique des optima n’annule pas nécessairement l’utilité du centre comme prior. En revanche, les constantes de normalisation et leurs seeds changent entre central et broad ; interpréter les deltas entre politiques au sein de chaque condition, sans attribuer une différence absolue entre conditions à un unique mécanisme.

Ce diagnostic ne mesure pas pourquoi A2 diffère de A1, ne démontre aucune nouveauté algorithmique et ne prescrit pas de remplacer le seed au milieu d’une expérience gelée. Il justifie de prévoir un contrôle raisonnable de l’initialisation dans une future expérience sur données nouvelles, et de distinguer gain d’AUC, précision terminale et valeur ajoutée du feedback. Le seed P1 reste exactement inchangé.

## Ressources et intégrité

- Recherche nouvelle : 3072/3072 appels objectif, 6144/6144 processus de proposition/rejeu, 96/96 trajectoires ; zéro allocation inutilisée.
- Référence uniforme reconstruite une fois pour chacune des 24 tâches : 3072 appels objectif distincts, séparés du budget de recherche. Les constantes sont exactement celles de B1.
- Audit indépendant après le run : 9216 appels d'intégrité supplémentaires,
  distincts de la recherche et de sa préparation (3072 observations variante,
  3072 observations contrôle, 3072 références). Aucun programme relancé : les
  valeurs objectives, les points/valeurs initiaux contre midpoint B1, les
  192 métriques et tous les résumés ont été vérifiés. Résultat PASS dans
  `b2/independent_review.json`, digest canonique
  `46fec532666d17529a4b23181f32e83de8b61761cd194c71b34b5b9b62d0cdb0`.
- Temps mur du run à huit workers : 63.289 s ; somme des temps de trajectoire : 494.876 s. Aucun appel externe ou de modèle, aucun token, aucun coût de fournisseur.
- Environnement : `/tmp/phase0-venv/bin/python`, Python 3.13.13. Le constructeur d’évaluation et la frontière de subprocess sont ceux déjà gelés. La frontière n’est pas un sandbox de sécurité OS.
- Les 96 hashes des contrôles, les inputs, les réglages et les sources ont été revérifiés après calcul ; les agrégats ont été recomputés depuis toutes les observations conservées. Les valeurs défavorables ne sont pas supprimées.
- Gel canonique : `6eae835547be85c560a4daa4c2bd3f714b7f16600307ef4657e3af628647f3db`.
- Résultat canonique : `c53db55f2856e6757d73501bc899c6fbb8f65602ec1d4f9f062b1a04fe9be0dc`.
- Source exacte exportée sans formatage : `b2/optimizer.py`, SHA256 `958fbb12279a15966bf1ffa45ddffc7cd5f946ca800a3fe2b8e132a150c5c190` ; identique au texte évalué dans le gel.
- Données nouvelles : `b2/raw/`; contrôles conservés : `raw/{central,broad}/seed/`; les sources, hashes et chemins de chaque paire figurent dans le gel et les résultats.

## Commandes et vérification

```bash
/tmp/phase0-venv/bin/python -m pytest -q artifacts/optimizer_discovery/investigation16/benchmark/test_first_point.py tests/unit_tests/test_investigation16_benchmark.py
/tmp/phase0-venv/bin/python -m artifacts.optimizer_discovery.investigation16.benchmark.first_point prepare
/tmp/phase0-venv/bin/python -m artifacts.optimizer_discovery.investigation16.benchmark.first_point run
/tmp/phase0-venv/bin/python -m artifacts.optimizer_discovery.investigation16.benchmark.first_point analyze
```

TDD : le premier test a échoué à la collecte, car le nouveau module n’existait pas encore. Après implémentation, **9 tests passent en 7,70 s** (six nouveaux et trois B1), sans skip. Black, Ruff et `git diff --check` passent pour les changements vérifiés. Le scan de motifs de clé OpenRouter passe pour les nouveaux fichiers B2, y compris les deux preuves de l'audit indépendant. Les tests d’interface sont hors budget de recherche et leurs propositions ne sont pas intégrées aux résultats B2. Les sorties de tests proviennent de l’outil d’exécution ; aucun log autonome n’a été créé.
