# B2 — contrôle indépendant après résultat

**Contrôle réussi.** L'amplitude du gain AUC annoncé est reproduite exactement à
partir des observations conservées. Aucune erreur de pairing, de normalisation,
de métrique, aucun fallback et aucune omission ne l'expliquent. Ce contrôle
confirme le résultat exploratoire sur les fixtures publiques B1 ; il ne le
transforme pas en confirmation sur de nouvelles tâches.

| Condition | AUC seed | AUC premier point au centre | Réduction relative | Paires améliorées / dégradées |
|---|---:|---:|---:|---:|
| central | 0,192299928 | 0,027569468 | 85,66 % | 37 / 11 |
| broad | 0,167905518 | 0,055168850 | 67,14 % | 33 / 15 |

Les 96 variantes et leurs 96 contrôles seed sont présents. Le source variante
diffère exactement par la branche de première proposition prescrite : le milieu
des bornes remplace le premier tirage uniforme. Tout le comportement à histoire
non vide identique reste inchangé. Le point initial et sa valeur correspondent
exactement aux 96 contrôles midpoint B1 de mêmes tâche et seed local.

Le contrôle a vérifié les sources gelées, les hashes de chaque contrôle et
variante, les 24 tâches et leurs normalisations, les seeds locaux, B=32, les
points finis dans les bornes, les 32 observations et propositions valides de
chaque trajectoire et les 64 exécutions subprocess déclarées. Toutes les valeurs
objectives sauvegardées ont été recalculées avec `B.objective`. Les 192 panels
de métriques ont été recalculés avec `B.metrics` et indépendamment par minima
successifs des valeurs normalisées. Chaque résumé sauvegardé, chaque paire,
strate, seed local, courbe moyenne et décomposition AUC a été reproduit
exactement sans appeler les helpers `compare`, `B.aggregate` ou `H._summarize`.

Le contrôle n'a exécuté aucun candidat et n'a appelé aucun modèle :
`B.propose_point` et `B.evaluate` étaient remplacés par des sentinelles qui
échouent immédiatement. Son travail supplémentaire, séparé du budget B2, est
de **9 216 appels objectif d'intégrité** : 3 072 pour les variantes, 3 072 pour
les contrôles seed et 3 072 pour reconstruire les références de normalisation.

L'interprétation doit rester précise. Les six strates et quatre seeds locaux
ont une AUC moyenne améliorée dans chaque condition, mais 26 des 96 trajectoires
ont une AUC dégradée. Le regret final se dégrade dans trois des six strates de
chaque condition et pour trois des quatre seeds locaux broad. L'amélioration
globale du regret final broad (0,011339954 → 0,008866973) dépend donc d'un effet
hétérogène. Le résultat montre une faiblesse de l'initialisation du seed pour
cette métrique et ce benchmark ; il ne prouve pas une amélioration uniforme de
la précision finale ni un avantage du feedback récursif.

La baisse du premier terme AUC ne représente que 0,026388612 des 0,164730459
de baisse central, et 0,019122630 des 0,112736668 de baisse broad. Les termes
suivants mêlent persistance du premier incumbent et propositions induites par
l'histoire modifiée : cette décomposition est arithmétique, pas une séparation
causale de ces deux mécanismes. La condition broad conserve par ailleurs un
prior symétrique autour du centre. Ces fixtures ont déjà été inspectées ; une
future comparaison doit conserver un contrôle d'initialisation et de nouvelles
données, sans promettre les mêmes pourcentages.

Commande de contrôle exécutée :

```bash
PYTHONPATH=. /tmp/phase0-venv/bin/python /tmp/investigation16_b2_independent_review.py
```

Toutes les assertions sont passées. Les identités des 96 paires, tous leurs
deltas et les comptes d'intégrité sont conservés dans
[independent_review.json](independent_review.json).

- Gel B2 : `6eae835547be85c560a4daa4c2bd3f714b7f16600307ef4657e3af628647f3db`.
- Résultat B2 inchangé : `c53db55f2856e6757d73501bc899c6fbb8f65602ec1d4f9f062b1a04fe9be0dc`.
- JSON de ce contrôle : `46fec532666d17529a4b23181f32e83de8b61761cd194c71b34b5b9b62d0cdb0`.

Aucun protocole, source gelée, résultat antérieur, fichier F1 ou fichier P1
n'a été modifié.
