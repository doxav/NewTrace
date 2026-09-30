# Enquête historique : gains reproduits et portée réelle

Date : 2026-09-09. Cette enquête est exploratoire. Elle rejoue des instances déjà
publiées et ne fournit donc aucune nouvelle preuve de généralisation. EXP-15,
Phase 0 et les résultats historiques restent inchangés.

## Résultat principal

Le gain historique de vitesse est **reproduit**, mais son mécanisme est un ordre
de recherche informé dans un menu manuscrit. Il ne démontre pas que le feedback
récursif produit de meilleurs programmes qu'une génération indépendante. Un
contrôle déclaré avant le replay, consistant à choisir directement `nearest`,
obtient exactement la même qualité avec **zéro évaluation méta**. C'est un contrôle
informé, adapté à ce menu précis ; ce résultat ne prouve pas qu'un tel choix serait
disponible sur n'importe quelle nouvelle famille inconnue.

Il existe aussi un gain historique sur la tâche elle-même : **5 programmes LLM
indépendants sur les 12 réponses originales VRPTW dépassent le meilleur des neuf
heuristiques manuscrites**. Les cinq performances sont reproduites exactement.
Cela valide la possibilité d'améliorer du code sur cette surface, sans établir
un avantage du feedback récursif.

## Ce qui a été réellement retesté

165 évaluations du vrai benchmark Trace-Bench, sans appel LLM :

- 9 politiques × 4 tâches × 3 répétitions = 108 évaluations de routage ;
- 4 templates initiaux ;
- 11 programmes LLM sauvegardés, sur leur tâche source ;
- les 22 transferts correspondants sur les deux tâches sœurs ;
- 14 politiques de packing/admissible-set et 6 entrées des menus prose historiques.

Les sources sauvegardées sont réutilisées telles quelles. Les 13 bundles utilisés
historiquement pour les knobs sont aussi rechargés pour vérifier leur structure.
L'analyse de W2 énumère exactement tous les sous-ensembles de chaque taille du
menu de neuf politiques : elle ne repose sur aucune simulation Monte-Carlo.

| Évidence historique | Replay actuel | Interprétation défendable |
|---|---|---|
| EXP-07 : `nearest` premier sur 4 tâches | 36 scores identiques aux originaux ; écart maximal 0 ; étendue des 3 répétitions 0 | Optimum partagé **dans ce menu**, sur ces fixtures |
| Corrélation des rangs | Spearman 0,512623 à 1,000000 ; six paires positives | Information transférable entre les tâches, avec CVRP/OVRP quasi redondants |
| EXP-08 : amortissement W2 | K*=2,25 pour Q=1 ; 2,571429 pour Q=0,99 ; 6 pour Q=0,95 | Coût amorti face à la recherche uniforme sans remplacement |
| Contrôle `nearest` fixé | q=1 à b=1, coût méta 0 | Le gain observé ne nécessite pas d'apprendre l'ordre sur ce menu précis |
| EXP-09 : code valide sur sa source | 11/11 encore valides ; tous les scores identiques | Le code sauvegardé est réellement exécutable sur sa cible |
| EXP-09 : transfert sur tâche sœur | 0/22 valides ; tous les échecs sont des incompatibilités de signature | Cause racine structurelle reproduite, avant toute comparaison de qualité |
| EXP-06 : packing | 4 scores distincts, étendue 2908,2 | Surface optimisable réelle, mais plusieurs politiques différentes textuellement sont équivalentes |
| EXP-06 : admissible-set | 4 scores distincts, étendue 390 | Même conclusion |
| Guard contre les menus de mauvais type | Les six entrées prose/vides ne remplacent pas les paramètres ; scores initiaux conservés | Le défaut historique de substitution est corrigé ; une recherche sur ce menu reste sans levier utile |
| EXP-10 : structure des données | Les 13/13 bundles ont exactement 1 input et 1 info | Les réglages de partition et d'ordre d'exemples n'ont pas de diversité externe à exploiter dans ce chemin |

Les échecs de transfert contiennent notamment : `takes 4 positional arguments but
6 were given` et `missing 2 required positional arguments`. Les valeurs −1e6
restent conservées dans l'évidence historique, mais le replay les marque
`valid: false` et n'en déduit aucun regret numérique.

## Vitesse : le contrôle change la portée du résultat

En conservant le template initial dans les deux pools, sur la famille dédoublonnée
TSP/CVRP/VRPTW :

| Propositions sur la cible | Recherche uniforme, E[q] | Ordre appris sur les deux sources | `nearest` fixé |
|---:|---:|---:|---:|
| 0 | 0,848948 | 0,848948 | 0,848948 |
| 1 | 0,887213 | 1,000000 | 1,000000 |
| 2 | 0,915186 | 1,000000 | 1,000000 |
| 4 | 0,951819 | 1,000000 | 1,000000 |
| 9 | 1,000000 | 1,000000 | 1,000000 |

L'ordre appris coûte 9 × 2 = 18 évaluations sur les tâches sources. À Q=1, la
différence de coût par tâche est 9−1=8, donc K*=18/8=2,25. Ce calcul est exact.
Il s'agit du budget nécessaire pour que **la qualité attendue** d'un meilleur de
b tirages atteigne Q ; ce n'est pas une mesure du temps moyen jusqu'au premier
succès dans un processus adaptatif. Les coûts source/cible sont comptés en
évaluations, pas en durée réelle comparable entre familles.

Deux réserves supplémentaires étaient faciles à manquer dans Git :

1. `nearest` est aussi **la première entrée écrite à la main** dans le menu. Les
   variantes historiques avec seulement 3 ou 5 candidats méta contiennent donc
   déjà le meilleur candidat ; elles ne prouvent pas qu'un petit budget découvre
   généralement le bon candidat dans un menu inconnu.
2. Le calcul de ce résultat est dans `probe_u_w2_sweep.py` et
   `probe_u2_default_baseline.py`. C'est une analyse d'une table déterministe,
   pas l'exécution d'une boucle de synthèse récursive via Control Plane. Un
   harness de production existe (`probe_v_harness.py`), mais son existence ne
   transforme pas les résultats U/U2 en preuve d'efficacité de ce harness.

## Performance sur la tâche : ce qui a vraiment progressé

| Tâche source | Réponses originales | Sources valides conservées et rejouées | Meilleure qualité normalisée au menu | Programmes meilleurs que le menu |
|---|---:|---:|---:|---:|
| TSP | 12 | 1 | 0,957414 | 0 |
| CVRP | 12 | 4 | 0,939698 | 0 |
| VRPTW | 12 | 6 | 1,383016 | 5 |

La normalisation historique vaut 1 au meilleur candidat du menu et 0 au pire.
**1,383 n'est donc pas une amélioration de 38,3 % du coût VRPTW** ; c'est un
dépassement de 0,383 fois l'étendue du menu. La génération d'origine utilisait
des appels à température 0 pour un échantillon et 1 pour les autres, un plafond
de 6000 tokens et une concurrence 6 : ses taux de validité ne sont pas des
  estimations actuelles pour le protocole EXP-15.

Le meilleur échantillon VRPTW sauvegardé (index 7) passe de la distance moyenne
26,476508 du `nearest` manuscrit à **20,702124**, soit **21,81 % de distance en
moins selon cet évaluateur**. Le code équilibre distance et urgence temporelle
avec `distance + 0.5 * slack`, au lieu de regarder uniquement la proximité. La
lecture du benchmark confirme que son score est l'opposé de la distance moyenne
des 16 instances. Cette amélioration respecte la sémantique de cet évaluateur
existant ; ce n'est ni un nouvel audit complet de conformité VRPTW industrielle,
ni une mesure hors échantillon, ni une preuve de nouveauté algorithmique.

Ces résultats suggèrent une surface favorable lorsque le programme possède une
décision de construction répétée, un historique pertinent et des contraintes qui
laissent plusieurs heuristiques plausibles. Ils ne permettent pas de prédire le
gain d'une nouvelle variante de `recursive_opt` sans comparaison contrôlée.

## Knobs, batch et types de méta-optimisation

Le précédent ne justifie pas une règle universelle « au moins 6 ou 7 exemples ».
Ce qui compte est l'information distincte qui atteint réellement la génération et
la sélection. Un exemple externe peut contenir de nombreuses instances internes ;
inversement, multiplier les lignes identiques ne crée aucun signal supplémentaire.
Pour étudier l'ordre des exemples, des minibatches ou une affectation du budget,
il faut que ces décisions soient **effectivement consommées par l'entraînement**.

De même, une politique fixe optimale sur tous les membres d'une famille est un
cas simple favorable au transfert, **pas une condition nécessaire de toute
méta-optimisation**. Une politique conditionnelle peut adapter son comportement
à l'historique et bénéficier de régularités partagées même si les meilleurs
heuristiques fixes diffèrent entre tâches. Il faut donc mesurer les signaux et
leur transférabilité, plutôt que rejeter une famille parce qu'elle n'a pas un
argmax fixe unique commun.

Le probe historique des 11 knobs notait lui-même `inner_steps=0` dans plusieurs
chemins de scoring. Des réglages de trainer/optimizer qui ne déclenchent aucun
update ne peuvent démontrer ni leur utilité ni leur inutilité. La répétition du
score seul n'est donc pas un test comparatif de différentes stratégies de
méta-optimisation. Nous avons revalidé la structure à une seule entrée, pas
reproduit une prétendue comparaison de trainers absente du protocole.

Les pistes historiques distinctes à séparer expérimentalement sont :

- **Réutilisation d'un ordre/prior entre tâches** : effet W2 reproduit sous son
  contrôle faible ; à tester contre un prior fixé informé et contre une vraie
  génération indépendante de même budget, en facturant l'apprentissage amont.
- **Optimisation directe de code** : surface non plate et programmes VRPTW
  meilleurs reproduits ; il faut un contrat portable et des traces exploitables.
- **Choix des exemples / minibatches / crédit / mémoire** : liveness à vérifier
  avant tout benchmark de qualité, avec plusieurs observations distinctes et une
  boucle d'updates réellement exécutée.
- **Sélection d'un trainer ou d'un moteur de recherche** : aucun des succès
  reproduits ci-dessus ne classe PrioritySearch, population, GEPA, OptoPrime ou
  une profondeur de récursion. Ce sont des comparaisons encore à faire, avec
  allocations contrôlées, pas des solutions historiquement validées.

### Contrôle actuel des stratégies réellement résolues

Un appel offline à `resolve_trainer` a vérifié une autre forme de menu dégénéré :

| Nom déclaré autorisé par `LevelConfig` | Classe résolue aujourd'hui |
|---|---|
| `MinibatchAlgorithm` | `ParetobasedPS` |
| `BeamsearchAlgorithm` | `ParetobasedPS` |
| `UCBSearchAlgorithm` | `ParetobasedPS` |
| `PrioritySearch`, `PrioritySearchMulti`, `ParetobasedPS` | Classe du même nom |
| `SequentialUpdate`, `SequentialSearch`, `BeamSearch` | Classe du même nom |

Les trois premiers noms ne correspondent pas à des classes exportées actuelles.
`opto/features/recursive_opt/optimize.py:resolve_trainer` applique explicitement
un fallback vers `ParetobasedPS`. **Un menu contenant ces trois labels ne compare
donc pas trois trainers.** L'observation est sauvegardée dans
`raw/trainer_resolution.json` ; ce fichier inspecte aussi la signature runtime,
masquée par le décorateur de configuration (`self, **kwargs`).

La lecture du corps de `PrioritySearch.train` confirme les réglages utilisables
sans créer un second framework : `num_candidates`, `num_proposals`,
`score_function="mean"/"ucb"`, `long_term_memory_size`,
`short_term_memory_size`, `memory_update_frequency`, `decouple_optimizers`, et
`use_best_candidate_to_explore`. Leur efficacité reste à comparer. Les wrappers
classiques emploient encore certains anciens noms (`memory_size`,
`validate_proposals`) alors que le corps actuel initialise les mémoires avec les
nouveaux noms : leur sémantique complète exige un test d'exécution avant adoption.

Pour le prochain diagnostic de largeur de recherche, la modification minimale est
donc de conserver `PrioritySearch` et de varier ses paramètres actuels vérifiés,
en comptant réellement toutes les propositions, les évaluations et les parents
explorés. Changer seulement le nom du trainer serait une expérience insuffisante.

L'évaluateur déterministe est un avantage métrologique, pas une impossibilité
d'améliorer la qualité. Le fait qu'il ait une variance de replay nulle ne supprime
ni la stochasticité de génération, ni l'incertitude de sélection sur peu de tâches.
L'ancien langage « W1 impossible à bruit nul » ne doit pas devenir une conclusion
générale d'impossibilité de gain à budget égal sur les objectifs déterministes.

## Claims historiques qui ne doivent pas revenir comme recommandations

Les historiques de retraction ont été inspectés, sans relancer de nouveaux appels
LLM sur les anciennes tâches prose :

- **UC4 +0,163** : le commit `40333f5f` documente des bras évalués sur des ensembles
  différents et des artefacts identiques. La comparaison corrigée dans Probe B
  rapporte une différence appariée −0,005965. C'est une relecture des traces et
  rapports préservés, pas une nouvelle estimation de la performance prose.
- **Probe K +4,8** : rétracté pour artefact vide et bruit lié à la concurrence.
  Le gain ne doit pas être utilisé pour projeter une accélération de recherche.
- **Probes prose** : les budgets d'évaluation bornés ont réduit l'écart-type
  GSM8K observé de 0,04145 à 0,01307 (ratio 3,17). C'est une amélioration de la
  mesure, pas un gain récursif ; le provider et ces expériences n'ont pas été
  rejoués ici. Aucun gain de latence actuel n'en est déduit.
- **EXP-11 / BBEH** : des effets proches de 1e−5 sur un score proche de 1 ne
  fondent pas une recommandation de batch. La comparaison à un plancher de bruit
  et le rejet des backends invalides restent nécessaires.

## Git, environnement et reproductibilité

Commits identifiés et lus :

| Sujet | Commit historique |
|---|---|
| Audit de mesure et UC4 | `40333f5ff4ca049163b4330a3f1150e481d5d9c2` |
| Menu et surface | `113336ef7e02b0225502f0df1cd80e5400355c1f` |
| Routing / optimum partagé | `c0d9e26685981f86780e641cf0db0ac0e01f4de8` |
| W2 avec template commun | `0f8c0ffbb240c1f62ac4c150bedfda2d77fe8cf8` |
| Generation et transfert réels | `a68d2139418ecaaca8ac6c811a9b2afee7268825` ; données enregistrées dans `f8a08541` |
| Limites des knobs et du pool | `4f086065853ca98fcd6750f8aa35b8d0b8806e9f` |

Replay : Trace `13ebda2242e1c18022591737b113030ca2ce2da2`, Python 3.13.13,
`/tmp/phase0-venv/bin/python`, Trace-Bench
`716cf82674d0dcb6a62c3f7b72a87b571bf35aa1`. Les fichiers non suivis préexistants
de Trace-Bench sont décrits dans la provenance ; aucune modification suivie n'y
est présente. Les hashes des sources, résultats d'origine et protocole sont dans
`raw/replay_01/provenance.json`. Le replay utilise le bridge courant et vérifie
ses sorties contre les données anciennes ; ce n'est pas la reconstruction d'un
ancien environnement Python complet.

Commandes exécutées :

```bash
/tmp/phase0-venv/bin/python -m pytest -q tests/unit_tests/test_investigation16_history.py
/tmp/phase0-venv/bin/python -m black --target-version py313 artifacts/optimizer_discovery/investigation16/history/retest_history.py tests/unit_tests/test_investigation16_history.py
/tmp/phase0-venv/bin/python -m ruff check artifacts/optimizer_discovery/investigation16/history/retest_history.py tests/unit_tests/test_investigation16_history.py
/tmp/phase0-venv/bin/python -m artifacts.optimizer_discovery.investigation16.history.retest_history artifacts/optimizer_discovery/investigation16/history/raw/replay_01
```

Les trois tests sont verts (1,65 s au lancement du replay). La phase rouge
précédente échouait à l'import du module encore absent, conformément au test-first.
Black et Ruff passent. Le script refuse de réécrire un répertoire de résultats
existant ; pour reproduire à nouveau, fournir un nouveau nom `replay_02`.

Évidence : `raw/replay_01/routing_summary.json`, `transfer_summary.json`,
`dataset_structure.json`, `routing/`, `transfer/`, `menu/`. Les sorties brutes
permettent de recalculer les tableaux et de vérifier la validité séparément des
objectifs. Aucun nouveau token modèle, aucune dépense provider.
