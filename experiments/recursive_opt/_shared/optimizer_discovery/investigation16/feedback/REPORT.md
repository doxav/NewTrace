# Audit du feedback et du chemin de recherche d’EXP-15

Statut : diagnostic exploratoire rétrospectif, limité aux sources, requêtes,
trajectoires TRAIN et VALIDATION déjà conservées. Aucun holdout n’est lu par
`audit.py`. Aucun appel réel au modèle, aucune modification d’EXP-15, aucun
changement de dépendance ou de code de production. Les tests avec client simulé
ci-dessous sont des tests du mécanisme logiciel, pas des résultats de découverte.

## Ce qui est établi

1. **L’objectif exact n’est communiqué dans aucune des 80 requêtes.** Les
   instructions demandent de minimiser une fonction avec 32 évaluations, mais
   n’expliquent ni l’objectif *anytime*, ni l’AUC, ni l’importance de trouver de
   bonnes valeurs tôt. Le modèle peut raisonnablement privilégier la valeur
   finale. Le trainer, lui, classe effectivement les parents par AUC TRAIN.
   Il existe donc un décalage vérifié entre l’information donnée au générateur
   et le critère utilisé pour choisir ses propositions.

2. **Le feedback A2 contient six tâches, pas une seule.** Une ligne du dataset
   Trace encapsule un panel de six trajectoires, soit 192 observations avec B=32.
   Chaque requête A2 montre seulement les observations 1, 2, 31 et 32 de chaque
   trajectoire, plus le minimum brut rencontré. Cela représente 24 observations
   sur 192, soit 12,5 %. Les instants de découverte du minimum, la courbe
   best-so-far et 28 observations intermédiaires par tâche sont absents.
   Passer arbitrairement de six à sept exemples ne corrigerait pas ce mécanisme.

3. **Cette projection peut effacer exactement une différence majeure de vitesse.**
   Un test appelle la vraie méthode `Experiment.feedback` sur deux trajectoires
   valides de la même fonction x² : la première atteint zéro à l’évaluation 3,
   la seconde à l’évaluation 31. Les premières valeurs valent 100. Les AUC brutes
   sont 6,25 et 93,75, soit un facteur 15, mais les feedbacks sont identiques.
   C’est une démonstration de perte d’information du résumé ; ce facteur 15
   n’est ni un gain d’EXP-15, ni une projection d’un futur gain du modèle.

4. **Le contenu textuel propagé par Trace n’est pas consommé par le générateur.**
   Le vrai `SlotOptimizer` exige une présence de feedback, récupère le parent,
   puis `owner.proposal` reconstruit son propre résumé par le cache du panel.
   Un test injecte un marqueur dans `EvaluationResult.feedback`, traverse le vrai
   chemin `generate → Control Plane → PrioritySearch → SlotOptimizer`, et constate
   que le marqueur n’est présent dans aucune requête. Le résumé reconstruit est
   toujours présent. A2 utilise bien les parents et les résultats d’entraînement
   de la production ; cela ne démontre pas que le riche contenu Trace est exploité.

5. **La troncature de caractères n’a pas contribué à EXP-15.** Les 40 feedbacks
   reconstruits correspondent exactement aux requêtes préservées. Leur longueur
   est comprise entre 2 894 et 6 001 caractères, sous la limite de 12 000.
   Zéro JSON tronqué. C’est distinct de la limite de tokens de génération, qui
   doit être examinée séparément.

6. **L’information du dernier essai est souvent difficile à attribuer.** Sur les
   35 requêtes après la première, 7 répètent exactement le feedback `current`
   dans `previous_attempt`. Dans 22 autres requêtes, le dernier programme est
   différent du parent et non vide, mais son code n’est pas inclus ; le modèle
   voit son résultat sans voir la modification qui l’a produit. Le code du parent
   reste disponible. Les autres six cas concernent notamment des réponses sans
   source, pour lesquelles une recherche d’absence d’une chaîne vide n’a pas de
   sens ; le compteur 22 les exclut explicitement.

7. **La recherche teste une configuration étroite de recursive_opt.**
   `num_candidates=1`, `num_proposals=1`, `use_best_candidate_to_explore=True`.
   La méthode `PrioritySearch.explore` reprend systématiquement le meilleur
   incumbent : le code d’exploration d’autres membres de l’archive n’est pas
   exécuté. Un test sur la vraie méthode vérifie que modifier la priorité d’un
   autre candidat ne change pas le parent ; passer à deux candidats permet une
   branche. `long_term_memory_size=None` signifie archive **illimitée**, et non
   absence de mémoire. Cette archive de recherche n’est pas une liste d’exemples
   transmise au modèle.

8. **Huit propositions ne signifient pas huit améliorations successives.**
   Vingt des 40 propositions A2 partent du seed. Les profondeurs des parents
   sont 0/1/2 pour respectivement 20/6/14 propositions ; aucune ne part d’un
   parent de profondeur supérieure à 2. Sept propositions seulement améliorent
   strictement l’AUC TRAIN de leur parent. Cela n’est pas un dysfonctionnement
   de sélection : le protocole choisissait précisément le meilleur parent TRAIN.

## Classement et sélection : fragilité mesurée, causalité encore ouverte

Les corrélations de rang et inversions ci-dessous portent uniquement sur les
programmes éligibles, seed inclus. Les programmes invalides restent tous dans
`audit_results.json` avec des valeurs manquantes typées. Le nombre de paires
inclut les égalités ; une égalité n’est pas comptée comme inversion. Les tâches
ou les candidats ne sont pas traités comme des réplications indépendantes.

| Seed | Bras | Taille du pool éligible | Spearman TRAIN/VAL | Inversions/paires | Meilleur TRAIN | Sélection VAL | Rang TRAIN de la sélection |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 11 | A1 | 5 | 1,000 | 0/10 | 1 | 1 | 1 |
| 11 | A2 | 7 | 0,000 | 10/21 | 2 | seed | 7 |
| 23 | A1 | 8 | 0,275 | 11/28 | 6 | 2 | 2 |
| 23 | A2 | 7 | 0,891 | 2/21 | 1 | 0 | 2 |
| 37 | A1 | 8 | 0,850 | 3/28 | 1 | 1 | 1 |
| 37 | A2 | 5 | 0,900 | 1/10 | seed | seed | 1 |
| 41 | A1 | 8 | 0,786 | 5/28 | 4 | 0 | 3 |
| 41 | A2 | 6 | 0,543 | 4/15 | 4 | 5 | 2 |
| 53 | A1 | 7 | -0,357 | 13/21 | 3 | 7 | 6 |
| 53 | A2 | 8 | 0,690 | 6/28 | 4 | 6 | 3 |

Les indices de proposition commencent à zéro. Seulement trois des dix pools
sélectionnent le même programme par TRAIN et VAL. Au total, 55 des 210 paires
présentent une inversion. Cela établit un désaccord sur les panels observés,
pas qu’une sélection TRAIN aurait amélioré la généralisation. Il faut séparer
variance d’instance, variance du seed local, spécialisation de programme et
effet du petit panel dans une réévaluation sur des données neuves.

Le décalage AUC/valeur finale apparaît aussi dans les résultats de candidats :
23 des 95 paires éligibles A2 sont ordonnées en sens opposé par l’AUC TRAIN et
le regret final TRAIN. Cette observation rend le décalage d’instructions
plausiblement conséquent, sans prouver qu’il explique la comparaison A2/A1.

Deux exemples importants sont les sélections A2 des seeds 41 et 53 : leurs
programmes ont un meilleur résultat VAL que leur parent mais un moins bon AUC
TRAIN. Ils sont donc conservés dans le pool final mais ne deviennent pas les
parents des propositions suivantes. Il serait erroné de les supprimer des
résultats ; EXP-15 les a bien conservés conformément au protocole.

## Ablations minimales recommandées

Ces pistes nécessitent encore des propositions réelles appariées et une validation
indépendante. Aucune n’est présentée comme un gain projeté acquis.

1. Donner **aux deux bras** le même objectif explicite : minimiser la moyenne
   best-so-far sur le budget, avec une définition de l’agrégation. Garder les
   constantes de normalisation et les optima hors de l’API des programmes.
   Tester séparément le contenu du feedback A2 pour ne pas confondre deux changements.
2. Comparer le résumé EXP-15 à un résumé déterministe qui donne un score TRAIN
   réellement aligné, la progression aux budgets 1/2/4/8/16/32, les instants
   d’amélioration et des propositions représentatives avec leur index. Ne pas
   exposer automatiquement les constantes de benchmark : utiliser des valeurs
   brutes ou une référence observable commune si nécessaire. Toute nouvelle
   information dérivée des optima doit être déclarée dans la nouvelle expérience.
3. Relier `previous_attempt` à un hash, un index et son code ou un diff borné.
   Dédupliquer explicitement quand il s’agit du parent courant. Un résumé de
   trois essais pertinents et contrastés est une ablation de mémoire possible ;
   ce n’est pas un mécanisme actuellement testé dans EXP-15.
4. Tester la robustesse de classement sur 12 ou 24 instances d’entraînement
   équilibrées et plusieurs seeds locaux, avant de conclure à un seuil de
   « six/sept traces ». Un simple accroissement de `batch_size` sur la ligne
   unique contenant tout le panel ne crée pas de diversité supplémentaire.
5. Comparer recherche gloutonne, deux parents et plusieurs propositions par
   parent via `PrioritySearch` existant. Fixer le même nombre total de réponses
   et d’allocations d’évaluation. Élargir le nombre de parents est nécessaire
   pour que des stratégies de priorité ou de diversité puissent agir dans
   cette configuration ; changer uniquement `score_function` ne suffit pas
   lorsque seul le meilleur incumbent est toujours exploré.
6. Adapter le nombre d’itérations en fonction du budget total réellement
   consommé : ne pas multiplier silencieusement réponses et évaluations en
   augmentant simultanément `num_candidates`, `num_proposals` et les batches.
   Auditer la vraie production, car une itération initiale et les tailles
   variables du pool rendent une simple formule théorique insuffisante.

Il n’est pas justifié de conclure que davantage de profondeur récursive,
un autre modèle, un budget intérieur supérieur ou un benchmark plus difficile
produirait un avantage. EXP-15 mesure une stratégie de génération de code,
avec un seul composant entraînable et un évaluateur de trajectoires ; il ne
compare pas l’ensemble des stratégies de méta-optimisation de recursive_opt.

## Reproduction

```
/tmp/phase0-venv/bin/python artifacts/optimizer_discovery/investigation16/feedback/audit.py
/tmp/phase0-venv/bin/python -m pytest -q artifacts/optimizer_discovery/investigation16/feedback/test_audit.py
/tmp/phase0-venv/bin/python -m pytest -q tests/unit_tests/test_recursive_exp15.py::test_production_trace_budgets_isolation_and_selection
```

Le script reconstruit les 40 feedbacks A2 octet pour octet, conserve les 80 slots,
et sauvegarde les hashes des requêtes/pools lus. Les fonctions de projection,
de propagation et d’exploration testées sont celles d’EXP-15 et de production.
Les conclusions sur l’efficacité du modèle restent ouvertes : ces tests
établissent les mécanismes logiciels et les désaccords observés, pas leurs
effets causaux sur une nouvelle comparaison scientifique.

## Module prospectif livré après l’audit

`rich_feedback.py` fournit une projection réutilisable, distincte d’EXP-15 :

```python
from artifacts.optimizer_discovery.investigation16.feedback.rich_feedback import (
    ANYTIME_OBJECTIVE, build_feedback, serialize_feedback,
)

payload = build_feedback(
    current_source, current_train_rows, bounds_by_trajectory,
    previous_source=previous_source, previous_rows=previous_train_rows,
)
text = serialize_feedback(payload, max_chars=64000)
```

Les arguments du dernier essai sont facultatifs et doivent être fournis ensemble.
Le code courant reste séparé dans le prompt ; son hash figure dans le payload.
Un dernier essai différent fournit son code exact et son hash ; un essai identique
est référencé sans dupliquer le panel. `ANYTIME_OBJECTIVE` doit être donné
identiquement aux bras comparables dans le contraste qui l’active.

Chaque trajectoire expose uniquement ses bornes légales, validité et statut
typiqué, budget, point initial, incumbent avec instant de découverte, tous les
événements d’amélioration, courbe best-so-far brute complète et compteurs de
diversité, répétition, frontière et stagnation. La moyenne anytime brute est
absente pour une trajectoire incomplète. Les poids, normalisations, optima,
familles, identifiants, seeds locaux et logs du host ne sont jamais copiés.
Un payload trop long est rejeté avec une erreur explicite, sans tronquer le JSON
ou supprimer des tâches. La limite doit être enregistrée avant le run ; elle ne
doit pas être augmentée en fonction des résultats scientifiques.

La compatibilité a été vérifiée sur les 90 candidats/pools TRAIN archivés
d’EXP-15, soit 540 trajectoires valides ou invalides. Tous les payloads courants
sont valides et conservent leurs six tâches ; le plus long fait 22 840 caractères.
`rich_compatibility.json` conserve le détail et les hashes des quatre fichiers
Python au moment de ce contrôle. Les essais prospectifs ne sont pas encore des
résultats validant un gain du modèle.

Contrastes proposés, sans confusion de variables : code seul avec instruction
legacy ; code seul avec objectif anytime explicite ; feedback sparse avec objectif
explicite ; feedback riche avec objectif explicite. Un même parent, le même panel
neuf et le même budget de réponse sont employés dans chaque bloc. La taille du
panel doit faire l’objet d’un contraste ultérieur distinct ; l’augmenter en même
temps que l’information de feedback empêcherait d’attribuer l’effet observé.

Développement : test rouge initial pour module absent ; trois échecs ciblés sur
les vrais statuts `shape_error`/`nonfinite` et sur un panel mal formé, corrigés
avant le vert. Le test d’injection de feedback a d’abord laissé une entrée de
registre globale qui faisait échouer la régression dans le même processus ;
son registre est maintenant isolé et restauré par `monkeypatch`. C’était un défaut
du nouveau test, sans changement de production ni impact sur des données live.

Commande finale exécutée :

```
/tmp/phase0-venv/bin/python -m pytest -q artifacts/optimizer_discovery/investigation16/feedback/test_audit.py artifacts/optimizer_discovery/investigation16/feedback/test_rich_feedback.py tests/unit_tests/test_recursive_exp15.py::test_production_trace_budgets_isolation_and_selection
```

Résultat : **27 passed in 10.32s**. Ruff sur les quatre fichiers Python : tous
les contrôles passent. Black `--check --target-version py313` : les quatre fichiers
restent inchangés. Aucun fichier EXP-15 ou de production n’a été édité.
