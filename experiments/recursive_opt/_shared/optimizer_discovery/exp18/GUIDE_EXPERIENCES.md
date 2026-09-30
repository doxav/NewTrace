# Comprendre les expériences d’optimisation de programmes

Ce guide explique les mécanismes enregistrés d’EXP-16, EXP-17 et EXP-18. Il ne présente aucun résultat principal et ne modifie aucun protocole. La question est : **quelle information aide un modèle à écrire un meilleur programme de recherche numérique ?** Elle se distingue de la question « combien de niveaux de récursion faut-il ? », qui n’est pas testée ici.

## Ce qui est optimisé

Le modèle écrit le contenu d’un fichier `optimizer.py`, dont l’interface est :

```python
def propose(history, bounds, seed):
    return x
```

Le programme reçoit les points déjà essayés et leurs valeurs, les bornes autorisées et une graine aléatoire locale. Il propose un nouveau point. L’évaluateur, dans le processus hôte, calcule sa valeur puis enrichit l’historique. Le programme ne reçoit ni fonction objectif, ni paramètres cachés, ni optimum, ni nom de famille ou de partition. L’implémentation du [contrat et du sous-processus](../../../../../opto/features/recursive_opt/optimizer_program.py#L208) et la [boucle d’évaluation](../benchmark.py#L219) matérialisent cette séparation.

Deux optimisations sont donc imbriquées :

```text
Recherche extérieure : modèle → nouveau code propose → évaluation TRAIN
                                  ↑                         │
                                  └── parent / retour ──────┘

Dans chaque évaluation : propose(history) → point x → valeur f(x)
                              ↑                          │
                              └──── historique ──────────┘  × 32
```

La surface modifiable est le **code de la politique** : exploration uniforme, exploitation du meilleur point, choix des coordonnées, rayon de perturbation, adaptation à l’historique, gestion des bornes, etc. Les fonctions numériques restent fixes. Le programme déployé ne rappelle pas le modèle.

## Les six types de problèmes

Il existe trois familles, chacune en deux et quatre dimensions : six combinaisons, avec plusieurs instances par combinaison. Pour une coordonnée, posons `z_i = (x_i − décalage_i) / échelle_i` et `y_i = 1 + z_i`.

| Famille | Fonction avant multiplication par l’amplitude | Difficulté visée |
|---|---|---|
| Sphere transformée, 2D et 4D | `Σ z_i²` | Trouver un centre caché ; les échelles peuvent déjà rendre les axes inégaux. |
| Quadratique anisotrope, 2D et 4D | `Σ poids_i × z_i²` | Adapter la recherche à des sensibilités très différentes selon les axes. |
| Rosenbrock transformée, 2D et 4D | `Σ [100(y_(i+1) − y_i²)² + (1 − y_i)²]` | Progresser dans une vallée courbe avec des coordonnées couplées. |

Les bornes sont `[-5, 5]` par coordonnée. Les décalages sont tirés dans `[-2, 2]`, les échelles dans `[0,75 ; 1,5]`, l’amplitude entre `0,1` et `10`, et les poids quadratiques entre `1` et `1 000` selon les distributions du [générateur](../investigation16/generation.py#L30). L’optimum connu de l’évaluateur est le décalage, de valeur zéro, à l’intérieur des bornes. Les [formules exécutées](../benchmark.py#L88) ne comprennent pas de rotation cachée des axes.

L’objectif est la performance **pendant** la recherche, pas seulement le dernier point : moyenne, sur les 32 évaluations, de la meilleure valeur déjà obtenue, normalisée par une référence indépendante. Plus cette regret-AUC est faible, mieux c’est. Une amélioration précoce contribue plus longtemps. La [normalisation et la métrique](../benchmark.py#L111) sont communes aux programmes et restent privées.

**Quel lien avec des applications complexes ?** Ce sont des problèmes élémentaires de diagnostic, pas un échantillon représentatif de la recherche scientifique. La quadratique correspond à une forme matricielle diagonale : elle permet d’étudier les différences d’échelle entre directions, mais ne teste ni des matrices denses inconnues ni des contraintes comme le rang ou la positivité d’une matrice. Rosenbrock ajoute des interactions non linéaires. Aucun de ces cas ne couvre les grandes dimensions, les observations bruitées, les expériences coûteuses, la sélection d’outils, les hypothèses scientifiques ou des décisions sur des centaines d’étapes. Le contrat portable et l’évaluation comparative sont des préalables réutilisables ; le transfert des gains éventuels vers ces applications demanderait d’autres expériences.

Les replays historiques de routage et de VRPTW de l’enquête EXP-16 constituent un volet séparé : ils réexécutent des politiques conservées sur leurs anciennes fixtures. Ce ne sont pas des bras ni des tâches ajoutées à EXP-17/18 ; voir le [bilan de l’enquête](../investigation16/REPORT.md).

## Tâches, trajectoires, programmes et répétitions

Ces quatre unités répondent à des questions différentes :

| Unité | Signification |
|---|---|
| Instance ou tâche | Une fonction avec dimension et paramètres fixés. |
| Trajectoire | Un programme sur une instance avec une graine locale : jusqu’à 32 évaluations. |
| Programme candidat | Une version complète de `propose`, testée sur plusieurs trajectoires. |
| Répétition extérieure | Une nouvelle recherche de programmes par le modèle ; l’unité de comparaison entre bras. |

Pour EXP-17 et EXP-18, le [plan commun](../exp17/PREREG_EXP17.md#L36) prévoit :

| Partition | Instances | Graines locales par instance | Trajectoires par programme |
|---|---:|---:|---:|
| TRAIN | 24 | 2 | 48 |
| Validation | 12 distinctes | 2 | 24 |
| Audit final | 12 autres | 2 | 24 |

TRAIN guide la recherche. Après toute la génération, la validation choisit dans chaque bras le programme final parmi le programme initial et les candidats admissibles. Toutes les sélections sont figées avant l’audit, qui mesure le comportement sur des instances réservées. Ces [barrières sont exécutées](../exp17/study.py#L350), pas seulement annoncées.

Les mêmes graines locales sont partagées entre programmes et bras d’une comparaison. Une graine extérieure ne rend pas les réponses du modèle déterministes. Les panels restent fixes dans chaque étude : 46 recherches ne signifient pas 46 nouveaux panels d’apprentissage. Un candidat invalide conserve son emplacement ; ses évaluations inutilisées ne financent pas des propositions supplémentaires.

## Le parent et les variantes comparées

Un **parent** est un programme antérieur dont le code est remis au modèle pour produire une nouvelle version. Exemple purement illustratif : un parent explore uniformément une fois sur quatre et perturbe sinon le meilleur point avec un rayon fixe. Un enfant pourrait diminuer ce rayon avec l’historique. Ce changement constitue une proposition à évaluer, sans présumer d’amélioration. Le parent retenu peut être ancien ; il n’est pas forcément le dernier enfant produit.

Dans le dispositif historique EXP-16 P1, les [messages effectifs](../investigation16/search_experiment.py#L672) distinguent :

| Bras | Information et organisation |
|---|---|
| I | Générations indépendantes ; contrat et code initial communs, aucun résultat antérieur. |
| C | Réécriture du parent sélectionné sur TRAIN, sans retour chiffré explicite dans le message. |
| R | Parent et retour détaillé sur ses trajectoires TRAIN. |
| W | Retour détaillé, avec deux parents et quatre cycles, au lieu d’un parent et huit étapes pour R. |

W change donc aussi l’organisation largeur/profondeur. EXP-17 reprend précisément I contre C : huit propositions par bras et 46 répétitions extérieures. **C reçoit une information indirecte par le choix du parent**, mais son [constructeur de messages](../exp17/study.py#L367) n’insère pas les scores ni les trajectoires.

EXP-18 enregistre quatre bras exploratoires, chacun avec un parent par étape, 16 propositions et six répétitions :

| Bras | Choix du parent | Contenu supplémentaire commun ou historique |
|---|---|---|
| L | Meilleur score TRAIN agrégé | Résumé compact du parent. |
| M | Identique à L | Même résumé et mémoire des essais précédents. |
| P | Tirage uniforme parmi les parents non dominés | Résumé compact. |
| PM | Identique à P | Même résumé et mémoire. |

Pour P/PM, un programme domine un autre s’il n’est moins bon sur aucune instance TRAIN et est meilleur sur au moins une. Le vecteur comporte 24 moyennes, chacune sur deux trajectoires. Le [protocole Pareto](PREREG_EXP18.md#L96) change ensemble filtrage et échantillonnage des parents. EXP-18 n’a pas de bras indépendant à 16 propositions ; il ne suffit donc pas à comparer ses mécanismes à cette référence.

## Ce que Trace transmet réellement

Le chemin de production est bien utilisé : le [composant source devient un nœud entraînable](../../../../../opto/features/recursive_opt/spec.py#L159), les opérations `@bundle` relient sortie et [évaluation](../../../../../opto/features/recursive_opt/spec.py#L1275), puis le trainer effectue `backward` et `step`. L’adaptateur lit effectivement [`trace_graph.user_feedback`](../investigation16/trace_schedule.py#L126).

Cependant, **le graphe complet n’est pas automatiquement envoyé au modèle**. Le programme généré s’exécute dans un sous-processus ; ses branches, variables intermédiaires et opérations ne deviennent pas chacune des nœuds Trace. L’optimiseur injecté construit ses propres messages, sans utiliser automatiquement une présentation complète du graphe de type OptoPrime.

Le retour détaillé R/W comprend, pour chaque trajectoire, premier point, meilleur point, améliorations, courbe du meilleur résultat, répétitions, stagnation et contact avec les bornes : voir la [projection exacte](../investigation16/feedback/rich_feedback.py#L63). Il ne contient pas tous les états internes du programme.

Dans EXP-18, le [retour compact](study.py#L127) contient seulement identité du code, état d’exécution, nombres de trajectoires allouées/observées/valides, statuts typés et AUC agrégée lorsqu’elle est définie. Les points et courbes détaillés en sont absents. Les `stdout`/`stderr`, capturés dans les [preuves d’exécution](../benchmark.py#L253), ne sont pas transmis. Un fichier nommé `trace.json` conserve plan, résultat et journaux de l’orchestration ; son existence ne démontre pas leur présence dans le prompt.

Ainsi, « feedback pauvre » dépend du bras : aucun retour explicite pour C, résumé global pour L/M/P/PM, trajectoires détaillées pour R/W. Leur richesse relative est descriptible ; son effet utile demande une comparaison dédiée.

## Pourquoi sept programmes ne constituent pas un curriculum de sept tâches

Le `CurriculumBuffer` fourni dans la discussion n’est pas intégré à ce chemin. Son mode pool fixe sélectionne des exemples dans un pool ; son autre mode associe l’exemple courant aux derniers exemples de l’historique. Avec `history_size=2`, cela représente au plus trois exemples disponibles, indépendamment d’un `batch_size` supérieur.

Deux détails du code sont essentiels. Dans le mode pool fixe, `sample_batch` n’utilise pas l’historique alimenté par `add_success_after_fail`. Dans le mode curriculum, cet historique ne sert que si l’entraînement appelle effectivement cette méthode. Son nom ne vérifie pas lui-même la transition échec → succès : le code appelant doit la détecter et lui transmettre l’exemple pertinent. EXP-17/18 n’effectuent pas cet appel ni cette sélection.

Ici, le [dataset extérieur](../investigation16/trace_schedule.py#L174) contient un descripteur du panel TRAIN. L’évaluateur développe systématiquement ce panel entier. Le batch extérieur n’échantillonne pas cinq des 24 tâches.

La [mémoire M/PM](memory_projection.py#L246) choisit au plus sept **sources complètes distinctes récentes**, dans le même bras et la même répétition, en excluant le parent déjà affiché. Elle conserve aussi les résumés typés des emplacements antérieurs. Le plafond est de 65 536 caractères ; une source non affichée est signalée, sans découpage silencieux. Ce choix utilise la récence, pas la pertinence, la diversité comportementale ni un événement `success_after_fail`.

Choisir judicieusement cinq tâches diagnostiques et choisir sept anciens programmes sont donc deux interventions différentes. Ces expériences ne montrent ni qu’il faut au moins sept exemples ni qu’au-delà de cinq ils sont inutiles. Leur nombre optimal et un curriculum fondé sur la pertinence restent non validés ici. Enfin, ces expériences utilisent une stratégie extérieure fixée : elles n’optimisent ni cette stratégie elle-même ni une profondeur de récursion supplémentaire. Les conclusions futures devront respecter ces limites, ainsi que celle du sous-processus : il ne constitue pas un sandbox de sécurité du système d’exploitation.
