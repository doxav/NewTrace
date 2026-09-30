# EXP-19 — traces, curriculum et vitesse d’apprentissage

**Suite désormais exécutée :** [EXP20 — résultats de la tâche documentaire à surfaces mixtes](../EXP20/RESULTS.md).
EXP19 reste la preuve historique et la fixture de diagnostic ; il n’est pas réécrit
comme preuve d’efficacité sur cette nouvelle tâche.

**Rechecked 16 September:** saved results reproduce exactly and the 47 new tests pass again. No live rerun is needed for recovery. See the [execution review and one-hour continuation plan](../_history/reviews/EXECUTION_PLAN.md).

**État : campagne terminée, résultats recalculés.** Les traces et le curriculum sont utilisables et testés. Le criblage initial ne montre pas d’accélération reproductible. Un diagnostic ultérieur isole un réglage utile du trainer : sélectionner le parent sur TRAIN pendant l’apprentissage, puis sélectionner la politique finale sur VALIDATION, atteint **100 % sur TEST en quatre réponses, sur trois graines**, contre **72,22 % en moyenne** pour le standard au même budget LLM. C’est un signal local sur une petite tâche synthétique, pas une preuve de supériorité récursive générale.

## Cause des imports et réparation minimale

Il y avait **deux causes distinctes**. `/tmp/phase0-venv` n’a pas le SDK OTEL ; `humanllm` l’a déjà. En outre, le `__init__.py` du checkout IO importait immédiatement le frontend `opto.features.graph.graph_instrumentation`, absent de cet arbre. Cela rendait indisponibles des fonctions OTEL pourtant indépendantes de ce frontend. La disponibilité annoncée par `traces.py` masquait donc plusieurs causes sous un même drapeau.

Les exports IO sont maintenant paresseux, les imports de types/frontends sont différés et sysmon ne dépend plus du SDK OTEL. Les bundles émettent de vrais spans lorsqu’une session est active. La capture entoure effectivement le forward et l’évaluateur du control plane. Les limites d’événements, la fermeture des sessions, les exceptions et les appels asynchrones sont testés. Aucun SDK ou autre paquet n’a été installé.

Preuves : [exports IO](../../../opto/trace/io/__init__.py), [capture](../../../opto/features/recursive_opt/traces.py), [intégration du dict](../../../opto/features/recursive_opt/spec.py), [tests des backends et du curriculum](../../../tests/unit_tests/test_o1_trace_curriculum.py).

## Ce qui est utilisable maintenant

| Fonction | Ce qui est vérifié | Limite précise |
|---|---|---|
| OTEL | Imports paresseux ; spans réels des bundles synchrones/asynchrones et d’un graphe LangGraph | SDK déjà présent dans `humanllm`, absent de `/tmp/phase0-venv` ; aucune installation |
| sysmon | Fonctionne sans SDK OTEL en Python 3.13 ; appels/retours réels, événements bornés, sessions fermées | Python ≥ 3.12 ; capture du thread observé, pas des processus enfants |
| hybride | Capture des trois sources ; projection équilibrée pour qu’une source ne masque pas les autres | Juxtaposition des trois vues, pas fusion causale parfaite ; résumé borné |
| dict par niveau | Deux niveaux d’un même plan exécutent des modes distincts | Le graphe natif nécessaire à OptoPrime subsiste : les modes règlent la capture supplémentaire |
| curriculum | `add_success_after_fail` est appelé sur de vrais résultats TRAIN, puis le batch suivant change | Trainers Trace basés sur `SearchTemplate` ; pool indexé et seuil explicite ; GEPA/externe non modifiés |
| erreurs | Les exceptions/non-valeurs ne deviennent pas des réussites apprises | La validation sémantique du score reste la responsabilité de l’évaluateur |

Configuration à placer dans le dict canonique existant :

```python
level = spec["levels"][0]  # configurable indépendamment dans chaque niveau
level["objective"]["trace_config"] = {
    "mode": "hybrid",  # internal / otel / sysmon / hybrid
    "detail": "full",  # summary = noms ; full = nœuds et valeurs
    "credit_horizon": "episode",
    "max_nodes": 48,
    "max_chars": 4000,
    "semantic_names": ["forward", "execute", "apply_rule", "evaluate"],
}
level["engine"]["config"].setdefault("trainer_kwargs", {}).update({
    "batch_size": 5,
    "curriculum": {"history_size": 4, "success_threshold": 1.0},
})
```

Le trainer réévalue le précédent batch TRAIN après une mise à jour, observe les échecs devenus réussites, puis construit le batch suivant. Celui-ci contient au plus quatre exemples récents et au moins une place d’exploration. Le succès signifie un score Guide fini ≥ `success_threshold` : un objectif de minimisation doit fournir un score orienté en conséquence. Le Guide fournit le score habituel ; il n’a pas à appeler une méthode spéciale du buffer. Les réévaluations supplémentaires sont comptées. Les résultats VALIDATION ne remplissent pas le buffer.

`credit_horizon` désigne ici une **projection locale** : step/truncated = un nœud, episode = jusqu’à huit, full = jusqu’à `max_nodes`. Ce n’est pas une rétropropagation temporelle entre épisodes. La trace complète reste archivée ; depuis S3, elle ne contourne plus la projection dans le payload visible par l’optimiseur. Le vieux frontend `instrument_graph(backend="trace")` dépend toujours de `graph_instrumentation`, absent du checkout ; le graphe Trace natif et les frontends OTEL/sysmon testés fonctionnent.

## Tâche, surfaces et niveaux — ce que l’on optimise réellement

| Élément | Définition |
|---|---|
| Tâche O0 | Apprendre six tables de vérité de quatre bits, puis les appliquer à des expressions composées |
| TRAIN | Les 24 observations primitives : six opérateurs × quatre couples d’entrées |
| VALIDATION | 24 expressions nouvelles de profondeur 2, pour choisir les politiques |
| TEST | 48 expressions nouvelles de profondeur 3, ouvertes après gel de toutes les sélections |
| Surface | Six paramètres texte, ou un JSON contenant les mêmes 24 bits ; interpréteur fixe avec nœuds `apply_rule` |
| O1 | Choisir batch, trace, présentation, feedback, goal, optimiseur ou trainer de l’apprentissage O0 |
| O2 imbriqué | Choisir l’optimiseur d’O1 ; l’évaluation d’O1 lance réellement O0 via le control plane |
| Performance | Exactitude et nombre de réponses LLM jusqu’à 90 % ; non-atteintes conservées comme censurées |

Le three-way compare : état initial sans apprentissage / apprentissage standard / apprentissage avec réglage O1. Le diagnostic O2 est supplémentaire.

Cette fixture est **synthétique, pas BBEH**. Elle teste une petite composition de décisions et la circulation du feedback. Elle ne mesure ni découverte de code libre, ni optimisation de matrices, ni recherche scientifique de long horizon. Les nouvelles graines changent l’ordre TRAIN et les compositions, mais pas les six concepts : ce n’est pas une généralisation à de nouvelles tables.

Une mémoire spécialisée peut apprendre les tables en stockant les labels primitifs observés, sans LLM. C’est un contrôle diagnostique indispensable : le benchmark ne justifie pas à lui seul une recherche générative complexe. Les fonctions numériques Sphere/Quadratic/Rosenbrock permettent un curriculum d’instances, mais ne remplacent pas un benchmark de compétences symboliques diverses.

## Ce qui a été comparé

| Axe | Contrastes | État scientifique actuel |
|---|---|---|
| Batch | Tailles 1/3/5/7 ; replay récent de taille 2/4 | Gain ponctuel à 7, non reproduit ; curriculum S1 souvent inactif, repris activement en S2 |
| Trace | internal/OTEL/sysmon/hybrid ; détail et horizon | Capture active ; défaut de projection S1/S2, corrigé et retesté en S3 ; aucun gain sur le diagnostic S3 |
| Surface | Six textes / un JSON | Même espace de 24 bits ; ne teste pas une surface dynamique ni du code libre |
| Feedback | Erreur localisée / score | Petite différence de présentation : sur une sortie binaire, le score révèle déjà le label |
| Goal | Générique / préserver les bits déjà appris | Pas d’amélioration observée sur le criblage ; un seul contraste exploratoire |
| Optimiseur | OptoPrimeV2 / OPROv2 | Les échecs à 8 000 tokens ne permettent pas d’écarter OPRO ; reprise à 16 000 |
| Trainer | SequentialSearch / PrioritySearch / ParetobasedPS | Les deux premiers peuvent se confondre à largeur 1 ; contraste Pareto séparé sans gain observé |
| Récursion | Exécutions réellement imbriquées O2→O1→O0 | Faisabilité et choix de configuration ; coût préalable séparé, aucune preuve générale sur la profondeur |

Le batch modifie aussi la taille de validation échantillonnée par le trainer actuel. Le panel final de sélection est fixe. Les tailles de batch ne constituent donc pas un contraste pur de couverture TRAIN, ni un budget égal d’évaluations de tâche. Les menus sont bornés : aucun « optimum global » n’est revendiqué.

## Protocole et versions à ne pas confondre

- **S1** : criblage de 17 réglages, graine 19021, quatre réponses, plafond 8 000 ; trois contrastes supplémentaires isolent détail, horizon et trainer Pareto.
- **S2** : curriculum réellement actif, graine 19024 ; combinaison et ablations sur 19022 ; plafond 16 000 après diagnostic de troncature. Interruption avant TEST à cause du défaut de projection. Le rapprochement final retrouve la réponse O2 et une requête enfant O1/OPRO restée sans reçu ; voir `s2_interruption_reconciliation.json`.
- **S3** : projection corrigée ; quatre diagnostics de trace sur 19025 ; nouvelle exécution imbriquée de développement ; comparaison appariée à huit réponses sur 19061/19073/19079. Configurations choisies avant TEST, ordre tournant, trois processus isolés.

Les sélections de politiques utilisent VALIDATION exclusivement et sont toutes sérialisées avant TEST. Incertitude descriptive : bootstrap des trois graines appariées, 10 000 tirages, RNG 19090. Trois graines ne suffisent pas à établir une supériorité générale. Les coûts du criblage et de la récursion sont conservés séparément des coûts d’apprentissage appliqué.

Modèle unique : OpenRouter `deepseek/deepseek-v4-flash-0731`, température 0,6, top_p 1, reasoning low via `extra_body`, timeout 300 s, cache client désactivé. Les réponses vides/tronquées consomment leur budget ; pas de remplacement pour mauvaise qualité. Routage automatique identique, fournisseur réel enregistré. Les éventuels retries transport sont bornés ; leurs événements sont enregistrés explicitement depuis S3.

## Résultats S3 : le criblage ne prouve pas l’accélération

Trois graines nouvelles, huit réponses par bras ; 72 réponses au total. La configuration O1 choisie sur développement ne change que la présentation du feedback en score. La combinaison et son ablation avaient toutes deux donné 50 % sur leur graine de développement. Il n’y avait pas plusieurs mécanismes gagnants justifiant une combinaison plus complexe.

| Graine | Initial | Standard | O1 choisi | Configuration issue du diagnostic récursif |
|---|---:|---:|---:|---:|
| 19061 | 35,42 % | 47,92 % | 47,92 % | 47,92 % |
| 19073 | 41,67 % | 43,75 % | 50,00 % | 50,00 % |
| 19079 | 41,67 % | 52,08 % | 52,08 % | 50,00 % |
| **Moyenne TEST** | **39,58 %** | **47,92 %** | **50,00 %** | **49,31 %** |
| Médiane TEST | 41,67 % | 47,92 % | 50,00 % | 50,00 % |
| Atteinte de 90 % en ≤ 8 réponses | 0/3 | 0/3 | 0/3 | 0/3 |

O1 − standard : **+2,08 points**, deltas [0 ; +6,25 ; 0], intervalle bootstrap descriptif [0 ; +6,25]. Cela ne démontre pas une accélération. Le standard n’atteint pas non plus le seuil ; aucune non-atteinte n’est convertie en temps observé.

Le diagnostic réellement imbriqué O2→O1→O0 a consommé **10 réponses** en préparation. O1 propose Pareto, sans battre le standard : égalité à 54,17 % en validation, standard conservé. O2 consomme **16 000 tokens de raisonnement sans texte exploitable**. La configuration `recursive_selected` est donc **identique au standard** : son écart de +1,39 point [−2,08 ; +6,25] n’est pas un effet de profondeur. Par rapport à O1, son écart est −0,69 point [−2,08 ; 0]. Le flag de validité de la politique conservée ne signifie pas que la génération O2 a réussi.

Preuves : [résultats S3](../_shared/o1_learning/s3_results.json), [choix gelés](../_shared/o1_learning/s3_selection_frozen.json), [diagnostic imbriqué](results/s3/recursion_result.json).

## Résultats S4 : le mécanisme précis qui change l’apprentissage

**S4 ne teste pas le curriculum.** Les six runs S4 ont `curriculum=None` et zéro
événement de replay. Il compare uniquement l’utilisation de VALIDATION pendant
le fitting. Le curriculum failed→solved a été exercé séparément en S2
(`active_curriculum`) : 2 événements pour curriculum3, 0 pour curriculum5 et
6 pour curriculum7, sur une seule graine. Son efficacité n’est pas établie.
Le succès de S4 ne peut donc pas être présenté comme un gain du curriculum.

Un **parent** est l’état des paramètres donné au modèle pour produire la prochaine proposition. Exemple réellement enregistré : dans `paired_s3/standard_19079`, la réponse 0 corrige D de `0000` à `1000` ; la requête 1 fournit de nouveau D=`0000`. La correction disparaît pendant la sélection du parent. Elle n’a pas été effacée par la réponse suivante du modèle. [Requêtes/réponse et hashes](../_shared/o1_learning/parent_return_diagnostic.json).

Sur cette tâche, la validation porte sur des **compositions**. Une primitive correctement apprise peut ne pas encore améliorer ces compositions si d’autres primitives restent fausses. Sélectionner trop tôt selon ce signal peut écarter un progrès partiel. C’est une explication du comportement observé ; le contraste ci-dessous teste le changement du mécanisme complet de sélection pendant le fitting.

Le [protocole S4](../_shared/o1_learning/stage_s4_protocol.json) est enregistré avant ses appels : nouvelles graines, quatre réponses par bras, batch 6, même modèle/plafond de 16 000. Le seul champ modifié est :

```python
fit_spec["levels"][0]["datasets"]["validation"] = []
```

Cela supprime la validation du **fitting seulement**. Dans les deux bras, toutes les politiques sont ensuite évaluées sur le même panel externe VALIDATION ; les sélections de chaque préfixe sont gelées avant TEST. Il ne faut pas supprimer cette sélection externe. Le runner [selection_diagnostic.py](../_shared/o1_learning/selection_diagnostic.py) applique ce changement au dict existant et réutilise le trainer de production.

Batch 6 permet quatre batches couvrant les 24 primitives sans reshuffle. L’audit vérifie **six identités TRAIN identiques dans chacune des 12 paires de prompts**. L’ordre des bras alterne entre graines (deux dans un sens, une dans l’autre), avec trois workers. [Contrôle effectif des batches](../_shared/o1_learning/s4_batch_match.json).

| Graine | Initial TEST | Standard TEST, 4 réponses | Parent sur TRAIN TEST, 4 réponses | Gain | Réponses pour 90 %, standard / modifié |
|---|---:|---:|---:|---:|---|
| 19083 | 54,17 % | 68,75 % | 100 % | +31,25 pts | non atteint / 4 |
| 19097 | 43,75 % | 77,08 % | 100 % | +22,92 pts | non atteint / 4 |
| 19109 | 50,00 % | 70,83 % | 100 % | +29,17 pts | non atteint / 4 |
| **Moyenne** | **49,31 %** | **72,22 %** | **100 %** | **+27,78 pts** | **0/3 / 3/3 atteintes** |
| Médiane | 50,00 % | 70,83 % | 100 % | +29,17 pts | — |

Intervalle bootstrap apparié descriptif à 95 % du gain final : **[+22,92 ; +31,25] points**. Trois graines, mêmes concepts, tâche choisie pour le diagnostic : ce signal ne justifie pas une affirmation générale de significativité. On démontre ici une atteinte dans le budget pour le bras modifié et une non-atteinte pour le standard ; **aucun facteur ×2 ou ×N d’accélération n’est estimable** sans son temps d’atteinte.

| Budget réalisé S4, total des trois graines | Standard | Parent sur TRAIN |
|---|---:|---:|
| Réponses LLM | 12 | 12 |
| Évaluations internes de tâche | 444 | 234 |
| Évaluations externes VALIDATION | 336 | 360 |
| Tokens effectivement rapportés | 120 349 | 122 514 |
| Coût rapporté, USD | 0,010873 | 0,009371 |
| Réponses vides / échecs d’exécution candidats observés | 0 / 0 | 0 / 0 |

Les évaluations internes diminuent de 47,3 %, car on retire la validation échantillonnée du fitting. Le total fitting + sélection diminue de 780 à 594 (23,8 %). Ces ressources ne sont donc **pas égalisées**, contrairement aux réponses LLM et aux exemples TRAIN consommés. C’est précisément une intervention sur le trainer ; elle n’isole pas indépendamment le critère de sélection et son coût d’évaluation.

Les trois politiques modifiées sélectionnées sont les mêmes six tables correctes. Hash canonique JSON : `a0c6aa5c09819b75f465c58eb8bbd5f801cce1fe935abbfd00127697be539152`. Les trois configurations exactes, les hashes des contrôles et leurs courbes sont conservés dans [les sélections gelées](../_shared/o1_learning/s4_prefix_selections_frozen.json), [les résultats](results/s4_results.json) et [l’audit](../_shared/o1_learning/integrity.json). Les candidats évalués n’ont pas été réparés manuellement.

**Ce résultat est une amélioration O1 diagnostiquée par l’expérimentateur et retestée prospectivement. Ce n’est pas un réglage découvert automatiquement par O2.** Il n’établit pas une nouveauté algorithmique. S3 reste un résultat sans accélération ; S4 ne le remplace pas.

## Précision causale du 23 septembre : que fait réellement la validation ?

« Consulter VALIDATION » pendant le fitting signifie exécuter des candidats sur
ces exemples, calculer des scores et les utiliser pour choisir le parent de la
prochaine proposition. Cela influence la trajectoire de recherche ; ce n’est pas
une simple mesure affichée, ni nécessairement des exemples ajoutés au prompt LLM.

La lecture du trainer et la reconstruction offline des preuves montrent deux
mécanismes distincts. D’abord, TRAIN teste des primitives alors que VALIDATION teste
leurs compositions : un progrès sur les premières peut temporairement dégrader
les secondes. Ensuite, **le standard ne fait pas une comparaison pure, commune et
fraîche sur VALIDATION** : le parent accumule des rollouts TRAIN et VALIDATION,
le nouveau candidat commence avec des rollouts VALIDATION, et les deux appels au
sampler consomment des lots différents. `mean_score` agrège les observations
stockées sans distinguer leurs populations. Cela introduit un défaut de
comparabilité dans les scores de sélection.

Exemple exact, graine 19097, première proposition identique dans les deux bras :

| Mesure | Parent initial | Nouvelle proposition |
|---|---:|---:|
| Même batch TRAIN de six exemples | 3/6 | 6/6 |
| Ensemble TRAIN de 24 primitives, recomputé | 10/24 | 13/24 |
| Ensemble VALIDATION de 24 compositions | 4/24 | 1/24 |
| Score interne utilisé par le standard | (3 TRAIN + 0 VALIDATION)/12 = 0,25 | 0 VALIDATION/6 = 0 |

Le standard repart du parent initial ; la variante TRAIN repart de la proposition.
Les deux lots VALIDATION internes de six exemples sont disjoints. Le score complet
VALIDATION est également inférieur pour la proposition : le bruit des petits lots
ne suffit donc pas, à lui seul, à expliquer son rejet. Une sélection gloutonne
fondée sur tout ce panel la rejetterait aussi.

[Reconstruction, assertions et hashes](../_shared/o1_learning/s4_mechanism_review_20260923.json) ; code :
[stockage/moyenne des scores](../../../opto/trainer/algorithms/priority_search.py#L78),
[évaluation des candidats](../../../opto/trainer/algorithms/priority_search.py#L714),
[ajout des observations TRAIN](../../../opto/trainer/algorithms/priority_search.py#L967).
Aucun nouvel appel modèle ni changement des sources évaluées.

**La conclusion reste : désactiver cette utilisation de VALIDATION pendant le
fitting améliore cette procédure sur cette fixture.** Le delta final n’est pas
attribuable séparément à la population VALIDATION, au mélange des scores, à leur
ancienneté ou aux lots distincts. S4 change ces mécanismes ensemble. Les résultats
numériques restent valides pour l’intervention enregistrée ; ils ne constituent
pas une preuve que « la validation nuit à l’apprentissage » en général.

Avant une extrapolation, comparer la sélection TRAIN et la sélection VALIDATION
avec des scores séparés, des lots communs à tous les candidats et une même règle
d’actualisation des scores. Le contrôle standard historique peut être conservé
comme troisième traitement. Rejouer les propositions sauvegardées teste les
choix locaux sans appels LLM ; mesurer les performances finales des nouvelles
trajectoires nécessite une nouvelle expérience, car un autre parent change les
propositions suivantes.

Le champ responsable (`datasets.validation` pendant le fitting) n’était pas dans
le menu initial des réglages. Le candidat « O1_selected » de S3 ne modifiait
finalement que le feedback localisé en feedback scalaire. L’absence de gain robuste
de ce choix n’établit donc pas l’inefficacité du réglage des hyperparamètres.
S4 est précisément une amélioration O1 identifiée manuellement et retestée ;
aucun optimiseur de niveau supérieur ne l’avait découverte automatiquement.

## Causes établies, hypothèses encore ouvertes

| Explication proposée | Ce que les preuves permettent de conclure |
|---|---|
| Trop peu de tokens | Défaut réel sur certaines réponses : trois sondes nouvelles à 16 000 produisent du texte, dont une dépasse 13 000 tokens. Mais S3 reste faible à 16 000 et O2 peut encore épuiser ce plafond. Ce n’est pas la cause unique. |
| Trop peu d’exemples | Les trois standards S3 ont rencontré les 24 primitives avant la fin et restent sous le seuil. Une mémoire spécialisée exploitant ces mêmes labels atteint 100 %. La couverture seule n’explique pas leur échec. |
| Parent qui perd les progrès | Transition prouvée dans les requêtes ; S4 améliore les trois graines en modifiant la sélection pendant le fitting. C’est le signal causal local le plus utile de cette campagne. |
| Curriculum non branché | S1 ne suffisait pas à le tester. Réévaluation TRAIN ajoutée ; S2 enregistre 2 et 6 transitions pour curriculum3/7, mais aucun avantage de performance. Le mécanisme fonctionne ; son intérêt reste ouvert. |
| Trace plus riche | Imports/capture/projection corrigés ; les prompts diffèrent effectivement. Aucun gain sur le petit diagnostic S3. Les traces primitives sont courtes : cela ne réfute pas leur utilité sur un workflow long. |
| Plus de niveaux de récursion | Appels imbriqués réels, mais budget minuscule et O2 tronqué. Ni efficacité ni inefficacité générale établies. |
| Mauvaise tâche | Fixture utile pour isoler le flux de données et le parent, trop simple pour évaluer toute la promesse du framework. Les 24 bits peuvent être appris directement sans LLM. Aucun transfert vers matrices, BBEH ou recherche longue n’a été mesuré. |

## Décision pratique et limites de portée

1. **Utilisable maintenant :** dict par niveau pour les backends de trace et le curriculum, avec les limites documentées ci-dessus. Aucun changement de Guide spécifique au buffer n’est nécessaire.
2. **Réglage mesuré pour cette tâche :** batch 6, conservation du progrès sur TRAIN pendant le fitting, sélection externe VALIDATION séparée. Garder le budget 16 000 ; cela règle une contrainte observée, sans garantir une réponse.
3. **À ne pas imposer globalement :** retirer la validation de tous les trainers. Sur d’autres tâches, elle peut prévenir un vrai surapprentissage. Le résultat concerne le décalage primitives/compositions testé ici.
4. **Prochaine évaluation utile :** reprendre le workflow symbolique BBEH/PAL existant, avec un lot distinct de problèmes, et tester d’abord ce seul choix de sélection. Ensuite seulement, comparer curriculum activé et trace sémantique compacte à budgets LLM et évaluations comptés. Les adaptateurs de datasets existants sont dans `experiments/recursive_opt/multiobjective_reasoning/` ; ils n’ont pas été exécutés dans EXP19.
5. **Non couvert :** optimum global des huit axes, surface qui évolue dynamiquement, code libre, horizon long, GEPA et tous les trainers externes, preuve d’amortissement et apport de profondeur. Les contrastes de surface, goal et feedback sont étroits ; leurs résultats négatifs ne ferment pas ces pistes.

## Coûts, incidents et intégrité

La campagne entière conserve **255 réponses réelles uniques**, **2 332 712 tokens** (1 554 725 prompt ; 777 987 completion, dont 688 882 reasoning), coût fournisseur connu **0,21856749005 USD**. Le plafond initial de 8 000 reprenait le réglage validé en Phase 0 ; ce n’était pas une limite imposée par le modèle. Il a été relevé symétriquement à 16 000 après les sondes, avant S2/S3/S4. Plafonds : 84 réponses à 8 000, 171 à 16 000. Neuf réponses n’ont pas de texte exploitable (3,53 %) : huit fins `length`, une fin `error`. Elles restent dans les budgets. S3 comporte une réponse vide dans `recursive_selected` ; S4 aucune.

Une proposition du diagnostic `detail_only` S3 contient des tables mal formées et provoque trois observations d’exécution invalides. Elles sont conservées ; elles ne deviennent pas des scores de mauvaise performance ni des exemples résolus. Les politiques finales S3/S4 sont exécutables. Les 6 399 scores internes de tâche enregistrés et les 6 744 évaluations externes de sélection sont comptés séparément ; ce sont les appels des runs, **pas un total incluant chaque recalcul offline, pilote interrompu et évaluation diagnostique**. L’audit conserve les trois tentatives d’évaluation sans score.

Depuis S3, deux échecs de transport suivis chacun d’un retry et d’une récupération sont enregistrés, tous en S4. S4 a donc 24 réponses pour 26 tentatives connues. Les événements transport S1/S2 n’étaient pas capturés : ne pas inventer un total exact des tentatives de toute la campagne. La requête enfant O1/OPRO interrompue n’a pas de reçu ; son traitement/facturation distante reste inconnu. Le coût ci-dessus n’est pas une facture exhaustive. Le routage est automatique et identique entre bras ; les fournisseurs effectivement utilisés sont archivés.

Trois pilotes initiaux échouent avant réponse modèle et sont conservés. Le défaut de projection S1/S2 est corrigé avant S3, dont les sources sont archivées séparément. Une réparation de `update_progress` après gel S3 ne change que le rendu d’une ligne de résultat imbriqué : diff, hashes et test sont conservés dans `s3_reporting_repair.*`. Le recalcul vérifie les sources gelées et cette seule exception documentaire. Le rapprochement final corrige explicitement l’identifiant de la requête S2 restée en attente, sans modifier le constat historique initial.

L’objectif d’environ une heure a été dépassé. L’horloge UTC présente deux interruptions d’environ 2 h et 2 h 46 ; elles sont documentées dans `clock_interruption*.json`. Le travail actif dépasse lui aussi une heure. Les durées murales affectées par ces interruptions ne servent pas à revendiquer une accélération d’apprentissage.

## Vérification et navigation

- Nouveaux tests : **47 passed** dans `humanllm`, dont les backends OTEL ; une dépréciation LangGraph existante.
- Même suite sans SDK : **34 passed, 13 skipped** dans `/tmp/phase0-venv` ; les 13 skips optionnels ont tous été exécutés avec succès dans l’autre environnement.
- Régressions affectées après les corrections S3 : **325 passed**.
- Suite plus large exécutée avant la dernière correction S3 : **524 passed, 3 skipped** (deux backends optionnels, un exemple historique nécessitant un appel live). Elle ne constitue pas une exécution complète de tout le dépôt après S3.
- Recalcul S3/S4 : mêmes sélections, hashes, courbes et intervalles ; deux audits indépendamment exécutés donnent un JSON identique. Les 12 paires de batches S4 sont vérifiées avant TEST ; un test vérifie que toutes les sélections sont persistées avant le premier score TEST.
- `git diff --check`, Black et **Ruff 0.15.12 de humanllm** sur les fichiers ciblés passent. Le Ruff global 0.16.3 active un ensemble de règles plus large et signale **154 points** sur ce même périmètre ; sa sortie est conservée, sans suppression de règles ni réécriture des sources gelées. Scan de 1 210 fichiers/entrées d’archives : aucune clé réelle ni chaîne de credential détectée ; aucun `.env` staged. Les warnings de style préexistants des anciens trainers ne sont pas masqués.
- SHA départ/final : `7e701b40485b9880ccfbb64c1faaabb22401c294`. Branche `codex/exp17-parent-selection-exp18-memory-pareto`. Aucun commit, push ou ajout de dépendance. Le patch staged IO utilisateur reste identique au point de sauvegarde ; EXP17 reste suspendue.

Commandes principales (les commandes complètes, sorties et exclusions sont dans `verification.json` et `verification_logs.zip`) :

```bash
/home/xav/miniconda3/envs/humanllm/bin/python -m pytest -q tests/unit_tests/test_o1_trace_curriculum.py tests/unit_tests/test_o1_learning_study.py
/tmp/phase0-venv/bin/python -m pytest -q -rs tests/unit_tests/test_o1_trace_curriculum.py tests/unit_tests/test_o1_learning_study.py
/home/xav/miniconda3/envs/humanllm/bin/python -m artifacts.o1_learning.audit
```

| À lire/utiliser | Rôle |
|---|---|
| **Ce fichier** | Point d’entrée : causes, usage, résultats et limites |
| [study.py](../_shared/o1_learning/study.py), [selection_diagnostic.py](../_shared/o1_learning/selection_diagnostic.py) | Tâche, dict et réglage effectivement testé |
| [analysis.py](../_shared/o1_learning/analysis.py), [recursion.py](../_shared/o1_learning/recursion.py) | Sélection, analyse et niveaux imbriqués de production |
| [integrity.json](../_shared/o1_learning/integrity.json), [verification.json](../_shared/o1_learning/verification.json) | Recalcul, budgets, hashes, tests et contrôle des secrets |
| `raw/`, `s3/`, `sources_S*.zip`, protocoles JSON | Preuves gelées ; aucun besoin de les lire comme une suite de rapports |
| `EXP19_protocol_history.md.gz` | Protocoles/amendements historiques, archivés sans multiplier les pages de lecture |
