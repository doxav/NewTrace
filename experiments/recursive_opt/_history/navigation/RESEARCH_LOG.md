> Historical document, retired from active navigation on 2026-09-29. Original location: `artifacts/RESEARCH_LOG.md`. Use the [canonical experiment index](../../README.md) and [current assessment](../../ASSESSMENT.md). The [original bytes](RESEARCH_LOG.md.original.gz) are preserved; relative links below were rebased for this location.

# recursive_opt — acquis, corrections et usages possibles

**Point d’entrée actuel, révisé le 13 septembre 2026.** Le contrôle des expériences
et l’interface de programmes sont réutilisables. Plusieurs gains locaux sont
mesurés. **L’avantage général du feedback récursif sur une recherche indépendante,
et celui d’ajouter des niveaux de récursion, ne sont pas établis.**

La campagne numérique antérieure reste suspendue à la demande de Xavier. Une nouvelle demande a autorisé **EXP-19 : traces, curriculum et vitesse O1**, désormais terminée ; voir [le bilan dédié](../../EXP19/RESULTS.md). EXP-17 conserve
**545/736 réponses**, sans sélection finale, sans audit réservé et sans résultat
d’efficacité. EXP-18 est terminée et inconclusive sur ses mécanismes.
[État exact de la suspension](../../_shared/optimizer_discovery/exp17/USER_REQUESTED_PAUSE.json).
Une requête déjà transmise au fournisseur peut encore être traitée/facturée ; les
processus locaux sont suspendus en mémoire, sans reprise automatique.

Trois lectures suffisent : ce bilan ; l’[index des fichiers](RESULTS_INDEX.md) pour
retrouver une preuve ; le [brief Patrick](../../_shared/optimizer_discovery/exp18/PATRICK_BRIEF.md)
pour une discussion externe. Les sections 9–14 ci-dessous sont des inscriptions
historiques conservées ; leurs anciennes priorités ne remplacent pas ce bilan.

## 1. Reconstitution A à I : ce qui s’est réellement passé

| Étape | Correction et informations manquantes | Preuves |
|---|---|---|
| **A — A/B/C/D/E, V2, puis les UC** | Oui : le programme explorait le code des composants, les configurations, les capacités, les outils, les traces, les politiques par famille et les priors transférables. O0/O1/O2/O3 désignaient des objets différents à optimiser. Les exemples hors ligne installaient souvent des candidats manuscrits : ils prouvaient une connexion et une surface, pas une découverte. | [Carte initiale](../../../../opto/features/recursive_opt/README.md), [V2](../../EXP00/notebooks/recursive_opt_phases_V2.ipynb), Git `56b1231b1`, `8c78e1b46` ; [ancienne suite locale](../../../../examples/XP_1stattempt/recursive_opt_use_cases.ipynb). |
| **A — three_way** | Ajouté dans Git `c3a0548c0` : A0 sans modification, A1 optimisation Trace standard, A2 mécanisme spécialisé ou plusieurs niveaux. Répartition prévue du même nombre de candidats entre niveaux, courbes, coûts et différences de code. **A1 signifiait alors Trace standard ; dans EXP-15 il signifie génération indépendante.** Le mot « recursive » réunissait plusieurs traitements, dont un solveur numérique sans LLM. Des comparaisons étaient défectueuses malgré le cadre. | [Helper, contrat en tête](../../../../examples/recursive_opt_three_way.py), [audit des comparaisons, §5](../../ASSESSMENT.md). |
| **B — surtout un gain de vitesse** | Partiellement vrai. UC1 conserve 1,0 pour les deux bras et 2 appels optimiseur contre 4, mais seulement deux points de courbe : on ne connaît pas l’itération exacte d’atteinte. UC13 économise les appels optimiseur, mais son verdict de vitesse utilise la première graine seulement ; le seuil du standard n’est atteint que sur **1/3** des graines archivées. Le gain W2 de routage est, lui, reproduit et précisément limité à un menu et un contrôle non informé. | [UC1 brut](../../../../examples/notebook_outputs/recursive_opt_use_cases/three_way_stage2_20260624_150102/uc1_code_bbeh_solver/three_way_report.json), [UC13 brut](../../../../examples/notebook_outputs/recursive_opt_use_cases/three_way_stage2_20260624_150102/uc13_numeric_head_to_head/three_way_report.json), [replay historique](../../_shared/optimizer_discovery/investigation16/history/HISTORY_REPORT.md), [nouvel audit §29](../../ASSESSMENT.md). |
| **C — recentrage sur UC2, le plus prometteur ?** | Ce classement n’est pas démontré. UC2 QASPER était un résultat bruité sans avantage établi. Le « flagship » de juin était **UC4**, puis son +0,163 a été retiré : les bras n’étaient pas évalués sur les mêmes tâches. La suite a privilégié des surfaces de code mesurables et portables ; l’optimizer discovery se rapproche davantage d’**UC1/UC5** que d’une confirmation d’UC2. | [Anciennes limites et rétractation](../../../../examples/recursive_opt_use_cases_CURRENT_LIMITS.MD), [audit §5.2](../../ASSESSMENT.md), [objectif Phase 0](../../_shared/optimizer_discovery/PHASE0_SPEC.md). |
| **D — même dict/control plane** | Oui, c’était une unification d’orchestration. Mais **normaliser tous les anciens fichiers n’a pas signifié refaire toutes les expériences**. Migration : 85 fichiers classés, 10 normalisés seulement, 6 non portables, 46 historiques, 23 sans dépendances suffisantes ; **zéro replay fidèle certifié**. Le notebook actuel a été remplacé par deux tests déterministes UC4/UC14, explicitement non historiques. | [ADR](../../_shared/control_plane_v2/control_plane_v2alpha.md), [migration](../../_shared/control_plane_v2/migration_report.md), [notebook actuel](../../../../examples/recursive_opt_use_cases.ipynb), Git `21a0ad3d2`, `c92f0af4a`, `5ed861065`. |
| **E — optimisation d’un optimiseur numérique** | Oui, réduction volontaire pour obtenir un premier instrument portable et une comparaison interprétable. Phase 0 valide un contrat ; EXP-15 compare des recherches de code. Cette réduction ne réalise ni la mémoire d’environnement de Projet 1, ni une recherche scientifique à long horizon. La définition précise de Projet 2 n’est pas retrouvée dans les sources de cet audit ; lui attribuer un livrable précis serait inventer le périmètre. | [Contrat](../../_shared/optimizer_discovery/OPTIMIZER_PROGRAM_V0.md), [portabilité](../../_shared/optimizer_discovery/exp15/PORTABILITY.md), [rapport EXP-15](../../_shared/optimizer_discovery/EXP15_REPORT.md). |
| **F — dérive EXP-15 → 16 → 17/18** | La concentration sur une seule famille de benchmark est réelle. EXP-16 a corrigé des défauts et testé plusieurs mécanismes **dans ce benchmark**, sans revenir à la diversité des UC. EXP-17 cherche à confirmer le signal post-hoc C−I. EXP-18 teste mémoire/Pareto, sans génération indépendante N16. Ni l’une ni l’autre ne couvre curriculum de tâches, OTEL, apprentissage de stratégie ou niveaux O2/O3. Leur volume ne prouve pas la généricité du programme de recherche. | [Matrice des causes](../../_shared/optimizer_discovery/investigation16/DECISION_MATRIX.md), [protocole EXP-17](../../_shared/optimizer_discovery/exp17/PREREG_EXP17.md), [rapport EXP-18](../../_shared/optimizer_discovery/exp18/REPORT.md). |
| **G — le parent existait déjà** | Exact. Modifier un programme courant à partir du résultat précédent est déjà une mécanique de Trace et du notebook PAL. **C n’est pas une invention algorithmique** : c’est un contrôle qui isole l’apport de la sélection du parent, sans texte de scores/trajectoires dans sa requête. La nouveauté éventuelle était la mesure contrôlée sur nouvelles instances, pas ce mécanisme. | [PAL, cellules 6–7](../../../../examples/OpenTrace_LangGraph_BBEH_boolean_expressions_PAL_curriculum_clean.ipynb), [construction des bras historiques](../../../../examples/recursive_opt_three_way.py), [inspection P1](../../_shared/optimizer_discovery/investigation16/production/PROGRAM_INSPECTION.md). |
| **G bis — documents devenus illisibles** | Oui. Les preuves brutes sont nécessaires ; les nombreux rapports intermédiaires ne doivent pas tous être des points d’entrée. L’index distingue désormais synthèses, preuves, protocoles gelés et anciennes propositions. Les références négatives et les sources évaluées restent conservées. | [Index réorganisé](RESULTS_INDEX.md). |
| **H — où sont vitesse, C et I ?** | Les quatre notions de vitesse sont distinguées en §4 : appels de recherche, itération d’atteinte, vitesse du programme déployé, débit de l’évaluateur. C−I vaut −0,029672 en P1, **exploratoire après résultats**. EXP-17 n’a pas produit son résultat confirmatoire. Affirmer que C a été confirmé serait faux. | [Calculs C/I et planification](../../_shared/optimizer_discovery/investigation16/production/FUTURE_DESIGN.md), [suspension EXP-17](../../_shared/optimizer_discovery/exp17/USER_REQUESTED_PAUSE.json), [T1 corrigé](../../_shared/optimizer_discovery/investigation16/throughput/REPORT_T1.md). |
| **I — temps utile et temps mal alloué** | Les réparations de comparabilité, de validité, de portabilité et les replays ont produit des acquis. Les anciennes proclamations sur des menus inactifs, les gains UC4 invalides, et l’accumulation de longs diagnostics avant de rattacher la mesure à un usage produit ont un mauvais rendement. Le recentrage aurait dû être réexaminé plus tôt ; la complétude expérimentale a pris le pas sur la question d’usage. Cela n’autorise ni à effacer un résultat négatif ni à promettre que le prochain mécanisme gagnera. | [Rétractations historiques §14/17/19](../../ASSESSMENT.md), [durées et ressources EXP-18](../../_shared/optimizer_discovery/exp18/RESOURCE_REVIEW.md), [limites du prochain design](../../_shared/optimizer_discovery/investigation16/production/FUTURE_DESIGN.md). |

Les commits ci-dessus ont été consultés dans l’historique local. Une date de
notebook ou un label d’UC n’est pas une preuve de validité. Le notebook PAL présent
est non suivi par Git ; sa lecture prouve son mécanisme local, pas sa date
d’introduction ni une performance reproduite.

## 2. Niveaux, tâches, surfaces : retrouver le programme initial

**O0** : programme/prompt qui résout une tâche. **O1** : manière d’optimiser O0
(exemples, mémoire, configuration, outils). **O2** : choix de cette politique par
famille. **O3** : prior transféré à une famille réservée. L’ancien V2 appelait aussi
« O2a » la réécriture du code d’un composant ; ce nom ne prouve pas deux boucles
imbriquées. Réécrire huit fois un même type de programme n’est pas huit niveaux.
[Définitions initiales](../../../../opto/features/recursive_opt/README.md),
[spine d’exécution](../../../../opto/features/recursive_opt/levels.py).

| UC / branche | Niveau et tâche | Surface réellement visée | Performance recherchée | État et limite d’usage |
|---|---|---|---|---|
| **UC1 / exemple B** | O0 code de solveur ; O1 si composant d’optimisation (« O2a » ancien). Batch selector, trace summarizer, BBEH booléen | Corps du code | Qualité du solveur ou du composant ; appels d’optimisation | Code réécrivable ; petits validateurs saturés. UC1 BBEH : économie historique d’appels, aucune supériorité de plafond établie. |
| **UC2 / exemple A** | O1 ; GSM8K, DROP, QASPER, mélange GSM8K/QASPER | Prompt initial, connaissances ; batch/trainer seulement avec entraînement actif | Score, coût d’apprentissage, adaptation de configuration | Bruit et champs inactifs selon le chemin. Pas de vainqueur établi. |
| **UC3 / exemple C** | O0 capacité, éventuellement O1 ; spécification et multiobjectif GSM8K | Texte de capacité, qualité/coût | Compromis multiobjectif, réutilisation de capacité | Connexion démontrée à certains endroits ; anciens échecs d’évaluateur/scoring et specs non portables. Pas de gain applicatif validé. |
| **UC4 / exemple D** | O2 politique par famille → O3 prior ; GSM8K/QASPER notamment | Configurations par famille et prior partagé | Transfert et amortissement | +0,163 retiré ; UC4 actuelle du notebook = test technique. Hypothèse de transfert encore ouverte. |
| **UC5** | O1 code d’outil ou politique d’outils ; selectors, retrieval, `note`/`trace_search` | Helper, choix d’outils, contexte du méta-optimiseur | Meilleurs updates / coût | Cas distincts regroupés sous un nom ; validateurs de décisions souvent artificiels. Pas de bénéfice démontré sur une campagne réelle. |
| **UC6** | O1 ; QASPER et variantes de horizon | `trace_type`, `credit_horizon`, contenu rendu | Utilité du feedback, coût | OTEL/hybrid non disponibles dans l’environnement audité. Aucun résultat valide de leur inutilité. |
| **UC7** | O1 routage de graphe vers sous-optimiseur ; petit problème numérique avec SciPy | Route faible / appel solveur ; coût d’appel | Routage conditionnel qualité/coût | Démonstrateur d’architecture ; pas preuve de recherche scientifique autonome. |
| **UC8** | O1/O2 politique de campagne ; cas de saturation/stagnation | Code décidant stop/restart/switch | Éviter les appels inutiles | Évaluation de décisions sur cas préparés ; économies de campagne réelle non démontrées. |
| **UC9** | O1 Agentic Trace ; signaux de difficulté/transfert | Politique outils + indication d’intention | Sélection d’information pour un update | Score de politique locale ; bénéfice downstream à tester. |
| **UC10** | O1 gouvernance des artefacts ; cas promote/retest/reject | Code de décision, confiance, cibles | Fiabilité des promotions | Guards réutilisables ; un score de conformité ne prouve pas de meilleures découvertes. |
| **UC11** | O0 code émetteur de prompt ; QASPER | Fonction qui produit le prompt du solveur | Qualité QA et transfert | Une autre représentation du prompt ; bruit historique et absence de preuve de gain récursif. |
| **UC12** | Transversal ; six primitives promues | Budgets, graines, causalité, contrôle, routing numérique, politique de recherche | Correct fonctionnement | Tests d’API et d’exécution, pas six découvertes de performance. |
| **UC13** | O1 ; configuration numérique du benchmark reasoning | `batch_size`, `batch_design`, solveur numérique vs génératif | Appels LLM/coût, qualité | Routage sans LLM optimiseur réutilisable. Ancienne surface quasi plate et verdict de vitesse limité à la première graine. |
| **UC14** | Transfert de code ; notamment politique outils → politique Agentic Trace, puis variantes | Code transmis puis adapté à une autre cible | Transfert, économies d’apprentissage | Plusieurs versions ; perte −0,07 conservée sur trois paires du petit évaluateur de juin. Aucun motif pour l’étendre à toute forme de transfert ; contrôle source/cible asymétrique. |
| **EXP-07/08/09** | Prior d’ordre / code de heuristique ; TSP, CVRP, OVRP, VRPTW | Menu de neuf heuristiques ou fonction de choix du prochain nœud | Coût de recherche et distance de tournée | W2 reproduit contre contrôle non informé ; meilleur code VRPTW reproduit sur sa fixture ; signatures incompatibles entre tâches. |
| **EXP-15/16/17/18** | Une recherche externe de code d’optimiseur ; Sphere, Quadratique, Rosenbrock, dimensions 2/4 | **Tout le code** `propose(history,bounds,seed)` ; à l’intérieur, points x en 2/4 dimensions | Regret au cours des 32 appels et regret final | Contrat portable réel ; espace de code large, problèmes numériques étroits. Pas d’expérience comparant O1/O2/O3. |
| **PAL curriculum** | O0 prompts/code d’agents LangGraph ; BBEH expressions booléennes | Prompt PAL + fonctions `parse_problem` / `execute_code` ; batch courant + succès récents | Précision et absence de régression sur exemples passés | Mécanisme de batch présent dans le notebook local ; pas de comparaison contrôlée curriculum/fixe ni preuve d’un gain produit. |

Sources de la carte : [suite de juin dans Git `8c78e1b46`](../../../../examples/XP_1stattempt/recursive_opt_use_cases.ipynb)
(la copie locale ne contient que les premiers UC ; pour UC7–14, consulter le commit),
[rapport historique des UC](../../../../examples/recursive_opt_use_cases_CURRENT_LIMITS.MD),
[évaluateur de politiques](../../../../opto/features/recursive_opt/decisions.py),
[replay routage](../../_shared/optimizer_discovery/investigation16/history/HISTORY_REPORT.md),
[guide numérique détaillé](../../_shared/optimizer_discovery/exp18/GUIDE_EXPERIENCES.md).

**Toutes les cibles numériques actuelles** : Sphere 2D, Sphere 4D, Quadratique 2D,
Quadratique 4D, Rosenbrock 2D, Rosenbrock 4D. TRAIN contient 24 instances, validation
12, audit réservé 12 ; chacune est jouée avec deux graines locales dans EXP-17/18.
TRAIN sert aux updates et au choix du parent ; validation à la sélection finale ;
audit uniquement après gel de toutes les sélections. Elles ne sont donc pas
toutes des tâches d’apprentissage. [Génération](../../_shared/optimizer_discovery/benchmark.py),
[configuration EXP-18](../../_shared/optimizer_discovery/exp18/exp18_manifest.json).

Sphere teste une géométrie séparable, Quadratique des courbures différentes selon
les axes, Rosenbrock des coordonnées couplées et une vallée courbe. Ce sont des
précurseurs élémentaires de difficultés numériques. Deux des trois familles sont
des quadratiques diagonales transformées. Il n’y a ni matrice dense à apprendre,
ni grande dimension, ni contraintes industrielles, ni environnement d’outils
persistant, ni planification scientifique à long horizon. **L’extrapolation vers
ces usages reste à construire**, elle n’a pas été validée par ces résultats.
[Audit géométrique](../../_shared/optimizer_discovery/investigation16/benchmark/GEOMETRY_REVIEW.md).

## 3. Hypothèses : ce qui est invalidé, et ce qui reste ouvert

« Claim invalidé » signifie que sa preuve est défectueuse. « Résultat négatif »
signifie que le traitement testé a moins bien réussi sous son protocole.
« Inconclusif » ne prouve ni équivalence ni inutilité. Une génération tronquée,
un batch inactif ou une signature incompatible ne réfute pas une idée générale.

| Hypothèse / ancien claim | Statut corrigé | Preuve, portée et explication non démontrée |
|---|---|---|
| **H1 : la récursion augmente le plafond** | **Non établi en général ; même maximum du menu en EXP-08** | q=1 désigne le meilleur des neuf candidats disponibles, pas l’optimum absolu du problème. Le replay VRPTW contient même des codes dépassant ce menu. Ancien `REFUTED` trop général. [Historique](../../_shared/optimizer_discovery/investigation16/history/HISTORY_REPORT.md). |
| **H2 : même qualité avec moins de recherche / W2** | **Soutenu conditionnellement** | Budget cible 1 contre 9, coût amont 18, K*=2,25 à Q=1 ; contrôle fixe `nearest` donne aussi q=1 sans apprentissage amont. Ni bénéfice propre au feedback ni amortissement général. [Résultats et calcul exact](../../_shared/optimizer_discovery/investigation16/history/HISTORY_REPORT.md). |
| **H3 : transfert d’artefact code** | **Ancien contrat invalidé pour le transfert ; hypothèse générale ouverte** | 11/11 exécutables sur leur source ; 0/22 sur leurs sœurs à cause des signatures, avant qualité. Le nouveau contrat se rejoue sur nouvelles instances en S1, sans résoudre tous les transferts. [Replay](../../_shared/optimizer_discovery/investigation16/history/HISTORY_REPORT.md), [S1](../../_shared/optimizer_discovery/investigation16/selection/REPORT_S1.md). |
| **H4 : transfert de knobs** | **Non testé utilement dans l’ancien pool** | 43/47 tâches avaient un seul exemple externe ; `inner_steps=0` rendait certains knobs inactifs ; BBEH presque saturé. Cela n’invalide pas un vrai curriculum. [Audit §21](../../ASSESSMENT.md). |
| **H5 : un optimum partagé par famille** | **Soutenu dans le menu routage** | `nearest` premier 4/4, corrélations positives. Ce n’est **pas une condition nécessaire** de toute méta-optimisation : une politique conditionnelle peut exploiter des régularités sans argmax fixe commun. [Replay](../../_shared/optimizer_discovery/investigation16/history/HISTORY_REPORT.md). |
| **H6 / W1 : réduction de variance** | **Non testé** | Le replay déterministe exclut le bruit de ce replay, pas la stochasticité de génération ni l’erreur de sélection. Aucun test général d’inutilité du feedback n’en découle. [Historique, section knobs](../../_shared/optimizer_discovery/investigation16/history/HISTORY_REPORT.md). |
| **H7 : UC4 +0,163** | **Claim invalidé / retiré** | Tâches différentes, identité arithmétique, artefacts identiques. Aucun surplus de tokens ne répare une soustraction non comparable. [Audit §5.2](../../ASSESSMENT.md), [run corrigé](../../EXP02/results/three_way_report.json). |
| **H8 : optimisation utile sur prose** | **Non résolu** | Évaluations bruitées et mélange score/coût ; EXP-12/13 historiques incomplets/non comparables à une reprise actuelle. Pas de preuve que la prose est inoptimisable. [Audit §11–14](../../ASSESSMENT.md). |
| **H9 : surface non plate** | **Soutenu sur les menus code ; non résolu sur prose** | Packing/admissible-set : plages 2908,2 / 390, replay stable. S/N prose <1 signifie signal non résolu sous ce plan, pas surface mathématiquement plate. [Replay](../../_shared/optimizer_discovery/investigation16/history/HISTORY_REPORT.md). |
| **Probe F +0,217 ; K +4,8 ; iteration 3** | **Claims retirés / sans preuve utilisable** | Bruit mal mesuré ; K avait un artefact vide ; menu effondré et artefacts inchangés dans iteration 3. Motifs distincts. [Audit §14/17/19](../../ASSESSMENT.md). |
| **UC13 : même optimum plus vite sur les graines** | **Résumé invalidé ; économie de LLM observable** | `_summarize` utilise la première ligne valide de chaque bras ; atteinte appariée seulement 1/3 dans le brut. « 0 appel optimiseur » ne signifie pas zéro coût d’évaluation. [Code](../../../../examples/recursive_opt_three_way.py), [brut UC13](../../../../examples/notebook_outputs/recursive_opt_use_cases/three_way_stage2_20260624_150102/uc13_numeric_head_to_head/three_way_report.json). |
| **Toutes les pertes des anciens UC sont du bruit** | **Généralisation à retirer** | Les cas QA et les validateurs déterministes ne partagent pas automatiquement le même plancher de bruit. UC14 conserve −0,07 sur 3 paires de son petit test ; portée étroite, pas réfutation universelle. [Brut UC14](../../../../examples/notebook_outputs/recursive_opt_use_cases/three_way_stage2_20260624_150102/uc14_code_transfer/three_way_report.json). |
| **3 000 tokens suffisent à générer le contrat** | **Échec constaté du réglage initial** | 0/3 code ; les tokens ont été consommés avant une source utilisable. Calibration 8 000/low : 10/10 valides sur fixture. Ce premier plafond était un réglage proposé pour un smoke, pas une contrainte scientifique nécessaire. [Phase 0](../../_shared/optimizer_discovery/PHASE0_REPORT.md). |
| **Les tokens expliquent seuls H15-B** | **Non démontré** | EXP-15 : 16/80 réponses tronquées sans code. EXP-16 passe à 32 000 et R perd encore ; changements combinés, donc pas estimation causale isolée du plafond. [Génération](../../_shared/optimizer_discovery/investigation16/generation/REPORT.md), [causes](../../_shared/optimizer_discovery/investigation16/DECISION_MATRIX.md). |
| **H15-A : A2 bat A0** | **Signal positif, dans EXP-15** | ΔAUC −0,022899, IC [−0,042456 ; −0,003341], n=5. Comparateur initial faible selon B2 ; cela ne transforme pas ce contraste en avantage propre au feedback. [EXP-15](../../_shared/optimizer_discovery/EXP15_REPORT.md). |
| **H15-B : A2 bat A1 indépendant** | **Inconclusif** | Δ −0,005068, IC [−0,037728 ; +0,024551], n=5. Ne pas appeler cela « réfuté » ni « confirmé ». [EXP-15](../../_shared/optimizer_discovery/EXP15_REPORT.md). |
| **Riche R bat I / C, P1** | **Signal négatif exploratoire, dans le protocole** | R−I +0,045257 [+0,003176 ; +0,085126] ; R−C +0,074929 [+0,034075 ; +0,116730]. Texte effectivement transmis, plafond 32 000, TRAIN24×2. Les causes contenu/longueur/service ne sont pas séparées. [P1](../../_shared/optimizer_discovery/investigation16/production/FUTURE_DESIGN.md). |
| **C bat I, H17** | **Hypothèse ouverte, confirmation interrompue** | P1 : −0,029672 post-hoc. EXP-17 : 545/736 réponses, aucun audit. Le parent est un contrôle connu, pas une nouveauté. [Protocole](../../_shared/optimizer_discovery/exp17/PREREG_EXP17.md), [pause](../../_shared/optimizer_discovery/exp17/USER_REQUESTED_PAUSE.json). |
| **Plus de largeur suffit** | **Inconclusif / facteurs mélangés** | W−R −0,009935, IC [−0,054857 ; +0,030305] ; W modifie largeur et nombre de tours. [P1](../../_shared/optimizer_discovery/investigation16/production/FUTURE_DESIGN.md). |
| **Mémoire des essais / Pareto améliorent la recherche** | **Inconclusif dans EXP-18** | Effet mémoire −0,004114 [−0,020651 ; +0,014910] ; Pareto +0,008512 [−0,015019 ; +0,035682]. Les mécanismes étaient actifs ; ni gagnants ni inutiles établis. [EXP-18](../../_shared/optimizer_discovery/exp18/REPORT.md). |
| **Un batch de 6/7 exemples, ou un curriculum, résout le problème** | **Non testé** | TRAIN24×2 est un panel fixe ; mémoire7 = programmes précédents. S1 prouve un effet de sélection dans une banque fixe, pas celui d’un curriculum. Aucun seuil universel 5/7. [S1](../../_shared/optimizer_discovery/investigation16/selection/REPORT_S1.md), [guide](../../_shared/optimizer_discovery/exp18/GUIDE_EXPERIENCES.md). |
| **OTEL/logs ou plus de niveaux améliorent le résultat** | **Non testé dans les études récentes** | Nœuds Trace présents à la frontière, pas branches internes du candidat ; `HAVE_TRACE_IO=False` ici ; pas d’ablation O1/O2/O3. [Traces](../../../../opto/features/recursive_opt/traces.py), [inspection P1](../../_shared/optimizer_discovery/investigation16/production/PROGRAM_INSPECTION.md). |

Ces corrections changent la **portée des affirmations**, pas les données ni les
rétractations. Les libellés historiques `REFUTED` de H1/H3/H9 restent consultables
dans Git et dans l’audit daté ; ils ne doivent plus servir de verdict universel.

## 4. Gains réels, séparés par type de performance

| Gain | Résultat mesuré | Ce qu’on peut utiliser | Limite |
|---|---|---|---|
| **Recherche informée : W2** | q=1 à budget cible 1 au lieu de 9 ; K*=2,25 après 18 évaluations amont | Prior/ordre de candidats sur cette famille | `nearest` fixé égale le résultat, zéro coût méta. q normalisé au menu, pas optimum global. |
| **Qualité du code VRPTW** | Meilleur code sauvegardé : distance 26,476508 → **20,702124**, soit −21,81 % ; 5 codes indépendants sur 12 réponses dépassent le menu | Exemple réel de code d’heuristique optimisable | Même fixture historique de 16 instances ; pas nouvelle généralisation, pas avantage récursif. |
| **Programme portable A2/41 d’EXP-15** | S1 nouvelles instances : AUC **0,075566**, seed 0,162685, midpoint 0,102004 ; A1/41 obtient 0,080689 | Démonstrateur exécutable déjà retesté, sans LLM au déploiement | Banque sélectionnée, pas nouvelle comparaison appariée des moteurs de génération. |
| **Sélection plus stable** | S1 : TRAIN6×1 →24×2, AUC du programme sélectionné −17,3 % | Évaluer assez de situations distinctes avant de choisir un programme | Banque fixe, sous-échantillons recouvrants ; pas +17,3 % de gain futur de feedback. |
| **Initialisation B2** | EXP-18 : AUC **0,040987** vs seed 0,144291 | Contrôle exigeant et simple : premier point au centre | Distribution favorable aux optima centraux ; aucune universalité. |
| **Débit local** | T1 corrigé : huit workers **×5,49**, mêmes trajectoires | Parallélisme de l’évaluateur, sur machine comparable | Deux mesures ; pas vitesse d’apprentissage ni gain du programme ; seize workers furent plus rapides mais avec moins de marge. |
| **Routing numérique UC13** | Zéro appel de génération optimiseur dans le bras numérique, huit dans le standard archivé | Déléguer des knobs réellement numériques à un solveur existant | Ne garantit ni score égal, ni gain de temps actuel, ni récursion utile. |

Preuves : [historique retesté](../../_shared/optimizer_discovery/investigation16/history/HISTORY_REPORT.md),
[S1](../../_shared/optimizer_discovery/investigation16/selection/REPORT_S1.md),
[B2/EXP-18](../../_shared/optimizer_discovery/exp18/REPORT.md),
[T1 corrigé](../../_shared/optimizer_discovery/investigation16/throughput/REPORT_T1.md),
[UC13 brut](../../../../examples/notebook_outputs/recursive_opt_use_cases/three_way_stage2_20260624_150102/uc13_numeric_head_to_head/three_way_report.json).
Ces pourcentages ne s’additionnent pas et ne prédisent pas le prochain résultat.

## 5. Dict, Trace et curriculum : générique à quel niveau ?

Le control plane fournit un dictionnaire normalisé, des références versionnées de
modules/évaluateurs/datasets, des dépendances entre niveaux, des cibles trainables,
des objectifs, des budgets, une mémoire et des barrières de sélection. **174 tests
ciblés passent à nouveau le 13 septembre**, notamment sur des dépendances causales
entre deux niveaux et sur les routes Trace/GEPA avec fournisseurs simulés.
Les fonctions d’enregistrement permettent d’ajouter un module ; elles ne prouvent
pas que tous les composants historiques ou tous les systèmes externes sont déjà
intégrés. [ADR et classification des champs](../../_shared/control_plane_v2/control_plane_v2alpha.md),
[tests actuels](../../../../tests/unit_tests/test_recursive_control_plane_v2.py),
[commande et portée §29](../../ASSESSMENT.md).

Les traces actuelles sont **différentes selon les études**. EXP-15 utilisait un
résumé pauvre reconstruit ; P1 a raccordé le feedback réellement propagé et ajouté
courbes, meilleurs points et événements d’amélioration. EXP-18 revient à un résumé
compact du parent pour tester mémoire et sélection. Aucun de ces traitements ne
transmet un graphe des branches internes du sous-processus. Les sorties standard
et erreurs sont conservées, mais ne deviennent pas automatiquement le feedback.
Le helper `MultiTraceSession.feedback_text` ne rend lui-même que sources et
comptages de nœuds/arêtes ; collecter un graphe ne suffit donc pas à l’exploiter.
[Audit du canal](../../_shared/optimizer_discovery/investigation16/feedback/REPORT.md),
[projection P1](../../_shared/optimizer_discovery/investigation16/production/PROGRAM_INSPECTION.md),
[traces.py](../../../../opto/features/recursive_opt/traces.py).

Dans le notebook PAL local, `last_successes` contient les succès du parcours TRAIN,
y compris les échecs corrigés. Ils alimentent réellement `sample_batch`, puis
`backward` et `step`. Le paramètre nommé `validation_set` reçoit ici ces succès
**d’apprentissage** ; il ne faut pas le confondre avec le `val_set` externe.
Le buffer est recréé et prérempli à chaque appel. Le code local utilise
`add_success`, pas uniquement `add_success_after_fail` : il ajoute aussi les
succès immédiats. Avec deux entrées en mémoire, on obtient le courant + au plus
deux anciens exemples. En mode pool fixe, cette mémoire n’est pas échantillonnée.
**Cette intégration existe dans ce notebook, pas dans EXP-17/18.**
[PAL, cellules 6–7](../../../../examples/OpenTrace_LangGraph_BBEH_boolean_expressions_PAL_curriculum_clean.ipynb).

Un futur test de curriculum devrait vérifier trois événements : échec puis succès,
insertion dans la mémoire, changement des identités réellement exécutées dans le
batch suivant. Filtrer seulement le texte d’un panel déjà évalué ne teste pas le
même mécanisme. Comparer des candidats sur des batches différents exige aussi un
panel commun de sélection, sinon le biais de comparabilité réapparaît. Cette
expérience n’est **ni mise en œuvre ni lancée** dans le présent audit.

## 6. Ce qui a été utile, et ce qui ne justifie plus de prolonger l’effort

**Acquis utiles** : découverte du faux gain UC4 ; refus des scores sentinelles
invalides ; détection des menus sans levier ; preuve de passage réel des paramètres
et du feedback ; contrat de programme portable ; sélection indépendante de l’audit ;
replay des bons codes historiques ; contrôle B2 ; mesure du débit. Cela évite des
conclusions fausses et laisse des composants utilisables.

**Mauvais usage de l’effort** : multiplier les lignes sans vérifier d’abord que les
knobs sont consommés ; traiter des démonstrateurs artificiels comme des résultats
produit ; présenter C comme une nouvelle piste technique sans rappeler le parent
Trace déjà existant ; maintenir trop de points d’entrée documentaires ; passer à
736 réponses pour confirmer un effet local avant de réexaminer son intérêt pour les
projets. EXP-18 a consommé **384 réponses et 5 599 563 tokens** pour un résultat
inconclusif ; son intervalle civil génération→audit est d’environ **53,66 h**, avec
attentes, suspension et interruption. Ce n’est pas 53,66 h de travail humain perdu,
mais cela montre que « cheap benchmark » ne signifiait pas boucle de recherche rapide.
[Ressources vérifiées](../../_shared/optimizer_discovery/exp18/RESOURCE_REVIEW.md).

**À écarter des claims actuels** : gain UC4 +0,163 ; plafond général réfuté ; transfert
de code universellement impossible ; nombre magique d’exemples ; effet garanti d’un
plafond de tokens supérieur ; OTEL jugé inutile sans backend ; bénéfice de récursion
profonde déduit d’une lignée de versions ; promesse d’un gain futur chiffré.

**À différer** : nouvelle grosse campagne numérique, nouvelle profondeur, comparaison
de plusieurs nouvelles traces et curriculum simultanément, refonte du framework.
Les échecs généraux ne sont pas établis ; simplement, aucune preuve actuelle ne
justifie d’en faire des actions promises gagnantes.

## 7. Ce qui peut servir maintenant aux projets et à Patrick

« Utilisable » signifie composant ou exemple vérifié. Cela ne garantit pas un gain
sur une nouvelle tâche. Projet 1 est défini ici comme optimisation agentique avec
mémorisation/rejeu d’environnement, d’après le périmètre communiqué. **Projet 2
reste à spécifier** ; les rapprochements indiqués sont conditionnels.

| Destination / UC | Niveau d’optimisation | Tâche et type de performance | Hypothèse et acquis réel | Surface et limites | Action minimale préalable ; résultat raisonnablement assuré |
|---|---|---|---|---|---|
| **Patrick — UC1/5, EXP-15/S1** | Code du programme de recherche, une boucle externe | Minimisation numérique ; qualité anytime/finale | Programme A2/41 portable retesté ; supériorité du feedback non établie | Source `propose` ; d2/4, B32 seulement | Fournir source exacte + `benchmark.evaluate` + petit manifeste côté hôte. On peut vérifier une exécution identique sans adopter Trace ; aucun gain d’un nouveau moteur garanti. |
| **Patrick — historique routage** | O0 heuristique de construction | VRPTW ; distance −21,81 % sur fixture | Cinq programmes indépendants meilleurs que le menu, replay exact | Fonction de choix prochain nœud ; signatures liées aux tâches | Montrer code et fixture comme preuve locale, puis normaliser un contrat de domaine avant toute étude de transfert. Adapter un contrat est une tâche d’ingénierie ; le gain hors fixture reste inconnu. |
| **Projet 1 — UC1/12, contrat portable** | Exécution/évaluation, pas encore méta-apprentissage d’environnement | Programme fichier ; intégrité, budgets, erreurs typées | Entrée/sortie et source exactes réutilisables | Remplacement du lanceur local ; aucun snapshot d’environnement ni sandbox OS | Vérifier le même contrat sur l’exécuteur déjà choisi par l’équipe, lorsqu’il sera disponible. La conformité est testable ; sa compatibilité ne peut pas être garantie sans son API. |
| **Projet 1 — UC7/9 et PAL** | O0 agents + O1 batch/outils | Workflow booléen ; correction et non-régression | Paramètres de graphe et curriculum réellement câblables | Prompt/code de deux agents, mémoire de quelques cas ; pas de tâche longue | Un test sans LLM de bout en bout : changer un paramètre puis vérifier sortie/Trace et exemple réinjecté. Le résultat assuré est un diagnostic de connexion, pas un gain d’agent autonome. |
| **Projet 2, si méta-optimisation réutilisable — UC2/13** | O1 configuration/routing | Paramètres numériques ; coût d’apprentissage | Route numérique sans appels optimiseur ; menus/trainer résolus auditables | Knobs réellement consommés ; les anciens alias peuvent tous tomber sur ParetobasedPS | Sur une tâche cible précise, prouver qu’un knob change le chemin exécuté et mesurer l’évaluateur avant optimisation. Pas d’engagement produit tant que Projet 2 n’est pas défini. |
| **Projet 2, si mémoire/connaissances — UC3/4/8/9/10** | O1/O2/O3 selon l’objet mémorisé | Choix d’expérience, réutilisation et décisions | Stockage, lignées et injection de connaissances testés ; bénéfice scientifique ouvert | Carte de connaissance/prior/politique ; pas une mémoire d’état d’environnement | Un contre-factuel « carte absente/présente » doit modifier le contexte consommé sur une tâche distincte. Vérification technique possible ; effet de transfert à mesurer ensuite. |
| **Tous — S1/T1** | Évaluateur et sélection | Débit et stabilité de classement | Mesures locales +17,3 % de qualité de sélection / ×5,49 de débit, distinctes | Panels et workers ; résultats dépendants de banque/machine | Réutiliser les contrôles, compteurs et tests d’identité ; ne pas appliquer ces multiplicateurs aux projections produit. |

Entrées concrètes : [programme A2/41 exact](../../_shared/optimizer_discovery/investigation16/selection/sources/1684f91acdc36c0ca6aac70afeb9cc2c4eed7ab847926d5880590e059266abb7.py.gz),
[évaluateur indépendant du moteur](../../_shared/optimizer_discovery/benchmark.py),
[frontière Projet 1](../../_shared/optimizer_discovery/exp15/PORTABILITY.md),
[code et scores historiques](../../_shared/optimizer_discovery/investigation16/history/HISTORY_REPORT.md),
[routing numérique](../../../../opto/features/recursive_opt/numeric_optimizers.py),
[mémoire](../../../../opto/features/recursive_opt/memory.py),
[tests de dépendances et injection](../../../../tests/unit_tests/test_recursive_control_plane_v2.py).

## 8. Décisions présentes et vérification

1. **Arrêter la course au résultat numérique.** EXP-17 est conservée incomplète,
   EXP-18 terminée. Aucun nouveau bras, pilote, modèle ou appel live dans cet audit.
2. **Réutiliser d’abord un acquis** : démonstrateur portable pour Patrick ; contrat
   d’exécution pour Projet 1 ; pour Projet 2, rattachement impossible à promettre
   sans sa définition. Ce sont des livrables plus sûrs qu’une nouvelle hypothèse
   de gain de feedback.
3. **Si une étude reprend plus tard**, choisir un seul usage et une seule question.
   C−I reste une question locale légitime, mais n’est plus présenté comme la meilleure
   prochaine action pour l’ensemble des projets. Trace/curriculum nécessite d’abord
   un test de consommation du batch, distinct d’un test d’efficacité.
4. **Préserver les preuves et réduire la lecture** : deux points d’entrée internes,
   un brief externe ; anciens protocoles/rapports accessibles via l’index. Aucun
   résultat défavorable ni source candidate n’est supprimé.

Audit du 13 septembre : historique Git, notebooks et JSON relus ; atteintes UC13
recalculées par graine ; 174 tests ciblés réussis ; backend OTEL absent constaté ;
alias trainer résolus vérifiés. Le détail des commandes, limites et corrections est
en [assessment §29](../../ASSESSMENT.md). Il ne s’agit pas de nouvelles
générations ni de nouvelles mesures de généralisation.

---

**Inscriptions historiques conservées ci-dessous.** Les décisions et statuts datés
ci-dessus font autorité pour la suite ; les hypothèses et résultats numériques des
expériences précédentes restent inchangés dans leurs fichiers originaux.


## 9. Optimizer discovery Phase 0 — 2026-09-08

The portable propose(history, bounds, seed) interface and deterministic evaluator are
ready. Future runs retain actual candidate behavior/evaluation evidence, with explicit
invalidity and unknown equivalence instead of silently certifying a collapsed menu.
This changes instrument validity, not any historical performance conclusion; see
[assessment §23](../../ASSESSMENT.md#23--optimizer-discovery-phase-0-instrument-and-interface).

Frozen OpenRouter DeepSeek calibration: **0/3** open-ended generation responses had
parseable code (all reached 3000 completion tokens). All failures are retained. A
separately preregistered interface-only midpoint request succeeded and executed at
seeds 0/1/2, budget 8 each, value 2.125 throughout. This demonstrates the interface,
not optimizer discovery or improvement. **Phase-1 search execution: NO-GO** under
the failed generation protocol; interface/protocol development can proceed.

EXP-12/13 remain historical/unresolved: model-name agreement does not establish
request comparability; the required ephemeral EXP-13 prompt is absent. Their raw
records and H1/H2/H3/H5 conclusions are unchanged. Full evidence:
[PHASE0_REPORT.md](../../_shared/optimizer_discovery/PHASE0_REPORT.md).

## 10. Optimizer generation readiness — separate calibration, 2026-09-08

After the user authorized prospective configuration calibration, a separately
preregistered 3,000-token control again returned no code: all 3,000 reported output
tokens were reasoning. An 8,000-token/default-reasoning pilot yielded 2/3 valid
history-responsive optimizers and failed its fixed gate. The 8,000-token/low-effort
pilot yielded 3/3 and was selected before fresh confirmation seeds 101–110.

Confirmation: **10/10 valid and history-responsive**, **effective menu size 7** on
actual common fixture trajectories. Each program completed 3 × 8 objective calls.
**Phase 0 generation readiness: GREEN** for the selected exact-model configuration;
proceed to Phase-1 preregistration. This does not establish optimization quality or
population reliability. All original failures and both new invalid requests remain.
The provider mix varied; observed readiness is not an isolated causal estimate of
reasoning effort. No historical hypothesis conclusion changed.

See [current Phase-0 report](../../_shared/optimizer_discovery/PHASE0_REPORT.md),
[selected settings](../../_shared/optimizer_discovery/selected_generation_config.json), and
[assessment §24](../../ASSESSMENT.md#24--optimizer-generation-readiness-calibration).


## 11. EXP-15 — first controlled optimizer-program discovery experiment

Preregistered H15-A tests selected A2 deployment against the unchanged seed. H15-B,
the central contrast, tests A2 against equal-response-budget independent code search.
Both were evaluated under the frozen confirmatory analysis. The Phase-1 pilot used
separate instances and outer seed 701; two of four completed responses produced
eligible programs, and validation retained the seed in both arms. All failures and
the transport retry remain recorded. Pilot results do not enter confirmation.

The frozen protocol (`0643691b`) uses five paired outer seeds, eight completed
DeepSeek responses per generative arm, 32 objective calls per trajectory and balanced
6/6/12 train/validation/holdout splits. All selections and the validation-selected
representative were frozen before any holdout evaluation. All five paired seeds and
80 response slots are retained. Invalid candidates and common deployment fallback
are explicit; no holdout fallback was needed.
See [preregistration](../../_shared/optimizer_discovery/PREREG_EXP15.md) and
[pilot evidence](../../_shared/optimizer_discovery/exp15/PILOT_REPORT.md).

Mean normalized anytime regret, lower better: A0 **0.139579**, A1 **0.121748**,
A2 **0.116680**. H15-A's paired delta is **−0.022899**, 95% bootstrap interval
**[−0.042456, −0.003341]**, a positive signal under the registered rule.
H15-B's delta is **−0.005068**, interval **[−0.037728, +0.024551]**, inconclusive.
A2 loses to A1 in two of five pairs. A1 has better sample mean final regret and
target attainment. Five outer seeds do not justify strong superiority claims.

A1 has 9/40 ineligible candidate slots; A2 has 12/40, including one program rejected
for nondeterministic execution. A2 retains the seed in two replications. All failures
remain, including the single transport timeout. There were 81 attempts for 80
completed responses, 573,191 reported tokens and USD 0.086824 reported cost, excluding
unknown billing on the timeout. Reporting-only latency/path-label corrections and
lossless archive packaging changed no scientific values or decisions. The frozen
protocol and complete aggregate pass the integrity audit; 772 offline tests pass.

The validation-selected representative is outer41/slot5, a Halton/incumbent/quadratic
hybrid, with exact source and lineage preserved. The result and portable evaluator
are ready for discussion with Patrick; the brief has not been sent. See
[full report](../../_shared/optimizer_discovery/EXP15_REPORT.md),
[machine-readable results](../../EXP15/results/exp15_results.json),
[Patrick brief](../../_shared/optimizer_discovery/PATRICK_BRIEF.md) and assessment §25.

This creates a portable-artifact venue. It does not retroactively overturn H3's
signature-bound historical setup, measure amortization, or identify a recursion-
depth effect. Historical H1/H2/H3/H5 conclusions and prior retractions are preserved.

## 12. EXP-16 — feedback root-cause investigation and prospective retest

This completed exploratory investigation preserves EXP-15 and tests several
explanations separately before a frozen production search comparison. The old
feedback omitted the explicit anytime objective and most incumbent evidence;
six tasks did exist, but one outer Trace row did not mean one training observation.
Increasing the completion cap improved observed eligibility in a small diagnostic,
without establishing a performance benefit or proving token exhaustion was the
central cause. More representative selection panels and parallel local evaluation
showed measurable benefits on fixed policies, not a demonstrated feedback gain.

P1 uses six paired outer seeds, eight completed responses per I/C/R/W search,
32 objective calls, 24 TRAIN/12 validation/12 audit instances with two local seeds,
the exact DeepSeek/OpenRouter model, 32,000 completion tokens and low reasoning.
I is independent; C rewrites a TRAIN-selected parent without explicit trajectory
feedback; R adds the actual propagated current-parent feedback; W allocates two
parents over four rounds. C therefore uses performance information indirectly.
Generation, validation selections and representative selection all precede audit.

Mean regret AUC, lower better: A0 **0.179329**, I **0.077260**, C **0.047588**,
R **0.122517**, W **0.112582**. Registered R−I **+0.045257** and R−C **+0.074929**
are negative signals; W−R **−0.009935**, CI **[−0.054857, +0.030305]**, is
inconclusive. R−A0 **−0.056813**, CI **[−0.090926, −0.023830]**, is positive.
Six outer replications on one fixed audit panel give fragile exploratory intervals.
C−I's promising mean **−0.029672** is post hoc, not a registered positive finding.

A separately registered fixed B2 control changes only the first seed proposal to
the bounds midpoint. It achieves AUC **0.031633**, **82.36% lower than A0**, on
the fresh P1 audit panel. Its final regret is worse than I/C, so it does not dominate
all metrics. This validates a weak-initialization mechanism for A0, not a causal
explanation of R−I. Merely beating the original seed is an insufficient target.

All 192 responses and 194 transport attempts are preserved. I/C/R/W eligibility is
46/44/41/43 out of 48; all 720 allocated audit trajectories are valid, without
deployment fallback. R selects the unchanged seed once despite eligible alternatives.
P1 reports 6,980,225 tokens and USD 0.488195; two failed transport attempts have
unknown billing. Source hashes, chronology, budgets, all cached numeric observations
and the full aggregate passed independent verification. No negative response was
replaced. All stages together comprise 242 completed model responses, separately
reported in the investigation; diagnostic and confirmatory evidence are not pooled.

Historical replay confirms conditional routing search savings against an uninformed
baseline, plus some genuinely improved stored programs on their original tasks.
It also reproduces the 22 sibling-signature failures and inert one-example knobs.
An informed handwritten policy removes the historical routing search advantage.
Neither this replay nor P1 establishes new amortization, recursion-depth benefits,
or literature novelty; earlier retractions and H1/H2/H3/H5 remain intact.

The next recommended question is whether TRAIN-selected parent rewriting C beats I
on wholly new instances and outer seeds, with B2 as a fixed comparator. This changes
the question explicitly. Richer feedback, rejected-attempt memory and other search
strategies require new bounded diagnostics; no numerical future feedback gain is
justified. See [full report](../../_shared/optimizer_discovery/investigation16/REPORT.md),
[future design](../../_shared/optimizer_discovery/investigation16/production/FUTURE_DESIGN.md),
[historical replay](../../_shared/optimizer_discovery/investigation16/history/HISTORY_REPORT.md)
and assessment §26. No new commit, push, merge, PR or message to Patrick was made.

## 13. EXP-17 — registered parent-rewriting confirmation in progress

The frozen comparison uses 46 new paired outer seeds, I/C with eight completed
responses each (736 allocated), and the common numerical evaluator. C rewrites
a TRAIN-selected parent without explicit score/trajectory text. The registered
primary contrast is C−I; A0 and midpoint-first B2 are fixed audit controls.
All generation must finish before validation, and all selections before audit.
No EXP-17 interim efficacy result is reported or used to change the run. See the
[protocol](../../_shared/optimizer_discovery/exp17/PREREG_EXP17.md) and assessment §27.

## 14. EXP-18 — completed exploratory memory/Pareto comparison

The separately frozen experiment completed all six outer seeds and all 384 model
responses, using 16 proposals per L/M/P/PM arm. TRAIN24/validation12/audit12
instances each have two local seeds and 32 objective evaluations. OpenRouter
`deepseek/deepseek-v4-flash-0731` used temperature0.6, top_p1, max_tokens32000,
native low reasoning, timeout300 and global generation concurrency1. The same
starting source, validity rules, selection and deployment fallback applied to all
arms. Pilots and prior experiments are not pooled into the result.

L uses a scalar-best TRAIN parent and compact current-parent feedback. M adds up
to seven recent distinct earlier program sources plus aligned typed attempt
summaries. P instead samples uniformly among nondominated 24-instance TRAIN
vectors; PM combines these interventions. The archive is not a task curriculum,
and generated internals/logs are not automatically serialized as Trace feedback.
The parent is an earlier program, not an earlier task. The
[French guide](../../_shared/optimizer_discovery/exp18/GUIDE_EXPERIENCES.md) explains these units.

Lower-is-better audit AUC means: A0 **0.144291**, B2 **0.040987**, L **0.035038**,
M **0.034315**, P **0.046941**, PM **0.039437**. The registered mechanism contrasts
are all inconclusive under the frozen paired-bootstrap rule:

| Contrast | Mean delta | Paired bootstrap 95% interval |
|---|---:|---|
| M−L | −0.000723 | [−0.012435, +0.016909] |
| P−L | +0.011903 | [−0.008491, +0.032761] |
| PM−P | −0.007504 | [−0.031867, +0.017142] |
| PM−M | +0.005122 | [−0.022718, +0.041892] |
| Memory main effect | −0.004114 | [−0.020651, +0.014910] |
| Pareto main effect | +0.008512 | [−0.015019, +0.035682] |
| Interaction | −0.006781 | [−0.025757, +0.015688] |

The bootstrap resamples the six outer seeds, not individual tasks. These are
fragile marginal exploratory intervals, without multiplicity guarantees. The
interaction is a departure from additivity, not proof PM is superior. All
generative-arm−B2 contrasts also include zero; the descriptive B2−A0 contrast
is **−0.103304 [−0.116482, −0.091377]**. Improvement over the original seed alone
does not establish the contribution of generative feedback.

Mechanism exposure was verified: seven complete sources appear in 33/96 M and
39/96 PM requests; 17 requests declare a character-budget omission. P and PM
choose a non-scalar-best parent in 73/96 and 63/96 requests. Thirty PM requests
combine seven complete sources and a non-scalar parent. These facts establish
exposure, not causal efficacy or an optimal memory length.

There are 23 static source failures and 25 further execution-ineligible programs,
48/384 ineligible overall. Every selected source is generated. All 864 audit
trajectories complete validly, with zero fallback. Every seed, slot and partial
invalid observation remains represented. The 399 attempts include 15 transport
failures with unknown remote usage/billing. Completed responses report 5,599,563
tokens and **USD0.905998544736 known cost**, not verified total spend.

The logical plan allocates 30,240 trajectories /967,680 objective evaluations;
the physical cache preserves 27,000 trajectories /804,709 actual objective calls
and 59,291 unused allocations. Incident011 leaves additional interrupted work
unknown; ceilings are not observed counts. Numeric verification recomputed every
cached objective and metric, and independent selection/resource analyses matched.
No scientific values or decisions were repaired. The final repository-wide
regression will be recorded when the separate EXP-17 run has finished.

The validation-frozen PM representative is outer18041/slot09, SHA256
`a787c80c41dbf44a073a0dcf3874d6237e64aabc6db9a2996e760fd4d0c13ced`, with exact
source, parentage and diffs preserved. It combines local quadratic modeling,
incumbent search and stagnation-dependent exploration; this is descriptive,
not a novelty or component-effect claim. See [report](../../_shared/optimizer_discovery/exp18/REPORT.md),
[program review](../../_shared/optimizer_discovery/exp18/PROGRAM_REVIEW.md),
[unsent Patrick brief](../../_shared/optimizer_discovery/exp18/PATRICK_BRIEF.md) and assessment §28.

EXP-18 has no independent N16 arm, no task curriculum and no extra recursion-depth
comparison. It does not justify projected memory/Pareto gains, prove their
equivalence, or establish amortization. Historical H3 and all retractions remain
unchanged. EXP-17 continues under its original freeze regardless of this outcome.


## 15. EXP-19 — traces, curriculum et vitesse O1, 13 septembre 2026

Cette campagne répond à la nouvelle demande d’ingénierie et d’évaluation ; elle
ne reprend ni ne remplace EXP-17/18. Le point d’entrée unique est
[EXP19.md](../../EXP19/RESULTS.md), avec protocoles, preuves et recalcul liés.

**Ingénierie validée.** Le checkout IO avait un import eager d’un frontend absent,
en plus du SDK OTEL absent d’un seul environnement. Les imports sont découplés,
OTEL/sysmon/hybrid capturent réellement l’exécution, et le dict configure la
projection consommée par niveau. Le curriculum observe les transitions TRAIN
échec→réussite, appelle `add_success_after_fail` et modifie le batch suivant ;
il ne nécessite pas de méthode spéciale dans le Guide. Ce sont des acquis de
fonctionnement, pas une preuve que ces mécanismes améliorent la performance.

**S3 : absence d’accélération démontrée.** Après criblage individuel, combinaison
et ablation, le three-way sur trois graines nouvelles, huit réponses par bras,
donne 39,58 % initial /47,92 % standard /50,00 % O1 sur TEST. Aucun bras n’atteint
90 %. O1−standard = +2,08 points, intervalle descriptif [0 ; +6,25]. Le diagnostic
O2 termine sans texte après 16 000 tokens ; la configuration récursive retenue
est identique au standard. Aucun gain de profondeur ne peut en être déduit.

**S4 : signal positif ciblé sur le trainer.** Les requêtes montrent la perte d’une
correction lors de la sélection du parent, avant la génération suivante. Un
nouveau protocole prospectif change uniquement la validation pendant le fitting,
et conserve une sélection externe VALIDATION identique avant TEST. Trois nouvelles
graines, quatre réponses par bras, six exemples TRAIN identiques par paire de
requêtes : TEST = 100 %/100 %/100 % pour le parent sélectionné sur TRAIN, contre
68,75 %/77,08 %/70,83 % pour le standard. Gain moyen +27,78 points, bootstrap
apparié descriptif [+22,92 ; +31,25]. Seuil 90 % atteint en quatre réponses sur
3/3 graines contre 0/3 dans ce budget. Ne pas en déduire un facteur d’accélération
sur des hitting times censurés. Les évaluations internes ne sont pas égalisées :
444 standard contre 234 modifié ; sélection externe 336 contre 360.

C’est un réglage O1 diagnostiqué manuellement et retesté, pas une découverte
récursive automatique. La tâche apprend 24 bits composés en expressions nouvelles ;
elle est synthétique, pas BBEH, et une mémoire spécialisée la résout sans LLM.
Les trois réplications n’utilisent pas de nouveaux concepts. Ce résultat ne
prouve ni nouveauté, ni amortissement, ni transfert à Projet 1/2. H1/H2/H3
historiques et les rétractations restent inchangés.

255 réponses réelles conservées, 2 332 712 tokens, coût connu USD0,21856749005 ;
9 réponses vides, une proposition mal formée donnant trois évaluations invalides,
deux retries transport documentés en S4. Une requête enfant S2 interrompue reste
sans reçu distant : le coût connu n’est pas une facture complète. S1/S2 comportent
un défaut de projection corrigé avant S3 ; sources et résultats antérieurs
préservés. Le rapprochement final de l’interruption et la réparation purement
documentaire du renderer sont explicités dans le bilan.

47 nouveaux tests passent avec OTEL ; 34 passent et 13 sont optionnellement sautés
sans SDK ; 325 régressions affectées passent après les corrections ; la suite plus
large antérieure compte 524 passes et 3 skips. Recalcul indépendant identique,
hashes et batches vérifiés. Pas de commit ni dépendance ajoutée ; checkout staged
utilisateur conservé. La durée visée d’une heure a été dépassée, indépendamment
des deux longues suspensions observées. EXP-17 reste suspendue.


**Précision causale EXP-19, 23 septembre.** La reconstruction offline de S4/19097
confirme qu’une proposition améliore les primitives TRAIN mais dégrade même la
VALIDATION complète ; le standard l’écarte et la variante TRAIN la poursuit. Elle
révèle aussi que le trainer compare des moyennes mélangeant TRAIN/VALIDATION et
des lots différents selon les candidats. Les scores finaux S4 restent inchangés,
mais le delta ne sépare pas ces mécanismes. La généralisation nécessite un
contraste avec scores séparés et lots communs. Voir la
[preuve et les limites corrigées](../../EXP19/RESULTS.md).

**EXP-20 — préparation, 23 septembre ; aucun résultat LLM.** Une tâche HotpotQA
documentaire est proposée pour tester O1 avec des paramètres code/texte/int/bool
et un petit lecteur distinct de l’optimiseur. Le pilote de qualification est prêt
pour validation utilisateur : 24 questions × quatre conditions, 96 réponses de
Gemma 3 4B, aucun appel d’optimisation. TRAIN proposé : 60 problèmes, batch 6,
mémoire de deux échecs devenus réussites. Une option du trainer compare sur le
même batch courant, sans changer le comportement historique par défaut.
Le diagnostic hors ligne retrouve les deux documents requis dans 5/24 cas avec
le ranker initial ; ce n’est ni une mesure du lecteur ni une preuve d’efficacité
du curriculum. Confirmation, cache commun d’apprentissage et sélection finale
restent à sceller après pilote. Aucun résultat/hypothèse antérieur n’est révisé.
Voir [le protocole didactique et ses limites](../../_shared/o1_learning/EXP20.md).

**EXP-20 — qualification exécutée le 23 septembre ; comparaison non lancée.**
Les 96 réponses du pilote sont conservées. Exact match : programme initial 7/24,
répétition 6/24, dix documents 5/24, documents pertinents fournis 8/24. Deux gates
échouent : oracle <12/24 et seulement deux échecs initiaux résolus, contre trois
requis. Format valide 96/96, un changement de correction à la répétition, aucune
troncature (maximum 39 tokens sur 192). Certaines réponses correctes sur le fond
échouent à l’exact match par verbosité/alias ; d’autres sont effectivement fausses.
Le recalcul brut reproduit tous les scores sans rescoring manuel. **Ce résultat
ne teste ni l’apprentissage ni le curriculum** : zéro appel d’optimiseur, zéro
évaluation TEST. Il interdit le lancement de la comparaison sous ce protocole,
sans invalider le potentiel de HotpotQA ou d’O1.

Une panne amont HTTP 429 a nécessité une reprise documentée : 192 tentatives
initiales sans réponse, puis 96 réponses scientifiques et un contrôle de
disponibilité. Total 299 tentatives, 80 119 tokens et coût rapporté USD0,00407705 ;
la facturation inconnue d’erreurs sans reçu n’est pas imputée à zéro. Le lanceur
arrête maintenant l’envoi après une vague défaillante et conserve des diagnostics
sanitisés. Aucun modèle, score ou gate n’a été changé. Sources, tentatives et
manifest initiaux conservés. Suite proposée : isoler l’instruction de réponse
dans un nouveau diagnostic de pilote, sans modifier plusieurs mécanismes à la
fois. Voir [résultats, limites et preuves](../../_shared/o1_learning/EXP20.md).

**EXP-20 — poursuite complète autorisée, bloquée par le fournisseur.** À la demande
explicite de l’utilisateur, la campagne entière est préparée malgré les deux gates
échouées, sans les réviser : six graines, six réponses d’optimiseur par bras appris,
60 TRAIN / 24 VALIDATION / 48 TEST. Les étapes logicielles de cache partagé, reprise,
curriculum, archivage, sélection et analyse sont testées. La première question du
pilote d’intégration et deux contrôles de disponibilité ont subi 18 refus HTTP429
`engine_overloaded` du pool partagé DeepInfra. Zéro nouvelle réponse scientifique,
zéro appel DeepSeek complété, zéro TEST. Le changement de lecteur reste une décision
explicite à prendre ; aucun remplacement n’a eu lieu. Il s’agit d’un blocage externe,
pas d’un résultat sur l’apprentissage. Les sources initiales, le pilote historique
et chaque refus sont conservés. Voir [l’état de campagne](../../EXP20/results/run_store/full/status.json).


**EXP-20 — campagne complète terminée, 23 septembre 2026.** L’utilisateur a
explicitement demandé le lecteur Qwen 2.5 7B ; l’optimiseur est resté
`deepseek/deepseek-v4-flash-0731`. Après pilotes séparés, le protocole
`EXP20-FULL-v1` a exécuté six graines × deux bras appris × six réponses = 72,
avec 60 TRAIN / 24 VALIDATION / 48 TEST. Toutes les sélections des préfixes 0–6
ont été gelées avant TEST. Le programme mêle code de classement, instruction,
`top_k` entier et expansion booléenne. Le curriculum ajoute réellement les
échecs devenus réussites : 33 transitions, couverture de feedback 22–36 questions
par chaîne. Les deux bras comparent parent et proposition sur le même batch TRAIN.

Score principal (moyenne TEST aux préfixes 0–6) : inchangé 28,47 %, standard
43,11 %, curriculum 39,73 %. Différences appariées et intervalles bootstrap
exploratoires : standard−inchangé +14,63 points [+11,41 ; +17,81] ;
curriculum−inchangé +11,26 [+8,13 ; +14,09] ; **curriculum−standard −3,37
[−8,53 ; +2,38], inconclusif à n=6**. Exactitudes finales respectives : 28,47 %,
47,92 %, 43,75 %. Aucun seuil de 70 % atteint. Ce résultat montre un gain du
programme appris sur ce panel ; il ne démontre pas d’accélération propre au
curriculum. Il ne mesure ni profondeur récursive supplémentaire, ni amortissement,
ni avantage sur une génération indépendante sans feedback. H1/H2/H3 historiques
ne sont pas révisées par extrapolation.

Huit des douze programmes finaux utilisent dix documents. L’absence de contrôle
Qwen fixe à dix documents empêche d’attribuer tout le gain au classement appris.
La diversité/répétition des questions reste une piste, pas une cause isolée du
delta curriculum. Quatre réponses vides, cinq fins `length` et trois artefacts
générés invalides sont conservés ; zéro fallback TEST. Coût principal rapporté
2,848424 USD, 7 060 réponses réelles dont 6 988 Qwen, 12 987 439 tokens. Les pilotes
restent séparés, dont un appel interrompu de facturation inconnue.

Le routage de débit et une correction de clé de cache ont été testés et figés
avant confirmation. Aucun changement scientifique après TEST. Recalcul brut des
7 128 lignes externes identique, 276 tests passent, aucun skip. Le contrôle staged
utilisateur est préservé ; aucun commit ou dépendance ajouté. Les gates Gemma
échouées et les anciens blocages sont conservés comme états historiques.
[Résultats, sources et limites](../../_shared/o1_learning/EXP20.md),
[preuve de recalcul](../../EXP20/results/run_store/full/confirmation/raw_recomputation.json).

## EXP-21 — huit axes / méta-optimisation, bloquée avant conclusion (23 septembre 2026)

Demande : comparer les huit axes, combiner/ablater puis optimiser automatiquement
les réglages (O1), enfin exécuter O2→O1→O0 et confirmer hors développement.
Lecteur Qwen 2.5 7B, optimiseur DeepSeek v4 Flash 0731 conservés. Menu déclaré de
21 configurations isolées, deux graines DEV et six prévues en confirmation.

État : 99/252 réponses de développement, 13/42 chaînes apprises, 10/42 mesures
terminées. Deux refus HTTP 403 et l’API de statut confirment le plafond cumulé de
clé OpenRouter à 10 USD, solde nul. Un `BrokenProcessPool` est aussi conservé,
cause exacte non établie ; huit requêtes sans réponse/refus nécessitent une
réconciliation avant reprise. Aucun appel O1/O2 réel ni confirmation/test final.

Sur la seule graine appariée disponible, les différences de moyenne de progression
contre standard sont batch 12 +7,14 points, OPRO +4,76, feedback score seul +4,76,
dernier parent valide +1,19, code seul −15,48, int/bool seuls −16,67. Ce sont des
observations de développement n=1, **aucun gagnant sélectionné, aucune hypothèse
confirmée ou invalidée**. Le curriculum a produit six transitions réelles dans
une autre chaîne ; sa mesure finale manque. Toutes les cases absentes sont visibles.

Pilotes compris : 115 réponses DeepSeek, 4 433 Qwen, 12 030 420 tokens,
3,597817 USD rapportés ; facturation des requêtes interrompues inconnue.
297 tests passent. Recalcul exact de 3 841 lignes, 84 sources vérifiées, paramètres
réels conformes, 156 lignes invalides conservées, zéro fallback dans la mesure
partielle. Les sources gelées EXP20 et le staged utilisateur restent inchangés.
[Point d’entrée et tableaux](../../_shared/o1_learning/EXP21.md),
[bilan d’interruption](../../EXP21/results/run_store/interruption_summary.json),
[audit brut](../../EXP21/results/run_store/partial_development_audit.json).

### EXP21 — reprise du 24 septembre 2026

Les vingt chaînes commencées sont désormais achevées et mesurées : **120/252
réponses de développement, 20/42 courbes**, zéro requête localement en attente.
Les dix anciennes tentatives restent archivées ; huit complétions distantes non
récupérables ont été explicitement réémises avant confirmation, sans remplacement
d’aucune réponse reçue. Leur facturation incertaine reste signalée. Des défauts
de comparaison de checkpoints (tuples/listes JSON et ordre/identités de trace)
ont été corrigés et testés ; les sources EXP20 sont intactes.

Recalcul de 6 396 lignes, 104 sources ; 416 lignes invalides conservées, aucun
fallback dans les mesures disponibles. Coût cumulé EXP21 : 4,435877 USD rapportés.
Le plafond relevé à 14 USD laisse environ 3 USD, insuffisants pour les quelque
20 USD d’optimiseur plus lecteur du scénario maximal restant. Les 132 slots de
développement jamais émis, combinaisons/ablations, O1/O2 et confirmation restent
à exécuter. Aucun nouveau verdict d’hypothèse, aucune sélection de gagnant à
partir des cases incomplètes. [État et reprise](../../_shared/o1_learning/EXP21.md).

### EXP21 — continuation du 24 septembre, crédit du compte épuisé

99 nouvelles réponses DeepSeek et 4 121 Qwen : **219/252 propositions de développement, 33/42 courbes complètes**. Cinq chaînes interrompues sur HTTP 402, quatre inédites. Le compte est à −0,364179 USD alors que la clé garde 0,624477 USD sous son plafond de 14 USD : relever seulement le plafond ne réapprovisionne pas le compte.

Standard : progression moyenne 49,70 %, exactitude finale 54,17 %, programme inchangé 27,08 %, sur deux graines de développement. Sysmon +0,60 point et goal explicite +0,30 ; batch12 −0,30, OPRO −2,98, curriculum2 −4,76. Pas de gagnant choisi sur les comparaisons incomplètes, aucune exécution réelle O1/O2 ni confirmation.

961 tests passent, deux exclusions documentées ; audit de 10 432 lignes et 186 sources. Cumul EXP21 6.817566 USD rapportés, distinct du compte global. Reprise des cinq rejets de paiement préparée et testée par amendement v8, sans changer les données ni paramètres scientifiques. [Rapport et reprise](../../_shared/o1_learning/EXP21.md), [preuve API](../../EXP21/results/run_store/provider_credit_402.json).
