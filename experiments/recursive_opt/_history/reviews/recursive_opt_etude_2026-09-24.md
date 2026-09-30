# Étude vérifiée de `recursive_opt` — état au 24 septembre 2026

## 1. Conclusion et périmètre

**Le travail a produit une infrastructure et plusieurs gains locaux vérifiables. Il n’a pas établi un avantage général de la récursion, du feedback riche, de la mémoire ou de niveaux O2/O3 supplémentaires.** Dire « aucun progrès » serait aussi inexact que présenter ces expériences comme une validation de l’optimisation récursive.

Les acquis les plus solides sont le contrôle des expériences, la distinction validité/performance, les programmes portables, les corrections de mesure et certains diagnostics de sélection. La standardisation du dict est un acquis important, mais elle était **déjà largement présente au dernier commit de `recursive_opt`**. Elle n’est donc pas un nouveau résultat d’EXP15–21. Les traces et le curriculum sont maintenant effectivement branchés dans l’arbre local ; leur utilité comparative reste limitée ou indémontrée selon l’expérience.

**Le lot local ne peut pas être intégré proprement par un simple commit de tout `opto/features`.** L’essai dans un checkout isolé échoue sur 5 des 6 tests ciblant les nouvelles fonctions : la télémétrie et le curriculum requièrent du code hors `features`. Même l’extension indépendante OpenRouter requiert une mise à jour du fichier de provenance pour que le test obligatoire reste vert. Un petit patch cohérent est préparé ; sa portée et les éléments à différer sont détaillés en §8. **Le workflow élargi reste toutefois rouge sur deux tests historiques d’Experiment-0, déjà en échec sur HEAD propre** (§5/9).

### Références Git et méthode

| Référence | État vérifié | Usage dans cette étude |
|---|---|---|
| `recursive_opt` et `origin/recursive_opt` | `846580defe935195c1f5f39d6336079ef6ff1e10`, 2 septembre 2026 | Point de départ demandé ; EXP01–14 sont déjà dans son historique |
| HEAD de Trace au début de l’audit | `7e701b40485b9880ccfbb64c1faaabb22401c294`, branche `codex/exp17-parent-selection-exp18-memory-pareto` | 40 commits après la référence ; distinguer ces commits du travail local non commité |
| Dernier commit ayant changé `opto/features/recursive_opt` | `fc31bb669a`, 8 septembre | Ce n’est pas le dernier commit de la branche `recursive_opt` |
| Worktree `Trace-experiment0` | `46c2e9b7de`, HEAD détachée, avec ses propres changements locaux | Comparaison de l’état réellement présent, pas seulement de son commit |

L’étude utilise les fichiers locaux, les diffs Git, les sources des runners, les reçus/mesures enregistrés et des tests hors réseau. **Aucun nouvel appel LLM, aucun changement de modèle, aucune reprise d’EXP17/21, aucun push.** Les scripts d’expériences archivés ne sont pas des instructions de relance.

Vérifications nouvelles :

- Recalcul des **180 trajectoires / 5 760 valeurs objectives d’EXP15**, **720 / 23 040 d’EXP16-P1**, **864 / 27 648 d’EXP18**, à partir des points sauvegardés et des fonctions cibles ; métriques et moyennes concordantes. Total : **1 764 trajectoires, 56 448 valeurs**. Ce contrôle couvre les audits finaux, pas tous les caches TRAIN.
- Recalcul dédié d’EXP20 : **7 128 évaluations externes**, 72 réponses optimiseur, 252 lignes invalides conservées, zéro fallback ; scores, contrastes, hashes, unicité du cache et gel avant TEST vérifiés.
- Recompte de **545 reçus `response.json` pour EXP17** et absence des barrières finales ; aucun classement intermédiaire calculé.
- Comparaison des deux worktrees et des principaux résultats d’Experiment-0 ; tests de l’arbre local et du patch isolé.

Preuves reproductibles : [script d’audit](../audit_20260924/verify.py), [résultat et hashes des entrées](../audit_20260924/verification.json), [recalcul brut EXP20](../audit_20260924/exp20-raw-recomputation.json). Les chiffres non recalculés ici sont explicitement rattachés aux rapports ou fichiers d’origine. Les intervalles ci-dessous sont ceux des protocoles enregistrés ; cette étude n’en invente pas de nouveaux.

## 2. Ce qui était expérimenté avant EXP15

Les identifiants EXP01–14 sont repris du registre à `846580defe:artifacts/RESEARCH_LOG.md`. Ils ne correspondent ni aux numéros de sections de l'ancien assessment, ni aux UC1–14, ni aux « Prompt 17/18/19 » de migration. EXP14, notamment, est un inventaire et non une expérience d’efficacité.

| Expérience / données | Test exact et volume | Résultat vérifiable | Analyse actuelle |
|---|---|---|---|
| **EXP01 / Probe A** — [brut](../../EXP01/results/probe_a_results.json) | 3 prompts × 3 répétitions × 2 tâches, GSM8K/QASPER ; `inner_steps=0`, `max_examples=2` | Rapport signal/bruit 0,96 et 0,74 ; bruit QASPER ≈0,0391 dans le sous-ensemble analysé | Ce petit instrument ne distingue pas proprement effet du prompt et variabilité. Il ne prouve pas que toute optimisation de prose est impossible. |
| **EXP02 / Probe B** — [three-way](../../EXP02/results/three_way_report.json) | UC4 corrigé : mêmes niveau et famille de holdout ; 3 graines prévues, 8 candidats/bras ; seulement **1 paire exploitable** standard/récursif | Moyennes disponibles standard 0,184 (n=2), récursif 0,167 (n=1) ; delta apparié ≈−0,006 | Le +0,163 historique comparait des tâches différentes ; il est retiré. Les moyennes non appariées ne constituent pas le test causal. |
| **EXP03 / Probe F** — [brut](../../EXP03/results/probe_f_results.json) | Optimisation du prompt GSM8K contre état initial ; 5 graines tentées, 10 lignes, **4 paires** exploitables | Deltas `[0,530 ; 0,00925 ; 0,03525 ; 0,13275]`, moyenne **+0,1768125** ; très hétérogènes | Le chiffre historique +0,217 n’est pas la moyenne de ce fichier. Le claim positif reste retiré : instrument/bruit et attribution qualité/coût insuffisants. |
| **EXP04 / Probe K** — [brut](../../EXP04/results/probe_k_results.json) | Réécriture de code bin packing ; 3 graines, 6 lignes, **2 paires** exploitables | Gain apparent +4,8 ; artefact annoncé optimisé vide ; bruit découvert à concurrence réelle | Résultat inutilisable comme preuve de gain. Le registre ancien indiquait « 3 » sans distinguer tentatives et paires. |
| **EXP05 / Probe L** — [brut](../../EXP05/results/probe_l_results.json) | Même artefact sur 8 tâches, concurrence 1/8, 6 répétitions par cellule | Packing : SD 0 en série, **4,40555** à concurrence 8 ; autres tâches plates/saturées selon les cas | La certification doit utiliser la concurrence réelle. Les autres contrôles ont également observé SD ≈3,15 ; ce n’est pas une constante universelle. |
| **EXP06 / Probe R** — [brut](../../EXP06/results/probe_r_results.json) | Menus de prose appliqués au code, puis menus de code adaptés : 8 heuristiques packing, 6 admissible-set, contrôles invalides | 4 scores distincts sur chaque surface ; étendues **2 908,2** et **390**. La prose détruit les paramètres de code | Montre une surface active et un ancien instrument invalide, pas une victoire du moteur récursif. |
| **EXP07 / Probes S,T,T2** — [table](../../EXP07/results/probe_t_routing_menu.json) | 9 heuristiques de routage × 4 tâches, contrôles invalides, répétitions ; sensibilité à `max_examples` 2/4/8/16 | `nearest` meilleur **4/4**, corrélations de rang 0,51–1 ; 7/9 scores distincts sur TSP, 9/9 ailleurs | Optimum partagé dans un menu fini. CVRP/OVRP sont presque doublons ; famille effective de 3 tâches. Ce n’est pas une condition nécessaire à toute méta-optimisation. |
| **EXP08 / W2, U/U2/V0** — [rapport](../probe_2026/RESULTS_W2_routing.md) | Ordre appris sur tâches sources contre tirage uniforme sans remplacement parmi 9 candidats ; espérances exactes ; 360 répétitions task×candidat, plus contrôles | Qualité normalisée q=1 à budget cible 1 contre 9 ; coût amont 18 évaluations ; **K*=18/(9−1)=2,25**, ou 6 au seuil q=0,95 | Économie réelle face au contrôle non informé, avec candidats manuscrits portables. Une règle fixe `nearest` atteint aussi le plafond sans ce coût amont : pas de bénéfice propre au feedback établi. |
| **EXP09 / W,X** — [brut X](../../EXP09/results/probe_x_llm_w2.json) | Génération indépendante : X comporte **36 réponses**, 12 par cible de routage ; transfert de 11 programmes source valides vers deux tâches sœurs chacun | **11/11** exécutables sur leur source, **0/22** sur les sœurs ; erreurs de signature. Replay historique : un code VRPTW réduit la distance de **21,81 %** sur sa fixture | Échec du contrat de transfert testé, pas réfutation de tout transfert. Il existe aussi un gain local de code issu de génération indépendante. |
| **EXP10 / knobs S** — [brut](../../EXP10/results/probe_s_knobs_results.json) | Variation des paramètres de config sur **13 bundles** LLM4AD ; inventaire élargi de 47 tâches | Pas de variation de score dans ce chemin ; 13/13 bundles n’exposent qu’un exemple externe ; 43/47 tâches dans cette situation, 2 chargements en erreur | Le chemin à `inner_steps=0` et ces données n’exercent pas utilement l’ordre/batch d’apprentissage. Ne pas généraliser à un curriculum actif ou à tous les hyperparamètres. |
| **EXP11 / Y,Y2** — [répétitions](../../EXP11/results/probe_y2_knob_repeats.json) | BBEH interne à 15 exemples ; contrôle n=10 ; tailles/order/horizon, 5 répétitions par valeur pour trois paramètres | Score ≈0,99998 ; batch size : étendue inter-moyennes **8,82e−5**, ratio inter/intra 2,11 ; order 0,74 ; horizon 0,47 | **Petit signal batch possible**, pas « tous les effets dans le bruit ». Surface presque saturée, aucun transfert utile établi. Les −1 des backends absents sont des invalidités, pas des performances de traces. |
| **EXP12 / G QASPER** — [brut](../../EXP12/results/probe_g_qasper_paired_smoke.json) | Deux graines appariées, 4 runs, concurrence 2 et même endpoint | Δ moyen −0,18918 ; SD appariée **0,25365** | Effectif insuffisant, variabilité importante ; paramètres historiques incomplets. Ne pas transformer ce smoke en preuve contre la récursion. |
| **EXP13 / AA** — [brut local non suivi](../../EXP13/results/probe_aa_results.json) | Prompt trouvé contre prompt vide, ordre intercalé ; **6+6 tentatives**, limite 480 s | **12 scores `null`, 0 score exploitable**, 480 s enregistrées pour chacune | Terminé sans mesure utilisable, et non encore « en cours ». Le texte du prompt est conservé dans le JSON (477 caractères), même si le script lit un ancien chemin `/tmp` disparu. Pas de verdict find/noise possible. |
| **EXP14 / backlog** — [catalogue](../../EXP14/results/spec_backlog_catalogue.json) | Audit de **90 specs = 18 variantes × 5 répétitions**, pas de nouvelle campagne | 8 variantes signalées non exécutables ; 70 niveaux à menu partageaient le même menu de 5 éléments | Travail de qualification/migration. Ni 90 expériences scientifiques, ni 90 résultats positifs. |

Les replays d’EXP06–11 sont documentés dans [HISTORY_REPORT](../../_shared/optimizer_discovery/investigation16/history/HISTORY_REPORT.md). Un replay de données déjà consultées vérifie l’exécution et les anciens chiffres ; il n’est pas une nouvelle confirmation sur des données réservées.

### Place des UC et des exemples antérieurs

Les notebooks A/B/C/D/E et UC1–14 précèdent ce registre. Leur rôle est cartographié dans [RESEARCH_LOG §2](../navigation/RESEARCH_LOG.md) ; ils ne constituent pas de nouvelles expériences après `846580defe`.

| Usages historiques | Objet testé | Bilan conservable |
|---|---|---|
| UC1, UC5, exemples B/C | Réécrire du code de solveur/composant, des outils ou capacités | UC1 : qualité 1,0 des deux bras sur 3 graines, 2 appels optimiseur contre 4 ; seulement deux points de courbe, donc pas de facteur de convergence précisément estimé. Autres scores souvent issus de petits validateurs. |
| UC2, UC3, UC6, UC11 | Prompt/configuration, capacités, multiobjectif, modes de traces, code émetteur de prompt | Mesures bruitées ou évaluateurs défectueux/inactifs selon version. Les backends absents ne prouvaient pas l’inutilité des traces. |
| UC4 | Politique par famille puis prior transféré | Gain phare retiré ; voir EXP02. Le smoke du notebook migré n’est pas cette expérience. |
| UC7, UC8–10, UC12 | Routage vers solveur, arrêt/restart, outils, promotions, primitives de contrôle | Démonstrations d’architecture et scores sur cas préparés ; efficacité d’une campagne réelle non établie. |
| UC13 | Solveur numérique contre recherche générative de knobs | Économie d’appels LLM potentielle ; ancien verdict de vitesse sur la première graine, seuil standard atteint sur seulement 1/3 des graines archivées. |
| UC14 | Transfert de code entre petites politiques source/cible | Delta historique −0,07 sur 3 paires dans cette version ; asymétrie source/cible, aucune généralisation à tout transfert. |

Le tableau original de juin et son bandeau de rétractation sont eux-mêmes historiques : ils ne doivent pas être fusionnés avec les corrections de septembre en un verdict unique. Les anciens commentaires « promote » ou « positive claim » ne sont pas l’autorité actuelle.

### Autres probes : ne pas les compter comme autant de confirmations

| Groupe | Ce qui a été fait | Ce qu’il permet de conserver |
|---|---|---|
| [C](../probe_2026/probe_c_results.json), [D](../probe_2026/probe_d_results.json), [I](../probe_2026/probe_i_results.json) | Décomposer métrique et tokens, varier la température, comparer prompt vide/terse/CoT ; C contient 12 mesures sur chacun de deux benchmarks internes | Diagnostic d’un score mêlant exactitude et longueur. Une réponse plus brève peut améliorer ce score sans meilleure résolution. |
| E/G/H/J | Certification initiale puis corrigée, vivacité des évaluateurs et extension du pool ; E couvre 8 tâches, G/H 5, chacun des deux fichiers J 10 entrées, avec recouvrements | Des erreurs du harness faisaient paraître certaines tâches mortes ; ne pas agréger les pools comme des tâches indépendantes ni reprendre la première certification. Voir [E corrigé](../probe_2026/probe_e_results_corrected.json), [G](../probe_2026/probe_g_liveness.json), [H](../probe_2026/probe_h_recertified_llmfree.json), [J](../probe_2026/probe_j_pool.json). |
| [M](../probe_2026/probe_m_results.json), [N](../probe_2026/probe_n_results.json), [O](../probe_2026/probe_o_results.json), [P](../probe_2026/probe_p_results.json) | Diagnostic d’Experiment-0 : M=24 évaluations ; N=246 évaluations/492 forwards, 818 s ; O explore 3 tâches ; P=52 exemples BBEH object-counting | N : 0 invalidité sur 246, borne supérieure Wilson ≈1,54 % ; P : 0/52 invalidité mais exactitude **11,54 %**. Zéro événement observé ne garantit pas un taux nul ; fiabilité du parsing et qualité sont distinctes. |
| [Q / iteration3](../probe_2026/iteration3_analysis.json) | 3 surfaces × 3 paires standard/récursif ; 18 lignes dans les trois JSON Q | Numeric/mixed : delta 0 ; packing : −3 ; artefacts identiques et menus effondrés. Expérience invalide pour tester les traitements prévus. |
| S/T/U/V/W/X/Y | Signatures, ordre transféré, contrôles de répétition, génération et knobs | Sous-expériences d’EXP07–11 ; les dépenses et observations ne doivent pas être comptées une seconde fois. Certains scripts ne disposent pas d’un résultat complet correspondant. |

## 3. `optimizer_discovery` : Phase 0, EXP15, EXP16, EXP17

### Objet commun et interprétation des métriques

Le modèle génère le code de `propose(history, bounds, seed)`. Le programme propose des points numériques ; l’hôte évalue une fonction cachée. Les familles sont Sphere, Quadratique anisotrope et Rosenbrock, en dimensions 2 et 4. Le programme déployé ne rappelle pas de LLM. La métrique principale est le regret normalisé moyen des meilleurs points successifs sur **32 appels objectif** : **plus bas est meilleur**.

Il faut distinguer budget de génération, trajectoires numériques, appels objectif et réplications statistiques. Un million d’appels numériques n’équivaut pas à un million de preuves indépendantes d’efficacité. Les graines externes sont les réplications ; les panels restent fixes dans chaque étude. Les optima sont centrés dans une partie des bornes, ce qui explique l’importance du contrôle manuscrit midpoint.

| Étude | Comparaison exacte et volume | Performance / validité | Conclusion défendable |
|---|---|---|---|
| **Phase 0 : interface** | 3 générations ouvertes à plafond 3 000, puis 1 smoke midpoint séparé ; 10 156 tokens | 0/3 sources exploitables pour la génération ouverte ; smoke valide sur 3 graines ×8 appels | Contrat et transport possibles ; protocole initial de génération NO-GO, aucune efficacité démontrée. |
| **Phase 0 : calibration** | 17 nouvelles réponses : contrôle 1, pilotes 3+3, confirmation 10 ; même modèle, passage à 8 000/low ; 41 503 tokens ; 360 appels fixture sur 15 programmes exécutables | Confirmation 10/10 valides et sensibles à l’historique, 7 trajectoires distinctes ; 240 appels pour cette confirmation | Faisabilité de génération dans ce contrat. 10/10 ne garantit pas une fiabilité population ≥90 %. Voir [rapport](../../_shared/optimizer_discovery/PHASE0_REPORT.md). |
| **EXP15** | A0 seed fixe ; A1 génération indépendante ; A2 recherche Trace avec feedback. **5 graines ×2 bras ×8 réponses =80**, 81 tentatives. TRAIN6/validation6/holdout12, 1 graine locale ; 180 trajectoires d’audit | AUC A0 **0,139579**, A1 **0,121748**, A2 **0,116680**. A2−A0 **−0,022899**, IC [−0,042456 ; −0,003341]. A2−A1 **−0,005068**, IC [−0,037728 ; +0,024551]. Inéligibles 9/40 A1, 12/40 A2 ; A2 garde le seed 2 fois | Gain contre seed initial, **avantage du feedback sur génération indépendante inconclusif**. A2 perd 2 paires sur 5 ; A1 meilleur en moyenne sur regret final/atteinte. [Rapport](../../_shared/optimizer_discovery/EXP15_REPORT.md), [brut](../../EXP15/results/exp15_results.json). |
| **EXP16-P1** | I indépendant ; C réécrit le parent choisi sur TRAIN sans scores/trajectoires explicites ; R ajoute feedback du parent ; W répartit deux parents sur 4 rounds. **6 graines ×4 bras ×8=192 réponses**, 194 tentatives ; TRAIN24/validation12/audit12 ×2 graines locales | AUC A0 **0,179329**, I **0,077260**, C **0,047588**, R **0,122517**, W **0,112582**. R−I **+0,045257**, IC [+0,003176 ; +0,085126] ; R−C **+0,074929** défavorable ; W−R inconclusif. 720 trajectoires d’audit valides | Le feedback riche testé fait moins bien que les contrôles I/C. C−I **−0,029672** est prometteur mais **post-hoc**. C utilise indirectement la performance pour choisir le parent : ce n’est pas « aucune information ». [Rapport](../../_shared/optimizer_discovery/investigation16/REPORT.md). |
| **EXP16 contrôle B2 séparé** | Même seed, seul premier point remplacé par le centre des bornes ; 144 nouvelles trajectoires sur l’audit P1 | AUC **0,031633**, soit −82,36 % contre A0 ; regret final moins bon que certains bras appris | Contrôle simple extrêmement compétitif. Battre A0 seul n’identifie pas la valeur du mécanisme génératif. [Résultats](../../_shared/optimizer_discovery/investigation16/production_baseline_control/results.json). |
| **EXP17** | Confirmation de C−I, **46 nouvelles paires ×2 bras ×8=736 réponses prévues** ; A0/B2 contrôles ; panels 24/12/12 ×2 ; 64 032 trajectoires /2 049 024 appels **alloués**, pas observés | **545 réponses reçues**, aucune barrière finale de génération/sélection, aucun audit final. Pilote disjoint prévu/exécuté séparément | **Suspendue, sans résultat confirmatoire.** Ni le signal post-hoc d’EXP16 ni EXP18 ne remplace les 191 réponses et l’analyse manquantes. [Protocole](../../_shared/optimizer_discovery/exp17/PREREG_EXP17.md), [pause](../../_shared/optimizer_discovery/exp17/USER_REQUESTED_PAUSE.json). |

EXP15 rapporte **573 191 tokens, 0,086824 USD connus**, hors facturation inconnue du timeout ; 28 800 appels objectif partagés effectivement comptés dans son bilan, plus comptabilité séparée de normalisation. Son pilote de 4 réponses est distinct des 80 réponses principales. EXP16-P1 rapporte **6 980 225 tokens, 0,488195 USD connus**, hors deux tentatives échouées. L’enquête EXP16 entière conserve **242 réponses** : G1 24, F1 24, pilote production 2, P1 192. Les coûts connus ne sont pas des factures exhaustives.

### Ce qu’EXP16 a réellement isolé

| Diagnostic | Volume / résultat | Ce qu’on peut en déduire |
|---|---|---|
| Audit S0/feedback | EXP15 : 4/32 observations retenues par tâche, incumbent absent dans 69,6 % des résumés, 0/80 prompts explicitant l’objectif anytime | Lacunes d’information certaines. Cela ne prouve pas que les corriger améliore le résultat. |
| G1 | 12 réponses par plafond ; admissibilité 8/12 à 8 000, 10/12 à 32 000 | Choix prospectif de faisabilité ; aucune réponse de ce diagnostic ne dépasse 6 581 tokens de complétion. Pas d’effet causal isolé du plafond sur la qualité. |
| B1 | 384 trajectoires valides ; politique adaptative fixe meilleure que contrôles simples | Surface non globalement saturée. |
| B2 public | 96 paires ; AUC −85,66 % sur distribution centrale, −67,14 % sur élargie ; 26 pertes individuelles conservées | Initialisation utile, sans domination de toutes les métriques. |
| S1 | Banque fixe, 200 sous-panels recouvrants ; score d’audit du programme sélectionné 0,091412 →0,081151 →0,075566 avec panels 6×1 /12×2 /24×2 | Meilleure sélection locale ; pas 200 réplications indépendantes de recherche. |
| T1 corrigé | Mêmes résultats à 1/4/8/16 workers ; ≈**5,49×** de débit à 8 workers | Gain d’ingénierie de l’évaluateur, distinct d’un gain de recherche et de l’amortissement. |
| F1 | 24 réponses à parents fixes ; rich−sparse **+0,049229**, IC [+0,000160 ; +0,103587] ; rich−code inconclusif | Aucun gain du feedback riche ici. Ce n’est pas une boucle d’apprentissage à plusieurs tours. |

Sources détaillées et limites : [rapport EXP16](../../_shared/optimizer_discovery/investigation16/REPORT.md), [matrice de décision](../../_shared/optimizer_discovery/investigation16/DECISION_MATRIX.md), [audit d’achèvement](../../_shared/optimizer_discovery/investigation16/COMPLETION_AUDIT.md). Un timeout réseau déclaré ne constitue pas automatiquement une borne absolue de temps mural.

## 4. EXP18–21 : prolongements nécessaires au bilan actuel

| Étude | Volume et intervention | Résultat principal | Limite de l’attribution |
|---|---|---|---|
| **EXP18, terminée** | 6 graines ×4 bras ×16 réponses =**384** ; 399 tentatives, 5 599 563 tokens ; L parent scalaire, M archive jusqu’à 7 codes antérieurs, P parent Pareto TRAIN, PM combinaison ; 864 trajectoires d’audit | AUC L **0,035038**, M **0,034315**, P **0,046941**, PM **0,039437**, B2 **0,040987**, A0 **0,144291**. Mémoire : −0,004114 [−0,020651 ; +0,014910] ; Pareto : +0,008512 [−0,015019 ; +0,035682] | Sept contrastes mécanistiques inconclusifs ; pas de bras indépendant N16. Les mécanismes sont exposés réellement, ce n’est pas un simple no-op. [Rapport](../../_shared/optimizer_discovery/exp18/REPORT.md). |
| **EXP19-S3** | Tâche synthétique de 6 tables booléennes/24 bits ; TRAIN24, VALIDATION24, TEST48 ; 3 graines ×3 bras appris ×8 réponses =**72** ; diagnostic imbriqué séparé 10 réponses | Initial 39,58 %, standard 47,92 %, O1 sélectionné 50 %, configuration issue de récursion 49,31 % ; aucun bras n’atteint 90 % | Pas d’accélération établie. O2 épuise 16 000 tokens sans texte ; sa configuration retenue égale le standard. L’écart observé n’est pas un effet de profondeur. |
| **EXP19-S4** | 3 graines ×2 bras ×4 réponses =**24** ; même batch TRAIN de 6 ; supprimer VALIDATION du fitting, garder sélection externe VALIDATION | TEST standard **72,22 %**, parent sur TRAIN **100 %** ; +27,78 points, IC descriptif [+22,92 ; +31,25]. Seuil 90 % atteint dans le budget 3/3 contre 0/3 | Gain local réel ; intervention diagnostiquée manuellement, pas découverte par O2. **Aucun curriculum** dans S4. La sélection standard mélange des scores TRAIN/VALIDATION et des lots différents : plusieurs mécanismes changent ensemble. |
| **EXP20, terminée** | HotpotQA, code de ranking + prompt + `top_k` + booléen expansion ; lecteur Qwen 2.5 7B, optimiseur DeepSeek ; **6 graines ×2 bras ×6=72 réponses** ; TRAIN60/VALIDATION24/TEST48 | Exactitude finale : initial **28,47 %**, standard **47,92 %**, curriculum **43,75 %**. Métrique principale (moyenne des préfixes 0–6) : 28,47/43,11/39,73 %. Curriculum−standard **−3,37 points**, IC [−8,53 ; +2,38] | Apprentissage utile contre initial. Curriculum actif (33 transitions) mais avantage non établi. 8/12 programmes choisissent 10 documents ; pas de contrôle Qwen fixe `top_k=10`, pas de génération indépendante, pas de découverte O1 automatique. |
| **EXP21, partielle** | 21 configurations ×2 graines ×6=252 réponses de développement prévues ; batch, traces, surface, feedback, goal, optimiseur, trainer, puis O1/O2 prévus | État de reprise : **120/252 réponses**, **20/42 chaînes mesurées** ; 132 slots de développement restants ; aucun O1/O2 réel, aucune confirmation, TEST fermé | Pas de gagnant global. Des tableaux partiels ou une seule graine appariée ne justifient pas de sélectionner les axes favorables. L’index resté à 99/252 et 10/42 est périmé. |

EXP18 : **48/384 programmes inéligibles**, zéro fallback d’audit, 0,905999 USD connus. Le plan alloue 967 680 appels objectif ; le cache enregistre **804 709 appels réels**, et des travaux interrompus supplémentaires restent inconnus. Ni budget alloué ni fichiers de reçus ne doivent servir de dénominateur statistique.

EXP19 entière : **255 réponses uniques, 2 332 712 tokens, 0,218567 USD connus**. S1/S2 ont des défauts de projection des traces ; S3 les corrige. Le curriculum failed→solved a été exercé séparément sur une graine en S2 (2/0/6 transitions selon réglage), sans preuve d’efficacité. Les gains de S4 ne doivent pas lui être attribués. La mémoire spécialisée des 24 labels constitue également un contrôle simple possible. [Rapport et limites corrigées](../../EXP19/RESULTS.md), [S3](../../_shared/o1_learning/s3_results.json), [S4](../../EXP19/results/s4_results.json).

EXP20 : **9 168 évaluations de politique** =2 040 apprentissage +5 880 TRAIN/VALIDATION externes +1 248 TEST ; **6 988 réponses lecteur**, 1 910 cache hits, 72 réponses optimiseur ; **12 987 439 tokens, 2,848424 USD** pour la campagne principale. Les appels et changements de lecteur des pilotes sont séparés. Le modèle optimiseur demeure `deepseek/deepseek-v4-flash-0731`. [Rapport](../../_shared/o1_learning/EXP20.md), [résultats](../../EXP20/results/run_store/full/confirmation/results.json).

EXP21 au point de reprise archivé : pilotes inclus, **136 réponses DeepSeek, 6 487 Qwen, 15 847 779 tokens, 4,435877 USD connus** ; 8 tentatives distantes perdues de facturation inconnue restent conservées. Ces chiffres sont partiels et ne s’ajoutent pas à une confirmation inexistante. [État primaire](../../EXP21/results/run_store/recovery_summary.json), [rapport actualisé](../../_shared/o1_learning/EXP21.md).

## 5. État final du control plane et réutilisation du dict

### Statut technique exact

Le contrat reste **`schema_version="recursive-opt/v2alpha"`, `kind="recursive_optimization"`**, avec blocs globaux et `levels[]` contenant notamment `surface`, `module`, `engine`, `objective`, `datasets`, `llm_roles`. Les extensions sont validées avant exécution ; les modules/évaluateurs passent par les registres.

La migration historique a classé **85 fichiers** : 10 `normalized_only`, 6 `local_nonportable`, 46 `historical_only`, 23 `missing_dependency`, **0 `execution_replayable`**. Normaliser ne signifie donc ni reconstituer les dépendances d’époque ni reproduire les anciens scores. Le notebook courant UC4/UC14 est un contrôle technique déterministe, pas un replay scientifique. [Rapport de migration](../../_shared/control_plane_v2/migration_report.md).

Les contrats testent réellement Trace et GEPA, les champs causaux, l’isolation du holdout, les budgets, les rôles LLM, la reprise et la provenance. Les tests ont une valeur d’ingénierie ; leur succès n’établit pas un gain d’optimisation. Les premiers défauts de migration et les corrections ultérieures doivent être lus chronologiquement dans [readiness_audit](../../_shared/control_plane_v2/readiness_audit.md) et [proof](../../_shared/control_plane_v2/proof.md).

**Le dernier statut machine local est `ready_for_prompt_18=false`, `required_gepa_ci=false`.** Le run CI ancien `33094071518` est enregistré réussi, mais `covers_current_tree=false`. Aucun CI distant récent n’a été exécuté ou vérifié par cette étude. Le hash local de l’arbre de travail est `f0d3218e…aee3` dans [prompt18_readiness.json](../../_shared/control_plane_v2/prompt18_readiness.json). Un historique « green » ne certifie pas le checkout actuel.

Attention : `spec._runtime_tree_sha256()` ne hache que les `.py` sous `opto/features/recursive_opt`. Ce hash **seul** ne certifie pas les modifications de `opto/trainer` ou `opto/trace`. Les snapshots de sources des expériences et des tests d’intégration restent nécessaires.

### Réserve supplémentaire : le workflow complet n’est pas vert

La vérification élargie donne **224 succès et 2 échecs**. Les deux échecs sont dans `experiments/recursive_opt/multiobjective_reasoning/tests/test_main_experiment.py` :

- `test_invalid_completed_core_result_resumes_without_provider_calls` attend `evaluation.status="constraint_failed"`, mais le résultat courant porte `"invalid"` ;
- `test_frozen_protocol_hashes_profiles_and_constraints_are_unchanged` attend le hash ancien de `evaluator.py`, qui a été modifié par les travaux historiques de diagnostic/amendement.

Ils sont reproduits **à l’identique sur HEAD propre `7e701b4048`**, puis sur le petit patch de routage. Ils ne sont donc pas créés par les modifications locales de curriculum ou par ce patch. Le premier mérite une revue du contrat de validité et de sa normalisation ; le second une distinction explicite entre source historique gelée et source amendée. **Ne pas simplement remplacer les assertions/hash pour obtenir du vert.** Cela empêche de déclarer le workflow requis globalement sain, même si la suite unitaire et les contrats propres du control plane passent.

### Le même dict a-t-il été utilisé dans EXP19, EXP20 et EXP21 ?

**Oui, même schéma et même point d’entrée `run_spec`, mais pas une implémentation gelée inchangée ni un fichier JSON universel autosuffisant.**

| Campagne | Preuve dans le code | Extensions / comportement effectif |
|---|---|---|
| EXP19 | [study.specification et appel `S.run_spec`](../../_shared/o1_learning/study.py) ; [diagnostic S4](../../_shared/o1_learning/selection_diagnostic.py) | Ajout de `objective.trace_config` dans `spec.py` ; `engine.config.trainer_kwargs.curriculum` vers le trainer ; modules booléens enregistrés. S4 remplace `datasets.validation` du fitting par `[]`, garde la sélection externe. |
| EXP20 | [task.specification](../../o1_qa/task.py), [campaign.fit](../../o1_qa/campaign.py) | Même `v2alpha`, module/évaluateur HotpotQA enregistrés ; `run_spec(..., resources={llm_factory, trainer: EXP20CampaignTrainer})`. Trace interne bornée, `selection_score_window="latest_train_batch"`, curriculum facultatif ; validation/test externes au fitting. `provider.sort` est accepté par un changement additionnel. |
| EXP21 | [axes.specification/fit](../../o1_qa/axes.py), [meta.specification](../../o1_qa/meta.py) | Repart du dict EXP20, modifie les champs par axe, enregistre `EXP21Trainer` ; prépare O1/O2 avec évaluateurs qui exécuteraient les niveaux inférieurs. Les appels imbriqués réels O1/O2 ne sont pas encore exécutés. |

Les registres locaux, clients injectés, trainers spécialisés, journaux et étapes externes de sélection sont donc partie du dispositif. Copier seulement une spec dans l’ancien checkout ne reproduit pas EXP19–21. Un ancien validateur `v2alpha` refusera notamment `trace_config` : version textuelle identique ne signifie pas compatibilité bidirectionnelle totale.

`credit_horizon` dans ce nouveau bloc est une limite de **projection locale de trace** : step/truncated=1 nœud, episode≤8, full≤`max_nodes`. Il ne réalise pas une mémoire inter-épisodes ou une rétropropagation temporelle. Les traces OTEL/sysmon ne couvrent pas automatiquement les processus enfants ; l’archive complète et le feedback projeté sont distincts.

## 6. Différences avec `Trace-experiment0`

`/home/xav/code/Trace-experiment0` est un **autre worktree du même dépôt**, pas le contenu de `Trace/artifacts`. Les données principales d’Experiment-0 sont dans :

- [sources et rapports](../../multiobjective_reasoning) ;
- [matrice principale enregistrée](../../../../outputs/recursive_opt/experiment_0/experiment-0-v2/main_after_transport_resilience_fix/main.json) ;
- [analyse](../../../../outputs/recursive_opt/experiment_0/experiment-0-v2/main_after_transport_resilience_fix/analysis.json).

Ces résultats existent dans les deux worktrees. Les quatre fichiers centraux `main.json`, `analysis.json`, `decision.json`, `episode_trajectory_audit.json` sont **identiques octet par octet**. La présence d’une copie dans Trace ne représente pas une seconde expérience.

### Résultat réel d’Experiment-0

Workflow GSM8K à deux instructions analyse/réponse ; contrôle fixe A, Trace B, GEPA C, ablation D = Trace sans gate de validation ; **5 graines ×2 budgets candidats (6/12) ×4 bras =40 unités**. Les budgets candidats ne sont pas des comptes garantis égaux d’appels optimiseur ; les résultats enregistrent séparément les appels réels.

| Bras | Exactitude moyenne | Ratio tokens forward moyen | Unités avec sortie invalide |
|---|---:|---:|---:|
| A fixe | 99,17 % | 1,01248 | 1/10 |
| B Trace | 97,92 % | 0,87902 | 1/10 |
| C GEPA | 95,00 % | 0,75508 | 1/10 |
| D Trace sans gate validation | 88,75 % | 0,62903 | 1/10 |

Les **40/40 unités sont terminées** et les gates d’infrastructure passent. Les critères scientifiques enregistrés de qualité/efficacité ne sont pas satisfaits ; 4/40 unités ont une invalidité sous la règle originale zéro tolérance. B−A : exactitude −1,25 point, IC [−4,17 ; +1,67], ratio tokens −0,13346, IC [−0,24897 ; −0,02295]. Réduction de tokens observée, **pas de gain de qualité prouvé**. Ce sont des prompts presque saturés, pas les programmes numériques d’EXP15–18 ni HotpotQA d’EXP20.

Les 20 runs optimisés n’avaient aucune `candidate_trajectory` persistée. La décision finale était `RETURN_TO_CONTROL_PLANE_FOR_TRAJECTORY_PROVENANCE`. Le hotfix ultérieur ajoute cette persistance pour les nouveaux runs ; **il ne reconstitue pas magiquement les lignées manquantes des 40 runs**. [Rapport de clôture](../../multiobjective_reasoning/reports/prompt18_r3f_main_completion_trajectory_stop.md), [hotfix](../../_shared/control_plane_v2/candidate_trajectory_provenance_hotfix.md).

### Différences du code présent

Comparaison des `.py/.md/.json`, hors `__pycache__`, entre les deux worktrees :

| Périmètre | Identiques | Différents / supplémentaires dans Trace |
|---|---:|---|
| `opto/features/recursive_opt` | 2 | 15 différents, 2 nouveaux : `measurement.py`, `optimizer_program.py` |
| `opto/trace` | 15 | `bundle.py` différent ; 12 fichiers `io` nouveaux |
| `opto/trainer` | 14 | 5 fichiers différents |
| `opto/optimizers` | 9 | 3 différents : extraction et invalidité des réponses dans OptoPrime/OptoPrimeV2/utils |
| `experiments/recursive_opt/multiobjective_reasoning` | 1 461 | 3 différents, 3 supplémentaires |
| `artifacts/control_plane_v2` | 31 | 3 différents : ADR, footprint, readiness |

Les trois sources d’Experiment-0 différentes sont `evaluator.py`, `main_experiment.py`, `specs.py` : journalisation bornée du texte invalide, maintien des invalides comme mauvaises réponses dans le dénominateur, et **amendement `invalid_rate <= 0` → `<= 0.1`**. Les anciens résultats restent inchangés. Une nouvelle exécution avec cet amendement n’aurait pas exactement la règle scientifique originale ; on ne peut pas déclarer rétroactivement les anciennes gates réussies. [Amendement](../../multiobjective_reasoning/manifests/invalid_rate_gate_amendment_v1.json).

## 7. Ce qui a changé hors `artifacts`, et les risques

### Entre le commit `recursive_opt` et HEAD, déjà commité

Le diff du code `opto` porte uniquement sur cinq fichiers de `features/recursive_opt`, avec six fichiers de tests associés. **Aucun changement de `opto/trace`, `opto/trainer` ou `opto/optimizers` dans cet intervalle Git.**

| Fichier | Changement conservé dans l’historique |
|---|---|
| `optimizer_program.py` | Nouveau contrat portable, validation de source/proposition, exécution isolée et bornée |
| `measurement.py` | Preuves des menus effectivement évalués, validité typée |
| `spec.py` | Enregistrement/intégration du programme optimiseur, observation des évaluations avant éviction, export de métriques/trajectoires et traitement des propositions invalides |
| `levels.py` | Transmettre le résultat nécessaire à la collecte de preuves |
| `tracebench.py` | Rendre accessibles les observations de mesure |

Ces changements sont **déjà commités** dans les 40 commits ; il ne faut pas les « recommiter » comme s’ils étaient du travail local inédit. Les modifications d’OptoPrime observées contre Experiment-0 sont plus anciennes que `846580defe` et ne contredisent pas ce constat.

### Changements locaux avant cet audit

| Emplacement | Portée réelle | Risque / décision |
|---|---|---|
| `features/recursive_opt/spec.py` | `trace_config`, capture autour de l’évaluation, projection dans feedback, export `curriculum_events`, exception contrôlée `provider.sort` | Plusieurs sujets mélangés. Extraire le seul routage indépendant ; différer le lot capture/curriculum. |
| `features/recursive_opt/traces.py` | Distingue OTEL/sysmon, capture stricte, validation des limites, projection bornée | Dépend du backend non commité ; nettoyage lors d’un échec partiel d’ouverture et erreurs de conversion TGJ à auditer. Le chemin `strict` conserve des warnings lors de certaines erreurs de conversion : il n’est pas un contrat universel fail-fast. |
| **`opto/trace/bundle.py`** | Instrumentation des appels sync/async des bundles lorsqu’une session est active | **Modification du cœur**, impact possible sur toutes les fonctions bundlées, erreurs/async/coût des spans à tester. |
| **`opto/trace/io/*.py`** | 12 nouveaux fichiers déjà indexés par l’utilisateur ; 4 004 lignes dans l’index initial ; certains fichiers ont aussi des modifications non indexées | **Modification du cœur**, imports optionnels, SDK, sessions/contexte, sys.monitoring et frontends. Ne pas embarquer ce gros checkout dans un commit incident. |
| **`opto/trainer/loader.py`** | Buffer failed→solved, sampling avec replay, validation des données | Compatibilité de sérialisation, diversité des batches, ancien dataset vide/array ; une régression de restauration d’ancien état est reproduite ci-dessous. |
| **`opto/trainer/sampler.py`** | Agrégation des scores curriculum sur les candidats ; exclusion d’ExceptionNode | Validité sémantique d’un score fini, alignement indices/subbatches, concurrence ; EXP21 ajoute justement un adaptateur de validité. |
| **`opto/trainer/search_template.py`** | Réévaluation du précédent batch TRAIN, branchement curriculum, `log_frequency=None` | Appels supplémentaires et comptabilité ; régression possible sur loaders personnalisés/anciens états sans attribut `curriculum`. |
| **`opto/trainer/algorithms/priority_search.py`** | Mode opt-in `latest_train_batch`, score commun parent/proposition, reset mémoire | Restriction à un parent/proposal/batch, objectif scalaire ; ne pas généraliser ce mode sans tester les invariants. Le défaut historique de mélange des populations reste dans le mode `history`. |
| **`opto/trainer/algorithms/classical_algorithms.py`** | Passage explicite des arguments `guide=` et `train_dataset=` dans trois wrappers | Petit correctif autonome potentiel ; devrait avoir son propre commit avec tests, hors demande stricte `features`. |
| `examples/recursive_opt_three_way.py` | Verdict de vitesse sur toutes les graines, non-atteintes conservées | Correctif de reporting utile, hors commit `features`. |
| `experiments/recursive_opt/o1_qa/`, tests O1/QA/EXP16–17 | Adaptateurs/runners/tests locaux non suivis | Nécessaires à la reproduction des expériences ; leur simple présence ne rend pas les fonctions génériques de la bibliothèque disponibles dans un clone propre. |
| Notebook PAL local, `outputs/...` | Démonstration et sorties locales non suivies | Conserver, ne pas confondre avec une campagne comparative vérifiée. |

Pas de modification de dépendances de production dans ce diff local. Les autres sous-packages `opto/features` sont inchangés. L’inventaire complet initial de l’index a été sauvegardé avant toute opération de commit ; les fichiers IO restent hors du lot proposé.

### Régression reproduite : ancien état `DataLoader`

Un état sauvegardé avant le curriculum ne contient ni `curriculum` ni `_last_indices`. `__setstate__` ne crée pas de valeurs par défaut ; `sample()` lit maintenant `self.curriculum` sans garde. Reproduction locale, en construisant l’ancien état puis en réattachant le dataset comme l’API le demande :

```python
from opto.trainer.loader import DataLoader

loader = DataLoader({"inputs": ["a"], "infos": [1]}, randomize=False)
state = loader.__getstate__()
state.pop("curriculum")
state.pop("_last_indices")
restored = DataLoader.__new__(DataLoader)
restored.__setstate__(state)
restored.dataset = {"inputs": ["a"], "infos": [1]}
restored.sample()  # AttributeError: 'DataLoader' object has no attribute 'curriculum'
```

**Ce défaut n’est pas corrigé dans cet audit**, pour ne pas modifier les sources expérimentales et le cœur hors du lot demandé. Avant intégration du curriculum : initialiser les champs absents lors de la restauration et ajouter un test de véritable ancien état, en plus des pickles nouvellement créés. Une suite verte n’exclut pas ce cas non couvert.

## 8. Garder, différer et commiter

| Décision | Éléments | Conditions concrètes |
|---|---|---|
| **Garder dans la bibliothèque** | `levels`, `effects`, budgets, mémoire, `optimize`, `measurement`, contrat portable, normalisation/validation du dict | Préserver les tests d’effet causal, de score valide, d’isolation, de budgets et de replay ; ne pas leur attribuer une efficacité générale. |
| **Garder comme contrôles réutilisables** | Seed midpoint B2, programme exact A2/41, programmes source valides, tâches et métriques versionnées | Conserver source+hash+contrat et domaine de validité. Le sous-processus du programme n’est pas une sandbox de sécurité. |
| **Garder comme résultats/archives** | EXP01–21, sorties invalides, gels, reçus, tentatives interrompues | Pas de suppression ou de « remplacement » des résultats défavorables. Pas d’audit d’EXP17 avant sa barrière prévue. |
| **Différer l’intégration globale** | `trace_config`/traces.py + IO/bundle ; curriculum + loader/sampler/search ; nouveaux modes de sélection | Lots autonomes de code **et tests**, avec compatibilité sans SDK, ancien état sérialisé, threads/async/erreurs, typage de validité, consommation budgétaire et tests sans activation. |
| **Ne pas promouvoir comme résultat gagné** | Curriculum, mémoire/Pareto, profondeur O2/O3, transfert universel, augmentation générale du plafond | Les expériences ne les établissent pas. Une absence de preuve n’est pas une preuve d’inutilité. |

### Patch minimal concret

Le [patch proposé](../audit_20260924/minimal_commit_proposal.patch) contient seulement :

1. **`opto/features/recursive_opt/spec.py`** : 8 lignes ajoutées /4 retirées ; autoriser uniquement `request_params.extra_body.provider={"sort": "price"|"latency"|"throughput"}` pour un profil OpenRouter ; les contrôles d’identité modèle/fournisseur/credentials/base URL restent refusés.
2. **`tests/unit_tests/test_recursive_control_plane_v2.py`** : 14 cas paramétrés dans le fichier existant ; valeurs permises, forme invalide, fournisseur incorrect, clés supplémentaires, conservation du modèle et effet sur fingerprint.
3. **`artifacts/control_plane_v2/prompt18_readiness.json`** : empreinte exacte du lot isolé et note de vérification ; `ready_for_prompt_18=false` et absence de CI couvrant cet arbre restent explicites.

Ce correctif ne change pas les paramètres par défaut d’un appel existant. Lorsqu’un utilisateur demande un tri, la politique de routage peut varier : c’est une option explicite et fingerprintée. Cela ne démontre ni accélération ni meilleure reproductibilité du fournisseur.

**Conflit de portée identifié avant commit :** un commit strictement limité à `features` ne peut pas embarquer le test de cette option ni actualiser l’empreinte exigée. Le test `test_35_source_provenance` échoue effectivement avec le code seul : ancien hash `656521f3…aa10`, nouveau `ceed8a309…80c5`. Le test n’a été ni désactivé ni affaibli. Le patch cohérent de trois fichiers est donc soumis comme exception minimale de portée ; sinon il doit rester un patch sans commit.

**Statut du commit : en attente de la réponse à cette exception de périmètre.** Aucun commit du cœur, aucun `git add .`, aucun amend/rebase/cherry-pick en masse. Les fichiers et portions déjà indexés par l’utilisateur doivent être préservés, y compris leurs différences avec l’arbre de travail.

### Ordre conseillé pour les lots suivants — non exécutés ici

1. Traiter séparément les correctifs de wrappers classiques/logging et le reporting three-way, avec les tests existants pertinents.
2. Faire du curriculum un lot cohérent **loader + sampler + search + tests de restauration/validité/budgets**. Corriger le défaut reproduit avant commit. Ne pas publier seulement son entrée de config.
3. Faire de la capture un lot **backend IO + bundle + traces + spec + tests**, sans rendre le SDK obligatoire pour le chemin interne. Tester explicitement les échecs de seconde ouverture, conversions et fermeture.
4. Intégrer le mode de sélection sur batch courant séparément ; contrôler le comportement `history` inchangé et refuser les configurations non prises en charge. Ne pas imposer globalement l’abandon de VALIDATION.
5. Actualiser la provenance correspondante et exécuter le workflow requis sur le commit exact avant de remettre readiness à vrai.

## 9. Vérifications exécutées et limites

Environnement de tests : `/home/xav/miniconda3/envs/humanllm/bin/python` (Python 3.12). Secrets fournisseur retirés de l’environnement des commandes ; sockets réseau bloquées dans pytest. Les sockets Unix/locales nécessaires au notebook/async sont autorisées explicitement lorsque requis. Aucun fichier `.env` ou secret n’est lu par cet audit.

| Vérification | Résultat | Portée |
|---|---|---|
| Arbre local complet, `tests/unit_tests` | **959 passed, 2 skipped**, 189,21 s ; 1 avertissement LangGraph | Inclut les travaux locaux du cœur et tests non suivis ; ne certifie pas leur disponibilité dans un clone du commit actuel. [Log](../audit_20260924/full-unit.log). |
| Copie des deux fichiers `features` seuls sur HEAD propre | **1 passed, 5 failed**, 24 deselected | Échec OTEL/sysmon/hybrid faute de IO ; échec curriculum. Pas un échec de l’arbre local complet. [Log corrigé pour sockets Unix](../audit_20260924/features-only-unix.log). |
| Petit correctif de routage seul, avant actualisation du hash | **14 passed, 1 failed**, 54 deselected | Cas fonctionnels verts, gate provenance rouge. [Log](../audit_20260924/narrow-before-provenance.log). |
| Patch cohérent minimal dans checkout isolé, suite unitaire | **787 passed, 4 skipped**, 63,86 s | 773 tests de base réussis +14 nouveaux ; skips Graphviz, deux backends absents, ancien exemple C nécessitant appel live. [Log](../audit_20260924/narrow-full-unit.log). |
| Control plane/hardening/transport + tests Experiment-0 + tests EXP18 hors dossier unitaire | **224 passed, 2 failed**, 52,14 s | Deux échecs historiques Experiment-0 ; aucun autre échec. Les tests sous `artifacts/.../test_*.py` ne sont pas inclus dans `tests/unit_tests`. [Log](../audit_20260924/contracts-and-exp18.log). |
| Reproduction des deux échecs sur HEAD propre, puis patch minimal | **2 failed dans chaque checkout**, 5,16 /4,80 s | Préexistants au patch de routage, aucun test affaibli. [Base](../audit_20260924/experiment0-baseline.log), [patch](../audit_20260924/experiment0-minimal-patch.log). |
| Recalcul numérique EXP15/16/18 | **PASS**, 56 448 valeurs et 1 764 trajectoires | Toutes les observations des audits finaux, aucune génération ou exécution de candidat. |
| Recalcul brut EXP20 | **PASS**, 7 128 lignes externes et 72 réponses | Conforme aux résultats conservés ; aucune facture externe vérifiée. |
| Lint et whitespace du patch isolé | **Ruff PASS**, `git diff --check` PASS | Une reformatisation Black sans rapport a été retirée ; AST identique attesté avant/après, seules les nouvelles lignes de tests sont conservées. |

Commandes exactes principales, depuis `/home/xav/code/Trace` (le checkout du patch est `/tmp/trace-recursive-audit-20260924/isolated`) :

```bash
git log -10 --oneline -- opto/features/recursive_opt
git diff --stat recursive_opt HEAD -- opto examples tests setup.py pyproject.toml
git diff HEAD -- opto/trace/bundle.py opto/trainer
git diff --cached --stat
git -C /home/xav/code/Trace-experiment0 status --short

env -u OPENAI_API_KEY -u OPENROUTER_API_KEY -u ANTHROPIC_API_KEY \
  -u GOOGLE_API_KEY -u TAVILY_API_KEY RECURSIVE_OPT_LIVE=0 \
  PYTHONHASHSEED=0 PYTHONPATH=. \
  /home/xav/miniconda3/envs/humanllm/bin/python -m pytest -q \
  --disable-socket --allow-hosts=127.0.0.1,localhost tests/unit_tests

env -u OPENAI_API_KEY -u OPENROUTER_API_KEY -u ANTHROPIC_API_KEY \
  RECURSIVE_OPT_LIVE=0 PYTHONPATH=. \
  /home/xav/miniconda3/envs/humanllm/bin/python \
  artifacts/recursive_opt_audit_20260924/verify.py \
  > artifacts/recursive_opt_audit_20260924/verification.json

# Dans le checkout isolé du patch minimal :
env -u OPENAI_API_KEY -u OPENROUTER_API_KEY -u ANTHROPIC_API_KEY \
  -u GOOGLE_API_KEY RECURSIVE_OPT_LIVE=0 PYTHONHASHSEED=0 PYTHONPATH=. \
  /home/xav/miniconda3/envs/humanllm/bin/python -m pytest -q -rs \
  --disable-socket --allow-unix-socket \
  --allow-hosts=127.0.0.1,localhost tests/unit_tests

/home/xav/miniconda3/envs/humanllm/bin/ruff check --output-format concise \
  opto/features/recursive_opt/spec.py tests/unit_tests/test_recursive_control_plane_v2.py
git diff --check
```

Commandes complémentaires exécutées avec le même environnement fournisseur désactivé :

```bash
/home/xav/miniconda3/envs/humanllm/bin/python -m pytest -q -rs \
  --disable-socket --allow-unix-socket --allow-hosts=127.0.0.1,localhost \
  tests/unit_tests/test_recursive_control_plane_v2.py \
  tests/unit_tests/test_recursive_final_hardening.py \
  tests/unit_tests/test_recursive_transport.py \
  experiments/recursive_opt/multiobjective_reasoning/tests \
  artifacts/optimizer_discovery/exp18/test_driver.py \
  artifacts/optimizer_discovery/exp18/test_memory_projection.py \
  artifacts/optimizer_discovery/exp18/test_pareto_selection.py \
  artifacts/optimizer_discovery/exp18/test_pareto_trainer.py \
  artifacts/optimizer_discovery/exp18/test_study.py

# Deux échecs reproduits séparément dans baseline/ puis isolated/ :
/home/xav/miniconda3/envs/humanllm/bin/python -m pytest -q \
  --disable-socket --allow-unix-socket \
  experiments/recursive_opt/multiobjective_reasoning/tests/test_main_experiment.py::test_invalid_completed_core_result_resumes_without_provider_calls \
  experiments/recursive_opt/multiobjective_reasoning/tests/test_main_experiment.py::test_frozen_protocol_hashes_profiles_and_constraints_are_unchanged

# Cas fonctionnels du routage et test de provenance, avant mise à jour du hash :
/home/xav/miniconda3/envs/humanllm/bin/python -m pytest -q \
  --disable-socket --allow-unix-socket \
  tests/unit_tests/test_recursive_control_plane_v2.py \
  -k 'openrouter_sort or test_35_source_provenance'

# Test de dépendances, avec uniquement spec.py et traces.py locaux copiés sur HEAD :
/home/xav/miniconda3/envs/humanllm/bin/python -m pytest -q \
  --disable-socket --allow-unix-socket \
  tests/unit_tests/test_o1_trace_curriculum.py \
  -k 'real_capture or requested_capture or real_trainer_receives_curriculum'
```

Le premier essai de ce dernier test bloquait aussi les sockets Unix d’asyncio ; il a été refait avec cette permission locale, sans modifier le code. Les **5 échecs** persistants du second essai sont ceux retenus pour l’analyse des dépendances.

Le recalcul EXP20 utilise son script existant avec `pytest_socket.disable_socket()` avant `runpy.run_path(..., run_name="__main__")` ; son stdout a été archivé, sans réécrire le résultat scientifique. Les contrôles ne constituent pas un audit exhaustif de sécurité du code généré, de tous les SDK facultatifs, ni de toutes les dépendances externes.

## 10. Décision scientifique et prochaine preuve utile

Les données autorisent à affirmer : **« le projet sait mieux définir, exécuter et auditer une recherche ; certaines interventions locales améliorent des scores ou des coûts »**. Elles n’autorisent pas : « la récursion apporte un surplus général sur une bonne recherche standard à budget total comparable ».

Pour les gains numériques, A0 est trop faible pour être le seul comparateur ; B2 est indispensable. Pour HotpotQA, le contrôle fixe Qwen à 10 documents et une instruction figée manque avant d’attribuer le gain à la réécriture de code. Pour EXP19, il faut comparer des parents avec scores séparés, lots communs et règles d’actualisation identiques avant d’attribuer l’effet à la validation elle-même. Pour la profondeur, il faut des exécutions O1/O2 effectives et compter **tout le coût amont** ; les labels « recursive » ne suffisent pas.

Une suite raisonnable ne consiste donc pas à relancer tous les anciens UC. Elle consiste à choisir un usage produit, un contrôle simple fort, une seule intervention causalement active et des données nouvelles ; figer le budget et le critère principal, puis publier tous les résultats, y compris les non-atteintes et échecs. **Cette étude ne lance aucune de ces campagnes.**

Volume physique constaté, hors `__pycache__` : `control_plane_v2` **34 fichiers /0,25 MB**, `probe_2026` **311 /2,95 MB**, `optimizer_discovery` **490 223 /2,586 GB**, `o1_learning` **85 402 /0,799 GB** (unités décimales, tailles logiques ; archives incluses). Ce volume documente le coût de conservation, pas la force statistique : les résultats centraux restent souvent à cinq ou six graines, et EXP17 reste incomplète.
