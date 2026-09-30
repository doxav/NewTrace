> Historical document, retired from active navigation on 2026-09-29. Original location: `artifacts/optimizer_discovery/exp18/DOCUMENT_UPDATE_MAP.md`. Use the [canonical experiment index](../../README.md) and [current assessment](../../ASSESSMENT.md). The [original bytes](EXP18_DOCUMENT_UPDATE_MAP.md.original.gz) are preserved; relative links below were rebased for this location.

> **Archive de planification documentaire du 10 septembre, remplacée pour la navigation.**
> État courant : [bilan et priorités](RESEARCH_LOG.md),
> [index des fichiers](RESULTS_INDEX.md), [rapport EXP-18 achevé](../../_shared/optimizer_discovery/exp18/REPORT.md).
> EXP-17 est suspendue avant audit depuis le 13 septembre. Les instructions et
> anciens numéros de sections ci-dessous ne constituent plus le plan de travail.

# EXP17/EXP18 — carte historique des mises à jour après résultats

Revue documentaire du 2026-09-10. Ce fichier prépare la rédaction finale ; il ne
rapporte aucun résultat d'efficacité EXP17/EXP18. Seuls les protocoles nouveaux,
les rapports historiques et leurs preuves sauvegardées ont été lus. Aucun modèle,
programme candidat ou objectif n'a été exécuté pour cette revue. Les fichiers
historiques, les sources gelées et leurs rétractions restent inchangés.

## 1. Ordre de mise à jour et sources faisant autorité

Les protocoles [EXP17](../../_shared/optimizer_discovery/exp17/PREREG_EXP17.md) et
[EXP18](../../_shared/optimizer_discovery/exp18/PREREG_EXP18.md) décrivent deux questions distinctes. Au moment de cette
revue, EXP17 dispose d'un protocole confirmatoire final ; EXP18 conserve encore
l'intitulé de protocole pilote et de plan exploratoire provisoire. Toute description
finale doit être dérivée de leurs gels effectifs, sans traiter cette carte comme
une nouvelle préinscription.

Après chaque étude, la rédaction doit attendre les barrières de génération et de
sélection complètes, l'audit terminé, les vérifications numériques/budgétaires et
le recalcul de l'analyse. Utiliser les `freeze.json`, `freeze_sha256.json`, reçus
de réponses, `selections_frozen.json`, `audit_results.json` et
`analysis_results.json` de **cette étude**. Une extension `.json.gz` conserve les
mêmes données logiques, mais une représentation physique unique doit être retenue.

Les rapports doivent présenter toutes les seeds, tous les slots, les échecs et le
repli. Une étude interrompue ou dont les sémantiques ont été invalidées ne devient
pas un résultat confirmatoire complet. Une source générée invalide, gérée selon
le protocole, reste une observation du procédé de recherche.

Ordre pratique : rapport propre à EXP17 ou EXP18 → inspection descriptive et
brief propres à cette étude → nouvelles sections de l'assessment → tableau de
statut du research log. Les sources exportées restent exactement celles évaluées,
avec hashes et parentés ; le représentant vient de la validation. Placer les
exports dans un répertoire résolu disjoint du répertoire scientifique d'entrée,
jamais dans son cache. L'[adaptateur d'inspection](../../_shared/optimizer_discovery/exp17/program_inspection.py)
impose ces conditions et n'exécute aucun candidat.

## 2. Research log : modifications précises

Référence : [artifacts/RESEARCH_LOG.md](RESEARCH_LOG.md).
Les numéros proposés ci-dessous supposent que §12 reste la dernière section au
moment de l'intégration ; vérifier les ajouts concurrents avant de numéroter.

| Emplacement actuel | Mise à jour après résultats | Éléments à préserver |
| --- | --- | --- |
| Date « Last updated » et §1, tableau des hypothèses | Ajouter **H17** avec le seul contraste confirmatoire C−I et sa règle enregistrée. Ajouter des entrées EXP18 explicitement exploratoires pour mémoire, sélection de parent et interaction, en conservant les identifiants du protocole final plutôt qu'en en inventant après les résultats. | H15-A et H15-B restent les conclusions propres à EXP15. Ne pas transformer son H15-B inconclusif en positif ou négatif rétroactivement. |
| §1, paragraphes après H15-A/B | Ajouter une synthèse datée EXP17/18 avec effet, intervalle et périmètre. Qualifier C comme réécriture d'un parent choisi sur TRAIN, sans texte de performance explicite. | Le résultat négatif de R−I dans EXP16 et son statut exploratoire. Les anciens H1/H2/H3/H5 gardent leur surface et leur contrôle historiques. |
| §2, G-A et G-C | Remplacer « nouvelle hypothèse à confirmer » par le résultat de H17, puis préciser séparément ce qu'EXP18 informe. | Une amélioration face à A0 ne suffit pas à établir l'intérêt face à I ou B2. Un résultat nul ou négatif constitue une réponse valable. |
| §2, G-B | Ajouter le résultat des contrôles de provenance, de budgets et de séparation des splits si les vérifications passent. | API isolée, environnement assaini et timeout ne prouvent pas un sandbox de sécurité du système d'exploitation. |
| §3, lignes EXP17 et EXP18 déjà présentes | Remplacer les statuts d'exécution par configuration finale, effectif enregistré/représenté, toutes les réponses prévues/terminées, effets et liens vers les rapports. EXP17 : 46 paires, N8, 736 réponses selon son gel actuel. EXP18 : six paires, N16, quatre bras, 384 réponses selon son plan actuel, à confirmer contre le gel final. | Ne pas ajouter une deuxième ligne au même ID ; séparer les pilotes. Ne pas exclure des seeds parce que le seed initial est sélectionné ou qu'une recherche n'a aucun remplacement éligible. |
| §3, « Retractions » et §4, défauts | Ajouter uniquement les défauts réellement découverts et leurs conséquences documentées. Distinguer réparation d'analyse/export et modification des sémantiques de génération/évaluation/sélection. | UC4, Probe F/K, Iteration 3 et le contrôle de types vacueux restent rétractés. Les défauts de l'exporteur corrigés avant tout export réel n'invalident pas une valeur scientifique. |
| §5, registre des instruments | Ajouter une ligne pour le benchmark numérique portable avec familles/dimensions, panels et garantie exacte de replay. | Les 43/47 tâches à une seule entrée concernent l'ancien pool ; elles ne décrivent pas les panels numériques actuels. Replay déterministe ne signifie pas génération LLM déterministe. |
| §8, « Next, by priority » | Ajouter les prochaines questions issues des nouvelles études ; dater et restreindre l'ancienne priorité EXP12/13 aux travaux prose concernés. Le texte « gates every remaining decision » est devenu trop général. Marquer le lieu portable comme établi. | Ne pas prétendre qu'EXP17/18 termine EXP12/13, teste H6/W1 ou annule les problèmes des anciens backlogs. |
| Après §12 | Ajouter **§13 EXP17** puis **§14 EXP18**, avec liens vers assessment §27/§28 et les preuves exactes. | Garder §11 EXP15 et §12 EXP16 comme comptes rendus de leurs expériences ; leurs prescriptions étaient prospectives à leur date. |

Le paragraphe EXP17 doit donner C−I avant les comparaisons secondaires C/I−A0/B2.
Le paragraphe EXP18 doit présenter les quatre contrastes simples, les deux effets
moyens et l'interaction, avec invalidité, repli et exposition réelle aux mécanismes.
Ne pas remplacer ces contrastes par un classement des moyennes des quatre bras.

## 3. Assessment : appendices nouveaux, histoire conservée

Référence : [recursive_opt_assessment.md](../../ASSESSMENT.md).

| Emplacement | Modification recommandée |
| --- | --- |
| Préambule, plage « §0–§22 », bloc « CURRENT », table « How to read this » | Actualiser la navigation vers les sections effectivement présentes et vers le statut courant du ledger. Dater explicitement le verdict initial du 29/30 août. Les expressions historiques « No. Not once » ou « no experiment … since the migration » ne doivent pas apparaître comme des résumés actuels. Ajouter une note de continuation plutôt que réécrire leurs preuves anciennes. |
| §0, verdict initial | Conserver son texte comme diagnostic historique ; renvoyer au ledger et aux nouvelles §27/§28 pour l'état présent. Une amélioration nouvelle n'efface pas le problème UC4 démontré dans §5.2. |
| §25, EXP15 | Aucun remplacement de métrique, statut H15, code ou intervalle. Au besoin, un court lien daté vers les études ultérieures. |
| §26.1–§26.3, diagnostics/P1/intégrité | Conserver les nombres et limites. Aucun pooling EXP16+EXP17 ni EXP16+EXP18. Ne pas relire l'absence de mémoire P1 comme la preuve causale de sa perte. |
| §26.4, « Decision after the negative feedback result » | Conserver la recommandation historique ; ajouter uniquement un renvoi daté indiquant où sa confirmation C−I et le diagnostic des mécanismes sont maintenant rapportés. |
| Nouvelle §27, EXP17 | Décrire question confirmatoire, 46 unités externes et nouveau panel, C/I/A0/B2, gel et amendements, C−I avec intervalle/règle, comparaisons secondaires, coûts réels, validité/repli, sources sélectionnées, limitations. Le seuil de planification 0,02 reste un objectif utile choisi avant les résultats, pas un gain prédit. |
| Nouvelle §28, EXP18 | Décrire le plan factoriel L/M/P/PM, toute modification avant le gel, ses sept contrastes exploratoires, l'exposition à la mémoire et aux fronts, coûts/validité/repli, programmes et parentés, puis une décision de prochaine expérience. Ne pas convertir les six seeds en centaines de réplications via les tâches ou points. |

L'ancien §3.1 sur `memory_policy` inactif et §21 sur les exemples uniques restent
des diagnostics de leurs adaptateurs. EXP18 ajoute explicitement du contexte au
propriétaire de génération et remplace étroitement le choix de parent : ce n'est
pas une activation rétroactive de ces anciens knobs. De même, les profondeurs
des chaînes de programmes ne sont pas des niveaux imbriqués de méta-optimisation.

## 4. Documents EXP16 : préserver, puis relier

Ces documents appartiennent à une investigation terminée. Les nouveaux résultats
doivent figurer dans leurs propres rapports. Si la navigation historique est
actualisée, ajouter un encadré daté clairement séparé, sans retoucher le contenu
gelé ou la conclusion ancienne.

| Document et section | Traitement lors de la continuation |
| --- | --- |
| [REPORT.md](../../_shared/optimizer_discovery/investigation16/REPORT.md), « Recommandation et gains projetés » | La confirmation proposée y est dite « non lancée ». Conserver ce fait au moment du rapport et ajouter un lien de continuation EXP17. Lier EXP18 pour les hypothèses nouvelles de mémoire/sélection, sans réécrire R comme gagnant. |
| [DECISION_MATRIX.md](../../_shared/optimizer_discovery/investigation16/DECISION_MATRIX.md), lignes mémoire de tentatives, erreurs rejetées, recherche étroite, noms de stratégies | Garder la colonne de preuve EXP16. Créer dans le nouveau rapport une matrice de continuation liée aux contrastes EXP18 ; ne pas fusionner un effet observé avec sa motivation rétrospective. |
| [FUTURE_DESIGN.md](../../_shared/optimizer_discovery/investigation16/production/FUTURE_DESIGN.md), « Bras à conserver… » et dimensionnement | Conserver C−I post hoc et les scénarios 46/182 paires comme calculs antérieurs. Un renvoi peut expliquer que 46 est devenu l'effectif EXP17 ; sa justification ne doit pas être recalculée en fonction du résultat obtenu. |
| [REJECTION_MEMORY_DESIGN.md](../../_shared/optimizer_discovery/investigation16/history/REJECTION_MEMORY_DESIGN.md), « Contraste causal… » | Conserver comme proposition antérieure. Le rapport EXP18 doit expliciter l'écart : résumé courant compact commun, quatre bras mécanistiques et aucun indépendant N16, au lieu d'affirmer avoir exécuté exactement le I/C/R proposé ici. |
| [PROGRAM_INSPECTION.md](../../_shared/optimizer_discovery/investigation16/production/PROGRAM_INSPECTION.md) | Préserver R/16411, son hash, les six programmes et ses limites de parenté. Les représentants EXP17/18 seront choisis dans leurs propres validations, jamais parmi les vainqueurs de l'audit historique. |
| [PATRICK_BRIEF.md](../../_shared/optimizer_discovery/investigation16/PATRICK_BRIEF.md) et le brief initial EXP15 | Préserver comme briefs de leurs dates ; créer un nouveau brief pour les nouvelles études. Aucun envoi n'est autorisé par cette tâche documentaire. |
| Protocoles, gels, données JSON/gzip, archives source et calculs historiques | Aucune modification. Ils restent les preuves des anciennes affirmations et des scénarios de planification. |

## 5. Motivations vérifiées contre les preuves persistées

Cette revue a recalculé des statistiques **sur les fichiers existants**, sans
nouvelle génération ni réévaluation. Les neuf hashes d'entrées consignés dans
[future_design_calculations.json](../../_shared/optimizer_discovery/investigation16/production/future_design_calculations.json)
correspondent encore à leurs fichiers physiques.

| Proposition | Vérification locale | Limite causale à conserver |
| --- | --- | --- |
| C mérite une confirmation indépendante | Les six valeurs dans [l'analyse P1](../../EXP16/results/production_run/analysis_results.json.gz) donnent C−I moyen **−0,029671523774**, SD échantillonnal **0,048136641921** ; quatre deltas sont favorables, deux défavorables. C−I est absent de ses contrastes enregistrés. | C−I était post hoc sur un seul panel fixe. Cette sélection de la question justifie de nouvelles seeds/instances et interdit de traiter le P1 favorable comme une confirmation supplémentaire. |
| Le texte riche ne bénéficie pas automatiquement à la recherche | La même analyse conserve R−I **+0,045257**, intervalle **[+0,003176 ; +0,085126]**, et R−C **+0,074929**, **[+0,034075 ; +0,116730]**. Le signe positif est défavorable à R. | R−C teste l'ajout de ce feedback dans une recherche qui évolue ensuite différemment. Il ne sépare pas ancrage, volume, représentation, réactions du modèle ou routage ; il ne démontre pas l'inutilité de tout feedback. |
| C exploitait une information TRAIN implicite | [L'inspection sauvegardée](../../_shared/optimizer_discovery/investigation16/production/program_inspection.json) compte 48 enveloppes C avec `visible_to_llm=false`, contre 48 visibles pour R et 48 pour W ; chacune décrit 48 trajectoires du parent. Les parents sont choisis par la recherche sur TRAIN. | C n'est pas un deuxième indépendant ni un bras « sans feedback » au sens large. EXP17 teste cette procédure de sélection et réécriture. |
| Un rejet pouvait disparaître du prochain contexte | Les [requêtes E1 slot00](../../_shared/optimizer_discovery/investigation16/production_engineering/raw/16601/R/slot_00/request.json) et [slot01](../../_shared/optimizer_discovery/investigation16/production_engineering/raw/16601/R/slot_01/request.json) ont exactement les mêmes messages et le même hash de parent. L'[E1](../../_shared/optimizer_discovery/investigation16/production/ENGINEERING_REPORT.md) conserve le premier échec syntaxique et une seule source éligible sur deux. | Cela démontre l'absence de cet exemple d'erreur dans le contexte suivant, pas le bénéfice de l'ajouter. Une réponse valide ensuite ne prouve pas une réparation informée par le rejet. |
| La mémoire des tentatives était absente de P1 | [L'inspection P1](../../_shared/optimizer_discovery/investigation16/production/PROGRAM_INSPECTION.md) documente un schéma limité au parent courant et à l'AUC TRAIN agrégé, sans codes/statuts alignés des précédents essais. | Une archive interne utilisée pour choisir des parents n'est pas une archive de plusieurs exemples exposée au modèle. Augmenter sa taille seul ne garantit aucun nouveau contenu du prompt. |
| W ne testait pas Pareto par instance | Les six recherches W sauvegardées ont des nombres de parents distincts par ronde **[1,2,2,2]**. Leur sélection est scalaire sur l'AUC TRAIN. W contre R vaut **−0,009935**, intervalle **[−0,054857 ; +0,030305]**. | La largeur était réellement exercée, mais deux parents/quatre rondes contre un parent/huit rondes mêle largeur et nombre d'updates. Ce résultat ne valide ni ne réfute la sélection par front d'instances. |
| Un label de trainer ne suffit pas | [trainer_resolution.json](../../_shared/optimizer_discovery/investigation16/history/raw/trainer_resolution.json) montre `MinibatchAlgorithm`, `BeamsearchAlgorithm` et `UCBSearchAlgorithm` résolus tous en `ParetobasedPS`. L'[adaptateur EXP18](../../_shared/optimizer_discovery/exp18/ADAPTER.md) documente pourquoi le parent unique de PrioritySearch ignore un simple changement de mode Pareto. | Le choix EXP18 inclut nondominance **et** tirage uniforme des membres du front. Les tests de sélection d'un spécialiste prouvent l'exécution de ce mécanisme, pas un gain sur la tâche. |
| Les anciennes accélérations ont un contrôle important | [routing_summary.json](../../_shared/optimizer_discovery/investigation16/history/raw/replay_01/routing_summary.json) conserve écart historique maximal **0**, étendue de replay **0**, K*=**2,25** à Q=1, et `nearest` fixé atteignant la même qualité avec **0 coût méta**. | Gain de recherche contre un ordre uniforme non informé dans un menu manuscrit connu ; aucune nouvelle preuve d'amortissement de synthèse LLM ou de Pareto. |
| Du code indépendant pouvait déjà améliorer une tâche | [transfer_summary.json](../../_shared/optimizer_discovery/investigation16/history/raw/replay_01/transfer_summary.json) conserve 11/11 sources exécutables sur leur cible et 0/22 transferts valides. Cinq des six sources VRPTW conservées battent `nearest` ; le meilleur index7 réduit la distance de **26,476508 à 20,702124**, soit **21,8095 %**. | Sources issues des 12 réponses VRPTW originales ; replay sur leurs fixtures historiques, sans nouvelle preuve hors échantillon. Les échecs de signature ne sont pas des mauvais regrets. Ni gain récursif ni nouveauté établis. |

Les anciens diagnostics de sélection S1 et de débit T1 restent également distincts :
**−17,3 %** d'AUC de la politique sélectionnée dans une banque fixe, et débit local
**×5,49** à huit workers, respectivement. Le contrôle B2 montre une faiblesse du
premier point du seed, mais pas une explication isolée de R−I. Ces gains ne
s'additionnent pas en une prévision de gain récursif. Voir
[assessment §26.1–26.3](../../ASSESSMENT.md#261-what-the-staged-diagnostics-isolate).

Empreintes physiques vérifiées lors de cette revue :

| Fichier historique | SHA256 |
| --- | --- |
| `production_run/analysis_results.json.gz` | `cfb3a7e123b27bf86affc46aac83e7d98c530c92f9e6c4532dd96cdc5206ba1f` |
| `production/program_inspection.json` | `f716ecf0491bb33c19e6ccb6660e79f46b8fd6cadcf9242e572ad7d63143ec00` |
| `history/raw/replay_01/routing_summary.json` | `74f8c4cd0d8cd1b346286d6e69df0022d5159f6e27758f69deb76836764f9a62` |
| `history/raw/replay_01/transfer_summary.json` | `9d2c9abb0c80348b9714baaee8aa5216fb3341fdb18ccdcdb738529699f824d4` |
| `history/raw/trainer_resolution.json` | `bf5862c2caac5dea7ddee05a0d749c4616555efa906b5f8ecf3d22d59fae12bd` |

## 6. Grille d'interprétation à appliquer aux nouveaux résultats

Pour **EXP17**, suivre exactement la règle du gel : intervalle bootstrap C−I
entièrement négatif → signal positif ; entièrement positif → signal négatif ;
tous les deltas nuls → aucune différence détectable ; sinon → inconclusif.
Présenter aussi la taille de l'effet face au seuil de planification 0,02 et les
contrôles A0/B2, sans changer la règle après observation. Une victoire de C
soutiendrait la sélection TRAIN suivie d'une réécriture au budget enregistré,
pas le texte riche de R, la profondeur récursive ou l'amortissement.

Pour **EXP18**, les contrastes enregistrés sont M−L, P−L, PM−P et PM−M ; les
effets moyens sont `((M−L)+(PM−P))/2` pour la mémoire et
`((P−L)+(PM−M))/2` pour la sélection ; l'interaction est `PM−P−M+L`.
Tous restent exploratoires à six seeds, sans garantie de multiplicité. Chaque
comparaison mémoire porte sur l'ensemble code/statut/résultat antérieur ajouté
au même résumé courant compact. Chaque comparaison P porte sur la politique de
front et d'échantillonnage complète. Une interaction éventuelle ne peut être
attribuée à un champ isolé du prompt.

Le rapport doit compter les sources antérieures distinctes réellement montrées,
les omissions, les slots invalides/sans code et les requêtes recevant au moins
six ou sept exemples complets. Avec N8, un seul appel peut suivre sept réponses
antérieures ; avec N16, neuf appels le peuvent, avant déduplication et exclusion
du parent déjà affiché. Ces disponibilités ne démontrent aucun seuil d'apprentissage.
Rapporter également taille des fronts, choix différents du meilleur scalaire,
parents effectivement reçus par le callback et décisions terminales inutilisées.
Un front singleton n'exerce pas une diversité de choix ; un grand front peut
rapprocher la politique d'une exploration uniforme de l'archive.

Il n'y a **pas de bras indépendant N16 dans EXP18**, ni de R riche dans ses
quatre bras. EXP18 ne peut donc établir un avantage de la mémoire sur indépendant
à N16, ni isoler l'effet « compact contre riche ». Comparer PM/N16 à I/N8
d'EXP17 ajouterait un changement de budget, de panel et de seeds ; cela ne
constituerait pas le contraste causal manquant. Enfin, des gains ou pertes dans
ces deux études ne classent pas FunSearch/OpenEvolve, d'autres modèles, des
familles nouvelles, d'autres dimensions ou horizons.

Les anciennes rétractions restent visibles quelle que soit l'issue. La conclusion
doit distinguer ce qui est observé, ce que le mécanisme effectivement testé permet
d'attribuer, et la prochaine hypothèse à enregistrer. Aucun gain numérique futur
ne doit être promis sur la seule base d'une moyenne favorable.
