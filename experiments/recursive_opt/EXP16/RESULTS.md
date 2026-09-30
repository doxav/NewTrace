# EXP-16 — enquête sur les conditions d’un gain du feedback

**Les diagnostics et P1 sont exécutés. Le feedback détaillé corrigé ne montre pas
le gain recherché : R−I = +0,045257 de regret-AUC, intervalle bootstrap exploratoire
[+0,003176 ; +0,085126], donc défavorable à R.** L'enquête établit des défauts
de l'ancien instrument et des améliorations utiles de sélection, de débit et de
seed. Elle ne permet pas de promettre un gain du feedback. Les pièces de clôture
et l'audit indépendant final en précisent les limites.

Base conservée : EXP-15, commit `13ebda2242e1c18022591737b113030ca2ce2da2`.
Branche de travail : `codex/investigation-feedback-exp16`. Les réponses, programmes
et conclusions d’EXP-15 sont conservés ; cette enquête est un nouvel EXP-16
exploratoire. Aucun contact externe, push, merge ou PR.

## Ce qui doit être expliqué

EXP-15 donne les AUC suivantes : A0 = 0,139579, A1 = 0,121748 et A2 = 0,116680.
A2−A0 vaut −0,022899,
avec l’intervalle bootstrap enregistré entièrement négatif : signal positif dans
ce protocole. Le contraste central A2−A1 vaut seulement −0,005068, avec un intervalle
[−0,037728 ; +0,024551] : l’avantage ajouté du feedback n’est pas établi.

Ce n’est pas une démonstration d’absence d’effet. Cinq réplications donnent une
estimation fragile : retirer descriptivement la seed 53 inverse le signe moyen en
faveur d’A1. Tous les résultats restent dans l’analyse principale. Il faut séparer
les défauts certains de l’instrument des hypothèses sur leur effet scientifique.

## Cinq quantités différentes derrière « davantage d’exemples ou de batch »

| Quantité | EXP-15 | Ce qu’elle contrôle |
|---|---:|---|
| Observations dans une trajectoire | B = 32 | Information disponible pour optimiser une instance |
| Panel d’apprentissage | 6 instances × 1 seed local | Précision et diversité de l’évaluation d’un programme |
| Propositions de programmes | N = 8 par bras/seed externe | Opportunités de recherche, invalides comprises |
| Tentatives antérieures montrées au modèle | Parent courant et résumé de la tentative précédente, sans son code distinct | Mémoire explicite des succès et échecs de la recherche |
| Réplications externes | 5 | Incertitude du contraste entre procédures de recherche |

La ligne externe du dataset Trace représente un **panel de six vraies tâches**,
pas une seule tâche numérique. Augmenter un paramètre de batch du trainer ne doit
pas être assimilé automatiquement à ajouter des instances ou des tirages locaux.
Il faut vérifier les données effectivement évaluées et les appels réellement faits.
Il n’existe pas ici de seuil démontré « six ou sept exemples suffisent ».

Si « six ou sept exemples » désigne six ou sept programmes antérieurs avec leur
code et leur résultat, ce mécanisme n’était pas présent. L’archive de recherche
conserve des candidats, mais cela ne signifie pas que tous entrent dans le prompt.
Dans P1, R/W reçoivent la trace détaillée du parent courant ; une mémoire explicite
de plusieurs tentatives rejetées n’est pas testée. Un futur contraste devrait
aligner chaque code, son hash, ses résultats et son statut, avec une règle fixe
de sélection des exemples. Son bénéfice ne peut pas être déduit du seul fait
d’augmenter le panel d’évaluation de six à vingt-quatre instances.

La [revue d’une mémoire explicite des rejets](../_shared/optimizer_discovery/investigation16/history/REJECTION_MEMORY_DESIGN.md)
identifie une extension limitée au propriétaire déjà appelé par Trace : projeter
les seuls slots antérieurs du même bras, avec code, hash, statut et résultats
TRAIN déjà disponibles, sans réévaluation ni résumé LLM supplémentaire. Il faut
la distinguer du feedback réellement propagé et figer son instantané pour la reprise.
Avec N = 8, sept réponses antérieures ne sont disponibles que pour l’appel 8 ;
N = 16 offrirait neuf appels avec cette disponibilité maximale. Déduplication,
absence de code et limites de contexte la réduisent. Ce scénario reste à tester,
avec le même N pour les comparateurs ; aucun seuil d’apprentissage n’est établi.

Il faut également distinguer trois sens de « vitesse » :

- **Vitesse de la politique déployée** : qualité atteinte après un nombre donné
  d’évaluations de l’objectif ; AUC anytime et temps d’atteinte en évaluations.
- **Débit de l’implémentation** : durée nécessaire pour exécuter les mêmes
  trajectoires ; T1 mesure ce point en conservant exactement leurs résultats.
- **Efficacité de la découverte** : qualité du programme obtenu par réponse du
  modèle, token, coût ou temps de recherche ; les budgets alloués et les dépenses
  réelles doivent être rapportés séparément.

Une accélération des sous-processus n’établit pas une découverte plus efficace.
De même, un gain de la politique déployée après une recherche coûteuse ne suffit
pas à établir l’amortissement de cette recherche sur de futurs usages.

Elle rend néanmoins plus abordable le panel plus fiable : un candidat EXP-15
avait 6 trajectoires de train et 6 de validation ; P1 en alloue 48 et 24, soit
six fois plus au total. Le débit mesuré à huit workers est environ 5,49 fois
celui de l’exécution série sur les diagnostics T1. Une extrapolation simple
rapproche donc le temps local par candidat des deux configurations. Ce n’est pas
une égalité de coût scientifique : les évaluations allouées restent six fois plus
nombreuses, les programmes peuvent être plus coûteux, et les appels au modèle
restent séquentiels.

## Résultats déjà vérifiés

| Diagnostic | Résultat | Portée et limite |
|---|---|---|
| [Audit du feedback](../_shared/optimizer_discovery/investigation16/feedback/REPORT.md) | Seulement 4/32 observations conservées par tâche ; point incumbent absent dans 69,6 % des résumés courants ; aucun des 80 prompts n’explicite l’AUC anytime | Perte d’information et décalage de consigne démontrés ; effet causal sur les programmes à mesurer |
| [Canal de production](../_shared/optimizer_discovery/investigation16/history/P1_SCAFFOLD_REVIEW.md) | L’ancien adaptateur vérifiait la présence de Trace, puis reconstruisait séparément le feedback du prompt | Ajouter des traces en amont ne garantissait pas leur consommation ; nouveau chemin testé sur le contenu effectivement propagé |
| [G1, plafonds de génération](../_shared/optimizer_discovery/investigation16/generation/REPORT.md) | Éligibilité 8/12 à 8 000, 10/12 à 32 000 ; fins length 1/0 | Choix de 32 000 selon règle préalable ; aucune réponse de ce groupe ne dépasse 6 581 tokens de complétion rapportés : pas de preuve de sauvetage individuel ni de gain de qualité |
| [B1, surface optimisable](../_shared/optimizer_discovery/investigation16/benchmark/B1_REPORT.md) | 384 trajectoires valides ; programme adaptatif fixe meilleur en moyenne que les trois diagnostics initiaux sur de nouvelles instances | Le benchmark n’est pas globalement saturé ; ce n’est pas un test de feedback génératif |
| [B2, initialisation du seed](../_shared/optimizer_discovery/investigation16/benchmark/B2_REPORT.md) | Une seule modification : premier point au centre. AUC centrale 0,192300 → 0,027569 ; plage élargie 0,167906 → 0,055169 | 96 paires sur fixtures publiques ; −85,66 % et −67,14 %, avec 26 pertes individuelles conservées. Le regret final ne s’améliore pas uniformément. Contrôle numérique indépendant passé |
| [S1, fiabilité de sélection](../_shared/optimizer_discovery/investigation16/selection/REPORT_S1.md) | Audit moyen de la politique sélectionnée : 0,091412 à 6 × 1, 0,081151 à 12 × 2, 0,075566 à 24 × 2 | −11,2 % et −17,3 % sur une banque fixe ; pas un gain projeté du feedback. Les 200 sous-panels se recouvrent et ne sont pas 200 réplications indépendantes |
| [S2, budget de propositions](../_shared/optimizer_discovery/investigation16/budget/S2_REPORT.md) | Deux remplacements A2 sélectionnés apparaissent aux réponses 6/7 ; à N = 8, 5,4 comportements observés distincts hors seed sur TRAIN par bras en moyenne | Un pilote à N = 2 ou N = 4 serait peu représentatif ; aucun avantage futur à N = 16 n’est démontré |
| [Historique rejoué](../_shared/optimizer_discovery/investigation16/history/HISTORY_REPORT.md) | Gains W2 reproduits ; nearest informé les égale sans méta-coût ; un programme VRPTW indépendant réduit la distance de 21,81 % sur la fixture historique | Surfaces de code et transfert d’ordre utiles ; nécessité du feedback récursif et généralisation nouvelle non démontrées par ces replays |
| [Débit de l’évaluateur](../_shared/optimizer_discovery/investigation16/throughput/REPORT_T1.md) | Résultats identiques à 1/4/8/16 workers ; gain corrigé d’environ 5,49 fois à 8 workers | Gain d’ingénierie sur cette machine, pas gain de recherche ni amortissement ; une mesure touchée par suspension a été répétée séparément |
| [Timeout client](../_shared/optimizer_discovery/investigation16/runtime/TIMEOUT_REPORT.md) | Le vrai chemin client accepte une réponse active plus longue que son timeout ; une réponse inactive expire | 300 s est un délai par opération/inactivité, pas une échéance absolue de génération |
| [F1, interventions sur le prompt](../_shared/optimizer_discovery/investigation16/feedback_experiment/REPORT.md) | 24 réponses ; rich − sparse : ΔAUC +0,049229, intervalle [+0,000160 ; +0,103587] ; rich − code : +0,012100, intervalle [−0,015993 ; +0,041424] | Signal défavorable exploratoire face au sparse ; pas de gain établi du riche. Six blocs à parent fixe, fallback commun et quatre sorties invalides conservées ; ce n’est pas une boucle itérative |
| [P1-E1, production réelle](../_shared/optimizer_discovery/investigation16/production/ENGINEERING_REPORT.md) | Deux réponses réelles, un programme éligible sur 48 trajectoires ; prompts de 141 807 caractères ; reprise sans client et compteurs vérifiés | Faisabilité technique, pas gain scientifique ni fiabilité démontrée après plusieurs changements de parent |

La [matrice de décision](../_shared/optimizer_discovery/investigation16/DECISION_MATRIX.md) relie chaque hypothèse à sa preuve,
son intervention et ce qui reste à démontrer. Les sources primaires externes sont
référencées dans [PRIMARY_SOURCES.md](../_shared/optimizer_discovery/investigation16/PRIMARY_SOURCES.md) et les rapports spécialisés.
La [revue des mécanismes publiés](../_shared/optimizer_discovery/investigation16/research/LITERATURE_MECHANISMS.md) distingue
notamment la sélection de programmes de FunSearch, les niches par instance de
GEPA, les critiques de Self-Refine et l’archive de designs d’ADAS. Leurs budgets,
objets et signaux diffèrent de ceux du présent benchmark ; leurs gains ne sont
pas transposés à `recursive_opt`.

## Une faiblesse du seed désormais isolée

B2 change uniquement la première proposition : le milieu des bornes remplace
le tirage uniforme. Pour toute histoire non vide identique, le reste du programme
produit exactement la même proposition que le seed d’origine. Les contrôles B1
sont réutilisés sans réexécution ; les 96 variantes passent le contrat et le même
évaluateur. Un audit indépendant a revérifié les entrées, 6 144 observations des
paires, leurs normalisations et leurs métriques, sans relancer les programmes.

L’effet sur l’AUC est grand et ne se réduit pas au premier terme du calcul :
ce terme représente environ 16–17 % de la différence arithmétique. Le reste
combine la persistance du meilleur point et les histoires ultérieures induites
par cette initialisation. Cela ne constitue pas une décomposition causale de ces
deux mécanismes. Le regret final se dégrade dans trois des six strates de chaque
condition, malgré un gain terminal moyen ; toutes ces pertes restent présentées.

La [figure des 96 paires](../_shared/optimizer_discovery/investigation16/benchmark/b2/paired_outcomes.png), également disponible
en [PDF](../_shared/optimizer_discovery/investigation16/benchmark/b2/paired_outcomes.pdf), montre les gains, pertes et égalités
sans supprimer de trajectoire. Dans la distribution élargie, B2 améliore le
regret final de 17 trajectoires, en dégrade 22 et laisse 9 égalités, malgré
l’amélioration de la moyenne. Les axes logarithmiques sont indiqués explicitement.

Ce résultat montre qu’un gain important face au seed d’origine peut venir d’une
modification très simple, sans LLM. Il renforce le besoin d’une référence
écrite à la main plus forte pour interpréter l’intérêt pratique de la découverte.
Il ne mesure pas l’effet du feedback récursif et ne justifie pas de remplacer
en cours d’expérience le seed commun déjà fixé. La généralisation de cette
variante depuis les fixtures publiques a été évaluée dans le contrôle prospectif
séparé décrit plus bas, sans modifier le seed ni les pools de P1.

La [dérivation géométrique du premier point](../_shared/optimizer_discovery/investigation16/benchmark/FIRST_POINT_GEOMETRY.md)
explique pourquoi ce mécanisme est plausible sur les deux familles quadratiques.
Avec des shifts symétriques dans `[-a,a]`, un premier tirage indépendant dans
`[-b,b]` et des coefficients indépendants des shifts, le rapport des espérances
brutes est `E[f(centre)] / E[f(uniforme)] = a² / (a²+b²)` : 13,8 % pour la
distribution centrale et 44,8 % pour la distribution élargie. Ces valeurs ne sont
ni une moyenne des regrets normalisés, ni une prévision d’AUC, ni un résultat
pour Rosenbrock. Elles n’affirment pas une indépendance exacte des fixtures
pseudo-aléatoires finies. Les 96 paires restent un diagnostic public distinct du
contrôle prospectif.

Le [relevé des premiers points EXP-15](../_shared/optimizer_discovery/investigation16/benchmark/EXP15_FIRST_POINTS.md) retrouve
zéro point exactement central dans chacun des bras A0, A1 et A2 (60 trajectoires
par bras). Certains programmes commencent par un point de Halton dont seule la
première coordonnée est centrale. B2 ne reproduit donc pas une initialisation
midpoint déjà adoptée par les programmes sélectionnés d’EXP-15.

Arithmétiquement, le premier terme contribue −0,004887 à la différence moyenne
A2−A0 de −0,022899, et −0,002773 à A2−A1 de −0,005068. Cette dernière contribution
est nulle pour quatre seeds sur cinq ; seule la seed 53 diffère. Retirer le premier
terme ne supprime pas les effets ultérieurs du point initial sur l’incumbent et
l’histoire. Le reste de l’AUC n’est donc pas un effet causal isolé du feedback.

## Ce que les diagnostics excluent ou nuancent

La trace n’était pas absente : elle était appauvrie et son canal canonique n’était
pas consommé directement. Aucun des 40 résumés JSON courants n’a subi la troncature
par caractères supposée. Une collision construite et testée produit le même ancien
feedback pour deux trajectoires dont l’AUC diffère d’un facteur 15.

L’archive n’était pas désactivée : `long_term_memory_size=None` signifie non bornée.
Le choix d’exploration restait néanmoins étroit, avec un incumbent, 20/40 parents
égaux au seed et une profondeur de parent maximale 2. Plusieurs anciens noms de
stratégies se résolvent à la même classe ; comparer leurs seuls labels serait trompeur.

Les fins de réponse à 8 000 tokens contribuaient aux pertes de propositions, mais
elles étaient plus nombreuses en A1 qu’en A2. Elles ne suffisent donc pas à expliquer
un désavantage propre au feedback. Des erreurs syntaxiques arrêtées normalement,
des contraintes lexicales non explicites et des exécutions non déterministes sont
des mécanismes différents. Deux copies exactes du seed appartiennent à A1, pas à A2.

Le plafond initial de 3 000 tokens était un choix de budget du premier protocole,
pas une limite intrinsèque du modèle. La Phase 0 a ensuite établi une configuration
exécutable à 8 000 tokens avec raisonnement `low`. L’enquête a porté le plafond à
32 000 selon une règle de faisabilité enregistrée avant comparaison. Le réglage
`low` ne garantit pas une quantité fixe de raisonnement. Les compteurs « reasoning »
et « completion » reçus de certains fournisseurs ne sont même pas emboîtés ; ils
sont conservés tels quels, sans les additionner ni en tirer un taux d’utilisation
du plafond artificiellement précis.

Le prior de centre est exploitable : dans B1, déplacer les optima vers une plage
plus large détériore fortement midpoint, mais conserve la qualité du programme
adaptatif fixe. Le critère anytime accorde un poids important aux premières
évaluations. Un meilleur résultat final peut donc coexister avec une moins bonne
AUC ; ce n’est pas une incohérence du calcul. Il faut expliciter ce critère au modèle
et conserver les métriques finales comme résultats complémentaires.

Cela n’exclut pas un autre mécanisme : sur de petites fonctions numériques
classiques, les connaissances préalables du modèle peuvent déjà rendre la
génération indépendante très compétitive. B1 démontre de la marge face aux
diagnostics simples ; il ne mesure pas un plafond théorique ni toute la marge
restant au meilleur programme indépendant. La pertinence de tâches plus riches
pour apprendre d’une trace reste donc une hypothèse, sans justification pour
remplacer après coup ce benchmark par une famille donnant un résultat favorable.

La [revue géométrique](../_shared/optimizer_discovery/investigation16/benchmark/GEOMETRY_REVIEW.md) précise cette limite :
« Sphere » utilise des échelles indépendantes par coordonnée et est donc une
quadratique diagonale légèrement anisotrope dans les coordonnées du candidat
(conditionnement entre 1 et 4). La famille Quadratic couvre la même classe
géométrique avec un conditionnement potentiellement plus élevé, jusqu’à 4 000.
Deux tiers du poids primaire sont ainsi des quadratiques diagonales ; Rosenbrock
apporte la géométrie couplée et non convexe. Ces transformations étaient
explicitement enregistrées : cette observation n’invalide aucun résultat.

## Interventions sur le feedback et comparaison prospective

F1 compare 24 réponses réelles, six blocs à parent fixe, quatre conditions : consigne
initiale/code seul ; consigne anytime/code seul ; anytime/feedback sommaire ;
anytime/feedback détaillé. Même parent et mêmes conditions invariantes dans un bloc,
ordre préenregistré, tous les programmes figés avant la validation indépendante.
Les échecs sont conservés et la même politique de repli définit les comparaisons.

F1 est terminé : les AUC moyennes sont 0,164111 pour legacy, 0,122130 pour
anytime/code seul, 0,085002 pour sparse et 0,134231 pour rich. Les contrastes
instruction seule et sparse − code restent inconclusifs, malgré leurs moyennes
favorables. Rich − sparse présente un signal négatif exploratoire fragile ;
rich − code est inconclusif. Les 24 réponses restent dans les comparaisons,
y compris deux épuisements du plafond, une mauvaise API terminée normalement
et une réponse terminée en erreur fournisseur. Le rapport distingue ces causes.

Ce résultat empêche d’assimiler « davantage de trace » à « meilleur apprentissage ».
La correction d’une perte d’information est justifiée techniquement ; son utilité
pour le modèle reste empirique. F1 évalue une proposition depuis un parent fixe,
sans mécanisme de conservation d’un bon incumbent. Le gain d’une recherche
itérative complète doit donc être mesuré séparément, selon le protocole P1 déjà
défini, sans le modifier pour contourner ce résultat défavorable.

P1-E1 vérifie deux vraies mises à jour de production avec un panel 24 × 2 et des
prompts longs. Son gel est `4be570f58e0522d76d17e9f67b9bb1338eed060dc800319c5b083f23fa1abe75`.
Une archive exacte conserve 93 fichiers de code/dépendances avant exécution.
Le contrôle est terminé et passe : une sortie a une erreur syntaxique malgré
une terminaison normale à 4 873 tokens ; l’autre complète les 48 trajectoires.
Les deux appels utilisent le même parent seed. Ils vérifient le chemin réel
et la reprise, sans établir la fiabilité des lignées longues.
Le deuxième prompt est même identique au premier : le R enregistré montre la
trace du parent conservé, sans transmettre le code et l’erreur syntaxique du
candidat rejeté. Obtenir ensuite une sortie valide n’est donc pas une preuve
de réparation apprise grâce à cette erreur. Une mémoire explicite des échecs
reste un mécanisme distinct à tester ; elle ne sera pas ajoutée au P1 gelé.

La [validation de production P1](../_shared/optimizer_discovery/investigation16/production/PROTOCOL_P1.md), conditionnée à ce gate,
compare I indépendant, C réécriture d’un parent sélectionné sans résultats dans le
prompt, R même recherche avec feedback, et W exploration de deux parents. Six
réplications × quatre bras × huit réponses = 192 allocations. Tous les bras
utilisent le même panel 24 × 2, la même validation 12 × 2, le même audit indépendant
12 × 2 et B = 32.

P1 a terminé les 192 réponses de son protocole gelé, sous le hash
`113f2eb03abfbb3f8e80e8ce5946beecd6b039fef28968b96171475a21d57e8e`.
La référence B2 est enregistrée séparément avant le premier appel P1, sous le hash
`f0aeb8b78d56745c2e54b02462877952bdda15ec4e184fc0346e13d0bb0e288f`.
Ses 144 trajectoires ont été évaluées après l’audit primaire complet.
Les sources et dépendances exactes sont conservées dans une archive de 97 fichiers.

L’[audit de l’ordre enregistré](../_shared/optimizer_discovery/investigation16/production/ORDER_AUDIT.md), réalisé sans consulter
les résultats P1, vérifie une précédence équilibrée pour R−I (3/3). En revanche,
C précède R et R précède W dans cinq réplications sur six. Une dérive du service
pourrait donc contribuer à ces contrastes secondaires ; elle n’est pas démontrée,
et le 3/3 ne garantit pas non plus son élimination pour R−I. L’ordre gelé reste
inchangé. W−R compare une répartition différente du budget entre largeur et
nombre de tours, sans isoler la seule largeur.
Le contraste R−I reste central ; R−C mesure l’apport du contenu évalué au-delà
de la réécriture d’un parent déjà sélectionné par entraînement. C contient donc
une information de sélection implicite. W−R compare deux répartitions du même
budget : deux parents sur quatre tours ou un parent sur huit tours. Il ne sépare
pas la largeur du nombre de mises à jour séquentielles. Aucune nouvelle étude d’efficacité
ne sera déclenchée uniquement pour inverser un résultat défavorable.

## Résultat prospectif de production P1

Les 192 réponses ont 192 identités uniques et 194 tentatives de transport.
Les 24 recherches sont complètes. Toute génération précède toute validation ;
les 24 sélections et le représentant R précèdent tout audit. Les 720 trajectoires
d'audit allouées sont valides, sans repli. Le vérificateur numérique a reproduit
exactement les observations et métriques des 14 592 évaluations mises en cache,
y compris 865 lignes partielles invalides. Ce travail d'intégrité n'a relancé
aucun candidat ou modèle. Voir les [données principales](results/production_run/analysis_results.json.gz),
le [contrôle B2](../_shared/optimizer_discovery/investigation16/production_baseline_control/results.json) et la
[figure appariée](../_shared/optimizer_discovery/investigation16/production/presentation/paired_results.png).

| Bras | AUC moyenne | AUC médiane | Regret final moyen | Cible 0,01 atteinte | Générés éligibles |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0, seed original | 0,179329 | 0,181088 | 0,027274 | 56,25 % | sans génération |
| I, indépendant | 0,077260 | 0,092659 | 0,006712 | 80,56 % | 46/48 |
| C, parent sélectionné sans trace détaillée | 0,047588 | 0,050507 | 0,002646 | 91,67 % | 44/48 |
| R, parent avec trace détaillée | 0,122517 | 0,120798 | 0,024585 | 61,11 % | 41/48 |
| W, deux parents avec trace détaillée | 0,112582 | 0,117316 | 0,027942 | 59,72 % | 43/48 |
| B2, contrôle fixe supplémentaire | 0,031633 | 0,031685 | 0,014285 | 63,19 % | sans génération |

| Seed externe | A0 | I | C | R | W | B2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16411 | 0,147066 | 0,088515 | 0,048731 | 0,056954 | 0,114780 | 0,032321 |
| 16423 | 0,143346 | 0,099153 | 0,073075 | 0,092900 | 0,124702 | 0,030316 |
| 16437 | 0,224167 | 0,019062 | 0,052283 | 0,095341 | 0,085727 | 0,032353 |
| 16441 | 0,199224 | 0,102539 | 0,016177 | 0,146255 | 0,158443 | 0,031469 |
| 16453 | 0,182605 | 0,057486 | 0,074348 | 0,182605 | 0,071989 | 0,031902 |
| 16467 | 0,179570 | 0,096802 | 0,020916 | 0,161048 | 0,119851 | 0,031438 |

| Contraste enregistré | ΔAUC moyen | IC bootstrap apparié 95 % | Lecture enregistrée |
| --- | ---: | --- | --- |
| R−I, central | +0,045257 | [+0,003176 ; +0,085126] | signal négatif |
| R−C | +0,074929 | [+0,034075 ; +0,116730] | signal négatif |
| W−R | −0,009935 | [−0,054857 ; +0,030305] | inconclusif |
| R−A0 | −0,056813 | [−0,090926 ; −0,023830] | signal positif |
| B2−A0, supplémentaire | −0,147696 | [−0,170401 ; −0,125731] | signal positif |
| R−B2, supplémentaire | +0,090884 | [+0,056326 ; +0,126515] | signal négatif |

Ces intervalles appliquent les 10 000 rééchantillonnages appariés enregistrés,
graine 1515. Six réplications externes partagent un panel fixe : les intervalles
sont exploratoires et fragiles, sans garantie de multiplicité. Les tâches et les
points d'une trajectoire ne sont pas des réplications supplémentaires. R perd face
à I dans quatre paires sur six et face à C dans les six. C a la meilleure moyenne
générative, mais **C−I n'était pas un contraste enregistré de P1** : son avantage
apparent reste une piste exploratoire à confirmer sur de nouvelles données.

R conserve le seed pour 16453 malgré six remplacements éligibles. Les autres
bras sélectionnent toujours un programme généré. Les 18 slots inéligibles ne
sont pas supprimés : I/C/R/W en ont 2/4/7/5. Les invalidités de source seules
sont 2/3/4/3 ; les échecs d'exécution expliquent le complément, dont un programme
R valide sur train mais pas sur toute la validation. Dix réponses épuisent encore
32 000 tokens ; les 182 autres terminent normalement. Aucun timeout candidat
n'est observé. Le repli prévu reste testé par les tests, mais n'est pas nécessaire
sur ces trajectoires d'audit.

La référence B2 confirme sa transférabilité dans ce périmètre : **−82,36 % d'AUC
moyenne face au seed original**, sans LLM. Son regret final reste moins bon que
celui d'I et C. Il ne faut donc pas transformer sa bonne initialisation en
supériorité universelle sur toutes les métriques. Ce contrôle ne remplace pas
rétroactivement A0 et ne réécrit aucun contraste P1.

Le représentant R choisi avant audit est 16411/slot 7, hash
`40567fda87a2734f31a680db90e54d5f975a73b7930bc487be821d0cc5c9f243`.
Il utilise notamment une exploration de Halton et un substitut gaussien construit
en bibliothèque standard. Son inspection, les autres sources sélectionnées et
leurs lignées sont documentées séparément. Son choix repose sur la validation,
pas sur sa position dans les résultats d'audit.

L'[inspection des 192 sources et prompts](../_shared/optimizer_discovery/investigation16/production/PROGRAM_INSPECTION.md)
confirme que les 96 demandes R/W comportaient chacune 48 trajectoires du parent
courant et leurs 32 valeurs best-so-far. Les coordonnées ne décrivaient toutefois
que le premier point et les améliorations : 19,10 % des observations pour R,
17,37 % pour W. Trente panels R sur 48 répétaient un parent déjà décrit ; aucun
historique cumulatif des tentatives rejetées n'était transmis. Les profondeurs
des lignées R sélectionnées sont 4/2/3/5/0/1. W utilise réellement deux parents
distincts après son initialisation, mais quatre rondes au lieu de huit. Ces faits
écartent une absence totale de trace ou une largeur purement nominale ; ils ne
prouvent pas qu'ajouter une mémoire, tous les points ou un Pareto par instance
ferait gagner R. Ces mécanismes restent à isoler prospectivement.

## Coûts, budgets et limites d'exécution

| Bras | Réponses | Tokens prompt | Tokens complétion | Tokens totaux | Coût rapporté USD |
| --- | ---: | ---: | ---: | ---: | ---: |
| I | 48 | 27 840 | 287 018 | 314 858 | 0,050225 |
| C | 48 | 79 715 | 373 741 | 453 456 | 0,084017 |
| R | 48 | 2 798 859 | 378 080 | 3 176 939 | 0,183054 |
| W | 48 | 2 722 485 | 312 487 | 3 034 972 | 0,170899 |
| Total | 192 | 5 628 899 | 1 351 326 | 6 980 225 | 0,488195 |

Les 1 147 429 tokens de raisonnement rapportés ne s'ajoutent pas au total.
Tous les champs principaux ont une valeur pour les 192 réponses. Deux tentatives
de transport ont une éventuelle facturation distante inconnue. Le même modèle
OpenRouter demandé et les mêmes réglages ont été conservés ; 15 fournisseurs
amont sont observés. Les comparateurs ont le même budget de réponses, pas le même
coût réalisé : R consomme environ dix fois les tokens totaux d'I. Ses prompts
dépassent en moyenne 58 000 tokens. Cette surcharge est mesurée ; son rôle causal
dans la dégradation reste à isoler, tout comme le routage et l'ordre des requêtes.

Les allocations primaires sont **16 272 trajectoires / 520 704 objectifs**.
Les références logiques comptent 483 201 valeurs effectivement évaluées et
37 503 allocations inutilisées. Après réutilisation déterministe du cache,
les évaluations physiques sauvegardées comptent **440 961 objectifs et
882 283 sous-processus**. Le cache a 66 096 consultations, 51 504 hits et
14 592 misses, sans évaluation physique orpheline ni miss répété enregistré.
B2 ajoute 4 608 objectifs et 9 216 sous-processus, sans appel génératif.
Les 6 144 points de normalisation uniques sont de la préparation partagée ;
leur nombre de reconstructions physiques n'est pas entièrement instrumenté.
Le contrôle numérique primaire ajoute séparément 447 105 évaluations d'intégrité,
jamais des opportunités de recherche. L'audit indépendant B2 en comptabilise
séparément ses propres vérifications.

La génération et ses évaluations train ont duré environ 11,96 heures actives,
la validation/sélection 73,74 minutes et l'audit 5,35 minutes. Les 67,32 minutes
de mise en veille sont séparées du temps actif. Le timeout réseau de 300 secondes
reste une limite par opération/inactivité, pas une durée maximale d'une réponse.
Ces mesures ne sont pas un test contrôlé de vitesse des fournisseurs.

Le contrat candidat reste `propose(history, bounds, seed)` avec bibliothèque
standard, sous-processus frais, environnement nettoyé, rejets typés et replay
déterministe. Ce mécanisme n'est pas un sandbox de sécurité du système
d'exploitation. Aucune infrastructure Project 1 n'a été ajoutée.

## Niveau de confiance requis pour les conclusions

Une erreur d’instrument peut être démontrée sans que son effet sur la performance
le soit. Les tests de collision du feedback et de consommation du canal Trace
établissent le premier point. F1 mesure séparément les changements de consigne et
de contenu ; P1 mesure une procédure de recherche complète sous les corrections
combinées. Un gain de P1 ne permettrait donc pas d’attribuer tout l’effet à une
seule correction, ni à une profondeur de récursion supplémentaire.

Le résultat de S1 concerne la sélection dans une banque de programmes déjà fixés.
Il justifie un panel plus fiable, mais son amélioration de 17,3 % ne doit pas être
présentée comme une prévision du gain R−I. De même, l’accélération de l’évaluateur
mesure son débit local ; elle ne démontre pas que la recherche atteint une cible
avec moins d’appels au modèle ou à l’objectif.

Tous les contrastes et toutes les seeds sont rapportés, y compris la perte face
à la génération indépendante. Une recommandation peut porter avec confiance sur
la correction du protocole sans promettre la supériorité du mécanisme corrigé.
Les scénarios chiffrés annoncent leur effet utile, leur variance et leurs hypothèses.

## Recommandation et gains projetés

**Ne pas relancer R inchangé en augmentant seulement les tokens ou le nombre de
seeds pour obtenir une victoire.** Le plafond accru, l'objectif explicite, la vraie
Trace et un panel huit fois plus grand n'ont pas suffi. C utilise le même chemin
de production avec une information implicite de sélection, sans les longues
traces. Ce résultat ne signifie donc pas que toutes les stratégies de
`recursive_opt` sont inefficaces.

| Modification | Décision | Gain défendable |
| --- | --- | --- |
| Objectif anytime explicite, vraie Trace, invalidité typée et compteurs vérifiés | Conserver ces corrections | Instrument fidèle ; pas de gain R−I causalement attribué |
| Train 24 instances × 2 seeds locaux, validation 12 × 2 | Conserver sur ce benchmark | S1 : sélection −17,3 % d'AUC dans une banque fixe, pas une prévision générative |
| Huit workers locaux, appels modèle séquentiels | Conserver | T1 : débit ×5,49 sur cette machine, pas moins d'appels scientifiques |
| Référence handwritten B2 | Ajouter comme contrôle exigeant ; garder A0 comme pont historique | −82,36 % d'AUC prospectivement face à A0 ; pas une supériorité sur toutes les métriques |
| Plafond 32 000 / raisonnement low, même modèle | Conserver pour la comparabilité | Faisabilité accrue ; 10/192 troncatures persistent, aucun gain qualitatif promis |
| Réécriture C d'un parent sélectionné | Piste à confirmer face à I | C−I moyen −0,029672, exploratoire et non enregistré comme contraste P1 |
| Largeur W | Différer dans la prochaine confirmation centrale | Aucun bénéfice établi ; largeur et nombre de tours confondus |
| Trace compacte ou mémoire alignée des rejets | Diagnostic séparé avant confirmation | Hypothèses non validées, aucun gain projeté établi |
| Autre modèle ou autres familles | Ne pas imposer comme solution démontrée | Aucun comparatif de modèles ; benchmark non saturé mais géométriquement étroit |

La prochaine confirmation pratique proposée, **non lancée**, testerait **C−I sur
de nouvelles instances et seeds**, avec B2 fixe et A0 comme références. Pour
conserver le sens de cette réplication, garder le seed original comme point de
départ commun, le contrat, les familles/dimensions, B = 32, N = 8, les panels
24/12/12 avec deux seeds locaux, et les réglages communs P1. Alterner I/C puis C/I
équilibre l'ordre pour un effectif pair. Toutes les sélections précèdent l'audit
neuf. Faire de B2 le seed commun serait une intervention supplémentaire à piloter
symétriquement et enregistrer ; les résultats P1 n'en prédisent pas l'effet.

Un scénario d'effet utile absolu de 0,020 d'AUC donne **46 nouvelles paires C/I**,
soit 736 réponses à N = 8. Il suppose une approximation normale bilatérale à 5 %,
une puissance nominale de 80 % et une variance future proche du SD exploratoire
C−I de 0,048137. Ce n'est ni une puissance garantie du bootstrap, ni un bénéfice
attendu. Pour 0,010, le même scénario donne 182 paires. Fixer l'effet minimal
utile et la procédure confirmatoire avant les appels ; ne pas compléter P1
opportunément avec de nouvelles seeds.

Si la question reste précisément l'utilité du **texte du feedback**, conserver
I/C/R et piloter une seule intervention nouvelle, compacte ou fondée sur des
tentatives code/hash/résultat/statut alignées. Le parent sélectionné apporte déjà
une information implicite : C n'est pas « sans feedback » au sens large. Une
archive interne ne constitue pas une mémoire de rejets dans le prompt. Aucun
bénéfice de six/sept tentatives montrées n'est établi ; N = 8 ne les rend
disponibles qu'à la fin. Ne pas augmenter aveuglément N sur cette seule intuition.

Le SD R−I est 0,057022. À variance supposée constante, les effets absolus
0,005/0,010/0,020/0,030 nécessiteraient environ 1 021/256/64/29 nouvelles paires
dans le scénario normal précédent. Passer de six à sept ne règle donc pas la
précision d'un petit effet. Les calculs et sensibilités sont dans
[FUTURE_DESIGN](../_shared/optimizer_discovery/investigation16/production/FUTURE_DESIGN.md). **Les gains B2, S1 et T1 ne
s'additionnent pas** en une projection de gain récursif.

La confiance porte sur les défauts logiciels isolés, les gains bornés de sélection
et de débit, la faiblesse du seed, et le résultat défavorable de R dans P1.
Longueur de contexte, ancrage, priors du modèle, routage et mémoire des rejets
restent des mécanismes à séparer. L'enquête ne prouve aucune cause unique et ne
prétend pas établir nouveauté algorithmique, profondeur récursive ou amortissement.
H15-A/H15-B et les anciennes rétractions gardent leur portée historique.

Les données antérieures figurent dans le [classeur documenté](../_shared/optimizer_discovery/investigation16/report_data/README.md).
Les tables exactes P1/B2 sont dans [production/presentation](../_shared/optimizer_discovery/investigation16/production/presentation/data.json).
Les [vérifications](../_shared/optimizer_discovery/investigation16/VERIFICATION.md), l'[audit final indépendant](../_shared/optimizer_discovery/investigation16/production/INDEPENDENT_FINAL_REVIEW.md)
et l'[inspection des programmes](../_shared/optimizer_discovery/investigation16/production/PROGRAM_INSPECTION.md) complètent
ce rapport. Aucun appel supplémentaire ne sera utilisé pour inverser le résultat.
