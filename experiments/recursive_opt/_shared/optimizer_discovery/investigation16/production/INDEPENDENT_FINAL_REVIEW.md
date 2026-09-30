# Revue indépendante finale de P1 et du contrôle B2 prospectif

**Verdict : PASS pour l'intégrité des résultats et de leur analyse.** La revue du
10 septembre 2026 retrouve le résultat défavorable de R face à I et C, ainsi que
l'avantage du contrôle B2 fixé avant les appels principaux. Elle ne transforme pas
ce diagnostic exploratoire à six réplications en confirmation définitive.

La preuve détaillée, les valeurs par réplication, les 24 sources sélectionnées et
les 192 identifiants de réponse sont dans
[INDEPENDENT_FINAL_REVIEW.json](INDEPENDENT_FINAL_REVIEW.json). Aucun candidat,
modèle ou réseau n'a été exécuté pendant cette revue ; aucune clé n'a été chargée.
Aucun gel, code gelé, résultat brut ou décision de sélection n'a été modifié.

## Protocole et provenance

La revue porte sur le [protocole P1](PROTOCOL_P1.md), son
[analyse enregistrée](analysis_protocol.md) et le
[contrôle B2 complémentaire](BASELINE_CONTROL_PROTOCOL.md).

| Élément | SHA-256 vérifié |
|---|---|
| Gel P1 | `113f2eb03abfbb3f8e80e8ce5946beecd6b039fef28968b96171475a21d57e8e` |
| Gel du contrôle B2 | `f0aeb8b78d56745c2e54b02462877952bdda15ec4e184fc0346e13d0bb0e288f` |
| Archive des 97 sources | `f5120ad690b842f6fd2ca1558d00c5c2e84f723984dbbecd336e60c76fc546eb` |
| Analyse principale conservée | `e137635f40869587bbb532eeb899bf406e080376c6cbfccef819becf03fa23c4` |
| Résultat du contrôle B2 | `65915bf729ac14c8fbdbed766ffb76ce7b3d7428f22aee85c67391f0904469f9` |

Les 97 fichiers archivés correspondent exactement aux sources présentes ; le ZIP
est intègre. Les gels précèdent la génération. Les 82 371 fichiers de preuve lus
ont conservé leur hash entre leur lecture et la fin du contrôle.

Les six graines externes sont `16411, 16423, 16437, 16441, 16453, 16467`. Chaque bras
génératif reçoit huit réponses par graine : I indépendant, C réécriture du parent
retenu sans texte de feedback, R avec feedback TRAIN riche et AUC agrégée, W avec
deux parents sur quatre rounds. A0 reste le seed inchangé. Le budget intérieur est
32 évaluations ; TRAIN comporte 24 instances et deux graines locales, VALIDATION
12 × 2, AUDIT 12 × 2. Les contrôles de configuration retrouvent le modèle exact
DeepSeek/OpenRouter, la limite de 32 000 tokens, le raisonnement `low` et la
concurrence générative de 1.

Les 192 identifiants de réponse sont uniques, leurs requêtes et sources sont
liées par hash, et les 194 tentatives sont présentes. Les deux tentatives de
transport échouées restent visibles avec l'incertitude de facturation distante.
Les 144 événements de mise à jour Trace correspondent exactement aux huit slots
de C, R et W pour les six graines. Les hashes du parent, de l'enfant et du feedback
propagé correspondent aux requêtes et réponses ; aucun de ces événements n'est
étiqueté comme rejeu d'une réponse terminée.

## Sélection et séparation des données

Les 24 choix finaux ont été recalculés à partir des pools complets : validité de
toutes les trajectoires TRAIN et VALIDATION, minimum de l'AUC de validation, puis
ordre des propositions en cas d'égalité, avec le seed à l'indice −1. Chaque source
sélectionnée est identique à celle de son pool et, si elle est générée, à celle de
sa réponse conservée.

Les horodatages vérifient l'ordre global suivant :

| Événement | Horodatage conservé, nanosecondes |
|---|---:|
| Dernière réponse terminée | 1789013655566143624 |
| Barrière de génération | 1789013851282709879 |
| Première évaluation VALIDATION persistée | 1789013969430852507 |
| Dernière sélection individuelle | 1789018381017383115 |
| Gel global de toutes les sélections | 1789018381017708272 |
| Première évaluation AUDIT persistée | 1789018423161506979 |
| AUDIT principal terminé | 1789018740977517804 |
| Début du contrôle B2 | 1789018851790037621 |

Le représentant R reste celui choisi par validation : graine `16411`, proposition
7, AUC de validation `0.07063484878471085`, source
`40567fda87a2734f31a680db90e54d5f975a73b7930bc487be821d0cc5c9f243`.
Son texte exact se trouve dans
[la sélection conservée](../../../../EXP16/results/production_run/raw/16411/R/selection.json), champ
`source`. Il n'a pas été choisi sur AUDIT.

## Recalcul indépendant des résultats

L'analyse gelée, rejouée en lecture seule, est exactement égale au résultat
conservé. Une seconde voie arithmétique utilise `math.fsum`, agrège les courbes
conservées au sein de chacune des six strates, puis donne le même poids à chaque
strate et à chaque graine externe. Le bootstrap apparié est reconstruit sans
appeler le helper de contraste : `random.Random(1515)`, 10 000 tirages de six
graines avec remise et interpolation des quantiles 2,5 % et 97,5 %. L'écart
arithmétique maximal est `3.47e-17`.

| Bras | AUC moyenne | Médiane |
|---|---:|---:|
| A0, seed inchangé | 0.179329 | 0.181088 |
| I, génération indépendante | 0.077260 | 0.092659 |
| C, réécriture du parent | 0.047588 | 0.050507 |
| R, feedback TRAIN riche | 0.122517 | 0.120798 |
| W, deux parents | 0.112582 | 0.117316 |
| B2, contrôle prospectif fixé | 0.031633 | — |

Une différence négative favorise le premier terme du contraste.

| Contraste enregistré | Différence moyenne | Intervalle bootstrap 95 % | Lecture enregistrée |
|---|---:|---:|---|
| R − I | +0.045257 | [+0.003176 ; +0.085126] | Signal négatif |
| R − C | +0.074929 | [+0.034075 ; +0.116730] | Signal négatif |
| W − R | −0.009935 | [−0.054857 ; +0.030305] | Inconclusif |
| R − A0 | −0.056813 | [−0.090926 ; −0.023830] | Signal positif |
| B2 − A0 | −0.147696 | [−0.170401 ; −0.125731] | Signal positif |
| R − B2 | +0.090884 | [+0.056326 ; +0.126515] | Signal négatif |

Le contrôle B2 conserve sa source fixée
`958fbb12279a15966bf1ffa45ddffc7cd5f946ca800a3fe2b8e132a150c5c190`.
Ses 144 trajectoires et leurs 4 608 valeurs objectives ont été recalculées avec
les fonctions déterministes du benchmark. Les 12 normalisations de référence et
les métriques ont été reconstruites. Un calcul arithmétique séparé confirme les
minima cumulés, l'AUC, le regret final, le seuil et la censure. Le regret final
moyen est `0.014284912520611377`, le taux d'atteinte `0.6319444444444444` et le temps
moyen plafonné `18.479166666666668` évaluations. Ce dernier conserve les
non-atteintes comme censurées, sans les convertir en temps observés.

## Invalidités, fallback et budgets

| Bras | Candidats générés éligibles | Sources invalides | Trajectoires générées TRAIN + VALIDATION invalides |
|---|---:|---:|---:|
| I | 46 / 48 | 2 / 48 | 144 / 3 456 |
| C | 44 / 48 | 3 / 48 | 288 / 3 456 |
| R | 41 / 48 | 4 / 48 | 433 / 3 456 |
| W | 43 / 48 | 3 / 48 | 360 / 3 456 |

La dernière colonne inclut les sources invalides et les défaillances
d'exécution : elle ne mesure donc pas uniquement les exceptions d'un programme
syntaxiquement valide. Les 865 lignes invalides du cache physique sont retenues
avec leurs observations partielles et une métrique absente. R sélectionne le seed
pour `16453`, bien que des remplacements générés soient éligibles. Aucun pool ne
manque entièrement de remplacement éligible. Il n'y a aucun fallback parmi les
720 trajectoires AUDIT logiques ni parmi les 144 trajectoires B2.

Les allocations logiques couvrent 16 272 trajectoires, soit 520 704 appels
objectifs alloués. Elles sont distinctes des 14 592 entrées physiques uniques du
cache : 440 961 appels objectifs effectivement conservés, 882 283 sous-processus,
25 983 évaluations non utilisées après invalidité. Les 66 096 accès comprennent
51 504 hits et 14 592 misses, sans miss répété ni entrée dépourvue de son événement
de création. Les références d'une même ligne dans plusieurs pools ne deviennent
pas des dépenses physiques supplémentaires. Les allocations inutilisées ne sont
pas recyclées en propositions supplémentaires.

Les 192 réponses déclarent 5 628 899 tokens d'entrée, 1 351 326 tokens de sortie,
soit 6 980 225 tokens au total et **0.488195055934 USD**. Les 1 147 429 tokens de
raisonnement sont inclus dans les sorties, pas ajoutés une seconde fois. I utilise
314 858 tokens, C 453 456, R 3 176 939 et W 3 034 972 : l'égalité du nombre de
réponses ne constitue pas une égalité de tokens. Les coûts des deux tentatives
distantes incertaines restent inconnus. Les fins de réponse sont `stop` pour 182
slots et `length` pour 10 ; ces derniers restent comptés.

Le contrôle B2 a consommé séparément 4 608 appels objectifs scientifiques et
9 216 sous-processus, avec 1 536 appels de référence comptés par son invocation.
La présente revue ajoute **6 144 appels mathématiques d'intégrité** : 4 608 valeurs
de contrôle et 1 536 points de normalisation. Aucun de ces appels n'est imputé au
budget de recherche. La vérification numérique principale antérieure a effectué
447 105 appels d'intégrité, dont 440 961 observations et 6 144 références ; elle
n'a pas été répétée ici. Son résultat PASS est relié aux 14 592 hashes actuels du
cache et à toutes ses métriques reconstruites.

## Commande et limites

La commande exécutée, avec succès dans la session d'outil `35013`, est :

```bash
PYTHONPATH=. /tmp/phase0-venv/bin/python /tmp/investigation16_final_review.py
```

Le script a pour SHA-256
`46549d99e03789d5cb234508232a0897fe55e024787c20aad67477204f25b567`.
Le JSON conserve la commande, le résultat et les preuves ; aucun fichier séparé
de sortie standard n'avait été créé. Deux vérifications complémentaires en
lecture seule ont relié les 144 callbacks Trace à leurs requêtes et recomposé les
métriques secondaires B2, sans nouveaux appels objectifs. Les entrées JSON
`production_trace_callback_check` et `control_secondary_metrics_check` en rendent
compte.

L'incertitude reste celle de six graines externes sur un même panel AUDIT : les
intervalles sont fragiles et non ajustés pour les comparaisons multiples. R ajoute
conjointement des trajectoires brutes et une AUC TRAIN agrégée ; W modifie à la fois
la largeur et le nombre de rounds. Les précédences C/R et R/W sont déséquilibrées
5 contre 1 dans l'ordre gelé. Cette revue ne prouve ni que la latence du fournisseur
a causé les différences, ni quelle partie du feedback explique leur direction.

La génération a duré 47 100,84 secondes calendaires, dont environ 4 039,23 secondes
de suspension détectées par comparaison des horloges ; le temps monotone actif est
43 061,61 secondes. Le temps calendaire ne doit donc pas être assimilé à la seule
latence du fournisseur. Les coûts physiques décrivent les travaux persistés ;
d'éventuels travaux perdus avant persistance et les reconstructions de
normalisation non instrumentées restent hors de leur précision garantie.

Le résultat prospectif B2 étend son diagnostic d'initialisation à de nouvelles
instances enregistrées. Il ne remplace pas rétroactivement le seed P1 et ne
modifie pas le contraste principal R − I. Rien dans cette revue ne démontre une
nouveauté algorithmique, un gain de profondeur récursive ou une amortisation, ni
ne renforce l'isolation du sous-processus en sandbox système. Les résultats et
rétractations d'EXP-15 restent inchangés.
