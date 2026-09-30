# EXP-18 — lecture des programmes sélectionnés

Les 24 recherches ont chacune sélectionné un programme généré, distinct du programme initial. Les sources retenues sont également toutes distinctes entre elles. Le représentant global, choisi **sur validation avant l’audit**, est **PM, répétition 18041, proposition d’indice 9**. Il combine un modèle quadratique local, l’exploitation du meilleur point observé et une exploration adaptée à la stagnation.

Cette revue décrit les artefacts et les mécanismes effectivement exposés au modèle. Elle ne démontre pas qu’une ligne de code, la mémoire ou Pareto cause un gain. Les comparaisons quantitatives appartiennent à l’[analyse enregistrée](../../../EXP18/results/run/analysis_results.json.gz). Aucun candidat, objectif ou modèle n’a été exécuté pour cette revue.

## Sources exactes et sélection

Les fichiers `.py.txt` contiennent les **octets exacts du Python évalué**, sans formatage ni réparation. Ils peuvent être copiés vers `optimizer.py` sans transformation. L’[inventaire complet](programs/PROGRAM_INSPECTION.md) donne les 24 sélections et leurs empreintes ; le [rapport machine](programs/program_inspection.json) contient sources, filiation, différences, provenance et observations descriptives.

| Bras | Représentant sur validation | Source exacte | Lignes / nœuds AST | Réécritures dans sa filiation |
|---|---|---|---:|---:|
| L | 18067, indice 15 | [Programme L](programs/selected/L/18067_optimizer.py.txt) | 165 / 1 721 | 7 |
| M | 18037, indice 15 | [Programme M](programs/selected/M/18037_optimizer.py.txt) | 77 / 760 | 4 |
| P | 18067, indice 12 | [Programme P](programs/selected/P/18067_optimizer.py.txt) | 174 / 1 871 | 3 |
| PM | 18041, indice 9 | [Programme PM](programs/selected/PM/18041_optimizer.py.txt) | 261 / 3 026 | 5 |

Les indices commencent à zéro : l’indice 9 correspond à la dixième réponse. Les nombres de lignes et de nœuds AST décrivent la taille du code ; ils ne sont pas des critères de sélection.

Empreintes SHA256 des quatre sources :

```text
L  66a330a63f1af3e24c83241fa1fc955f4859f28f1318cf3d8f3238dfc9a951e0
M  d5b3ea6d1529815bfe9524accb95854d55e80b0be6ccfe8d4a9ff381920b7e23
P  0bc881c4b03dc545f4a9fbf7fa2dbca07c940598dc86e55e73dda9ed50da1c0c
PM a787c80c41dbf44a073a0dcf3874d6237e64aabc6db9a2996e760fd4d0c13ced
```

Le représentant global n’a pas été choisi après comparaison des résultats d’audit. Son identité figure déjà dans la [barrière de sélection](../../../EXP18/results/run/selections_frozen.json). Il n’est donc pas présenté comme le meilleur programme sur toutes les tâches possibles.

## Ce que fait le représentant PM

Le programme utilise uniquement `history`, `bounds` et `seed`. Toutes ses adaptations se reconstruisent depuis l’historique à chaque appel ; aucune mémoire externe ni appel LLM n’est nécessaire.

1. **Premier point : le centre des bornes.** Ensuite, le programme identifie le meilleur point déjà observé, son incumbent.
2. **Exploration globale.** Sa probabilité dépend du nombre de points et de la dimension. L’indicateur de stagnation peut la porter à 35 %. Le programme cherche alors un point uniforme assez éloigné des observations précédentes, pendant au plus dix essais, puis accepte une nouvelle proposition uniforme. Ce filtrage examine des distances ; il ne demande aucune nouvelle valeur de l’objectif.
3. **Modèle local.** Lorsqu’il dispose d’assez d’observations, il ajuste une approximation quadratique par moindres carrés régularisés. Il commence, quand le nombre de points le permet, avec les termes par axe, puis utilise aussi les termes croisés entre coordonnées. Les seuils sont respectivement 5 et 6 observations en 2D, 9 et 15 en 4D. Ces modèles ne sont pas nécessairement utilisés à chaque appel : la branche d’exploration peut être prise auparavant.
4. **Choix d’un pas.** À partir du modèle, il compare des déplacements de type Newton régularisé, de descente et des directions aléatoires. Leur taille est bornée. Les déplacements sont évalués par le modèle approché, puis un seul point est retourné à l’évaluateur réel.
5. **Solutions de repli internes.** Si le modèle quadratique ne fournit pas de proposition, le code essaie une approximation linéaire puis une perturbation aléatoire autour de l’incumbent. Ces branches internes diffèrent du remplacement de secours par le programme initial prévu par le protocole de déploiement.

Le rayon de recherche diminue avec la longueur de l’historique, augmente lorsque l’indicateur de stagnation atteint son seuil et possède un plancher. Les points sont ramenés dans les bornes. Le programme ajoute un petit bruit à la proposition du modèle quadratique ; cette ligne permet une perturbation locale, sans démontrer qu’elle apporte une amélioration.

L’idée concrète est donc : **apprendre une approximation numérique locale à partir des valeurs déjà observées, tout en conservant une possibilité de repartir explorer**. Le programme n’accède pas à la formule cachée. Cette approximation est une conséquence du code exécuté à l’intérieur d’une trajectoire, distincte de la mémoire des anciens programmes fournie au modèle générateur.

## Sa filiation réelle

La filiation du représentant PM est non ambiguë au niveau des sources :

```text
programme initial → indice 1 → indice 3 → indice 4 → indice 6 → indice 9
```

| Transition | Modification lisible dans les différences conservées |
|---|---|
| Initial → [1](programs/selected/PM/18041_slot_1.diff.json) | Ajout d’un solveur et d’une approximation quadratique avec termes croisés, au-delà des perturbations du programme initial. |
| 1 → [3](programs/selected/PM/18041_slot_3.diff.json) | Premier point au centre ; indicateur de stagnation ; exploration évitant les points proches ; restriction possible aux observations proches de l’incumbent. |
| 3 → [4](programs/selected/PM/18041_slot_4.diff.json) | Deux réglages locaux : seuil de distance exploratoire de 5 % à 6 % de la diagonale ; bruit après proposition quadratique de 1 % à 0,5 % de l’étendue par axe. |
| 4 → [6](programs/selected/PM/18041_slot_6.diff.json) | Rayon adapté à la stagnation ; modèle quadratique diagonal utilisable plus tôt ; ajout d’une branche de réflexion de points. |
| 6 → [9](programs/selected/PM/18041_slot_9.diff.json) | Retrait de cette branche de réflexion ; unification des ajustements quadratiques ; comparaison de plusieurs déplacements via le modèle ; seuil exploratoire de stagnation ramené de 40 % à 35 %. |

Une transition ne signifie pas une amélioration causée par chacune de ses modifications. Plusieurs changements peuvent intervenir ensemble. La mémoire peut également présenter des programmes hors de cette chaîne : la filiation donne le **parent explicitement utilisé**, pas la totalité des influences possibles sur une réponse.

Pour produire l’indice 9, la [décision Pareto enregistrée](../../../EXP18/results/run/raw/18041/PM/parent_decisions/slot_09.json) choisit l’indice 6 dans une frontière de quatre parents, alors que le meilleur scalaire enregistré est l’indice 5. Le [contexte exact](../../../EXP18/results/run/raw/18041/PM/slot_09/current_context.json) contient **six anciens codes complets**, et non sept : exclusion du parent, source dupliquée et réponse sans code expliquent la disponibilité réduite. Les statuts d’échec antérieurs restent présents. Le [message envoyé](../../../EXP18/results/run/raw/18041/PM/slot_09/request.json) contient bien ce parent et cette mémoire.

## Les trois autres représentants

**L construit un modèle de substitution de type processus gaussien.** Il commence au centre, ajoute quelques points de couverture déterministe, normalise les coordonnées, puis ajuste une matrice de covariance. Il compare 300 propositions internes mêlant mouvements locaux, modification d’une coordonnée, voisinages de points passés et exploration globale, avec un critère d’amélioration espérée. Une inversion impossible déclenche une perturbation de l’incumbent. Les 300 comparaisons utilisent le modèle, pas 300 appels de l’objectif.

**M utilise une procédure plus courte, adaptée aux variations par axe.** Il essaie le centre, puis deux points symétriques sur chaque axe. Ces mesures servent à proposer un minimum quadratique coordonnée par coordonnée. Le reste du budget alterne exploration uniforme et mouvements autour de l’incumbent, avec réaction à la stagnation. Le code exploite une structure séparable pour cette estimation initiale ; cela ne prouve pas qu’il résout une vallée couplée comme Rosenbrock.

**P construit également un modèle à noyau avec amélioration espérée.** Après le centre et quatre points de couverture, il utilise une longueur caractéristique issue des distances de l’historique. Il compare jusqu’à 800 propositions internes, globales ou proches de bons points observés. Sa filiation enregistrée est courte : initial → 2 → 3 → 12. Des étapes de recherche nombreuses n’impliquent donc pas autant de réécritures successives dans le programme finalement choisi.

Ces descriptions viennent de la lecture du code exact. Les matrices, régularisations et paramètres d’acquisition sont des choix des programmes générés ; cette revue ne certifie pas leur calibration numérique ni une implémentation de référence d’un algorithme publié.

## Observations conservées et limites de la lecture

Pour les quatre représentants, les 24 trajectoires d’audit commencent toutes au centre. Sur l’ensemble des **24 programmes sélectionnés**, 19 commencent ainsi sur leurs 24 trajectoires ; cinq utilisent un autre départ. Ce trait ne doit donc pas être attribué exclusivement à la mémoire ou à Pareto.

Les observations existantes des 24 sélections couvrent 576 trajectoires et 18 432 points évalués. Elles ne montrent ni invalidité du candidat sélectionné ni recours au programme de secours sur ces trajectoires. Elles comportent 56 répétitions exactes de points. Ces constats sont limités au panel enregistré : ils ne signifient pas absence universelle d’échec ou exploration systématiquement diversifiée.

Les branches internes n’étant pas instrumentées une par une, nous ne pouvons pas annoncer le nombre de pas réellement issus du modèle quadratique, d’une exploration uniforme ou d’une perturbation de secours. Lire une branche prouve son existence ; les sorties enregistrées ne suffisent pas toujours à identifier la branche qui les a produites.

Par ailleurs, une autre sélection, PM/18023/indice 14, possède une ambiguïté de filiation entre des origines de sources identiques : profondeur comprise entre 4 et 5. Le rapport conserve cette ambiguïté. Ces profondeurs mesurent des réécritures de programmes, **pas des niveaux de méta-optimisation récursive**.

## Mémoire et Pareto ont-ils réellement été exercés ?

Un contrôle exhaustif en lecture seule couvre les six répétitions et les 16 requêtes de chaque bras concerné. Il compare les décisions et contextes scellés aux requêtes consommées ; les décisions terminales ne comptent pas comme des générations.

| Exposition mesurée | M | P | PM |
|---|---:|---:|---:|
| Requêtes avec mémoire | 96 | — | 96 |
| Dont sept codes complets affichés | 33 | — | 39 |
| Avec omission explicite pour plafond de caractères | 12 | — | 5 |
| Décisions Pareto consommées | — | 96 | 96 |
| Frontière comportant plusieurs parents | — | 89 | 87 |
| Parent choisi différent du meilleur scalaire | — | 73 | 63 |
| Taille maximale de frontière consommée | — | 15 | 13 |
| Décisions terminales inutilisées, exclues ci-dessus | — | 6 | 6 |

Trente requêtes PM réunissent simultanément sept codes complets et un parent différent du meilleur scalaire. La mémoire sérialisée atteint au maximum 65 490 caractères pour M et 64 747 pour PM, sous le plafond de 65 536. Un nombre inférieur à sept vient notamment du début de recherche, des doublons, de l’exclusion du parent, des réponses sans source ou du plafond déclaré ; il ne faut pas compter les résumés d’essais comme des codes complets supplémentaires.

Le contrôle vérifie : présence exacte du texte mémoire dans la requête, empreintes des sources, exclusion du parent, récence et déduplication, couverture de tous les emplacements précédents, horodatages antérieurs au contexte, reçus TRAIN associés. Il reconstruit indépendamment la dominance et le tirage déterministe depuis les vecteurs TRAIN, puis compare le parent retenu au code réellement présent dans la requête. Les fichiers sont confrontés à la [barrière de génération](../../../EXP18/results/run/generation_frozen.json).

Cela exclut une explication purement nominale du type « la frontière n’avait jamais plusieurs membres » ou « aucun ancien code n’a été fourni ». **Cela ne démontre ni l’utilisation mentale de la mémoire par le modèle, ni la supériorité de son contenu, ni le nombre optimal d’exemples.** L’exposition observée et l’effet comparatif restent deux objets différents.

## Provenance de cette revue

Le rapport d’export authentifie 2 299 fichiers d’entrée. Cette revue a également recalculé les empreintes des 129 fichiers exportés listés dans son manifeste, dont les 24 sources exactes ; le 130e fichier est le rapport JSON lui-même. La [vérification numérique existante](../../../EXP18/results/run/numeric_verification/attempt_001.json.gz) porte le statut `PASS` ; elle n’a pas été relancée ici.

```text
Protocole gelé, SHA256 JSON canonique :
ce88c4b2d9530456e63b497f7bbde7608fbf3d8aca441c13c086d77b6beeea8a
programs/program_inspection.json, SHA256 fichier :
77f5272d149c37e75b418556f34883727a5896ff795ae999eda8b75b52add1ca
run/selections_frozen.json, SHA256 fichier :
ded3281b54b46055dca32088bf00216c8c8e1a17a4363d82b73ff762701bb038
run/generation_frozen.json, SHA256 fichier :
7a8186436ea6933ae36b78025b3aad157db9dee7d7221d3704441e9ffedb9d53
run/numeric_verification/attempt_001.json.gz, SHA256 fichier :
c7ce88095b91a59bc3d02d0ed6fbb18109fda8178da772698fc0338d79b4483e
Contrôle d’exposition : 1 376 fichiers lus ; SHA256 du JSON canonique
associant chaque chemin relatif dans run/ à son empreinte physique :
ab934322b00f9e74fb28f757e14956b5c85dbf9c2dc633acf5f62a750377e416
```

Aucun code évalué n’a été modifié. Aucune nouveauté dans la littérature, aucun gain de profondeur récursive et aucune amortisation ne sont établis par cette inspection. Le contrat reste portable, avec les limites connues du sous-processus local : ce n’est pas une isolation de sécurité du système d’exploitation.
