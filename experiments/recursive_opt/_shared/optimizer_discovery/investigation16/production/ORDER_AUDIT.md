# Ordre des bras P1 : portée de la rotation enregistrée

Cette vérification descriptive lit uniquement le manifeste, pendant la génération
et avant toute validation ou audit. Elle ne modifie ni l’ordre enregistré ni
l’analyse primaire. Les comptes exacts sont dans [order_audit.json](order_audit.json).

Les six ordres sont I/C/R/W, C/R/W/I, R/W/I/C, W/I/C/R, I/C/R/W et C/R/W/I.
La rotation évite qu’un bras occupe toujours la même position, mais elle ne rend
pas toutes les précédences par paires équilibrées.

| Contraste | Ordre observé dans le protocole | Conséquence |
|---|---|---|
| R−I, central | I avant R dans 3 réplications ; R avant I dans 3 | Précédence équilibrée, sans garantie d’éliminer toute dérive temporelle du service |
| R−C | C avant R dans 5 réplications ; R avant C dans 1 | Une dérive de service pourrait contribuer à ce contraste secondaire |
| W−R | R avant W dans 5 réplications ; W avant R dans 1 | Même limite pour ce contraste entre deux répartitions du budget, qui change largeur et nombre de tours |

Il s’agit d’une limite de contrôle, pas d’une preuve de dérive réelle ni d’un motif
pour supprimer des résultats. Les réglages et la politique de routage restent
communs ; les identités fournisseur, horloges et ressources sont conservées.
La rotation prévue est exécutée sans amendement après le premier appel.

Pour une étude future, apparier chaque ordre complet à son ordre inverse permet
d’équilibrer toutes les précédences par paires avec six réplications (trois ordres
et leurs inverses), tout en conservant les dépendances natives de chaque recherche.
Ce choix d’organisation n’établit aucun gain d’optimisation et n’est pas appliqué
au P1 en cours. Un entrelacement plus fin demanderait de vérifier explicitement
son intégration au moteur ; il ne faut pas créer une seconde boucle pour cela.
