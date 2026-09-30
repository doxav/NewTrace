# P1-E1 — validation indépendante du chemin de production

**Gate d’ingénierie passé.** `D.require_engineering()` a été exécuté à nouveau
sur les preuves conservées : deux réponses complétées, une proposition générée
éligible sur tout le panel d’apprentissage, reprise sans client vérifiée. Aucun
blocage d’ingénierie identifié par ce contrôle. Ce résultat ne mesure aucun
avantage du feedback et ne constitue pas une comparaison de performances.

Le [protocole gelé](PROTOCOL_P1_E1.md) fixe le namespace `P1-E1`, le seul bras R,
la seed externe 16601, 24 instances d’apprentissage × 2 seeds locaux, B = 32,
huit workers d’évaluation et une génération séquentielle. Ni les programmes ni
les données de ce contrôle ne sont transférés dans P1.

- Gel : `4be570f58e0522d76d17e9f67b9bb1338eed060dc800319c5b083f23fa1abe75`.
- Résultat canonique : `7ea3c7acdd68b4397554cda189c7663635c4347abcd0af0781d581d801017f20`.
- Archive des 93 fichiers sources et dépendances :
  `ea11026cc21bf4fe5c589522ba97f57899f363e34dd5803dbbdc7430b2bbf04a`.

## Réponses, prompts et échec conservé

Les deux requêtes utilisent exactement `deepseek/deepseek-v4-flash-0731`,
température 0,6, `top_p=1`, plafond de 32 000 tokens, champ natif
`reasoning.effort=low`, timeout 300 s, cache désactivé et aucun retry automatique
pour réponse vide. Le reçu de chaque génération identifie OpenInference et
rapporte le modèle résolu `deepseek/deepseek-v4-flash-20260731` ; les deux
identifiants sont conservés, sans substitution décidée par l’expérimentateur.

| Mesure | Slot 00 | Slot 01 |
|---|---:|---:|
| Caractères des deux messages | 141 807 | 141 807 |
| Tokens de prompt rapportés | 58 435 | 58 435 |
| Tokens de complétion rapportés | 4 873 | 5 531 |
| Tokens de raisonnement rapportés | 3 493 | 4 748 |
| Tokens totaux rapportés | 63 308 | 63 966 |
| Coût rapporté, USD | 0,001683524 | 0,001788804 |
| Durée active de génération, secondes | 224,751 | 247,621 |
| Fin de réponse | `stop` | `stop` |
| Statut du source | `syntax_error` | `valid` |
| Trajectoires d’apprentissage valides | 0/48 | 48/48 |

Le premier source contient, ligne 17, `if ranges[i] == 0 for i in range(dim):`,
expression syntaxiquement invalide. La réponse finit normalement sous le plafond :
cet échec précis n’est pas une troncature par la limite de tokens. Ses 48
allocations restent invalides typées, sans valeur objective inventée, sans
exécution de sous-processus et sans réparation manuelle. Le slot suivant est la
seconde proposition déjà allouée, pas un remplacement supplémentaire.

Les deux prompts contiennent les mêmes messages : 2 603 et 139 204 caractères,
soit un total inférieur au plafond gelé de 524 288. Ils consomment le feedback
réellement propagé par Trace ; leurs hashes de parent et de feedback concordent.
Le parent reste le seed initial après l’échec du slot 00. **Ce contrôle n’a donc
pas exercé un prompt construit après plusieurs remplacements du parent.** Une
réussite sur deux propositions n’est pas une estimation fiable du taux de
réussite futur, et le contrat n’est pas garanti par l’augmentation du plafond.

## Ordonnancement et comptabilité réels

Le plan conservé utilise Control Plane, `PrioritySearch`, un parent par tour,
une proposition par parent et trois itérations : initialisation puis deux
callbacks de mise à jour. Les deux événements `trace_update` correspondent
exactement aux deux réponses, sans replay de réponse pendant la génération.
Une ligne d’entrée Trace représente ici le panel complet de 48 trajectoires.
Les vérifications internes des nouveaux candidats utilisent l’apprentissage ;
elles ne sont pas des évaluations du split VALIDATION.

| Ressource | Quantité vérifiée |
|---|---:|
| Propositions / tentatives de transport | 2 / 2 |
| Retries de transport / réponses remplacées | 0 / 0 |
| Entrées de pool, seed compris | 3 |
| Trajectoires logiquement allouées | 144 |
| Évaluations objectives allouées | 4 608 |
| Évaluations objectives réellement conservées | 3 072 |
| Allocations objectives inutilisées après invalidité | 1 536 |
| Sous-processus, replay déterministe compris | 6 144 |
| Entrées de cache distinctes | 144 |
| Consultations de cache / hits / misses | 1 104 / 960 / 144 |
| Réponses avec usage et reçu fournisseur conservés | 2/2 |

Les consultations comprennent la préparation du contexte et les accès répétés
du chemin de production : 720 pour le seed et 192 pour chaque source généré.
Ces accès n’ajoutent aucune opportunité de recherche ni évaluation objective.
Le seed et le second programme terminent chacun leurs 48 trajectoires ; aucun
timeout, exception d’exécution ou fallback n’est enregistré. La fraction de
programmes générés invalides est 1/2, distincte des 48/144 trajectoires invalides
dans le pool qui inclut le seed.

Les 24 instances impliquent séparément 3 072 appels pour un design unique de
normalisation. Les reconstructions physiques de ce design entre processus ne
sont pas entièrement instrumentées : ce chiffre ne doit pas être présenté comme
le nombre exact de tous les appels de référence réellement exécutés.

## Reprise, isolation et portée temporelle

Le contrôle enregistré rejoue le run terminé sans client et compare les preuves.
La vérification indépendante a en outre interdit, en mémoire, `B.evaluate`,
`B.objective`, la création de client, le chargement de clé et l’accès réseau avant
d’appeler `D.require_engineering()`. Aucun de ces chemins n’a été invoqué et les
1 277 fichiers E1 sont identiques octet par octet avant et après. Les sources,
requêtes, réponses, allocations, cache et plan Trace passent les contrôles gelés.
Cette preuve de reprise d’un run terminé ne simule pas toutes les interruptions
possibles pendant une requête distante.

Les datasets de validation et de holdout du plan sont vides. Toutes les entrées
de cache et tous les événements d’évaluation portent `train` ; aucun gel de
sélection, résultat d’audit ou évaluation de validation n’existe dans ce run.
Aucun résultat de fitness n’est utilisé dans ce rapport ou pour changer P1.

Les horloges monotone, murale et de démarrage concordent pendant les deux
générations : aucune suspension ni remise à zéro détectée. Leur durée active
totale est 472,372 s. De la première génération à la fin de recherche :
499,848 s, dont 27,271 s après la seconde réponse pour l’évaluation finale et les
écritures associées. La préparation préalable du seed a pris 22,668 s ; l’attente
entre cette préparation et le lancement ne constitue pas du temps d’exécution.
Le cumul des durées d’évaluation des trajectoires est d’environ 376,888 s, en
parallèle : il ne faut pas l’additionner comme une durée séquentielle au run.

## Extrapolation de ressources, sans promesse de gain

Le total observé est de 116 870 tokens de prompt, 10 404 de complétion,
127 274 tokens totaux et 0,003472328 USD. Les 8 241 tokens de raisonnement sont
rapportés séparément et ne sont pas ajoutés aux tokens totaux. Les reçus emploient
aussi des compteurs normalisés différents ; ceux-ci ne remplacent pas les
compteurs natifs alignés avec les réponses et ne sont pas additionnés entre eux.

Si 192 réponses avaient exactement le même profil moyen, cela représenterait
environ 12,60 h de génération, 12,22 millions de tokens et 0,3333 USD. **C’est une
extrapolation descriptive sur deux appels au même fournisseur, pas un devis ni
une borne.** I/C ont des prompts différents ; les futurs parents et traces de
R/W peuvent être plus longs, et le service peut changer de latence ou de prix.
Le timeout 300 s n’est pas une échéance absolue de génération.

À cela s’ajoutent les évaluations, validations, audit, écritures et éventuels
retries. La projection T1 de 1,86 h pour 16 272 trajectoires à huit workers reste
fondée sur des politiques fixes ; le panel final E1 a pris environ 27,27 s pour
48 trajectoires, écritures comprises. Ni ce panel ni T1 ne bornent le coût de
programmes futurs plus complexes. E1 permet de franchir la condition préalable
d’ingénierie enregistrée ; il n’établit ni fiabilité à grande échelle, ni
généralisation, ni amélioration scientifique du feedback.

Preuves : [résultat E1](../production_engineering/engineering_results.json),
[allocations](../production_engineering/raw/16601/R/allocations_train.json),
[ordonnancement](../production_engineering/raw/16601/R/schedule.json),
[préparation du contexte](../production_engineering/context_preparation.json).
