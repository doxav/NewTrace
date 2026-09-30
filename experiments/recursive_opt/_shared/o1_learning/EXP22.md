# EXP22 — apprendre le code et l’instruction d’un optimiseur

**Exécution autorisée le 25 septembre 2026. Pilote imbriqué interrompu par la limite financière de la clé OpenRouter ; confirmation non gelée et non exécutée.** Ce document ne remplace aucun résultat d’EXP20/21 et n’autorise pas à déclarer EXP21 terminé. Les résultats nouveaux figurent ci-dessous ; la proposition initiale et ses limites sont conservées dans les sections suivantes. Aucun changement de production : adaptateur expérimental séparé, chemin Trace existant réutilisé.

## État d’exécution — à lire en premier

Hypothèse mécanique retenue **avant la recherche O1** : PC a le potentiel le plus large, car il associe sélection des preuves et instruction ; P a une surface plus simple et peut être plus fiable à six propositions. PC−B reste le contraste principal. Aucun bras n’est éliminé selon ses premiers scores.

Le [pilote enregistré](exp22/pilot_protocol.json) porte sur 24 nouvelles questions, avec exactement quatre documents remis au lecteur. Ses sources initiales, ses identités de données, les interventions et les critères sont conservés avant appels.

| Intervention fixe, écrite à la main | Exact-match pilote | Bonnes réponses / 24 |
|---|---:|---:|
| Programme initial | 33,33 % | 8 |
| Code de classement amélioré | 41,67 % | 10 |
| Instruction lecteur améliorée | 41,67 % | 10 |
| Code + instruction lecteur | 50,00 % | 12 |
| Documents justificatifs connus — diagnostic privilégié | 58,33 % | 14 |

**Les trois critères de marge passent.** Ces résultats montrent que des interventions admissibles changent la performance ; ils ne démontrent pas un gain O1, une synergie statistique, ni la supériorité de PC. [Résultats bruts du pilote](exp22/pilot_diagnostic/summary.json). Coût fournisseur rapporté : 120 réponses Qwen, 75 597 tokens, 0,0076469 USD.

Le raccordement expérimental apprend deux vrais champs (`update_instruction`, `selector_source`) et réutilise OptoPrimeV2/PrioritySearch. Le résumé natif de Trace, les événements par question et les requêtes exactes sont archivés ; cela ne signifie pas une capture OTEL illimitée. Seuls les exemples sélectionnés et l’agrégat du batch entrent dans le prompt O0. La génération indépendante redémarre depuis le même artefact initial. Les tests vérifient ces propriétés avec réseau simulé, séparément du pilote réel.

### Pilote O1 réel : état exact, sans classement prématuré

| Bras | Proposition O1 reçue | Résultat de génération | Apprentissages enfants |
|---|---:|---|---|
| P — instruction | 1/1 | Mise à jour extraite par le parseur natif | 4/6 réponses sur chaque épisode ; mesures finales incomplètes |
| C — code | 1/1 | Mise à jour extraite et sélecteur exécuté | 3/6 et 4/6 réponses ; mesures finales incomplètes |
| PC — combinaison | 1/1 | Aucune mise à jour extraite par le parseur natif | Baseline conservée ; aucune proposition de remplacement |
| I-PC — indépendant | 1/1 | Source du sélecteur syntaxiquement invalide | Baseline conservée ; aucune réparation manuelle |

Le contrôle B a terminé ses deux épisodes : scores moyens de progression **41,07 % et 37,50 %**, exactitude finale **37,50 % sur chacun**. Ces valeurs sont des mesures pilotes de B, pas un gain relatif à P/C/PC. La couverture détaillée du feedback est de 17 et 21 questions uniques, distincte des 36 questions TRAIN évaluées pour chaque candidat. Une petite mémoire ne garantit pas une couverture de vingt exemples.

**Aucun gagnant O1 n’est établi.** Les propositions P/C montrent que les deux surfaces peuvent être réellement modifiées ; leur efficacité n’a pas encore été mesurée complètement. Une seule proposition PC invalide n’invalide pas la combinaison. [Bilan machine](exp22/execution_summary.json).

### Tokens : diagnostic séparé, aucun ancien résultat remplacé

Les 33 réponses DeepSeek reçues comprennent 27 mises à jour O0, quatre propositions O1 et **deux diagnostics supplémentaires**. Onze finissent à la limite de longueur. Les premières réponses vides examinées ont consommé leurs 16 000 tokens en raisonnement malgré le paramètre natif `reasoning.effort=low` effectivement transmis.

Un [diagnostic enregistré à 32 000 tokens](exp22/compatibility_32000/protocol.json) reprend le premier prompt de chacun des deux épisodes, avec ce seul changement de configuration. Il ne remplace aucune proposition du pilote. Résultat : deux réponses non vides ; une contient 20 683 tokens de complétion, mais omet `</reasoning>` et reste inexploitable par le parseur natif. L’autre, à 11 100 tokens, donne un programme valide sur six exécutions locales de classement. **Le plafond supérieur résout parfois l’absence de texte ; il ne suffit pas à démontrer une génération fiable.** [Validation corrigée avec le parseur natif](exp22/compatibility_32000/local_validity_v2.json). Aucun réglage du pilote initial n’a été changé silencieusement.

L’audit initial de ces deux diagnostics avait mal déballé le dictionnaire retourné par le parseur et vérifiait le programme initial. Ce rapport est conservé et explicitement remplacé par v2 ; aucune génération, sélection ou mesure scientifique n’a utilisé cet audit erroné.

### Blocage, corrections et reprise

**Le fournisseur a refusé quatre requêtes avec HTTP 403 « Key limit exceeded ».** Au dernier [précontrôle expurgé](exp22/credit/002.json), la clé a consommé **14,073360487 USD** pour une limite de **14 USD** ; sa marge est nulle. Le compte dispose encore de **7,316813831 USD**, mais cette clé ne peut plus les utiliser. Le premier précontrôle signalait déjà cette contrainte ; les derniers appels concurrents expliquent le dépassement limité enregistré par le fournisseur.

Cumul propre à EXP22 : **33 réponses DeepSeek + 730 réponses Qwen**, **1 102 402 tokens**, **0,697837908 USD rapportés**, quatre refus conservés et aucune requête de complétion distante indéterminée encore en vol. Les deux diagnostics à 32 000 tokens sont inclus dans ces nombres.

La reprise a également révélé deux défauts de métadonnées, corrigés par tests : tuple/liste JSON dans l’archive du parent O1, et collision entre rapport interrompu et rapport final. L’[amendement d’ingénierie](exp22/engineering_resume_amendment.json) conserve les sources avant/après et les preuves initiales. Les prompts, tâches, modèles, scores et budgets ne changent pas. Les refus explicites sont autorisés séparément [par chemin et hash](exp22/resume_rejections.json) ; une réponse complétée ou une requête ambiguë ne peut pas être remplacée.

Après augmentation de la limite de la clé, la commande suivante reprend le pilote, avec **neuf réponses O0 restant à obtenir**, puis les évaluations TRAIN/VALIDATION/MESURE associées :

```bash
/home/xav/miniconda3/envs/humanllm/bin/python -m experiments.recursive_opt.o1_qa.open_optimizer pilot
```

La campagne complète reste à faire : clôture du pilote et décision documentée sur la fiabilité de génération, gel final, 24 propositions O1 de découverte, sélection META-VALIDATION, puis confirmation sur six épisodes. Relever la limite de clé vers 50 USD et disposer d’environ 30 USD de crédit de compte donnerait une marge de planification ; ce n’est pas une dépense cible. **Ne pas appeler la confirmation terminée, positive ou nulle tant que ces étapes manquent.**

Vérification finale hors ligne : **981 tests réussis, deux ignorés**, dont les vingt nouveaux tests EXP22 ; durée 190,15 s. Les exclusions concernent Graphviz absent et un exemple nécessitant un vrai fournisseur ; un avertissement de dépréciation LangChain subsiste. Ruff, Black et `git diff --check` passent. Les hashes de sources correspondent à l’amendement, les comptes sont recalculés depuis les reçus, le travail déjà indexé est inchangé et le scan de la valeur réelle de la clé ne trouve aucune fuite. [Commande exacte et contrôles](exp22/verification.json), [journal complet des tests](exp22/final_offline_suite.log). Aucun commit créé.

La recommandation est de tester trois variantes O1 : **instruction libre**, **code de sélection des preuves**, puis **les deux ensemble**. L’objectif est d’obtenir une meilleure courbe d’apprentissage O0 sur de nouvelles questions, à nombre d’appels comparable. Le potentiel est motivé ; un gain important n’est pas acquis.

## 1. Ce que l’audit établit réellement

| Question | Conclusion | Preuve et limite |
|---|---|---|
| EXP21 apprend-il des surfaces ouvertes au niveau O1 ? | Non : il prévoit de choisir des valeurs dans des menus. Le produit cartésien brut représente **51 840 configurations**, avec des choix parfois redondants ou conditionnels. | [`DOMAINS`, axes.py](../../experiments/recursive_opt/o1_qa/axes.py), recalcul par AST. Trois propositions O1 étaient prévues : elles ne couvrent évidemment pas cet espace. |
| Et au niveau O2 ? | Seulement **quatre configurations** : deux optimiseurs × deux mémoires. Une comparaison exhaustive serait plus lisible pour cette question étroite, si son coût descendant est acceptable. | [`specification`, meta.py](../../experiments/recursive_opt/o1_qa/meta.py). Deux propositions O2 prévues. |
| Ces restrictions expliquent-elles un échec observé d’O1/O2 ? | **Pas démontré : O1/O2 n’ont pas été exécutés réellement dans EXP21.** Les résultats disponibles comparent des configurations choisies à l’avance. | [Bilan primaire](exp21/continuation_summary.json) : 219/252 réponses de développement, 33/42 chaînes complètes, O1/O2 et confirmation non commencés. |
| L’instruction de méta-optimisation a-t-elle été oubliée ? | **Non. Elle est adaptée, mais fixe.** O1 reçoit une instruction sur la vitesse d’apprentissage de l’optimiseur inférieur et les domaines autorisés. | [`meta.specification`](../../experiments/recursive_opt/o1_qa/meta.py), puis [`OptoPrimeV2.problem_instance`](../../opto/optimizers/optoprime_v2.py). Vérification hors ligne : instruction présente dans le prompt rendu pour O1 et O2. |
| Toutes les expériences antérieures étaient-elles limitées à des boutons ? | Non. EXP15–18 généraient du code `propose(history, bounds, seed)`. EXP20 apprenait du code, une instruction, un entier et un booléen. EXP19 utilisait une table de vérité finie : une chaîne de caractères ne constitue pas nécessairement une surface linguistique ouverte. | [EXP15](../optimizer_discovery/EXP15_REPORT.md), [EXP19](EXP19.md), [EXP20](EXP20.md). Le code libre est déjà testé ; il ne suffit pas à garantir l’avantage du feedback. |
| Le gain EXP20 prouve-t-il la qualité du code appris ? | Il prouve une amélioration du **programme complet**, pas l’apport isolé de son code. **5/6** programmes standard sélectionnés utilisent les dix documents ; **8/12** en incluant le curriculum. | Recalcul des six [sélections brutes](exp20/full/confirmation/selection). Il manque un contrôle Qwen adéquat isolant le passage à dix documents. Leur ordre peut encore compter : le classement n’est pas rendu totalement inutile. |
| Les tokens ou les traces sont-ils innocentés ? | Non. EXP21 conserve 15 réponses vides et 18 fins `length` sur 219, catégories pouvant se recouper. Une troncature peut empêcher l’apprentissage ; cela ne prouve pas qu’un plafond supérieur produirait un gain scientifique. | [EXP21](EXP21.md). Vérifier séparément la complétude de sortie, la troncature du code présenté et l’information effectivement transmise. Augmenter le plafond ne répare pas un mauvais signal. |
| La supériorité bayésienne est-elle démontrée localement ? | **Non : aucune comparaison correspondante n’a été réalisée ici.** C’est un contrôle pertinent pour les menus finis, pas une conclusion tirée de ces résultats. | Séparer l’efficacité de sélection de configurations et la capacité à inventer de nouveaux programmes. |

Deux résultats historiques empêchent les conclusions excessives : EXP20 standard passe de 28,47 % à 47,92 % d’exactitude finale, mais sans attribution causale entre ses quatre composants. EXP15, déjà en code libre, donne A2−A1 = −0,005068 de regret-AUC, intervalle [−0,037728 ; +0,024551] : pas d’avantage clair du feedback récursif sur la génération indépendante dans ce protocole.

## 2. Quel outil pour quelle surface ?

| Surface | Approche à privilégier | Ce qu’on ne peut pas en déduire |
|---|---|---|
| Quatre choix prédéfinis | Énumération avec répétitions appariées | Inutile d’attribuer de la créativité au choix d’un nom d’optimiseur. |
| Nombreux paramètres numériques/catégoriels et conditionnels | Recherche aléatoire comme référence ; SMAC/TPE ou autre optimisation adaptée si le budget le permet | Une méthode bayésienne n’est pas automatiquement supérieure avec trois observations bruitées. |
| Nouvelle instruction en langage naturel | Génération et réécriture sémantique par LLM | Une meilleure formulation ne crée pas les preuves absentes du contexte. |
| Nouveau programme dépendant de l’historique | Synthèse de code, exécution réelle, feedback et tests de validité | Du code qui renvoie toujours une constante reste, fonctionnellement, un réglage constant. |
| Combinaisons de programmes/prompts déjà proposés | Sélection statistique possible, y compris bayésienne | Elle sélectionne des candidats ; elle n’invente pas nécessairement leur contenu. |

Cette séparation est cohérente avec [SMAC3](https://www.jmlr.org/papers/v23/21-0888.html). Les approches sont aussi complémentaires : [MIPRO](https://arxiv.org/html/2406.11695v2) combine propositions d’instructions par LLM et sélection de combinaisons par TPE. [Promptbreeder](https://arxiv.org/abs/2309.16797) fait évoluer des instructions de tâche et des instructions de mutation. Ces travaux motivent EXP22 ; ils ne prédisent pas ses gains et interdisent de présenter l’idée générale comme nouvelle.

## 3. Les niveaux, avec un exemple concret

```text
Question + dix documents
        ↓
programme O0 : code de classement → quatre documents → instruction du lecteur
        ↓                                              ↓
traces TRAIN                                    réponse Qwen / exact-match
        ↓
optimiseur O0 : choisit les preuves → instruit DeepSeek → modifie le programme
        ↑
O1 apprend le CODE qui choisit les preuves et/ou l’INSTRUCTION de cet optimiseur
```

Exemple : le lecteur échoue car les deux documents nécessaires n’arrivent pas ensemble dans son contexte. Un optimiseur utile doit reconnaître cette cause et proposer une modification du classement ; réécrire seulement « raisonne mieux » ne restitue pas un document absent.

Il faut distinguer quatre objets :

1. `answer_instruction` : instruction au lecteur Qwen, dans le **programme O0**.
2. `update_instruction` : instruction à DeepSeek pour modifier ce programme ; **artefact appris par O1**.
3. Instruction d’O1 : explique comment améliorer l’optimiseur O0 à partir de ses apprentissages mesurés ; elle reste fixe dans EXP22.
4. Score exact-match, règle de sélection et budgets : évaluateur de confiance, jamais modifiable par les candidats.

**Apprendre un “goal” signifie ici apprendre la manière de demander une amélioration, pas changer la définition de la réussite.** La méta-optimisation O1 exécute réellement des apprentissages O0. EXP22 n’ajoute pas O2 ; un résultat positif ne démontrerait pas l’intérêt d’une profondeur supplémentaire.

## 4. Les trois variantes proposées

Toutes partent de la même instruction déjà adaptée à la tâche et du même sélecteur de preuves écrit à la main. Le contrôle ne doit pas être rendu artificiellement faible.

| Variante | Ce qu’O1 peut inventer | Ce qui reste fixe | Hypothèse testée |
|---|---|---|---|
| **P — instruction** | Texte libre `update_instruction`, sans menu de formulations | Code de sélection des preuves | Une instruction de modification apprise améliore la vitesse d’apprentissage O0. |
| **C — code** | Fonction Python `select_evidence(events, limit)` qui choisit et ordonne des preuves TRAIN | Instruction d’optimisation adaptée | Une sélection contextuelle des erreurs, réparations et régressions améliore les modifications proposées. |
| **PC — combinaison** | Les deux composants précédents, dans le même budget O1 | Évaluateur, modèle, captures, rendu, appels et paramètres de recherche | Leur combinaison apporte un gain utile ; une interaction positive n’est pas présupposée. |

Le code de C est une **partie réelle de l’optimiseur**. Par exemple, il peut regrouper les échecs de récupération similaires, conserver une régression après modification du prompt et choisir un exemple réparé servant de garde-fou. Il ne choisit pas simplement `trace_mode="full"` dans un menu.

C’est volontairement un composant délimité : EXP22 ne teste pas toutes les architectures d’optimiseur. Réécrire le trainer, le parseur et l’évaluateur en même temps rendrait la cause d’un gain difficile à identifier.

**Contrôles nécessaires, en plus des trois variantes :**

- **B — optimiseur fixe exigeant** : instruction adaptée et sélection stratifiée déterministe de preuves. Écriture et gel avant toute comparaison P/C/PC.
- **I-PC — propositions O1 indépendantes** : même surface code+instruction, même point de départ, mêmes informations invariantes et six réponses ; aucun résultat ou candidat précédent dans la génération. Évaluations et sélection identiques à PC.
- Programme O0 initial sans apprentissage : référence commune, sans recherche supplémentaire. Chaque variante se lit ainsi en three-way : initial / B / variante apprise.

PC−B mesure le bénéfice du composant appris. **PC−I-PC** est nécessaire pour discuter l’apport du feedback O1, avec la limite de réplication exposée plus bas. P−B et C−B isolent les surfaces autorisées. Le contraste PC−P−C+B décrit leur interaction à budget de recherche fixé ; ce n’est pas une preuve générale de synergie.

## 5. Tâche et traces : une surface réellement utile

Je recommande de conserver le harness HotpotQA, mais avec **contexte contraint**, plutôt que de reconstruire encore un benchmark. Qwen `qwen/qwen-2.5-7b-instruct` reste le lecteur ; DeepSeek `deepseek/deepseek-v4-flash-0731` reste l’optimiseur aux deux niveaux.

- O0 apprend seulement `ranker_source` et `answer_instruction`.
- `top_k=4`, `bridge_expansion=False`, un appel lecteur par réponse, limite de contexte et de sortie fixes. Aucune possibilité de transmettre les dix documents par un autre champ ; instruction indépendante de la question, pas de réponses mémorisées.
- Questions nouvelles, équilibrées liaison/comparaison ; exclure **tous** les ensembles EXP20/21 et les doublons de question/paire justificative. Les documents individuels peuvent se recouper, à documenter.
- Chaque épisode : 36 TRAIN, 24 VALIDATION, 48 MESURE. Batch TRAIN de six, calendrier partagé ; six mises à jour O0. Toutes les sources, réponses invalides et baisses restent conservées.
- Données de diagnostic : documents classés/retenus, rappel des pièces justificatives sur TRAIN, réponse, format, exact-match, changement de composants, avant/après sur la **même question** lorsque réellement évalué. Une erreur de réponse malgré des documents pertinents reste un diagnostic, pas une preuve parfaite de la cause interne du lecteur.

Le sélecteur reçoit des événements TRAIN immuables de l’apprentissage en cours. Il renvoie **au plus cinq identifiants uniques** ; le rendu fixe ajoute les détails des événements correspondants. Le score agrégé du batch reste toujours présent. Le code ne peut inventer ni score ni observation ni nouvelle instruction : cela préserverait mal la séparation C/P.

Le contrôle B choisit, lorsqu’ils existent, une régression récente, un échec de récupération, un échec de réponse malgré la présence des pièces, un cas réellement réparé, puis un cas distinct récent ; déduplication et règle de départage fixes. C peut apprendre une meilleure allocation de ces cinq places.

Les transitions échec→réussite proviennent d’évaluations avant/après comparables ; pas d’étiquetage déduit de scores globaux. C choisit les **preuves montrées**, pas les questions évaluées : c’est une mémoire de feedback, pas une nouvelle comparaison du CurriculumBuffer qui remplace des éléments du batch. Modifier aussi ce batch constituerait une expérience différente.

Capture interne complète et horizon couvrant les six mises à jour, avec archivage des données brutes. Les vues courtes sont construites à partir de ces données. Vérifier les appels et valeurs réellement capturés ; le nom « full » ou OTEL ne garantit pas leur présence. Même budget de texte pour tous les bras. Le sélecteur ne doit pas être contourné par des exemples supplémentaires dans `#Inputs`, `#Others`, la mémoire d’OptoPrime ou un autre champ du prompt. Publier les identifiants et le nombre d’exemples uniques réellement vus.

HotpotQA est ici un précurseur de recherche avec récupération d’informations et erreurs de plusieurs composants. **Ce n’est ni une tâche d’optimisation matricielle, ni une validation de recherche scientifique à long horizon.** Les fonctions numériques d’EXP15 conviennent au code d’une politique numérique, moins aux interactions retrieval/instruction visées ici. BBEH reste une autre famille possible, mais aucun classement empirique ne permet d’affirmer aujourd’hui qu’elle donnerait plus de gain O1.

## 6. Pilote et critères d’entrée

Avant la recherche O1, utiliser des questions pilotes séparées pour vérifier :

1. **Marge accessible** : comparer le programme initial à un classement lexical raisonnablement amélioré et à une instruction lecteur adaptée, tous écrits et figés avant mesure. Décomposer code seul / prompt seul / combinaison. Cela indique si des interventions admissibles changent effectivement le résultat.
2. **Plafond du lecteur** : fournir les pièces justificatives connues dans quatre documents comme diagnostic privilégié uniquement. Ne jamais les exposer au programme de récupération ni utiliser ce contrôle comme concurrent à information égale.
3. **Action effective de P/C** : deux instructions produisent des requêtes O0 différentes ; deux sélecteurs produisent des preuves différentes dans ces requêtes, sans modifier scores ni données cachées.
4. **Chaîne imbriquée réelle** : une proposition pilote par variante et par contrôle indépendant, évaluée sur les épisodes pilotes ; aucun mock comme preuve scientifique.
5. **Faisabilité** : mesurer validité, sorties `length`, troncature des entrées, temps et coût de la chaîne entière, pas seulement de DeepSeek. Le plafond d’EXP21 est déjà 16 000 tokens : ne pas repartir à 3 000. Toute modification de plafond ou routage est symétrique et gelée après ce pilote.

Si même les interventions écrites à la main n’améliorent pas le programme et que le diagnostic du lecteur est bas, cette tâche ne justifie pas une campagne méta coûteuse. Si l’oracle seul progresse, la marge existe, mais son accessibilité par les surfaces proposées reste incertaine. Ces constats motivent une révision documentée avant gel, pas un déplacement du but après les résultats.

Pas d’arrêt anticipé d’un bras parce qu’il perd. Arrêt pour défaillance d’intégration, données inexploitables ou fournisseur indisponible ; reprise des seuls créneaux non complétés. Aucun essai complet invalide remplacé gratuitement.

## 7. Budget proposé, portée et analyse

Le budget ci-dessous vise d’abord **la transférabilité des optimisateurs sélectionnés**, pas une estimation robuste de toute la distribution des recherches O1. Une recherche O1 par variante est une limite explicite, même avec six épisodes de confirmation. Affirmer une supériorité générale de la méthode de recherche demanderait plusieurs découvertes O1 indépendantes.

| Étape après pilote | Allocation proposée |
|---|---|
| Découverte O1 | Six réponses pour chacun de P/C/PC/I-PC : **24 réponses O1** |
| Évaluation de découverte | Chaque candidat apprend O0 sur deux épisodes META-TRAIN séparés ; six réponses O0 par épisode |
| Sélection O1 | Tous les candidats et B évalués sur un épisode META-VALIDATION distinct ; sélection selon la courbe moyenne, B puis proposition la plus ancienne en cas d’égalité |
| Confirmation | B/P/C/PC/I-PC sur **six nouveaux épisodes appariés**, six réponses O0 chacun ; aucune nouvelle modification O1 |
| Référence | Programme O0 initial évalué dans chaque épisode, sans réponse d’optimisation |

Sans réutilisation entre candidats, le maximum hors pilote est **630 réponses O0 + 24 réponses O1 = 654 réponses d’optimisation** : découverte `4×6×2×6 + 2×6 = 300`, sélection `4×6×6 + 6 = 150`, confirmation `5×6×6 = 180`. Les termes supplémentaires évaluent B, partagé. Les candidats identiques peuvent être réutilisés selon une clé complète gelée ; ils consomment tout de même leur proposition. Les refus réseau et retries restent séparés.

Il faut ajouter les appels Qwen : **654 n’est pas le nombre total d’appels**. Une borne provisoire conservatrice de 828 réponses lecteur par chaîne O0 couvre sept candidats sur 36 TRAIN + 24 VALIDATION, sept préfixes sur 48 MESURE et 72 évaluations internes supplémentaires. Sur 105 chaînes cela représente jusqu’à **86 940 réponses Qwen**, hors pilote. Le pilote doit auditer le calendrier réel, supprimer les doublons par cache sûr et remplacer cette borne par un budget vérifié avant gel. Si le chemin de production exige davantage d’évaluations, ne pas les masquer.

Cela explique pourquoi une méta-optimisation est coûteuse même avec peu de propositions. **Je ne garantis pas cette confirmation complète en une heure.** Utiliser 8–10 workers pour les épisodes/candidats indépendants, respecter les dépendances O1 et les limites fournisseur ; le parallélisme ne réduit pas le coût total. Toute réduction de portée doit être décidée avant confirmation, jamais après observation d’un résultat défavorable.

META-TRAIN/META-VALIDATION désignent des **épisodes d’apprentissage entiers**. Chaque épisode contient lui-même TRAIN/VALIDATION/MESURE : O0 apprend sur TRAIN et sélectionne ses préfixes sur VALIDATION. Sur META-TRAIN, les scores MESURE constituent le feedback de recherche O1. Sur META-VALIDATION, ils servent seulement à sélectionner l’artefact O1, sans revenir dans sa génération. Sur META-TEST, ils restent fermés jusqu’au gel de tous les artefacts O1 et de tous les préfixes O0. Ainsi, « MESURE » n’est pas toujours le holdout scientifique final : c’est le rôle de l’épisode qui le détermine.

Score principal par épisode : moyenne des exactitudes MESURE des sept politiques choisies sur VALIDATION après `0…6` réponses O0. Les réponses vides ou invalides consomment un créneau ; le dernier programme valide reste disponible. Pas de sélection du meilleur point sur MESURE.

Contrastes : P−B, C−B, PC−B et PC−I-PC ; valeurs positives favorables. Publier les six différences par épisode et leur moyenne. Bootstrap apparié sur les épisodes, 10 000 tirages, graine proposée 22099 ; intervalle à 95 %, descriptif et conditionnel aux artefacts O1 sélectionnés. Ne pas traiter les questions, préfixes ou tokens comme des réplications de découverte O1. Ne pas choisir après coup le contraste le plus favorable.

**Objectif pratique proposé**, non prévision : au moins **+5 points** de score principal PC−B, avec borne inférieure de l’intervalle bootstrap supérieure à zéro ; et, secondairement, atteindre en quatre mises à jour la performance finale du contrôle à six, sans sacrifier la performance finale. Le contraste PC−B est principal ; les autres restent secondaires et sont tous publiés. Même si ce critère est atteint, six épisodes et une découverte O1 n’autorisent pas une affirmation générale de supériorité. Un seuil d’atteinte absolu peut être fixé sur pilote ; les non-atteintes restent censurées. Un effet positif contre B mais incertain contre I-PC ne démontre pas l’utilité du feedback O1.

Rapporter les coûts de découverte et d’apprentissage transféré séparément, les tokens réels, invalidités, complétions tronquées, temps, diversité comportementale et sélection éventuelle de B. Une courbe O0 meilleure ne prouve pas un amortissement du coût O1. Celui-ci ne peut être calculé qu’à partir d’économies mesurées à qualité comparable et d’un nombre explicite de réutilisations.

## 8. Implémentation minimale à préparer après validation du périmètre

Réutiliser `task.py`, l’enregistreur/reprise, le contrôle plane, les évaluateurs et la sélection d’EXP20/21 ; **ne pas modifier leurs versions scientifiques gelées**. Ajouter uniquement l’artefact O1 à deux champs et son raccordement testé.

- Le texte appris est passé explicitement à `levels[0].engine.config.optimizer_kwargs.objective` de l’optimiseur enfant avant son instanciation. Modifier seulement `objective.intent` ne suffit pas.
- La source apprise est exécutée pour sélectionner les identifiants de preuves, puis un rendu commun alimente le prompt d’OptoPrimeV2. Un adaptateur étroit au point de construction du prompt peut être nécessaire ; cette fonction n’existe pas déjà sous prétexte qu’on a écrit son nom dans un dict.
- Conserver le chemin de mise à jour, le parseur et le trainer de production. Pas de deuxième moteur de recherche.
- Source Python vérifiée, exécution bornée en sous-processus, environnement sans clés, entrées JSON et sorties typées. Cette frontière n’est pas un sandbox de sécurité système. Aucun appel LLM ou I/O candidat.
- Tester changements réels de requêtes, absence de fuite validation/test, reprise, hashes, compteurs, limites et exécution des mêmes évaluations dans tous les bras. Un sélecteur invalide est une invalidité de candidat ; aucune réparation manuelle.
- Inclure B dans tous les pools. Une source invalide ou qui échoue pendant les évaluations de découverte/sélection reste conservée et n’est pas éligible ; aucun score artificiel ne lui est imputé. Pendant le transfert, si le sélecteur choisi échoue, utiliser B pour la sélection de preuves et journaliser le fallback, sans appel ou budget supplémentaire. Une panne du mécanisme de confiance est un défaut d’ingénierie.

L’instruction O1 fixe doit demander d’améliorer **l’apprentissage de nouveaux programmes**, expliquer la métrique de courbe, les surfaces code/instruction, les traces disponibles et les coûts descendants. Elle ne doit pas demander de résoudre directement les questions. La représentation Trace et le format XML peuvent rester fixes ; leur généralité n’est pas en elle-même un défaut.

## 9. Vérifications réalisées pour cette proposition

- Lecture des sources de domaines, spécifications O0/O1/O2 et construction effective des prompts.
- Recalcul direct dans les sélections brutes EXP20 : `top_k=10` dans 5/6 standard, 3/6 curriculum.
- Inspection AST du domaine O1 : 51 840 combinaisons brutes.
- Construction hors ligne de prompts OptoPrimeV2 : instruction adaptée O1 présente (1 012 caractères), O2 présente (594 caractères) ; objectif O0 explicite transmis aux kwargs. Aucun modèle appelé.
- Commande : `env -u OPENROUTER_API_KEY -u OPENAI_API_KEY /home/xav/miniconda3/envs/humanllm/bin/python -m pytest tests/unit_tests/test_o1_qa_axes.py -q -k 'registered_variants or training_spec or meta_validation or real_meta_engine or actual_control_plane'` : **5 passed, 21 deselected**, 11,18 secondes. Les tests simulent le réseau ; ils vérifient le branchement, pas les gains.

La prochaine action utile est le pilote de marge et de raccordement décrit ci-dessus. Finir EXP21 reste possible comme question distincte ; ce n’est pas une condition nécessaire pour tester cette nouvelle surface. Aucun résultat antérieur n’est invalidé uniquement parce qu’EXP22 propose une meilleure question.
