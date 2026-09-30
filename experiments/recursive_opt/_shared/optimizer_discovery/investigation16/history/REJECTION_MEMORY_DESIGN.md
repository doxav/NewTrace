# Mémoire des tentatives rejetées : intervention future à tester

Cette revue porte uniquement sur les sources du dépôt. Aucun résultat, aucune réponse et aucune évaluation de P1 n’ont été consultés pour la rédiger. Elle ne modifie ni P1 ni ses fichiers gelés et ne constitue pas un nouveau protocole exécuté.

Le diagnostic d’ingénierie communiqué pour E1 — deux contenus de prompt identiques après une proposition rejetée pour erreur de syntaxe — est compatible avec le fonctionnement actuel : le prochain appel reçoit le code du parent retenu et son feedback TRAIN. Si ce parent reste inchangé, le code défaillant de l’enfant n’est pas nécessairement présenté au prochain appel. Cela établit une limitation du canal d’information ; cela ne démontre ni que ce canal supplémentaire améliore la recherche, ni qu’il explique à lui seul les résultats d’EXP-15.

## Où intervenir, en réutilisant le chemin existant

Les points d’extension suivants sont identifiés dans les sources, sans les modifier :

| Point existant | Rôle actuel | Modification future minimale |
| --- | --- | --- |
| `trace_schedule.py:generate_recursive`, classe locale `SlotOptimizer._step` | Appelle l’optimiseur depuis le vrai moteur Trace/Control Plane et transmet `self.trace_graph.user_feedback` au propriétaire. | Conserver ce chemin et son nombre d’appels. Aucun second moteur de recherche. |
| `search_experiment.py:SearchExperiment.proposal_from_trace` | Vérifie et préserve le feedback réellement propagé, lié au hash du parent. | Construire ici un contexte supplémentaire à partir des seules tentatives antérieures autorisées. Préserver séparément le texte Trace et son hash. |
| `SearchExperiment._proposal` | Assemble et enregistre les messages exacts, puis appelle le recorder immuable. | Ajouter une section de mémoire déterministe, son hash et son instantané de provenance au prochain prompt. |
| `SearchExperiment.panel` | Évalue, ou relit un cache vérifié, selon la tâche, la source, les seeds, le budget et la version. | Extraire le calcul et la validation de la clé dans un petit helper partagé ; fournir une lecture stricte sans évaluation sur cache manquant. Enregistrer un reçu des évaluations TRAIN effectivement réalisées. |
| `feedback/rich_feedback.py:build_feedback` | Projette les résultats bruts autorisés ; accepte déjà un `previous_source` avec ses `previous_rows`, vérifiés ensemble. | Réutiliser ce constructeur pour l’alignement d’une tentative antérieure, puis une petite enveloppe bornée pour plusieurs tentatives. |
| `search_experiment.py:verify_arm_responses` et barrières de gel | Reconstruisent exactement les prompts et vérifient leur chronologie et leur identité. | Étendre la reconstruction au contexte de mémoire et à ses reçus ; conserver tous les contrôles actuels. |

Ajouter simplement une liste à `SearchExperiment.feedback()` ne suffit pas. Le graphe du parent peut avoir été construit avant le dernier rejet ; son feedback propagé peut alors rester identique. La projection au moment de `proposal_from_trace()` dispose du registre des réponses déjà terminées. Elle devient un **contexte explicite ajouté dans l’appel de production**, et non une prétendue propagation rétroactive de tous les enfants rejetés dans le graphe Trace.

`exp15.py:_training_evaluator` et `_trace_engine` fournissent déjà l’évaluation TRAIN et l’intégration Control Plane nécessaires. Il n’est pas nécessaire de remplacer leur moteur, de réévaluer un candidat pour remplir la mémoire ou d’ajouter un critique LLM. Un reçu à la fin du panneau TRAIN du propriétaire suffit à documenter les résultats déjà disponibles. La finalisation de `SearchExperiment.generate()` complète certaines allocations après toute la génération du bras : un slot antérieur n’implique donc pas automatiquement qu’un panneau TRAIN complet était disponible au prochain appel.

## Source de vérité et alignement

Le registre immuable des réponses terminées doit rester la source de vérité. La projection ne prend que les slots du **même bras, du même seed externe et antérieurs au seuil enregistré**. Elle ne parcourt pas librement le cache partagé : cela pourrait introduire des informations d’un autre bras ou calculées plus tard.

Chaque entrée associerait :

- l’identité stable du slot et le hash de son parent ;
- le hash du contenu exact de la réponse et celui du code candidat ;
- le code exact, ou une référence explicite si ce code figure déjà intégralement dans le prompt ;
- l’étape et le statut typés : extraction, contrôle de source, exécution TRAIN valide ou invalide, résultat encore indisponible ;
- les nombres de trajectoires TRAIN allouées, observées et valides ;
- les progrès bruts et erreurs autorisés, avec éventuellement le seul AUC TRAIN agrégé si toutes les trajectoires requises sont valides.

Une invalidité garde une métrique absente, jamais un mauvais score artificiel. Une source syntaxiquement invalide reste un exemple utile de code et d’erreur ; elle ne doit pas être exclue parce qu’elle est inéligible à la sélection finale. Une réponse sans code conserve son statut et son slot mais ne compte pas comme exemple de code complet. Un résultat TRAIN indisponible reste indisponible : aucune évaluation supplémentaire ne doit être déclenchée pour combler cette lacune.

Les clés de cache et les identités internes de tâches restent dans la provenance hôte. Le prompt reçoit uniquement la projection autorisée, sans paramètres cachés, optimum, constantes de normalisation, noms de famille par trajectoire, validation ou audit. Ne pas joindre des valeurs normalisées par tâche aux valeurs objectives brutes : cette association pourrait révéler les normalisations. Le scalaire TRAIN agrégé est un champ distinct, déjà permis dans le propriétaire courant.

Le terme « rejeté » exige une preuve de la décision : un candidat différent du parent courant n’est pas forcément éliminé définitivement, surtout avec plusieurs parents. Le registre peut toujours affirmer son statut d’exécution et son résultat TRAIN ; il ne doit inférer une décision de sélection que si la trace la documente.

## Sélection bornée et reprise

Une règle simple à préenregistrer serait : au maximum sept sources antérieures distinctes, les plus récentes d’après l’indice du slot, sans classement par performance. Conserver séparément les identités des slots dupliqués et leur nombre. Le code du parent courant et le seed déjà présents sont référencés explicitement ; leur duplication ne crée pas de nouvel exemple indépendant. Les réponses sans code restent comptées parmi les échecs mais ne gonflent pas le nombre de sources distinctes.

Cette règle reste une proposition à valider techniquement, pas un choix empirique optimal. Les politiques de déduplication, d’omission et de résumé doivent être déterministes et gelées. Un code tronqué ne doit pas être présenté comme la source complète correspondant au hash : soit conserver l’intégralité, soit signaler précisément une omission ou un extrait qui ne compte pas comme exemple complet. Les échecs et omissions restent dans les données, même si le prompt n’en contient qu’un résumé borné.

La limite actuelle de source est de 65 536 octets (`benchmark.py:source_status`). Sept sources peuvent donc représenter 458 752 octets avant les résultats TRAIN et le prompt courant. La garde de 524 288 caractères du driver ne garantit pas à elle seule l’adéquation à la fenêtre de contexte du fournisseur. Répliquer sept panneaux riches de 48 trajectoires serait coûteux. Une projection historique plus courte peut réutiliser les observations existantes avec un nombre fixé de points de progression, des erreurs typées et un agrégat ; ses règles doivent être enregistrées avant usage. Compter les octets, caractères et tokens réels, les sources complètes effectivement transmises et les omissions.

Avant chaque requête, préserver un instantané avec le seuil de slots, les reçus disponibles et leurs hashes. Une reprise relit cet instantané ; elle ne l’enrichit pas avec les résultats apparus après l’interruption. L’identité des requêtes terminées, la politique des appels ambigus et le refus d’écraser les réponses restent inchangés.

Pour une variante à plusieurs parents, un instantané commun au début du tour évite que le second appel profite artificiellement d’un enfant du premier, simplement parce que les requêtes sont séquentielles. Avec `p` parents et `q` propositions par parent, le seuil proposé est le début du groupe de `p × q` slots. Les tests doivent confirmer que ce groupe correspond au tour effectivement exécuté par Trace. La mémoire et la largeur constituent deux interventions distinctes.

## Six ou sept exemples : montée en charge, pas seuil universel

Avec un parent et un appel par mise à jour, la requête numéro `j` ne peut utiliser que `j − 1` réponses antérieures au maximum :

| Budget par bras | Requêtes ayant au moins six réponses antérieures | Requêtes ayant au moins sept réponses antérieures |
| --- | ---: | ---: |
| `N = 8` | 2 : appels 7 et 8 | 1 : appel 8 |
| `N = 16` | 10 : appels 7 à 16 | 9 : appels 8 à 16 |

Ce sont des plafonds de disponibilité, avant déduplication, réponses sans code, résultats TRAIN encore absents et contraintes de contexte. Exclure le parent déjà fourni réduit encore le nombre d’autres programmes disponibles. Sept réponses antérieures ne sont ni sept programmes distincts, ni sept exemples valides, ni sept comportements différents. Les invalidités correctement documentées peuvent être informatives, mais cela reste à mesurer.

Un scénario futur `N = 16` permettrait plusieurs mises à jour avec une mémoire déjà remplie. Il ne démontre pas que seize propositions sont nécessaires ou suffisantes, ni que six ou sept exemples constituent un seuil d’apprentissage. Ne pas ajouter ce changement à P1 en cours.

## Pourquoi les mémoires existantes ne suffisent pas à activer ce contraste

`PrioritySearch.update_memory` (`priority_search.py:819`) conserve des candidats évalués perdants dans sa file. Cependant, `explore` en retire, les tailles et fusions la modifient, et la propriété `memory` peut elle-même fusionner et remettre à zéro une file. Ce n’est pas un registre exhaustif de slots. De plus, `compress_candidate_memory` (`:941`) conserve seulement `score` et `score_dict` dans les rollouts : malgré sa docstring, il met notamment `feedback`, `x`, `info` et `target` à `None`. Augmenter `long_term_memory_size` ne crée donc pas un prompt de sept codes avec leurs erreurs.

Un hook dans `PrioritySearch.update_memory`, avant compression et suivi de l’appel normal au parent, pourrait observer les candidats évalués. Il est moins complet que le registre des réponses : le chemin générique de `propose` peut filtrer une mise à jour vide avant de créer un `ModuleCandidate`. Le `SlotOptimizer` actuel renvoie une table de mise à jour même si la chaîne source est vide ; cette nuance interdit d’affirmer que tous ses échecs disparaissent de la file. La garantie d’exhaustivité doit néanmoins venir des slots persistés.

`OptoPrimeV2.construct_prompt` (`optoprime_v2.py:522`) affiche une FIFO puis ajoute les variables et le feedback du parent avant l’évaluation de l’enfant. Avec des optimiseurs copiés par candidat, les échecs d’un enfant ne remontent pas automatiquement dans la mémoire du parent retenu. Le propriétaire courant utilise `SlotOptimizer` : régler une taille de mémoire OptoPrime ne change pas ce chemin. `OptoPrime.summarize` est une agrégation déterministe du graphe, distincte du résumeur LLM du trainer.

`MemoryLite.record_artifact` accepte le statut `rejected` mais requiert un score numérique ; `retrieve` filtre par défaut les artefacts `promoted`. Introduire ce stockage nécessiterait une gestion explicite de l’invalidité, sans inventer de score, alors que les preuves requises existent déjà dans les fichiers du propriétaire.

Enfin, activer le résumeur POLCA changerait plusieurs facteurs : le `Summarizer` construit son propre `LLM()`, effectue un appel supplémentaire, échantillonne aléatoirement jusqu’à cinq candidats, utilise `id(candidate)` et un seuil de score par défaut à zéro pour distinguer succès et échec. Cela ne représente pas la validité du programme. `POLCA.propose` appelle aussi `optimizer.set_context`, dont aucune définition n’a été trouvée dans les optimiseurs du dépôt lors de cette revue. Sa compatibilité doit être testée avant tout usage ; ce n’est pas le raccourci minimal pour isoler l’effet d’une mémoire de rejets.

## Contraste causal et tests à prévoir

Le premier contraste mécanistique serait **R courant** contre **R courant + mémoire antérieure alignée**, à parent unique, même nombre de réponses, mêmes sources initiales, panneaux TRAIN, sélection, modèle, paramètres et allocations. Conserver I au même budget donne la comparaison pratique centrale. Si l’on veut distinguer la valeur des codes antérieurs de celle de leurs résultats, un contrôle « codes antérieurs seuls » est nécessaire ; le contraste R avec/sans mémoire estime autrement leur effet conjoint. Un contrôle à une seule tentative précédente pourrait distinguer rappel immédiat d’erreur et mémoire cumulative, sans présumer qu’un grand ensemble de bras est toujours justifié.

Ne pas comparer R avec seize propositions à I avec huit pour attribuer un gain à la mémoire. Ne pas changer simultanément mémoire, largeur, benchmark et seed dans une ablation causale. Un test `N = 16` doit appliquer ce budget aux comparateurs et être identifié comme tel. Les bras mécanistiques peuvent relever d’un pilote séparé ; une confirmation centrale I/R n’a pas à les répéter systématiquement. Son nombre de réplications externes doit dépendre d’un effet minimal utile et de scénarios d’incertitude sur la variance après P1, pas d’un choix automatique de six seeds.

Avant un essai réel, des clients scriptés dans le vrai moteur devraient vérifier : enfant invalide puis parent inchangé avec l’erreur désormais visible ; alignement code/hash/résultat ; aucune lecture de validation ; aucun appel ou évaluation supplémentaire ; échecs partiels et réponses sans code conservés ; comportement à mémoire vide ou surdimensionnée ; instantané identique entre frères d’un tour ; reprise identique même si des fichiers ultérieurs existent. Les tests de compte doivent porter sur les callbacks réellement exécutés par Trace, pas seulement sur une boucle simulée.

Le coût des prompts plus longs, leur latence, la dilution du signal et l’ancrage sur des programmes défaillants sont des effets possibles à mesurer. L’égalité de réponses ne garantit pas l’égalité des tokens. Aucun résumé, critique, réparation générative ou embedding non compté ne doit être introduit. **L’efficacité de cette intervention reste entièrement non validée ; aucune amélioration chiffrée de R par rapport à I n’en découle aujourd’hui.**
