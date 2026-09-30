# Inspection des programmes et du feedback effectivement utilisé dans P1

Le représentant R choisi par validation est un programme de 177 lignes combinant une initialisation de type Halton, un modèle de substitution à noyau gaussien et une sélection par amélioration espérée. Sa lignée comprend quatre modifications depuis le seed. D'autres recherches R restent beaucoup moins profondes : deux produisent leurs huit propositions directement depuis le seed. P1 teste donc une procédure de recherche adaptative, avec des lignées effectivement différentes, et non huit améliorations successives garanties.

Cette inspection est descriptive et postérieure à `pipeline_complete.json`. Elle ne choisit aucun programme à partir de l'audit. Le choix du représentant a été recalculé exclusivement avec les AUC de validation et correspond exactement à `selections_frozen.json` : **16411, slot 07**. Le gel des sélections précède `audit_started.json`. Les valeurs objectives proviennent uniquement des observations déjà enregistrées ; aucun candidat, objectif ou modèle n'a été exécuté pour cette inspection.

Les vérifications reproductibles et les sources exactes figurent dans [program_inspection.json](program_inspection.json). Les références logiques terminant par `.json` peuvent être stockées physiquement sous `.json.gz` ; le vérificateur enregistre le chemin physique et son hash. Les copies `.py.txt` ci-dessous sont des exports de présentation **octet pour octet**, sans formatage, renommage interne ou réparation. [Intégrité des exports](selected_R/export_integrity.json).

| Graine externe | Slot retenu, indexé depuis zéro | Lignée depuis le seed | AUC de validation | Source exacte exportée |
|---|---:|---|---:|---|
| 16411 | 7 | seed → 1 → 5 → 6 → 7 | 0,070635 | [16411_optimizer.py.txt](selected_R/16411_optimizer.py.txt) |
| 16423 | 6 | seed → 0 → 6 | 0,091810 | [16423_optimizer.py.txt](selected_R/16423_optimizer.py.txt) |
| 16437 | 4 | seed → 0 → 2 → 4 | 0,093871 | [16437_optimizer.py.txt](selected_R/16437_optimizer.py.txt) |
| 16441 | 7 | seed → 0 → 4 → 5 → 6 → 7 | 0,136120 | [16441_optimizer.py.txt](selected_R/16441_optimizer.py.txt) |
| 16453 | −1 | seed inchangé | 0,141393 | [16453_optimizer.py.txt](selected_R/16453_optimizer.py.txt) |
| 16467 | 2 | seed → 2 | 0,146404 | [16467_optimizer.py.txt](selected_R/16467_optimizer.py.txt) |

Les six hashes SHA-256 des sources évaluées sont :

```text
16411 40567fda87a2734f31a680db90e54d5f975a73b7930bc487be821d0cc5c9f243
16423 65a614e9dbde2d2400ba8832c7416d1ac30ca39730de3d195aef73b087b9a2b4
16437 273282cda608b8c4822f5485a57996977dfa3960c439debd646ccdfb692ebecf
16441 fcabb956294ef66c414d4a20c8be3a3f59914940703da35481f2e2aa1244ade9
16453 5b74e5a3fe2fc90fcb42a603b058fa7befbf65f38775caa77acd30cdf619b640
16467 9a8dfe5890b2ae04ff00175218d1ecea91beee1189aa6ce0524acbefe53500e2
```

Les sources faisant autorité restent les champs `source` des fichiers `raw/<graine>/R/selection.json` et de leurs réponses de génération correspondantes. Par exemple : [sélection du représentant](../../../../EXP16/results/production_run/raw/16411/R/selection.json), [requête finale](../../../../EXP16/results/production_run/raw/16411/R/slot_07/request.json), [réponse finale](../../../../EXP16/results/production_run/raw/16411/R/slot_07/response.json), [feedback propagé final](../../../../EXP16/results/production_run/raw/16411/R/slot_07/propagated_feedback.json).

**Comportement du représentant 16411.** Les six premières propositions en dimension 2, et les dix premières en dimension 4, suivent une séquence Halton commençant à l'indice 1. Cette phase ignore les valeurs observées et la graine ; la suite du programme utilise `Random(seed + len(history))`. Après cette initialisation, il retourne un point uniforme avec probabilité 0,10. Sinon il ajuste un noyau gaussien sur toute l'histoire, standardise les valeurs et inverse la matrice régularisée avec une routine Gauss–Jordan écrite dans le programme.

Les échelles du noyau dérivent des distances médianes par coordonnée, avec bornes proportionnelles à la largeur du domaine. Le programme calcule ensuite une acquisition de type amélioration espérée sur **471 points internes** : 200 uniformes, 150 autour de l'incumbent, 120 autour des trois autres meilleurs points et l'incumbent lui-même. Ces calculs n'appellent pas l'objectif : une seule proposition finale est retournée. En cas d'échec d'inversion, une perturbation gaussienne de l'incumbent prend le relais. Les perturbations locales sont écrêtées aux bornes ; un léger bruit tente d'éviter les doublons exacts.

Il s'agit d'une politique de type optimisation bayésienne simplifiée, sans prétention de nouveauté, de calibration statistique du modèle de substitution ou d'équivalence avec une bibliothèque reconnue. Le surcroît de calcul interne peut affecter le coût d'exécution sans consommer davantage d'évaluations objectives. Le nombre 32 apparaît dans ses calendriers de pas et la liste des bases est finie : les résultats ne garantissent pas la portabilité à des horizons ou dimensions arbitraires.

| Transition du représentant | Modification observable dans la source | AUC TRAIN avant → après | Diff exact |
|---|---|---:|---|
| seed → slot 1 | Remplacement de la recherche aléatoire locale par noyau gaussien, inversion et acquisition EI ; exploration initiale uniforme | 0,152764 → 0,089264 | [−1 → 1](selected_R/16411_slot_-1_to_1.diff.json) |
| slot 1 → 5 | Échelle du noyau tirée des distances, régularisation accrue, mélange de points globaux et locaux | 0,089264 → 0,085285 | [1 → 5](selected_R/16411_slot_1_to_5.diff.json) |
| slot 5 → 6 | Initialisation Halton ; échelles par coordonnée ; recherche autour de plusieurs bons points ; tentative d'éviter les doublons | 0,085285 → 0,062508 | [5 → 6](selected_R/16411_slot_5_to_6.diff.json) |
| slot 6 → 7 | Exploration uniforme 0,05 → 0,10 ; pas et échelles élargis ; plus de points internes globaux/locaux | 0,062508 → 0,054450 | [6 → 7](selected_R/16411_slot_6_to_7.diff.json) |

Ces différences TRAIN décrivent une lignée sélectionnée ; elles n'isolent pas l'effet causal de chaque modification. Le slot 7 modifie simultanément plusieurs constantes. Dans cette recherche, les slots 0 et 3 n'ont pas de source utilisable, tandis que 2 et 4 produisent des programmes moins bons sur TRAIN que leur parent. Les quatre demandes 2 à 5 reprennent le même parent et son même feedback : le modèle ne reçoit pas les résultats des trois tentatives précédentes. Tous ces essais restent dans les données ; les invalides ne deviennent pas des AUC extrêmes.

**Les autres politiques R.** Le programme 16423 commence par `2 × dimension` points Halton, puis alterne exploration uniforme et gaussiennes autour de l'incumbent. Il augmente la probabilité d'exploration après stagnation, réduit progressivement le pas, double parfois celui-ci et vérifie les doublons. La dernière transition abandonne la recherche coordonnée cyclique du parent et retire le décalage Halton dépendant de la graine. C'est une réécriture de comportement, pas seulement un réglage numérique.

Le programme 16437 utilise huit points Halton avec décalage dépendant de la graine, puis un mélange de perturbations locales, de différences entre points historiques et de déplacements entre les deux meilleurs points. La distance moyenne à l'incumbent détermine une échelle de pas ; la stagnation modifie les probabilités des branches. L'étiquette DE du commentaire doit être comprise comme une inspiration : le code n'impose pas qu'au moins une coordonnée mutante soit retenue. La dernière transition ajoute notamment une branche directionnelle et passe de dix à huit points initiaux.

Le programme 16441 conserve une initialisation Halton décalée par la graine, puis mélange direction du deuxième meilleur point vers l'incumbent, perturbation gaussienne complète et perturbation d'une coordonnée. La stagnation accroît fortement les retours uniformes. Le pas dépend de l'horizon 32 et de la récence de l'amélioration. Dans la dernière transition, les redémarrages Halton deviennent généralement uniformes, l'adaptation par dispersion des meilleurs points disparaît et le croisement de coordonnées est retiré. Des imports et variables inutilisés subsistent ; aucune correction manuelle n'a été appliquée.

La graine 16453 retient le **seed**, malgré six propositions admissibles après TRAIN et validation. La graine 16467 retient une perturbation gaussienne simple avec exploration et pas décroissants, choix occasionnel d'un autre point historique comme centre et sauts plus larges. Ce dernier programme avait un AUC TRAIN **moins bon** que le seed, 0,160912 contre 0,143672 : il n'est jamais devenu parent de génération, mais la validation l'a sélectionné dans le pool complet. Ce cas confirme que « rejeté par la recherche » ne signifie pas « absent de la sélection finale ».

L'observation des trajectoires d'audit confirme des différences de comportement sans identifier quelle branche interne a été prise à chaque appel. Sur 24 trajectoires de 32 points par programme retenu :

| Graine R | Octets / lignes / nœuds AST | Points touchant une borne | Points exactement répétés dans leur trajectoire | Premiers points distincts, dimensions 2 / 4 |
|---|---:|---:|---:|---:|
| 16411 | 5 744 / 177 / 1 743 | 11 / 768 | 0 | 1 / 1 |
| 16423 | 2 842 / 86 / 687 | 61 / 768 | 0 | 1 / 1 |
| 16437 | 3 570 / 108 / 919 | 117 / 768 | 1 | 12 / 12 |
| 16441 | 3 448 / 97 / 862 | 36 / 768 | 0 | 12 / 12 |
| 16453, seed | 545 / 10 / 159 | 74 / 768 | 0 | 12 / 12 |
| 16467 | 1 100 / 32 / 255 | 141 / 768 | 1 | 12 / 12 |

Les 144 trajectoires R sélectionnées sont valides, sans fallback de déploiement. Aucune ne commence exactement au milieu du domaine. Le premier point commun de 16411 et 16423 est `[0, −5/3]` en dimension 2 ; la dimension 4 ajoute les coordonnées `−3` et `−25/7`, aux arrondis flottants près. Leur bonne performance éventuelle ne saurait donc être attribuée au seul modèle de substitution : initialisation, exploration et exploitation changent ensemble. Ces statistiques d'audit sont descriptives, jamais des critères de choix du représentant.

**Ce que les générateurs ont reçu.** Les 192 requêtes contiennent la même instruction explicite de performance anytime, le même seed, le même contrat et les mêmes restrictions statiques. I ne reçoit que cette information invariante. C reçoit aussi le code du parent choisi par la recherche. R et W reçoivent ce code et le texte réellement propagé par Trace. C utilise donc un feedback de performance **indirect**, via le choix du parent sur TRAIN ; il n'est pas un second bras indépendant.

Les 96 prompts R/W contiennent chacun **48 trajectoires courantes : 24 instances TRAIN × 2 graines locales**, non 48 tâches distinctes. Chaque projection a été confrontée aux observations de son propre parent dans le pool : index et valeur du premier point, incumbent, tous les événements d'amélioration et les 32 valeurs de la courbe best-so-far correspondent exactement. L'AUC TRAIN agrégé est également présent et correspond au calcul enregistré. Les 96 panels sont valides : aucun feedback de parent invalide n'a effectivement atteint ces générations.

Cette correction résout plusieurs limitations de S0/EXP15 : le critère anytime est explicite ; le résultat est lié au code exact courant ; le modèle reçoit la courbe complète de l'incumbent ; l'intégration utilise réellement le feedback propagé. Mais **courbe complète ne signifie pas histoire complète**. Les coordonnées et valeurs individuelles des points qui n'améliorent pas l'incumbent sont absentes ; des statistiques d'étendue, de répétition, de bord et de stagnation en résument une partie. Les coordonnées exposées représentent 14 079 / 73 728 observations pour R, soit **19,10 %**, et 12 804 / 73 728 pour W, soit **17,37 %**, en comptant les panels répétés. Une mauvaise région explorée n'est donc pas décrite point par point.

Les courbes par trajectoire restent en valeurs brutes. Le modèle connaît un AUC normalisé global, mais ni les normalisations par tâche, ni leur contribution individuelle au score global, ni les familles ou paramètres cachés. Cette restriction respecte le protocole ; elle limite toutefois le diagnostic direct de l'origine d'un score global. La projection ne doit pas être confondue avec un graphe d'exécution détaillant les branches et variables internes du programme candidat, lequel s'exécute dans un sous-processus distinct.

**Mémoire et profondeur effectives.** R émet 21 / 48 demandes depuis le seed. Ses panels exacts présentent 30 répétitions au-delà du premier exemplaire par parent et graine externe ; W en présente 25 / 48. Cela n'implique pas des réponses déterministes : les seeds de requête, les tirages et le service sont stochastiques. Cela établit en revanche que les tentatives échouées ou inférieures ne produisent pas de nouvel exemple visible tant que le parent reste identique.

| Bras | Parents de profondeur 0 / 1 / 2 / 3 / 4 | Propositions complètes sur TRAIN | Signatures de points TRAIN distinctes, comptées par recherche |
|---|---|---:|---:|
| I | 48 / 0 / 0 / 0 / 0 | 46 / 48 | 46 |
| C | 14 / 11 / 13 / 9 / 1 | 44 / 48 | 44 |
| R | 21 / 16 / 5 / 5 / 1 | 42 / 48 | 42 |
| W | 22 / 14 / 9 / 3 / 0 | 43 / 48 | 43 |

La profondeur compte les arêtes de génération depuis le seed, pas une profondeur de récursion entre niveaux de méta-optimisation. Les signatures utilisent les propositions enregistrées sur toutes les mêmes entrées TRAIN ; les invalides restent distincts de cette comparaison. Aucun effondrement exact des trajectoires valides n'est observé à l'intérieur de ces recherches. Cette diversité ne mesure ni nouveauté algorithmique, ni diversité utile sur de nouvelles familles.

W alloue deux parents par ronde et une proposition par parent, soit quatre rondes. La première ronde utilise deux copies du seed dans chaque recherche. Les trois rondes suivantes ont **deux hashes de parents distincts dans chacune des six recherches** : sa largeur est effectivement exercée, 18 rondes sur 24 au total. Les prompts restent séparés par parent ; W ne montre pas plusieurs programmes ensemble au modèle. La largeur modifie aussi la profondeur disponible à budget huit : W contre R ne sépare donc pas chaque dimension de la stratégie de recherche.

La mémoire d'archive intervient dans le choix des parents W. Elle n'est pas un historique de rejets transmis au modèle. Le schéma effectivement reçu ne contient que `current` et `aggregate_training_auc`. Il n'existe ni panel d'échecs antérieurs, ni résumé cumulatif, ni mémoire partagée de critiques. La classe de production peut évoquer Pareto, mais le plan enregistré utilise une sélection **scalaire de l'AUC TRAIN agrégé**, sans niches par instance, objectif de diversité ou front multi-critères. [Adaptateur de scheduling gelé](../trace_schedule.py), [construction et validation du feedback gelées](../search_experiment.py).

**Portée du test.** R contre I porte sur l'ensemble « génération conditionnée par des parents choisis adaptativement et leur feedback TRAIN » face à des propositions indépendantes. R contre C approche plus directement la valeur du feedback textuel explicite, avec des chemins de recherche ensuite différents. W contre R teste une largeur modeste contre davantage de rondes à allocation commune. Aucune comparaison ne démontre l'avantage d'une mémoire des rejets, d'une critique LLM supplémentaire, de plusieurs niveaux de récursion ou d'un Pareto par instance.

F1 examinait une modification unique depuis des parents fixés, avec six trajectoires et sans AUC agrégé fourni. P1 combine une recherche itérative, davantage d'instances et répétitions, une projection riche native et un AUC agrégé. Les résultats des deux étapes ne sont donc pas des réplications du même contraste. Les inspections S0, F1 et P1 permettent de distinguer les défauts corrigés des mécanismes encore absents ; elles ne permettent pas d'attribuer une variation de performance entre étapes à un seul changement. La tâche reste limitée aux familles et dimensions gelées, à 32 évaluations et six réplications externes.

**Vérification.** Le helper [program_inspection.py](program_inspection.py) n'importe que la bibliothèque standard : JSON/gzip, hashes, AST sans exécution, diffs et calculs sur les lignes sauvegardées. Il vérifie les 24 sélections et la règle du représentant, les hashes des sources des 192 réponses, les 144 enveloppes natives C/R/W, les 6 912 projections de trajectoires correspondantes, les 96 prompts riches exacts et l'identité de 582 fichiers avant/après lecture. Il ne remplace pas l'audit numérique et budgétaire indépendant.

```bash
/tmp/phase0-venv/bin/python -m pytest -q artifacts/optimizer_discovery/investigation16/production/test_program_inspection.py
/tmp/phase0-venv/bin/python artifacts/optimizer_discovery/investigation16/production/program_inspection.py
/tmp/phase0-venv/bin/python -m black --check --target-version py313 artifacts/optimizer_discovery/investigation16/production/program_inspection.py artifacts/optimizer_discovery/investigation16/production/test_program_inspection.py
/tmp/phase0-venv/bin/python -m ruff check artifacts/optimizer_discovery/investigation16/production/program_inspection.py artifacts/optimizer_discovery/investigation16/production/test_program_inspection.py
git diff --check -- artifacts/optimizer_discovery/investigation16/production/
```

Résultats : **4 tests réussis**, recomputation strictement identique, Black/Ruff et contrôle des espaces réussis. Les diffs sont emballés en JSON pour conserver les espaces de contexte sans introduire de warnings de formatage ; le texte du diff reste exact.

Les tests portent sur les ancêtres futurs et identités de sources répétées, la fidélité des événements/courbes, les champs non déclarés, l'enveloppe native des invalides et les statistiques géométriques. Les erreurs attendues ont été observées avant implémentation des gardes. Aucun ancien fichier scientifique ni source candidate n'a été modifié.
