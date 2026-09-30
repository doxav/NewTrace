# EXP-18 — Mémoire des essais et sélection Pareto des programmes parents

Rapport du 12 septembre 2026. **L’expérience exploratoire est exécutée et vérifiée ; elle n’établit pas de gain de la mémoire ni de la politique Pareto sous ce protocole.** Les 384 réponses prévues, les 24 sélections et les 864 trajectoires d’audit sont présentes. Les sept contrastes mécanistiques ont tous un intervalle bootstrap qui inclut zéro. Ce résultat est interprétable ; il ne justifie ni de retoucher les programmes ni de changer les métriques pour obtenir un résultat favorable.

Les contrôles d’intégrité, le recalcul indépendant et la vérification numérique finale du pipeline sont PASS. La dernière exécution du pipeline s’est terminée avec le code 0. La régression globale de clôture du dépôt reste à réaliser : ce rapport ne la déclare pas passée. Les résultats d’EXP-17 ne sont ni consultés ni combinés ici.

## 1. Question et portée

Lorsqu’un modèle réécrit successivement un optimiseur numérique, est-il utile de lui montrer les essais antérieurs, y compris les échecs ? Est-il utile de choisir le programme parent parmi les programmes non dominés sur les différentes instances, plutôt que de retenir le meilleur score TRAIN moyen ? EXP-18 teste ces deux interventions et leur combinaison, avec six recherches appariées par bras.

Le modèle écrit une politique portable `def propose(history, bounds, seed): return x`. L’évaluateur lui transmet uniquement les points et valeurs précédemment observés, les bornes et une graine locale. Le programme propose un point ; l’hôte évalue la fonction cachée. La surface optimisable est donc le code de recherche — exploration, choix des coordonnées, rayon, utilisation de l’incumbent et de l’historique — et non la fonction cible. Le programme déployé ne rappelle pas de LLM.

Le [guide des expériences](../_shared/optimizer_discovery/exp18/GUIDE_EXPERIENCES.md) explique ces unités et les informations réellement transmises. Il distingue notamment sept programmes historiques de sept tâches d’apprentissage : la mémoire testée ici sélectionne des programmes récents, pas un curriculum de tâches. Le chemin de production Trace/Control Plane/PrioritySearch reste utilisé ; l’intérieur du sous-processus candidat ne devient pas pour autant un graphe détaillé envoyé au modèle.

EXP-18 ne contient pas de génération indépendante à 16 propositions. Il ne peut donc pas établir que ces mécanismes battent une référence indépendante de même budget. L reçoit déjà un résumé TRAIN explicite : M−L estime l’apport supplémentaire de l’archive, pas l’effet de tout feedback par rapport à aucun feedback. P modifie ensemble le filtrage des parents et leur tirage uniforme.

## 2. Protocole prospectif et implémentation figée

Le [protocole final](../_shared/optimizer_discovery/exp18/PREREG_EXP18.md) et le [manifeste](../_shared/optimizer_discovery/exp18/exp18_manifest.json) ont été fixés après des pilotes disjoints, avant toute réponse de l’étude principale. Le brouillon est préservé dans [protocol_versions](../_shared/optimizer_discovery/exp18/protocol_versions/PREREG_EXP18_draft_01.md). Les amendements ont explicité les budgets, règles et décisions d’ingénierie ; ils n’ont pas modifié les paramètres scientifiques après observation des résultats principaux.

| Élément | Choix fixé |
| --- | --- |
| Étude | EXP-18 ; namespace `EXP18-MECHANISMS-v1` ; diagnostic exploratoire |
| Répétitions extérieures | 18011, 18023, 18037, 18041, 18053, 18067 |
| Propositions | 16 réponses terminées × 4 bras × 6 graines = 384 ; une réponse invalide consomme son emplacement |
| Modèle | OpenRouter `deepseek/deepseek-v4-flash-0731` |
| Génération | temperature 0,6 ; top_p 1 ; max_tokens 32000 ; `extra_body={"reasoning":{"effort":"low"}}` |
| Client | timeout 300 s ; cache false ; empty_response_retries 0 ; client retries 0 ; génération simultanée 1 |
| Retries transport | Trois retries au maximum par invocation bornée, délais 2/4/8 s ; reprise explicite uniquement du travail inachevé, tentatives conservées |
| Évaluation locale | 32 appels objectif par trajectoire ; 2 graines locales par instance ; 8 workers ; timeout par proposition 2 s |
| Sélection | Tous les TRAIN et validations valides ; minimum AUC validation, puis index le plus ancien ; seed à index −1 |
| Analyse | Graine extérieure comme unité ; bootstrap apparié 10 000 tirages, Random1515, percentiles linéaires 2,5/97,5 |

La limite de 32 000 tokens est un plafond commun de faisabilité, pas une garantie de code ni de gain. Le pilote court a conservé une réponse sans source après consommation de ce plafond. Les graines de requête ne rendent pas la génération du modèle déterministe. Les paramètres et la politique de routage sont communs aux bras ; les jetons effectivement dépensés ne sont pas égaux lorsque la mémoire allonge les prompts.

Pour lire le fichier de gel : `config` et `tasks` décrivent cette exécution. Le bloc hérité `benchmark_manifest` conserve aussi des champs historiques d’EXP-15, notamment son plafond de 8 000 tokens et ses anciennes graines extérieures. Ces champs historiques ne pilotent pas la génération d’EXP-18. Les requêtes effectives ont été vérifiées contre la configuration active à 32 000 tokens et les six graines indiquées ci-dessus ; le bloc ancien est préservé pour la provenance.

Les deux pilotes propres à EXP-18 ont produit 10 réponses M et 6 réponses L/P/PM ; respectivement 10 et 5 candidats étaient admissibles. Le pilote M a naturellement atteint des requêtes montrant sept sources historiques complètes. Les données pilotes ne figurent dans aucun résultat ci-dessous. Voir le [rapport d’ingénierie](../_shared/optimizer_discovery/exp18/ENGINEERING_REPORT.md).

## 3. Benchmark, métrique et contrôle de l’information

| Partition | Instances nouvelles | Trajectoires par programme | Utilisation |
| --- | --- | --- | --- |
| TRAIN | 24, soit 4 par famille/dimension | 48 | Guide la génération et le choix du parent |
| Validation | 12, soit 2 par famille/dimension | 24 | Choisit les programmes après la fin de toute génération |
| Audit réservé | 12, soit 2 par famille/dimension | 24 | Mesure la performance après gel de toutes les sélections |

Les six strates croisent Sphere transformée, Quadratique anisotrope axis-alignée et Rosenbrock transformée avec les dimensions 2 et 4. Les bornes sont [−5,5] par coordonnée. Les décalages sont dans [−2,2], les échelles dans [0,75;1,5], les amplitudes entre 0,1 et 10 ; les poids quadratiques sont entre 1 et 1 000. L’optimum de valeur zéro est faisable au décalage caché. Il n’y a ni bruit ni rotation dense des axes. Les partitions sont distinctes, figées et identiques pour tous les bras d’une comparaison.

Pour l’instance j, la normalisation s_j est la moyenne des excès à l’optimum sur 128 points uniformes de référence tirés indépendamment. Elle est commune aux bras, strictement positive et cachée au candidat. Le regret à t vaut `(meilleure valeur observée à t − optimum) / s_j` ; l’AUC est sa moyenne sur les 32 évaluations. Les faibles erreurs négatives d’arrondi suivent la tolérance figée ; les mauvais résultats ne sont pas plafonnés à 1. On moyenne dans chaque strate, puis on donne le même poids aux six strates. **Une AUC plus faible est meilleure.**

L’optimum, les constantes de normalisation, les paramètres, les noms de fonctions/partitions et le code objectif restent privés. Toutes les 384 générations précèdent la validation. Les 24 sélections et chaque représentant par bras précèdent toute évaluation d’audit. Les horodatages, reçus et hashes des barrières ont été contrôlés. Les graines locales sont dérivées de SHA256, partagées entre programmes et distinctes des graines de génération.

Une source ou trajectoire invalide garde son statut typé et ses observations partielles ; on ne lui attribue pas de faux score numérique. Le programme initial reste dans chaque pool. Si aucun remplacement n’est admissible, il est sélectionné. En audit, un programme qui échoue doit céder définitivement la trajectoire au seed avec l’historique réel et le budget restant. Cette règle commune n’a été activée sur aucune trajectoire d’audit.

## 4. Bras et mécanismes réellement comparés

| Bras | Programme parent / politique | Information supplémentaire |
| --- | --- | --- |
| A0 | Seed manuscrit inchangé, sans génération | Exploration uniforme et perturbations gaussiennes décroissantes autour du meilleur point |
| B2 | Contrôle manuscrit fixé avant cette étude | Même base de recherche, avec premier point au milieu des bornes |
| L | Meilleur AUC TRAIN agrégé | Code parent et résumé compact de validité/AUC du parent |
| M | Même choix scalaire que L | Même résumé + archive des essais précédents |
| P | Parent tiré uniformément parmi les non-dominés TRAIN | Même résumé compact, sans texte d’archive |
| PM | Même choix de parent que P | Même résumé + même archive que M |

M/PM peuvent montrer au plus sept sources complètes distinctes récentes, hors parent déjà affiché, et les résumés typés des emplacements précédents. Le plafond est de 65 536 caractères ; toute omission est explicite, sans coupe silencieuse. L’archive conserve les essais rejetés mais ne sélectionne ni tâches pertinentes, ni sources par diversité comportementale.

Le vecteur Pareto contient 24 moyennes par instance, chacune sur deux graines locales. Seuls les panels TRAIN entièrement valides sont admissibles comme parents. La dominance est composante par composante ; les doublons source/vecteur gardent leur première origine. Les décisions terminales sans proposition suivante sont distinguées des parents effectivement utilisés. L’ordre des bras est apparié avec son inverse, ce qui équilibre chaque précédence à 3/3 : L/M/P/PM ; PM/P/M/L ; M/PM/L/P ; P/L/PM/M ; P/L/M/PM ; PM/M/L/P.

La revue indépendante des prompts et décisions confirme une exposition effective aux mécanismes :

| Exposition | M | PM |
| --- | --- | --- |
| Requêtes montrant sept sources complètes | 33/96 | 39/96 |
| Requêtes avec omission explicite due au plafond de caractères | 12/96 | 5/96 |

| Choix de parent consommé par une génération | P | PM |
| --- | --- | --- |
| Frontière contenant plusieurs parents | 89/96 | 87/96 |
| Parent différent du meilleur scalaire | 73/96 | 63/96 |

Trente requêtes PM réunissent sept sources historiques et un parent différent du meilleur scalaire. Les douze décisions terminales sans génération suivante sont exclues des 192 décisions consommées. La [revue des programmes](../_shared/optimizer_discovery/exp18/PROGRAM_REVIEW.md) documente la reconstruction de dominance/tirage depuis les reçus TRAIN, les cutoffs, la récence et le texte exact des prompts. Les lignées sont aussi dans l’[export inspecté](../_shared/optimizer_discovery/exp18/programs/PROGRAM_INSPECTION.md). Ces comptes démontrent l’exposition, pas l’utilité des mécanismes : l’absence de gain ne peut pas simplement être attribuée à une absence totale d’archive ou de choix de parent différent. Aucun nombre de tâches, taille de batch ou politique de mémoire n’a été retuné à partir des résultats d’audit.

## 5. Résultats des six répétitions sur les instances d’audit

| Graine | A0 | B2 | L | M | P | PM |
| --- | --- | --- | --- | --- | --- | --- |
| 18011 | 0,126226 | 0,044433 | 0,022212 | 0,015653 | 0,067420 | 0,104640 |
| 18023 | 0,147571 | 0,038897 | 0,037422 | 0,021301 | 0,085416 | 0,036173 |
| 18037 | 0,183566 | 0,053564 | 0,020276 | 0,014787 | 0,021518 | 0,019867 |
| 18041 | 0,135404 | 0,038820 | 0,042750 | 0,041827 | 0,048887 | 0,014208 |
| 18053 | 0,145371 | 0,034601 | 0,068352 | 0,053086 | 0,040082 | 0,017526 |
| 18067 | 0,127607 | 0,035607 | 0,019217 | 0,059239 | 0,018325 | 0,044209 |

| Bras | AUC moyenne | AUC médiane | Regret final moyen | Cible atteinte /144 | Temps cible moyen censuré |
| --- | --- | --- | --- | --- | --- |
| A0 | 0,144291 | 0,140387 | 0,022720 | 86/144 | 22,50 |
| B2 | 0,040987 | 0,038858 | 0,016283 | 93/144 | 18,94 |
| L | 0,035038 | 0,029817 | 0,002636 | 135/144 | 9,75 |
| M | 0,034315 | 0,031564 | 0,013191 | 114/144 | 13,32 |
| P | 0,046941 | 0,044485 | 0,006949 | 121/144 | 15,01 |
| PM | 0,039437 | 0,028020 | 0,003415 | 129/144 | 12,48 |

La cible est un regret normalisé ≤0,01. Le temps cible vaut 33 pour une non-atteinte : c’est une convention censurée B+1, pas un temps réellement observé. Les 144 trajectoires par bras proviennent de six graines ×12 instances ×2 graines locales. Les six graines, pas les 144 trajectoires, sont les répétitions statistiques.

Les moyennes de tous les bras génératifs sont inférieures à celle d’A0. Cependant, B2 obtient déjà une forte amélioration avec une modification fixe du premier point. Une comparaison au seed initial seul surestimerait donc ce que l’on pourrait attribuer aux mécanismes de génération. Les intervalles bras−B2 ci-dessous ne permettent pas de distinguer ces bras du contrôle B2 avec cette taille d’échantillon.

![EXP-18 : résultats appariés des bras génératifs et contrastes exploratoires](../_shared/optimizer_discovery/exp18/figures/exp18_mechanisms.png)

La figure montre les six répétitions des quatre bras génératifs et les contrastes mécanistiques enregistrés ; les contrôles A0/B2 restent dans les tableaux. [Version vectorielle](../_shared/optimizer_discovery/exp18/figures/exp18_mechanisms.svg).

## 6. Treize contrastes, dont sept effets mécanistiques

Pour les différences entre bras, une valeur négative favorise le premier bras. Les deltas sont calculés graine par graine avant rééchantillonnage. Les 10 000 tirages portent conjointement sur les six graines, avec les mêmes indices pour chaque contraste. Les tâches fixes ne sont pas rééchantillonnées comme des répétitions extérieures supplémentaires.

Les sept contrastes mécanistiques enregistrés sont exploratoires :

| Contraste | Delta moyen | Delta médian | Intervalle bootstrap 95 % | Lecture |
| --- | --- | --- | --- | --- |
| M-L | -0,000723 | -0,006024 | [-0,012435 ; 0,016909] | Inconclusif |
| P-L | 0,011903 | 0,003690 | [-0,008491 ; 0,032761] | Inconclusif |
| PM-P | -0,007504 | -0,012103 | [-0,031867 ; 0,017142] | Inconclusif |
| PM-M | 0,005122 | -0,004975 | [-0,022718 ; 0,041892] | Inconclusif |
| Effet moyen mémoire | -0,004114 | -0,010686 | [-0,020651 ; 0,014910] | Inconclusif |
| Effet moyen Pareto | 0,008512 | -0,002400 | [-0,015019 ; 0,035682] | Inconclusif |
| Interaction | -0,006781 | -0,010713 | [-0,025757 ; 0,015688] | Inconclusif |

Effet mémoire = ((M−L)+(PM−P))/2 ; effet Pareto = ((P−L)+(PM−M))/2 ; interaction = PM−P−M+L. L’interaction négative décrit un résultat combiné inférieur à l’attente additive, sans démontrer que PM bat un bras particulier. Les quatre effets simples ont le rôle JSON figé `descriptive_secondary`, les trois effets factoriels `exploratory_factorial` ; le protocole regroupe bien ces sept questions comme mécanistiques exploratoires.

Les six comparaisons supplémentaires ont toutes le rôle descriptif secondaire :

| Contraste | Delta moyen | Delta médian | Intervalle bootstrap 95 % | Lecture |
| --- | --- | --- | --- | --- |
| PM-L | 0,004399 | -0,000829 | [-0,026732 ; 0,040554] | Inconclusif |
| L-B2 | -0,005948 | -0,008932 | [-0,021783 ; 0,012069] | Inconclusif |
| M-B2 | -0,006672 | -0,007295 | [-0,024951 ; 0,012323] | Inconclusif |
| P-B2 | 0,005954 | 0,007775 | [-0,014159 ; 0,026525] | Inconclusif |
| PM-B2 | -0,001550 | -0,009899 | [-0,022736 ; 0,025804] | Inconclusif |
| B2-A0 | -0,103304 | -0,102629 | [-0,116482 ; -0,091377] | Signal positif descriptif |

La seule borne supérieure strictement négative concerne B2−A0, comparaison descriptive des deux contrôles. Aucun des sept effets mécanistiques n’exclut zéro. Le détail des treize vecteurs de deltas par graine figure dans la [revue indépendante](../_shared/optimizer_discovery/exp18/INDEPENDENT_RESULT_REVIEW.md), avec les mêmes valeurs que l’[analyse machine](results/run/analysis_results.json.gz).

Les règles d’interprétation étaient figées : intervalle entièrement négatif → signal positif ; entièrement positif → signal négatif ; tous les deltas nuls → aucune différence détectable ; sinon inconclusif. Il n’y a ni garantie simultanée pour ces treize intervalles, ni test confirmatoire de supériorité, ni langage de significativité. Avec n=6, les intervalles restent fragiles ; une absence de différence démontrée n’établit pas l’équivalence.

## 7. Validité, échecs et fallback

| Bras | Sources statiquement invalides | Programmes admissibles | Programmes non admissibles | Trajectoires TRAIN+validation invalides |
| --- | --- | --- | --- | --- |
| L | 11/96 | 82/96 | 14/96 | 972/6912 |
| M | 5/96 | 91/96 | 5/96 | 360/6912 |
| P | 5/96 | 81/96 | 15/96 | 952/6912 |
| PM | 2/96 | 82/96 | 14/96 | 648/6912 |

Au total, 23/384 sources sont statiquement invalides (5,99 %) : 17 absences de source, quatre erreurs de syntaxe et deux violations de protocole. Vingt-cinq autres sources passent ce filtre mais deviennent inadmissibles à l’exécution. Ainsi 336/384 programmes sont entièrement admissibles et 48/384 ne le sont pas (12,50 %). Les invalidités d’exécution ne sont pas assimilées à de mauvais scores objectifs.

Les 2 932 trajectoires générées invalides au sens logique correspondent à 2 068 lignes physiques distinctes. Parmi ces dernières, 1 204 conservent des observations partielles, pour 6 885 appels objectif inclus dans les ressources. Les statuts terminaux sont source absente, syntaxe, protocole, exception ou non-déterminisme ; aucun statut terminal timeout n’apparaît. Cette dernière observation ne garantit pas le succès de tous les seconds replays, qu’un statut de non-déterminisme peut aussi recouvrir. Les tableaux complets par graine et statut figurent dans les [revues de validité et ressources](../_shared/optimizer_discovery/exp18/RESOURCE_REVIEW.md) et [indépendante](../_shared/optimizer_discovery/exp18/INDEPENDENT_RESULT_REVIEW.md).

Chaque bras a retenu un programme généré pour chacune des six graines : aucun seed sélectionné et aucune recherche sans remplacement admissible. Les 864 trajectoires d’audit, dont 576 pour les quatre bras génératifs, sont complètes et valides. Zéro invalidité candidat et zéro fallback en audit, pour toutes les graines. Ce succès du déploiement ne fait pas disparaître les échecs observés durant la recherche.

## 8. Programmes retenus et portabilité

Les 24 programmes sélectionnés, leurs sources exactes, origines et différences successives sont préservés dans [programs/](../_shared/optimizer_discovery/exp18/programs/PROGRAM_INSPECTION.md). Les fichiers `.py.txt` contiennent le code évalué sans reformatage ni correction ; leurs octets peuvent être copiés vers `optimizer.py`. L’[inspection détaillée](../_shared/optimizer_discovery/exp18/PROGRAM_REVIEW.md) doit être lue comme une description, pas comme une preuve causale ou une revue de nouveauté.

| Représentant du bras | Graine / slot (index zéro) | Source exacte SHA256 |
| --- | --- | --- |
| L | 18067 / 15 | `66a330a63f1af3e24c83241fa1fc955f4859f28f1318cf3d8f3238dfc9a951e0` |
| M | 18037 / 15 | `d5b3ea6d1529815bfe9524accb95854d55e80b0be6ccfe8d4a9ff381920b7e23` |
| P | 18067 / 12 | `0bc881c4b03dc545f4a9fbf7fa2dbca07c940598dc86e55e73dda9ed50da1c0c` |
| PM | 18041 / 9 | `a787c80c41dbf44a073a0dcf3874d6237e64aabc6db9a2996e760fd4d0c13ced` |

Le représentant global fixé par la validation est [PM/18041/slot09](../_shared/optimizer_discovery/exp18/programs/selected/PM/18041_optimizer.py.txt), hash `a787c80c41dbf44a073a0dcf3874d6237e64aabc6db9a2996e760fd4d0c13ced`. Sa lignée source vérifiée est seed →1→3→4→6→9. La requête qui l’a produit contenait six codes historiques complets ; son parent 6 avait été choisi parmi quatre candidats de frontière, alors que le meilleur scalaire était le slot 5. La lecture décrit une régression quadratique locale, diagonale puis avec termes croisés, l’utilisation de l’incumbent, une adaptation à la stagnation et une exploration évitant la proximité immédiate. Ces traits ne sont pas isolés par une ablation ; ils ne sont donc pas des causes démontrées du résultat.

L’évaluateur commun reste indépendant du moteur de recherche : [benchmark.evaluate](../_shared/optimizer_discovery/benchmark.py) accepte source, tâche, graine locale, budget et règle de déploiement, puis produit une ligne typée. Un autre moteur peut fournir la même source et utiliser les mêmes panneaux, sélection et métriques. Le sous-processus dispose d’un répertoire frais et d’un environnement assaini ; cette frontière protège l’API et les credentials transmis, sans constituer un sandbox de sécurité du système d’exploitation. Aucun mécanisme Project-1 de snapshots ou de redémarrage n’a été ajouté ici.

## 9. Ressources observées et coûts inconnus

| Bras | Réponses / tentatives | Jetons prompt | Jetons complétion | Jetons totaux | Coût connu des réponses (USD) |
| --- | --- | --- | --- | --- | --- |
| L | 96 /101 | 190 561 | 914 016 | 1 104 577 | 0,237342534028 |
| M | 96 /103 | 1 072 759 | 778 771 | 1 851 530 | 0,261415000300 |
| P | 96 /98 | 164 760 | 766 259 | 931 019 | 0,165487488904 |
| PM | 96 /97 | 1 001 071 | 711 366 | 1 712 437 | 0,241753521504 |
| Total | 384 /399 | 2 429 151 | 3 170 412 | 5 599 563 | 0,905998544736 |

Les 2 570 545 jetons de raisonnement sont inclus dans les jetons de complétion ; on ne les ajoute pas une seconde fois. Les compteurs figés des réponses concordent avec les compteurs natifs des 384 reçus fournisseur. Les champs non natifs des reçus diffèrent et ne les remplacent pas. La somme `total_cost` des reçus vaut 0,905998501 USD ; l’écart avec la représentation des réponses est conservé et inférieur à 4,4×10⁻⁸ USD au total.

**0,905998544736 USD est le coût connu des réponses terminées, pas une dépense totale vérifiée.** Les 15 tentatives de transport échouées n’ont pas de coût ni d’usage mesuré et conservent une possibilité de génération distante/facturation supplémentaire. Cette incertitude ne prouve pas que quinze générations supplémentaires ont eu lieu. Les 384 reçus sont présents, avec 22 noms de route observés ; aucun nouvel appel de métadonnées n’a été fait pour les revues.

| Comptabilité | TRAIN | Validation | Audit | Appels objectif enregistrés | Allocation inutilisée | Sous-processus enregistrés |
| --- | --- | --- | --- | --- | --- | --- |
| Logique, seed inclus | 19 584 trajectoires | 9 792 | 864 | 880 741 | 86 939 | 1 762 830 |
| Cache physique distinct | 17 424 trajectoires | 8 712 | 864 | 804 709 | 59 291 | 1 610 766 |

Le budget enregistré totalise 30 240 trajectoires et 967 680 appels objectif alloués. La réutilisation des sources identiques réduit le cache physique à 27 000 trajectoires et 864 000 allocations. Les sommes logiques comptent les réutilisations ; elles ne sont pas de nouveaux calculs. Le journal compte 140 832 accès cache : 113 832 hits et 27 000 misses, sans ligne physique sans miss ni miss répété. Les 28 événements Trace de replay ne sont pas des réponses LLM supplémentaires. Aucune allocation abandonnée après invalidité n’a financé un essai supplémentaire.

Les 48 tâches ×128 points de normalisation représentent 6 144 évaluations de référence uniques, séparées du budget de recherche. La vérification numérique finale a effectué 810 853 évaluations supplémentaires d’intégrité — 804 709 valeurs conservées et 6 144 références — sans exécuter de candidat ni appeler le modèle. Les reconstructions intermédiaires de références et le travail interrompu non persisté n’ont pas été instrumentés exhaustivement.

Le temps civil entre début de génération et fin d’audit est de 53,658 heures. La génération contient environ 7,103 heures de suspension estimée de l’hôte. Les durées de génération rapportées par le fournisseur totalisent 64 510,461 secondes, tandis que les durées de tentatives côté client incluent aussi l’attente du verrou commun et du transport. Les temps de trajectoires sont sommés sur des workers concurrents. Ces mesures se recouvrent et ne s’additionnent pas en un prétendu temps total ; le timeout transmis de 300 s n’est pas une échéance absolue du slot. Voir la [revue des ressources](../_shared/optimizer_discovery/exp18/RESOURCE_REVIEW.md) pour les comptes, conventions et limites exacts.

## 10. Interruptions, reprise et provenance

Les incidents réseau ont produit quinze tentatives échouées — notamment résolution DNS, timeout, connexion fermée ou TLS — conservées avec leurs classifications et reprises explicites. Aucun échec de transport n’a conduit à remplacer une réponse terminée ou à changer le modèle, le plafond de tokens, la sécurité TLS ou la règle d’analyse.

Une interruption locale distincte a arrêté la sélection le 12 septembre à 16:34:42 UTC : `inspect.getsource` n’a pas pu relire du code de confiance destiné au worker. La génération était déjà complète, 21 sélections étaient préservées et le panel validation 18067/M/slot11 avait 0/24 lignes persistées. Cette exception d’infrastructure précède le lancement du worker concerné ; elle n’a pas été enregistrée comme une invalidité scientifique du candidat. Les panneaux achevés et les réponses sont restés inchangés ; seuls les travaux encore absents ont repris avec les mêmes sources et graines.

La cause de la lecture de source défaillante n’est pas identifiée. Les hashes actuels et les préflights authentifient les sources gelées, sans prouver tous leurs états transitoires antérieurs. Une trajectoire non persistée peut avoir déjà consommé des calculs en mémoire. Pour l’incident EXP-18 identifié, le plafond conservateur est 768 appels objectif et 1 536 sous-processus potentiellement perdus ; **les valeurs réelles restent inconnues**. On n’ajoute pas ces plafonds aux mesures et on ne les remplace pas par zéro. Le [diagnostic d’infrastructure](../_shared/optimizer_discovery/exp18/operational_status_checks/infrastructure_source_read_001.md) et l’[inventaire incident011](../_shared/optimizer_discovery/exp18/operational_status_checks/incident_011.json) conservent ces limites.

La base de l’étude est `13ebda2242e1c18022591737b113030ca2ce2da2`, branche `codex/exp17-parent-selection-exp18-memory-pareto`, avec le travail scientifique antérieur préservé. L’environnement figé utilise Python 3.13.13, Linux et LiteLLM 1.75.0 ; son inventaire complet est dans le manifeste de gel. Un checkpoint documentaire extérieur, `7e701b40485b9880ccfbb64c1faaabb22401c294`, daté du 12 septembre 16:34:22 UTC, est ensuite apparu. Son diff ne modifie que le journal de recherche et l’assessment ; il n’a pas été créé par l’agent racine. Il a été conservé, sans reset. La proximité temporelle de ce commit ou de mtimes ne démontre pas la cause de l’incident.

Aucune modification de sémantique scientifique après gel, aucun remplacement d’issue négative et aucune invalidation de résultat n’ont été nécessaires au vu des contrôles réalisés. Si une preuve ultérieure établissait un changement d’évaluateur ou de worker pendant des calculs antérieurs, la politique de défaut gelée s’appliquerait aux preuves affectées. Le présent rapport n’efface ni les rétractions ni les conclusions historiques d’autres expériences, notamment H3, et ne prétend pas avoir mesuré l’amortissement.

## 11. Vérifications réalisées et état de clôture

| Contrôle | Résultat documenté |
| --- | --- |
| Pilotes disjoints et préflight avant étude | PASS ; sources, archives, grilles et paramètres gelés |
| Génération/chronologie/sélections | PASS ; 384 réponses uniques, 24 sélections, audit après gel |
| Recalcul indépendant des résultats | PASS ; 30 207 fichiers stables ; six graines/six bras et treize contrastes conformes |
| Comptabilité indépendante | PASS ; 226 comparaisons de champs ressources/validité conformes |
| Vérification numérique finale attempt001 | PASS ; 27 000 lignes, dont 2 068 invalides/partielles ; 48 références ; zéro réparation |
| Pipeline EXP18 launch008 | Complet, six étapes, code de sortie 0 |
| Régressions globales avant les pilotes | 844 tests passés, 3 skips optionnels, un module externe exclu |
| Régression globale finale du dépôt | À effectuer ; non déclarée PASS dans ce rapport |

La [vérification numérique](results/run/numeric_verification/attempt_001.json.gz) a conservé les 172 580 fichiers d’entrée et les sources gelées inchangés. Le [journal terminal](results/run/runtime/launch_008/launch_finished.json) confirme la réussite des six étapes. La revue indépendante précédait ce contrôle et le laissait explicitement en attente ; son PASS s’appliquait à son périmètre propre. Le présent rapport prend acte du résultat numérique désormais disponible, sans le relancer.

Commande de régression globale pré-pilote conservée dans le [WORK_LOG](../_shared/optimizer_discovery/exp17/WORK_LOG.md) et son [journal](../_shared/optimizer_discovery/exp17/baseline/full_unit_regression_01.log) :

```bash
/tmp/phase0-venv/bin/python -m pytest -q -rs --disable-socket --allow-hosts=127.0.0.1,localhost tests/unit_tests --ignore=tests/unit_tests/test_recursive_opt_review_regression.py
```

Résultat : 844 passed, 3 skipped, 230,82 s. Les skips concernent Graphviz et deux backends optionnels graphe/télémétrie ; le module exclu dépend d’un backend Trace-Bench/fournisseur externe. Ce résultat pré-live n’est pas présenté comme une nouvelle régression postérieure au run. La rédaction de ce rapport n’a déclenché ni test lourd, ni candidat, ni objectif, ni génération ; le contrôle Markdown/espaces est ciblé sur le document.

## 12. Ce que le résultat permet de retenir

Cette étude démontre qu’une comparaison prospective des archives d’essais et d’une politique de parent Pareto peut être exécutée jusqu’au déploiement, avec tous les programmes invalides, budgets et reprises conservés. Elle fournit aussi un résultat scientifique limité : sous ces tâches, ce modèle, ces informations compactes et seize propositions, les gains mécanistiques restent non établis. Plus de contexte ou un meilleur classement numérique des moyennes ne suffit pas à prouver une amélioration.

Les essais portent sur des fonctions numériques bon marché, déterministes, en deux ou quatre dimensions, avec panneaux fixes. Ils ne représentent ni une recherche scientifique complexe, ni des matrices denses inconnues, ni des évaluations bruitées/coûteuses, ni un curriculum adaptatif. Ils n’optimisent pas la stratégie de méta-optimisation elle-même et ne testent pas une profondeur de récursion additionnelle. Toute nouvelle question — sélection de souvenirs pertinents, curriculum de tâches, référence indépendante N=16, autres familles ou moteurs — demande son propre protocole prospectif. Les résultats actuels ne sont pas retunés pour choisir ce qui aurait gagné.

Pour réutiliser l’instrument avec FunSearch/OpenEvolve, la prochaine intégration consiste à faire produire au moteur externe la même source `optimizer.py`, à la soumettre au même évaluateur et à enregistrer un nouveau bras avec budget comparable et sélection protégée. Il n’est pas nécessaire de lui imposer l’infrastructure recursive_opt. Cette comparaison n’a pas encore été exécutée et aucune supériorité sur ces moteurs n’est revendiquée.

### Repères des preuves

| Preuve | Identité SHA256 |
| --- | --- |
| Gel canonique EXP18 | `ce88c4b2d9530456e63b497f7bbde7608fbf3d8aca441c13c086d77b6beeea8a` |
| Archive sources gelées | `c29b8ba5a5538d6af62bc5696a8ef2841838016906e1384c563138bdf671a4eb` |
| Protocole final | `0f84fe008dff9d0795de0e061e157142a16dc39b80d34e2000b8e53ed49dd07c` |
| Analyse finale, octets gzip | `1654ab1ff03bf1d2713bb3ec6bfe870f603cc8ae6e77b32579a074d20f94925a` |
| Vérification numérique attempt001, octets gzip | `c7ce88095b91a59bc3d02d0ed6fbb18109fda8178da772698fc0338d79b4483e` |
| Revue indépendante des résultats | `7e0c9d15a88a68661441583fc7dd45c42f2c64f3f3ec1367c32c655337b50ee8` |

Les digests canoniques JSON et les hashes des octets gzip désignent des représentations différentes. Les valeurs non arrondies, toutes les graines, sources et erreurs restent dans le [run préservé](results/run/analysis_results.json.gz), avec la [revue indépendante](../_shared/optimizer_discovery/exp18/INDEPENDENT_RESULT_REVIEW.md), la [revue des ressources](../_shared/optimizer_discovery/exp18/RESOURCE_REVIEW.md), la [revue des programmes](../_shared/optimizer_discovery/exp18/PROGRAM_REVIEW.md) et le [guide explicatif](../_shared/optimizer_discovery/exp18/GUIDE_EXPERIENCES.md).
