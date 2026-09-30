# Recommandations pour une future expérience EXP-17

**Les diagnostics justifient de contrôler une meilleure initialisation du seed,
d'améliorer la mesure et de
tester des stratégies de recherche réellement distinctes. Ils ne permettent pas
de prévoir un gain du feedback récursif sur la génération indépendante.** Le
contraste R − I reste inconnu. Les modifications proposées ci-dessous visent une
expérience plus informative, avec un résultat positif, nul ou négatif également
acceptable.

Cette note utilise uniquement les replays historiques, B1, B2, S1, S2, T1 et la
revue géométrique publics. Aucun résultat F1, P1 ou P1-E1 en cours n'a été consulté
pour la rédiger. Aucun nouveau calcul expérimental ni appel de modèle n'a été
lancé. EXP-17 est ici un nom proposé, à vérifier dans le registre avant réservation.
**Le plan chiffré à six réplications et quatre bras génératifs est uniquement un
scénario exploratoire d'architecture et de budget. Ce n'est pas la configuration
confirmatoire finale d'EXP-17 et il ne fournit aucune promesse de puissance.**

| Levier | Gain effectivement mesuré | Portée et limite |
|---|---|---|
| Premier point du seed | B2 : AUC 0,192300 → 0,027569 en central, soit −85,66 % ; 0,167906 → 0,055169 en broad, soit −67,14 % | Intervention unique sur les fixtures B1 déjà vues ; 26 pertes individuelles conservées. Aucun gain R − I mesuré. |
| Diversité du panel de sélection | S1 : 6 tâches × 1 seed local → 24 × 2 réduit l'AUC d'audit moyenne des politiques sélectionnées de 0,091412 à 0,075566, soit −17,3 % | Banque fixe de 11 programmes ; le panel révèle de meilleures politiques déjà présentes, sans en générer. |
| Diversité à coût égal | S1, 24 trajectoires : 24 tâches × 1 donne 0,077907 ; 12 × 2 donne 0,081151 ; 6 × 4 donne 0,085921 | La diversité des tâches aide davantage que les seules répétitions locales dans cette banque ; ce n'est pas une loi universelle. |
| Parallélisme de l'évaluation | T1 corrigé : huit workers donnent 5,49× le débit du serial sain ; seize donnent 7,29× | Gain de temps d'évaluation sur ce matériel, sans changement des trajectoires. Les anciens ratios 7,08× et 9,40× sont supersédés après l'audit de suspension. |
| Réutilisation d'un ordre de recherche | Replay W2 : qualité attendue q = 1 avec une proposition cible, contre neuf pour le menu uniforme ; seuil d'amortissement K* = 2,25 tâches à Q = 1 | L'ordre coûte 18 évaluations sources. Le contrôle informé `nearest` fixé obtient aussi q = 1 avec une proposition et zéro coût méta. |
| Programmes construisant une solution | Replay VRPTW : meilleur programme sauvegardé à 20,702124 de distance contre 26,476508, soit −21,81 % ; cinq réponses originales sur douze dépassent le menu | Résultat reproduit sur les anciennes instances, issu de génération indépendante. Pas de nouvelle généralisation ni d'avantage récursif. |

Ces pourcentages portent sur des interventions, des métriques et des populations
différentes. **Il ne faut ni les additionner ni les multiplier pour annoncer le
gain attendu d'EXP-17.** Sources : [B2](../benchmark/B2_REPORT.md),
[S1](../selection/REPORT_S1.md), [T1 corrigé](../throughput/REPORT_T1.md),
[replays et commits historiques](HISTORY_REPORT.md).

1. **Contrôler prospectivement une référence de départ plus exigeante.**
   La variante B2 est un candidat au rôle de seed commun, dont l'adoption reste
   conditionnelle à un contrôle prospectif sur de nouvelles instances et de
   nouveaux seeds locaux, selon des critères fixés avant calcul. B2 utilisait des
   fixtures déjà vues ; son gain ne suffit pas à valider ce remplacement pour une
   confirmation. L'intervention reste unique : milieu des bornes quand l'histoire
   est vide, puis seed original inchangé. Source exacte :
   [optimizer.py](../benchmark/b2/optimizer.py), SHA256
   `958fbb12279a15966bf1ffa45ddffc7cd5f946ca800a3fe2b8e132a150c5c190`.
   Si ce contrôle justifie son adoption, tous les bras devront recevoir cette même
   source et l'inclure dans leur pool final ; elle deviendra également leur
   fallback commun. L'ancien seed peut rester
   un contrôle descriptif préenregistré. Cela évite de présenter la correction
   d'une initialisation faible comme une preuve de feedback utile. Ce changement
   est proposé uniquement pour une nouvelle expérience : aucun seed gelé actuel
   ne doit être remplacé.

   B2 ne force pas tous les programmes générés à commencer au centre. Leur liberté
   de proposer un autre premier point reste compatible avec OptimizerProgramV0.
   Une initialisation commune imposée à toutes les politiques serait une autre
   intervention, à enregistrer séparément avec son coût objectif. De même,
   supprimer simplement le premier terme de l'AUC n'enlève pas l'effet du premier
   point : il peut rester incumbent pendant la suite et modifier toute l'histoire.
   En B2, seuls 16–17 % du delta AUC proviennent arithmétiquement du premier terme.

2. **Financer davantage de tâches distinctes avant de multiplier les répétitions.**
   Allocation à examiner dans le scénario exploratoire : 24 instances de train
   × 2 seeds locaux par instance,
   équilibrées sur les six strates ; 12 instances de validation × 2 ; 24 nouvelles
   instances de test × 2, toutes disjointes. Le choix train 24 × 2 est soutenu par
   S1 : même politique sélectionnée qu'avec 24 × 4, à moitié du coût objectif,
   dans la banque étudiée. Les tailles validation et test proposées sont des
   allocations pratiques, pas des optima démontrés par S1. Un budget train limité
   à 24 trajectoires appelle plutôt 24 × 1 que 6 × 4 dans ces données.

   Conserver B = 32, la normalisation indépendante, les poids égaux par strate,
   l'AUC complète comme métrique principale et le regret final comme mesure
   secondaire explicite. B1 et B2 montrent que les classements anytime et finaux
   peuvent diverger : B2 dégrade le regret final dans trois strates sur six dans
   chacune des deux conditions, malgré une AUC meilleure dans toutes les strates.
   Aucune preuve ne justifie encore d'allonger B uniquement pour favoriser R.
   Les exemples « utiles » sont des observations distinctes réellement consommées,
   pas un seuil magique de six ou sept lignes. L'historique des knobs comportait
   13 bundles à une seule entrée externe ; certains contenaient des instances
   internes, mais les réglages d'ordre externe n'avaient aucune diversité à agir.

3. **Séparer feedback explicite, évolution du code et largeur de recherche.**
   Pour confirmer le contraste central I/R, conserver A0, I et R ; ne pas imposer
   de nouveau C et W si les diagnostics précédents ont suffisamment traité leurs
   questions mécanistiques. Ces deux bras sont des options exploratoires à inclure
   seulement si une incertitude résiduelle justifie leur coût.

   Le scénario exploratoire complet comprend A0 et quatre bras génératifs déclarés
   avant résultats : I, génération indépendante ; C, évolution du code avec
   incumbent choisi sur train, sans scores ni traces transmis dans le prompt ;
   R, même procédure avec retour déterministe des résultats et observations train ;
   W, retour de même nature que R mais deux parents et quatre cycles de propositions.
   I et R forment le contraste central ; pour l'AUC, un delta R − I négatif
   favorise R. R − C teste l'apport du retour explicite
   au-delà de l'évolution du code ; C reçoit déjà une information implicite via
   le choix de son parent, ce n'est donc pas un contrôle sans aucune sélection.
   W − R teste une modification conjointe de largeur et de profondeur séquentielle,
   pas une dose pure de feedback ni une profondeur supplémentaire de récursion.

   Illustration de budget pour ce seul scénario : huit réponses complétées par
   bras et six réplications externes appariées entièrement nouvelles, soit
   192 réponses pour
   quatre bras génératifs. **Six réplications ne sont pas recommandées comme
   confirmation suffisamment précise.** Les seeds et l'ordre intercalé du futur
   protocole confirmatoire restent à réserver après son dimensionnement.

   Après P1, définir un effet minimal utile sur le contraste apparié R − I,
   puis choisir n selon l'objectif de précision ou de puissance et des scénarios
   couvrant l'incertitude de la variance des différences. Ne pas utiliser une
   seule estimation ponctuelle issue d'un petit échantillon comme une variance
   connue, ni reprendre automatiquement n = 6. Les réserves de S0 sur la précision
   à faible réplication restent applicables ; aucun calcul de dimensionnement
   confirmatoire n'est effectué dans cette note. Geler ce choix avant toute donnée
   confirmatoire d'EXP-17 et réévaluer le budget avec le nombre final de bras.
   L'unité d'incertitude est
   la réplication externe ; ni les tâches, ni les points de trajectoire, ni les
   200 sous-échantillons chevauchants de S1 ne la remplacent.

   S2 montre que deux sélections A2 historiques apparaissent aux réponses six et
   sept. Un mini-pilote à deux ou quatre propositions ne teste pas cette profondeur.
   Il ne montre toutefois aucun avantage attendu à N = 16. Les courbes de meilleur
   score de validation par préfixe s'améliorent mécaniquement quand le pool grandit.
   Les deux sélections finales du seed dans A2 disposaient de quatre et six
   candidats éligibles : disponibilité et supériorité sur le seed sont distinctes.
   Source : [S2](../budget/S2_REPORT.md).

4. **Vérifier le mécanisme exécuté avant de comparer son nom.**
   Réutiliser la voie de production et ses paramètres actuels de `PrioritySearch`
   pour le contraste de largeur. Exiger que les journaux montrent huit callbacks
   génératifs réels, les parents effectivement explorés et tous les programmes
   proposés, y compris rejetés ou invalides. Une configuration écrite « deux
   parents × quatre cycles » ne prouve pas que cette allocation a eu lieu ; des
   sources dupliquées peuvent aussi rendre la largeur nominale trompeuse.

   Le replay du resolver montre que `MinibatchAlgorithm`, `BeamsearchAlgorithm`
   et `UCBSearchAlgorithm` résolvent tous vers `ParetobasedPS`. Les mettre dans un
   menu ne compare pas trois algorithmes. Certains wrappers emploient aussi des
   noms de paramètres anciens. Un test d'exécution doit démontrer que le knob
   modifie bien les updates, parents, mémoires ou exemples consommés ; `inner_steps=0`
   ne peut mesurer l'intérêt d'une stratégie d'update. La diversité de source ne
   suffit pas : S2 trouve 5,4 comportements train distincts en moyenne dans chaque
   bras, malgré des nombres de sources éligibles différents.

   Geler un retour train borné, déterministe et compréhensible qui relie les
   observations disponibles à l'objectif de recherche ; vérifier qu'il atteint
   effectivement la requête. La taille du panel S1 améliore la sélection mesurée,
   mais ne garantit pas qu'un LLM exploite un prompt plus long. Validation, test,
   fonctions objectives, paramètres cachés, optima et constantes de normalisation
   restent hors du chemin de génération. Conserver la même information invariante
   et les mêmes allocations d'évaluation dans tous les bras comparables.

5. **Utiliser le gain de débit pour améliorer la mesure, sans créer des budgets cachés.**
   Huit workers d'évaluation constituent un choix conservateur fondé sur T1 ; seize
   était plus rapide dans l'échantillon mesuré. Conserver la génération LLM à une
   requête à la fois. Pour le seul scénario exploratoire à quatre bras génératifs
   et six réplications, avec seed plus huit candidats par bras et
   72 trajectoires train/validation par programme, l'allocation logique atteint
   15 552 trajectoires de sélection et 1 440 de test, soit 16 992 au total :
   543 744 appels objectif à B = 32, avant réduction par cache ou arrêt invalide.
   Les 60 tâches uniques ajoutent 7 680 références partagées de normalisation.
   Les allocations de seed partagées, cache hits, arrêts invalides et appels
   d'intégrité doivent être distingués des appels effectivement consommés.

   Le débit T1 à huit workers donne environ 1,94 heure pour cette seule allocation
   exploratoire d'évaluation, et non pour une confirmation EXP-17 dimensionnée.
   C'est une extrapolation d'ingénierie depuis deux programmes fixes,
   pas une borne de runtime pour de nouveaux codes ; elle exclut la latence LLM.
   Capturer horloges réelle, monotone et de démarrage permet d'identifier une
   suspension. Ne pas reprendre les ratios T1 supersédés ni relancer une mesure
   simplement parce qu'elle est lente. Aucun de ces diagnostics ne permet de
   fixer un plafond de tokens, un modèle ou un réglage de raisonnement meilleur :
   ces choix nécessitent leur propre preuve de faisabilité de génération, commune
   aux bras, avant le gel futur.

6. **Conserver la géométrie pour le premier contraste corrigé et expliciter sa portée.**
   Les familles ne sont pas trois géométries indépendantes : Sphere est déjà une
   quadratique diagonale légèrement anisotrope dans les coordonnées du candidat ;
   Quadratic élargit cette distribution de conditionnement. Elles représentent
   quatre strates sur six et deux tiers du poids. Rosenbrock est distincte,
   couplée et non convexe. Les bornes de conditionnement possibles ne sont pas
   celles observées systématiquement sur les instances. La revue confirme la
   concordance avec la spécification, sans défaut d'implémentation.

   B1 et S1 établissent une marge de performance pour des politiques fixes, même
   au-delà du midpoint constant. Ils n'établissent pas que les connaissances
   préalables du LLM sur cette géométrie expliquent R − I. Ajouter rotations,
   dimensions, familles multimodales ou une tâche de construction pourrait changer
   la valeur informative des traces ; l'effet reste hypothétique. Le faire ensuite
   comme intervention explicitement motivée par la cible scientifique, avec
   données nouvelles et pilote structurel indépendant des victoires d'un bras.
   La piste VRPTW possède un gain de code historique reproduit, mais son passage
   à un contrat commun portable exigerait un travail distinct : les 22 transferts
   historiques de code entre tâches sœurs échouaient tous par signature avant
   même la mesure de qualité. Source : [géométrie](../benchmark/GEOMETRY_REVIEW.md).

7. **Enregistrer ce que l'expérience pourra conclure, et ce qu'elle ne mesure pas.**
   Geler les sélections avant tout test, conserver invalidité et fallback séparés
   des valeurs objectives, et accepter le résultat de chaque réplication. Une
   amélioration R − I autoriserait une conclusion sur le feedback et ce budget,
   sous ce benchmark et cette procédure de déploiement. Elle ne démontrerait ni
   nouveauté algorithmique, ni amortissement, ni avantage d'une récursion plus
   profonde. Une largeur supérieure n'est pas une récursion supplémentaire.

   Pour tester l'amortissement, mesurer dans une expérience dédiée le coût de
   l'apprentissage d'un prior et son réemploi sur plusieurs tâches nouvelles,
   face à un prior fixe informé et une génération indépendante de même budget.
   Le précédent W2 reproduit un mécanisme d'ordre dans un menu, où `nearest`
   manuscrit était déjà premier ; il ne classe pas les moteurs méta. Ne pas
   ressusciter UC4 +0,163, Probe K +4,8 ou la réduction de bruit prose comme des
   succès récursifs : les premiers sont rétractés et la dernière concerne la mesure.
