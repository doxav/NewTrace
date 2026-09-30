# Après P1 : précision et prochaine question scientifique

**Ne pas relancer R inchangé dans l’espoir qu’un effectif plus grand le fasse
gagner.** P1 fournit un signal exploratoire défavorable au feedback riche tel
qu’il a été déployé. Les corrections d’instrument sont utiles et validées, mais
elles n’établissent pas un avantage de ce mécanisme. La piste pratique suivante
la mieux motivée est de confirmer, sur données nouvelles, la recherche C fondée
sur la réécriture de parents sélectionnés par l’apprentissage, face à I. Cette
proposition change explicitement la question ; elle ne requalifie pas P1 en
réussite du feedback riche.

Cette note utilise uniquement les résultats complets conservés. Les
[calculs analytiques](future_design_calculations.json) enregistrent les sources,
leurs empreintes, tous les deltas et les hypothèses. Aucun nouvel appel de modèle,
aucune exécution de candidat, aucune simulation de nouveaux résultats et aucune
modification des gels ne sont effectués.

## Ce que disent les six réplications

Le contraste central enregistré R−I vaut **+0,045257 d’AUC**, avec intervalle
bootstrap **[+0,003176 ; +0,085126]**, donc un signal négatif exploratoire pour R
selon la règle gelée. Son écart-type apparié est **0,057022** et son erreur
standard descriptive **0,023279**. Quatre deltas sont défavorables à R et deux
favorables. Le bootstrap reste fragile avec six unités, sur un seul panel
d’audit fixé ; il ne justifie aucune généralisation universelle.

| Seed externe | R−I | Moyenne des cinq autres seeds |
|---|---:|---:|
| 16411 | −0,031562 | +0,060621 |
| 16423 | −0,006253 | +0,055559 |
| 16437 | +0,076279 | +0,039053 |
| 16441 | +0,043716 | +0,045566 |
| 16453 | +0,125118 | +0,029285 |
| 16467 | +0,064246 | +0,041460 |

Le signe de la moyenne est donc conservé dans les six retraits descriptifs.
Ce contrôle de sensibilité ne retire aucune seed de l’analyse et ne constitue
pas un nouveau test de significativité. Il contraste avec EXP-15, où le retrait
descriptif de la seed 53 inversait le signe moyen du contraste A2−A1.

Les autres contrastes enregistrés doivent rester visibles : R−C = +0,074929,
IC [+0,034075 ; +0,116730] ; W−R = −0,009935,
IC [−0,054857 ; +0,030305] ; R−A0 = −0,056813,
IC [−0,090926 ; −0,023830]. R améliore le seed initial tout en perdant face aux
deux contrôles génératifs I et C. W n’établit pas de gain sur R. W−R change
simultanément la largeur et le nombre de tours séquentiels.

Les programmes sélectionnés ne déclenchent aucun fallback pendant l’audit P1.
R possède 41 candidats éligibles sur 48, I 46/48, C 44/48 et W 43/48. Aucun
pool R n’est dépourvu de remplacement éligible ; une seed conserve néanmoins
le programme initial selon la sélection enregistrée. Les invalidités réduisent
les occasions de recherche, mais « aucun programme disponible » n’explique
pas ce résultat. Leur contribution causale précise n’est pas isolée.

## Dimensionnement : scénarios, pas gains projetés

Pour une étude future **inchangée dans ses propriétés de variance**, posons
`σ = 0,057022`, estimé sur les six deltas R−I. Une approximation normale avec
test bilatéral à 5 % et puissance nominale de 80 % donne
`n ≈ ceil(((z_0.975 + z_0.8) × σ / |δ|)²)`.
Pour viser une demi-largeur `h` d’intervalle normal à 95 %,
`n ≈ ceil((z_0.975 × σ / h)²)`.

| Effet absolu utile δ, ou demi-largeur h | Paires pour effet δ, puissance approximative 80 % | Paires pour demi-largeur h |
|---|---:|---:|
| 0,005 | 1 021 | 500 |
| 0,010 | 256 | 125 |
| 0,020 | 64 | 32 |
| 0,030 | 29 | 14 |

Les effectifs sont des **nombres totaux de nouvelles paires**, pas des seeds
à ajouter opportunément à P1. Les effets δ sont des hypothèses de planification
choisies pour leur intérêt pratique ; ce ne sont pas des bénéfices attendus.
Augmenter n peut confirmer une perte aussi bien qu’un gain.

À variance constante, les demi-largeurs normales approximatives seraient
0,045626 à n = 6 ; 0,042242 à n = 7 ; 0,032263 à n = 12 ;
0,022813 à n = 24 ; 0,016131 à n = 48 ; 0,011176 à n = 100.
Passer simplement de six à sept réplications ne résout donc pas un problème
de précision visant de petits effets. Ces valeurs **ne recalculent ni ne
remplacent les intervalles bootstrap enregistrés**.

Hypothèses indispensables : paires externes indépendantes et comparables,
processus de génération/sélection stable, variance future proche de celle
estimée et même portée conditionnelle du benchmark. Ni la couverture ni la
puissance du bootstrap futur ne sont garanties par cette approximation.
Une erreur de facteur 1,5 sur σ fait passer le scénario δ = 0,020 de 64 à
144 paires ; un facteur 2 le porte à 256. Ces multiplicateurs sont des tests
de sensibilité, pas des bornes de confiance sur σ.

L’écart-type EXP-15 de 0,038660 était plus petit. Les études changent à la fois
les données, la génération et le processus de recherche : cette différence ne
démontre pas que l’agrandissement du panel a augmenté la variance. Les nombres
calculés pour R−I ne doivent pas non plus être transférés aveuglément à C−I ou
à une nouvelle représentation de feedback.

## Bilan des explications concurrentes

| Preuve | Conclusion soutenue | Limite |
|---|---|---|
| S0 | Trace appauvrie, objectif anytime non explicite, ancien canal reconstruit séparément | Une correction d’instrument n’implique pas un gain du modèle |
| G1 | 8/12 puis 10/12 candidats éligibles ; plafond 32 000 choisi par la règle préalable | Petit diagnostic de faisabilité ; ne prouve pas qu’augmenter encore le plafond améliorerait R |
| F1 | Riche−sparse = +0,049229, IC [+0,000160 ; +0,103587] : signal négatif exploratoire | Un parent fixé, une proposition, fallback commun ; ne teste pas la recherche complète |
| S1 | Le panel 24 × 2 stabilise les choix dans la banque fixe ; amélioration descriptive de sélection de 17,3 % face à 6 × 1 | Pas une projection d’effet du feedback génératif |
| P1 | Canal réel, panel plus grand, objectif explicite et plafond accru n’établissent pas l’avantage de R ; R perd face à I et C | Corrections combinées ; contenu, longueur, validité et service ne sont pas causalement séparés |
| B2 prospectif | AUC 0,031633, B2−A0 = −0,147696 ; R−B2 = +0,090884, défavorable à R dans les six seeds | Référence fixe, même petit benchmark ; pas une découverte récursive ni une preuve sur des familles nouvelles |

B2 confirme sur le panel indépendant que battre A0 peut provenir d’une simple
initialisation. Cela explique la faiblesse de cette référence, **pas la cause
isolée de R−I**, puisque les bras partageaient le seed. La métrique compte :
le regret final moyen de C est 0,002646, celui de B2 0,014285, alors que B2
possède une meilleure AUC moyenne. Garder les deux métriques évite d’assimiler
qualité terminale et rapidité initiale.

F1 et P1 concordent sur une absence de bénéfice démontré de ces formes de trace
riche ; ils n’identifient pas une cause unique telle que la longueur du prompt,
les priors du modèle ou une famille d’objectifs trop simple. Dans P1, R utilise
3 176 939 tokens contre 314 858 pour I et 453 456 pour C. Le contenu ajouté a
un coût réalisé substantiel ; sa seule quantité ne mesure pas son utilité.
Une mémoire alignée de plusieurs essais rejetés n’a toujours pas été testée.

## Bras à conserver et conditions d’une prochaine confirmation

**Priorité proposée : I et C, avec B2 comme référence fixe exigeante et A0
comme lien historique peu coûteux.** C exploite déjà une information implicite :
le parent est choisi à partir des évaluations d’apprentissage. Il ne faut donc
pas le qualifier de recherche sans feedback au sens large. Sa différence
moyenne C−I de −0,029672 est prometteuse descriptivement, mais **ce contraste
n’était pas enregistré parmi les contrastes P1**. Il devient une hypothèse à
confirmer sur nouvelles seeds et nouvelles instances, pas une conclusion acquise.

Pour ce contraste futur, σ descriptif vaut 0,048137. Les mêmes scénarios normaux
δ = 0,005/0,010/0,020/0,030 donneraient 728/182/46/21 paires. Ils restent
conditionnels à six observations et ne constituent pas une recommandation
automatique de taille. Choisir d’abord un effet minimal d’intérêt et un budget
acceptable ; si la précision utile est inabordable, l’annoncer plutôt que
déclarer cinq ou six réplications suffisantes par défaut.

Si la prochaine question porte explicitement sur le feedback textuel, conserver
**I, C et R**. Tester une représentation plus compacte ou plusieurs tentatives
alignées demande d’abord un diagnostic prospectif séparé, symétrique et borné ;
aucune de ces solutions ne peut être annoncée gagnante. W peut être différé
pour concentrer les ressources : P1 n’a pas isolé un bénéfice de largeur et son
contraste mélange deux paramètres. Ce choix futur n’efface aucun résultat W.

Avant toute nouvelle confirmation, fixer une comparaison centrale, les critères
de décision et la gestion des comparaisons secondaires ; utiliser de nouveaux
panels et seeds sans fusion opportuniste avec EXP-15/P1 ; conserver invalidités,
budgets de réponses égaux, sélection isolée et repli commun. Garder B2 comme
comparateur est distinct d’en faire le seed commun : ce dernier changement
modifie le point de départ de tous les bras et doit être enregistré comme tel.

Apparier les ordres de bras à leurs inverses équilibre leurs précédences.
Dans P1, R−I était équilibré 3/3, mais C précédait R et R précédait W dans 5/6
réplications. La dérive du service reste une confusion possible, non démontrée.
Des panels d’audit indépendants supplémentaires seraient nécessaires pour élargir
la portée au-delà du panel fixe ; leurs tâches ne deviennent pas pour autant
des réplications externes de génération.

On peut recommander avec confiance des instruments corrects, un contrôle initial
plus fort et une question plus précise. **Aucun gain chiffré futur du feedback
récursif n’est actuellement justifié.** Cette enquête n’établit ni nouveauté
algorithmique, ni avantage de profondeur récursive, ni amortissement.

Sources principales : [analyse P1](../../../../EXP16/results/production_run/analysis_results.json.gz),
[contrôle B2](../production_baseline_control/results.json),
[protocole P1](PROTOCOL_P1.md), [analyse gelée](analysis_protocol.md),
[audit de l’ordre](ORDER_AUDIT.md). Les hashes des fichiers JSON physiques,
y compris les versions gzip, figurent dans le fichier de calculs associé.
