# EXP-15 : premiers points et premier terme de l’AUC

**Aucune des 180 trajectoires holdout EXP-15 ne commence exactement au centre des bornes.** Ce constat inclut les trois bras et les cinq seeds externes. B2 ne reproduit donc pas une initialisation midpoint effectivement sélectionnée dans ces déploiements EXP-15 ; il teste une autre intervention sur le seed, avec des fixtures publiques différentes.

Ce complément est descriptif et rétrospectif. S0 conservait déjà les contributions du premier terme, mais pas le décompte des points centraux. Les calculs ci-dessous utilisent exclusivement les observations et courbes EXP-15 sauvegardées. Aucun nouvel objectif, candidat, modèle ou ensemble F1/P1/E1 n’a été utilisé ; aucune étape prospective n’est créée.

Un point est compté au centre si toutes ses coordonnées sont exactement égales à `(low + high) / 2`, soit zéro pour les bornes gelées `[-5,5]`. Une seule coordonnée nulle ne suffit pas. Chaque case contient 12 trajectoires.

| Seed externe | A0 au centre | A1 au centre | A2 au centre |
|---:|---:|---:|---:|
| 11 | 0/12 | 0/12 | 0/12 |
| 23 | 0/12 | 0/12 | 0/12 |
| 37 | 0/12 | 0/12 | 0/12 |
| 41 | 0/12 | 0/12 | 0/12 |
| 53 | 0/12 | 0/12 | 0/12 |
| Total | 0/60 (0 %) | 0/60 (0 %) | 0/60 (0 %) |

Par exemple, les premières propositions A1/41, A2/41 et A2/53 sont identiques à dimension fixée : `[0, −1,666…]` en 2D, et `[0, −1,666…, −3, −3,571…]` en 4D. Elles ne sont pas le centre complet. Les coordonnées exactes et les observations initiales de chaque trajectoire sont conservées dans le JSON.

La décomposition utilise le regret normalisé déjà enregistré, sans recalcul de normalisation : pour chaque trajectoire, `AUC = r(1)/32 + Σ[t=2..32] r(t)/32`. On moyenne dans chaque strate, puis à poids égaux entre les six strates et entre les cinq seeds externes. Un delta négatif favorise A2.

| Bras | AUC moyenne | Contribution du premier terme | Contribution des 31 termes suivants |
|---|---:|---:|---:|
| A0 | 0.139578879 | 0.031592540 | 0.107986340 |
| A1 | 0.121748122 | 0.029477907 | 0.092270215 |
| A2 | 0.116680353 | 0.026705128 | 0.089975225 |

| Seed externe | Contraste | ΔAUC totale | Δ du premier terme | Δ des termes suivants |
|---|---|---:|---:|---:|
| 11 | A2-A0 | 0.000000000 | 0.000000000 | 0.000000000 |
| 11 | A2-A1 | 0.015787335 | 0.000000000 | 0.015787335 |
| 23 | A2-A0 | -0.016704959 | 0.000000000 | -0.016704959 |
| 23 | A2-A1 | -0.006095567 | 0.000000000 | -0.006095567 |
| 37 | A2-A0 | 0.000000000 | 0.000000000 | 0.000000000 |
| 37 | A2-A1 | 0.038199285 | 0.000000000 | 0.038199285 |
| 41 | A2-A0 | -0.054212155 | -0.010573164 | -0.043638991 |
| 41 | A2-A1 | -0.007632385 | 0.000000000 | -0.007632385 |
| 53 | A2-A0 | -0.043575517 | -0.013863892 | -0.029711625 |
| 53 | A2-A1 | -0.065597510 | -0.013863892 | -0.051733619 |
| Moyenne | A2-A0 | -0.022898526 | -0.004887411 | -0.018011115 |
| Moyenne | A2-A1 | -0.005067769 | -0.002772778 | -0.002294990 |

Pour A2 − A1, la différence du seul premier terme est nulle dans quatre seeds sur cinq ; seule la seed 53 contribue à sa moyenne non nulle. En particulier, A1/41 et A2/41 ont le même premier terme, puis des AUC différentes. Toutes les trajectoires favorables ou défavorables sont présentes, y compris les contrastes positifs des seeds 11 et 37.

**Cette identité arithmétique n’est pas une médiation causale.** Le premier point peut rester incumbent et influencer les propositions ultérieures ; ces effets sont inclus dans les 31 termes suivants. Les politiques comparées peuvent également différer sur de nombreux autres comportements. Retirer `r(1)/32` ne simule donc pas une initialisation commune, et le reliquat ne mesure pas à lui seul un effet du feedback sur l’adaptation. Cette décomposition ne remplace ni une ablation prospective ni l’analyse d’incertitude confirmatoire EXP-15.

Le lien défendable avec B2 est limité : B2 démontre un levier d’initialisation du seed sur ses propres fixtures, tandis que ce décompte précise que le gain numérique moyen observé dans EXP-15 ne correspond pas à une adoption du centre complet. Ces deux résultats ne quantifient pas le gain futur récursif − indépendant.

Vérifications réalisées sans exécution scientifique : 180/180 lignes distinctes ; 12 tâches par bras et seed, deux par strate ; mêmes identités de tâches et seeds locaux entre bras ; budgets et longueurs de courbe égaux à 32 ; sources identiques aux hashes sélectionnés ; sélections antérieures au holdout ; zéro ligne invalide ou fallback dans ce jeu. Les hashes physiques des cinq fichiers holdout correspondent à ceux de S0. Les AUC reproduisent les agrégats EXP-15 et les premiers termes reproduisent S0, avec contrôle arithmétique de la décomposition.

Les helpers existants `statistics.audit.read`, `benchmark.digest`, `benchmark.source_hash` et `benchmark.aggregate` ont été réutilisés. Aucun helper d’évaluation ni de génération n’a été appelé. Les assertions de contrôle ont passé ; aucune réécriture d’une preuve antérieure n’a été effectuée.

Preuve complète : [EXP15_FIRST_POINTS.json](EXP15_FIRST_POINTS.json), digest canonique `e9e4293076443bbc697368aa1d1d28d6d9dc2882f621d09f68eccd862dd0f673`. Le fichier garde toutes les 180 lignes descriptives, leurs hashes bruts, observations initiales, identités et sources, les agrégats par seed et les hashes des entrées. Sources antérieures : [audit S0](../statistics/diagnostics.json), [rapport S0](../statistics/REPORT.md), [rapport B2](B2_REPORT.md).
