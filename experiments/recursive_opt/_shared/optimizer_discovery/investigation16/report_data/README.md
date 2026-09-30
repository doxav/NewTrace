# Classeur des diagnostics terminés

Le classeur `outputs/01a07fdf-c364-79d1-bfe2-c467a140e0b7/optimizer_discovery_data.xlsx`
regroupe EXP-15 et les diagnostics terminés S0, G1, B1, B2, S1, T1 et F1.
Il fournit notamment les valeurs des figures scientifiques B1, B2 et S1.
Cet export a été construit pendant P1 et ne charge aucune de ses données.
P1 est maintenant terminé ; ses résultats figurent dans le supplément séparé
`optimizer_discovery_p1.xlsx`. Aucun des deux classeurs ne projette un gain récursif.

L'export utilise `@oai/artifact-tool`, sans exécuteur expérimental. Les 204 sources
sont autorisées explicitement et leurs hashes d'octets sont contrôlés avant
l'export. Le classeur conserve les 80 propositions EXP-15, 24 réponses G1,
96 paires B2 et 24 réponses F1, invalidités et pertes comprises. L'onglet Sources
permet de remonter aux fichiers exacts ; les hashes canoniques B2 sont distincts
des hashes des fichiers comprimés.

Les 17 onglets comportent des valeurs numériques typées, des deltas et des
agrégats par formules. Les intervalles statistiques viennent des analyses
enregistrées ; Excel ne remplace pas leur procédure de calcul. Une cellule vide
reste une valeur manquante. Les indicateurs booléens sont codés 1/0 uniquement
pour la validité et la conservation, jamais pour imputer un regret.

## Exécution et contrôles

Depuis la racine du dépôt :

```bash
/home/xav/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node --check artifacts/optimizer_discovery/investigation16/report_data/build_workbook.mjs
/home/xav/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node artifacts/optimizer_discovery/investigation16/report_data/build_workbook.mjs
```

Le runtime JavaScript existant fournit la dépendance ; aucune dépendance de
production du dépôt n'a été ajoutée. Le builder remplace seulement son export
et ses contrôles dérivés, pas les données scientifiques.

`build_03.log` enregistre un code de sortie zéro, 332 résultats de formules
comparés indépendamment et 738 comparaisons numériques au total.
`formula_error_scan.ndjson` rapporte zéro erreur Excel reconnue.
`workbook_checks.json` conserve tous les hashes, les conventions et les compteurs.
`visual_review.json` atteste la revue des 26 rendus : un aperçu de chaque onglet
et neuf plages supplémentaires. Les grands tableaux restent consultables par
défilement horizontal. La police Liberation Sans a été vérifiée localement avec
`fc-list`.

La revue indépendante ZIP/XML (`independent_workbook_review.json`) vérifie
l'archive et les 17 onglets, puis recalcule les **557 formules effectivement
exportées**, y compris celles qui n'avaient pas de résultat attendu enregistré
par le builder. Leurs valeurs en cache concordent ; aucune erreur de cellule,
macro ou liaison externe n'est détectée. Les 204 sources sont inchangées et les
tables EXP-15/G1/B2/F1 sont comparées aux données sources, pertes et invalidités
comprises. Cette revue ne lance pas Excel et ne remplace pas les audits
scientifiques des expériences.

SHA-256 de l'export historique contrôlé dans `independent_workbook_review.json` :
`3b0375d5354ba6c03255a3e8d2fbd7fe248042b26be7389b86c9b5396f3527b8`.

Le 10 septembre, une reprise du même builder a réexporté ce fichier dérivé avant
la coordination du supplément P1. Le builder et les données sources sont inchangés ;
le hash binaire de l'archive a changé. L'export actuellement livré a pour SHA-256
`ef871459db0413a0adda0d4a44f6f27f93b633f8fed2e09c870943f2bc58b2ce`.
Le contrôle indépendant a été relancé sous un nouveau nom de preuve,
`independent_workbook_review_r2.json` : PASS, 557 formules, 204 sources et
4 259 cellules projetées identiques. Les 26 aperçus ont exactement les mêmes
hashes que ceux de `visual_review.json`. Les anciennes revues restent conservées
avec leur hash historique ; aucune donnée scientifique n'a été remplacée.

```bash
/tmp/phase0-venv/bin/python artifacts/optimizer_discovery/investigation16/report_data/verify_workbook.py --output artifacts/optimizer_discovery/investigation16/report_data/independent_workbook_review_r2.json
```

## Corrections d'export conservées

Le premier essai (`build_01.log`) a échoué avant export : le critère booléen de
`COUNTIFS` donnait zéro au lieu de huit éligibles G1. Le codage explicite 1/0,
documenté dans le Guide, a corrigé cette représentation. Le deuxième essai a
passé les contrôles ; le troisième corrige la largeur du Guide, l'affichage des
identifiants de blocs et utilise une police réellement disponible. Les valeurs
scientifiques sources sont restées identiques. Aucun appel modèle, candidat ou
objectif n'a été ajouté par ces corrections.
