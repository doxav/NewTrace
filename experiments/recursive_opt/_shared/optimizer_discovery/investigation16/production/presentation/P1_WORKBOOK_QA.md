# Vérification du supplément P1

Statut : **PASS**. Ce nouvel export est séparé du classeur des études précédentes. Son builder ne lit ni ne modifie `report_data/` ou `optimizer_discovery_data.xlsx`.

Fichier livré : `outputs/01a07fdf-c364-79d1-bfe2-c467a140e0b7/optimizer_discovery_p1.xlsx`.

SHA-256 : `7b37934a397fa98f8ecf06dfff0966a8b175b1286afccc2709ef26b54b2407d4`.

| Onglet | Contenu et contrôle visuel |
|---|---|
| Données | 36 lignes, six seeds × six politiques ; indices B2 non applicables laissés vides ; sources exactes et tableau des moyennes. Titres, valeurs, unités, hashes et en-têtes lisibles. |
| Contrastes | Six contrastes, leurs 36 deltas appariés et IC bootstrap importés inchangés. Les quatre pertes de R face à I restent visibles ; direction et portée exploratoire explicites. |
| Recherches | 24 pools, huit réponses allouées chacun ; 18 générations non éligibles et une sélection du seed conservées. Allocations train/validation incluant le seed explicites. |
| Usage | Quatre bras, 192 réponses, 6 980 225 tokens et coût connu de 0,488195055934 USD. Les tokens de raisonnement ne sont pas additionnés au total. |
| Sources | Neuf fichiers et leurs SHA-256 ; définitions, censure, portée des bras et provenance de la figure. Aucun hash tronqué. |

Dix rendus PNG dans `workbook_previews/` ont été inspectés visuellement. Les seeds et noms de bras ont été centrés après le premier rendu pour éviter leur juxtaposition. Cette retouche ne change aucune donnée. Le classeur final ne présente pas de texte tronqué ni d’erreur de formule dans les plages inspectées.

Le contrôle indépendant `check_p1_workbook.py` lit uniquement CSV, JSON et ZIP/XML du fichier Excel. Il vérifie 658 cellules CSV/JSON, 670 cellules Excel/projection, les 138 formules et leurs valeurs en cache, les neuf sources et tous les hashes d’entrée. Les six seeds, six cellules B2 non applicables, pertes, invalidités et absence de repli sur l’audit sont conservés. Aucune erreur de formule détectée. Résultat détaillé : `p1_workbook_independent_checks.json`.

Les moyennes, médianes, deltas, fractions et totaux restent recalculables dans Excel. Les intervalles bootstrap sont copiés sans réestimation ; cette vérification ne prétend pas reproduire indépendamment le benchmark ou le bootstrap scientifique. Aucun modèle, candidat ni objectif n’a été exécuté.

Commandes de vérification :

```bash
/home/xav/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node --check artifacts/optimizer_discovery/investigation16/production/presentation/build_p1_workbook.mjs
/tmp/phase0-venv/bin/python artifacts/optimizer_discovery/investigation16/production/presentation/check_p1_workbook.py
/tmp/phase0-venv/bin/python -m black --check --target-version py313 artifacts/optimizer_discovery/investigation16/production/presentation/check_p1_workbook.py
/tmp/phase0-venv/bin/python -m ruff check --no-cache artifacts/optimizer_discovery/investigation16/production/presentation/check_p1_workbook.py
```

Ces commandes passent. Le contrôle des espaces de fin de ligne et marqueurs de conflit sur les trois nouveaux fichiers passe également. Le marqueur de création du skill Spreadsheets a été exécuté une fois avant le premier export de ce supplément. Le builder utilise le runtime Node et `@oai/artifact-tool` déjà installés ; aucune nouvelle dépendance de production.
