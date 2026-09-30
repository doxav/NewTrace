# Revue indépendante du classeur des diagnostics terminés

**PASS.** Le contrôle porte sur le fichier XLSX exporté, avec les modules ZIP/XML de la bibliothèque standard Python et un évaluateur restreint de ses formules. Aucun runner de recherche, logiciel Excel, candidat, objectif, modèle ou mécanisme de chargement de clé n’a été exécuté. Aucune donnée P1 n’a été lue.

Classeur : [optimizer_discovery_data.xlsx](../../../../EXP16/presentation/optimizer_discovery_data.xlsx).
SHA-256 : `3b0375d5354ba6c03255a3e8d2fbd7fe248042b26be7389b86c9b5396f3527b8`.
Builder vérifié : `45e015bc5b361845c4aff7804122fbb4abbd52ff7c57ad726e5d78c0561e2087`.

| Vérification | Résultat |
| --- | --- |
| Archive | 67 membres uniques, CRC valides ; tous les documents XML et fichiers de relations sont analysables ; 47 relations internes résolues. |
| Structure | 17 onglets visibles, noms et ordre conformes ; 26 rectangles de tables et leurs colonnes correspondent exactement au registre du builder. Aucune ligne masquée. |
| Formules | Les 557 formules ont une valeur enregistrée : 365 numériques, 192 textuelles. Toutes ont été recalculées indépendamment et concordent avec leur cache. |
| Contrôles du builder | Les 332 résultats de formules explicitement contrôlés correspondent aussi au recalcul ; ses 738 comparaisons numériques respectent leurs tolérances. |
| Erreurs et contenus externes | Aucun type de cellule erreur ni texte d’erreur Excel détecté ; aucune relation externe, référence de formule externe, macro, feuille macro ou pièce VBA détectée. |
| Provenance | Les 204 hashes de fichiers sources correspondent à la fois au registre et à l’onglet Sources, y compris les octets gzip. Les chemins autorisés ont été contrôlés avant lecture. |
| Préservation | Les 204 sources, le builder et le XLSX sont inchangés avant/après. L’analyse n’a écrit que ses propres fichiers de revue et son vérificateur. |

Les valeurs ont été comparées aux projections des sources dans 4 259 cellules : résultats EXP-15 par seed, générations EXP-15, G1, les paires B2, F1, ses contrastes et son usage. Les valeurs absentes sont distinguées strictement des zéros ; la tolérance numérique est `1e-12 × max(1, |attendu|)`.

| Données à préserver | Couverture vérifiée dans le classeur |
| --- | --- |
| EXP-15 | 80 slots distincts ; 21 candidats inéligibles et 16 fins `length` conservés. |
| G1 | 24 slots distincts ; six candidats inéligibles, avec leurs six cellules AUC laissées vides. |
| B2 | 96 paires distinctes ; 26 pertes AUC et 37 pertes de regret final conservées et correctement étiquetées ; 192 contrôles/variantes valides. |
| F1 | 24 slots distincts ; quatre sorties inéligibles, 24 trajectoires de repli, une terminaison fournisseur `error`, 26 tentatives et les quatre contrastes conservés. |

Le contrôle exact des bornes de tables exclut que ces nombres proviennent seulement d’un préfixe lu dans un tableau plus grand. Les hashes des sources dans B2 restent distincts de ceux des fichiers comprimés dans Sources. Les formules de libellé « gain/perte/égalité » conservent les issues défavorables ; les intervalles statistiques sont importés, sans nouveau bootstrap.

## Reproduction

Le programme [verify_workbook.py](verify_workbook.py) ne dépend que de Python standard. Par défaut, il contrôle sans écrire ; `--output CHEMIN_NOUVEAU` permet de conserver un nouveau compte rendu et refuse de remplacer un fichier existant.

```bash
/tmp/phase0-venv/bin/python artifacts/optimizer_discovery/investigation16/report_data/verify_workbook.py
/tmp/phase0-venv/bin/python -m black --check --target-version py313 artifacts/optimizer_discovery/investigation16/report_data/verify_workbook.py
/tmp/phase0-venv/bin/python -m ruff check artifacts/optimizer_discovery/investigation16/report_data/verify_workbook.py
```

Ces trois commandes passent. [independent_workbook_review.json](independent_workbook_review.json) conserve les hashes exacts, la liste des sources et des tables, les compteurs, les limites et le hash du programme reproductible. Le premier contrôle employait un script temporaire dont le hash est également préservé ; la version sauvegardée intègre le contrôle des tables et le mode sans écriture. Aucun changement ne concerne les chiffres scientifiques.

## Limites

Il s’agit d’un contrôle du contenu ZIP/XML et de la grammaire exacte des formules présentes, pas d’un test de toutes les versions d’Excel ou de LibreOffice, ni d’une validation universelle du format OOXML. Aucun moteur bureautique n’a été lancé. La mise en page, les rendus, l’impression et l’accessibilité relèvent de la revue visuelle séparée du coordinateur.

Les hashes prouvent l’identité des fichiers, pas à eux seuls leur validité scientifique. Toutes les formules ont été recalculées, mais les cellules importées des tableaux B1/S1/S0/T1 n’ont pas toutes fait l’objet d’une seconde projection indépendante ici ; elles bénéficient des hashes et contrôles supplémentaires du builder. Aucun objectif, aucune trajectoire et aucun bootstrap n’a été rejoué.

Le classeur couvre volontairement les diagnostics terminés. Sa validation ne clôture pas l’enquête EXP-16 ni le P1 en cours et ne transforme aucun gain descriptif en avantage récursif établi.
