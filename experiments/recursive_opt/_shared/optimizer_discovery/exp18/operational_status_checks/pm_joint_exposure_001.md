# EXP18 — exposition conjointe mémoire/Pareto, contrôle 001

Instantané : 2026-09-11T02:36:39.908435+00:00 → 2026-09-11T02:36:39.925108+00:00 UTC ; fin locale 2026-09-11T04:36:39.925108+02:00.

Périmètre : **PM / outer18011 / slots00–14**, quinze réponses terminées sous `EXP18-MECHANISMS-v1`. Gel `ce88c4b2d9530456e63b497f7bbde7608fbf3d8aca441c13c086d77b6beeea8a`. Les slots15/16 et les autres bras ne sont pas inspectés.

**Sept sources historiques complètes figurent effectivement dans chacune des sept requêtes consommées08–14.** Le contrôle détaillé porte sur08 (première disponibilité de sept sources) et09 (première de ces requêtes choisissant un parent différent du meilleur scalaire). Ce choix de cas dépend de l’exposition enregistrée, jamais d’un résultat d’efficacité.

Ce relevé prolonge [le contrôle P](pareto_exposure_001.md) et [le contrôle mémoire M](memory_exposure_001.json), sans répéter leurs audits ni étendre leurs conclusions. Il vérifie cette fois les deux mécanismes dans les mêmes requêtes PM.

## Couverture réelle des requêtes consommées

| Slot | Sources antérieures complètes | Caractères du bloc mémoire sérialisé |
|---:|---:|---:|
| 00 | 0 | 281 |
| 01 | 1 | 3618 |
| 02 | 2 | 6366 |
| 03 | 2 | 6773 |
| 04 | 3 | 12638 |
| 05 | 5 | 19142 |
| 06 | 6 | 23214 |
| 07 | 6 | 25120 |
| 08 | 7 | 27868 |
| 09 | 7 | 29405 |
| 10 | 7 | 29813 |
| 11 | 7 | 35122 |
| 12 | 7 | 42151 |
| 13 | 7 | 47934 |
| 14 | 7 | 49707 |

Pour ces quinze requêtes : source de chaque exemple complet identique à la réponse antérieure, hash correct, source du parent exclue, choix des sept sources distinctes non vides les plus récentes, cutoff strict, hash/longueur du texte et occurrence exacte à la fin du second message vérifiés. Aucun exemple supplémentaire n’a été créé pour atteindre sept.

## Deux requêtes authentifiées en détail

| Contrôle | Slot08 | Slot09 |
|---|---|---|
| Front reconstruit | [-1, 1, 2, 7] | [-1, 1, 2, 7] |
| Origine choisie | 7 | -1 |
| Meilleur scalaire (origine) | 7 | 7 |
| Origines des sept codes complets | [6, 5, 4, 3, 2, 1, 0] | [8, 7, 6, 5, 4, 3, 2] |
| Tentatives référencées comme parent courant | [7] | [] |
| Slots antérieurs | 8 | 9 |
| Sources complètes | 7 | 7 |
| Omissions par limite de sources | 0 | 2 |
| Omissions par limite de caractères | 0 | 0 |
| Slots de source dupliquée | 0 | 0 |
| Slots sans code | 0 | 0 |
| Caractères mémoire | 27868 | 29405 |
| Octets UTF-8 mémoire | 27870 | 29405 |

L’origine `-1` désigne le seed initial. Les indices ci-dessus sont des identités de programmes, pas des scores ni des rangs de qualité.

Au slot08, le parent7 venait d’être généré : il reste dans les résumés de tentatives avec la mention de parent courant, et son code est exclu du bloc historique. Les sept autres sources0–6 sont conservées intégralement. Le parent choisi appartient au front de quatre membres et coïncide ici avec le meilleur scalaire.

Au slot09, le même front de quatre membres produit un choix différent : le seed est effectivement transmis alors que la meilleure origine scalaire enregistrée est7. Le bloc historique contient les sources8–2 ; les deux sources plus anciennes sont explicitement omises par la limite de sept. Les deux facteurs sont donc exercés dans cette requête terminée. Cela ne mesure pas leur bénéfice.

La limite porte sur **65 536 caractères sérialisés**, avec au maximum sept sources. Les octets UTF-8 sont comptés séparément. Les nombres de codes complets ne signifient pas sept programmes valides, ni sept comportements distincts ; les statuts typés originaux restent dans le contexte.

## Provenance et reconstruction

- Aux slots08/09, reconstruction de chaque record historique depuis sa réponse, sa requête et son reçu TRAIN d’origine : même bras/outer, source exacte, parent, slot_id, hashes de réponse/reçu/record, statut, compteurs et projection autorisée. Les métriques lues pour vérifier cette projection ne sont pas reproduites dans ce rapport.
- Vérification des reçus disponibles au cutoff original : réponse terminée avant son reçu, reçu observé avant snapshot_ns, aucun index futur. Recalcul du digest ordonné de tous les records, des entrées de provenance, des exclusions/dédoublonnages et de tous les compteurs sauvegardés. Aucun résultat ultérieur n’est ajouté au snapshot.
- Reconstruction indépendante de la dominance stricte sur les seuls vecteurs TRAIN antérieurs, de la déduplication, du meilleur scalaire, du digest d’archive et du tirage uniforme SHA256/Random. Concordance avec les deux décisions sauvegardées et l’alias du hook PrioritySearch.explore gelé.
- Hash du parent choisi identique au code réellement présent dans le second message, au current_context et à la décision. Le texte propagé par Trace, sa projection compacte et l’intégralité du bloc mémoire correspondent exactement à la requête sauvegardée.
- Les quinze décisions reliées aux réponses contrôlées portent will_generate=true. Aucune décision terminale inutilisée n’est comptée comme consommée ; le terminal prévu16 est hors périmètre.
- 82 entrées physiques ont été lues puis re-hashées sans modification. Seul ce nouveau rapport opérationnel est écrit.

## Identités des deux requêtes

| Champ | Slot08 | Slot09 |
|---|---|
| Snapshot original (ns) | `1789091736851848507` | `1789092103917453578` |
| SHA256 parent | `d7b81a2b63be928f80bfc360cf6d2bb22b2a668986c06ebe8b1f3f4addbc31ad` | `5b74e5a3fe2fc90fcb42a603b058fa7befbf65f38775caa77acd30cdf619b640` |
| SHA256 mémoire UTF-8 | `19b63fb54785418fa6f6da5b2ba5aa90ebb0a027dbb170ddce05da51220163ce` | `642baefc7b22161bd8aef023419c2f047aa07ec3598524665cd8746097b9325b` |
| SHA256 fichier request.json | `202653386d78dde7f81cfe3c19e02ea0c36ce68ef576ba91e706135a9381276d` | `9ad96c57c56cfe1a714af8a8ac365e5f6d17e09854aef65cc2a695605e02aad0` |

## Limites

Ceci est une validation d’exposition, pas une analyse d’efficacité. Aucun regret, score TRAIN, tendance de performance ou contraste inter-bras n’est rapporté. Aucun accès à la validation, à l’audit/holdout ou aux lignes du cache ; les définitions de splits présentes dans le gel n’ont pas été utilisées. Aucun appel modèle, objectif, candidat, test ou interaction avec les processus en cours.

Les vecteurs des reçus ont seulement servi à reconstruire les fronts08/09 ; les observations brutes sous-jacentes ne sont pas réévaluées. La lecture statique et les sources gelées documentent le hook qui retourne un vrai candidat conservé ; les fichiers ne recréent pas rétrospectivement l’identité mémoire de l’objet Python. La politique Pareto combine filtrage et tirage uniforme. La présence de sept sources ne prouve aucun seuil universel d’apprentissage.

## Empreintes des entrées

Chemins relatifs à la racine du dépôt ; hashes physiques avant décodage gzip éventuel.

| Entrée | SHA256 |
|---|---|
| `artifacts/optimizer_discovery/exp17/study.py` | `f16cfd1eac80f1aa27584e6f654ff5c43abe46aeb57265fd67496bc4aeb48b5b` |
| `artifacts/optimizer_discovery/exp18/memory_projection.py` | `0dc1a6d414b60176e737f8d914561cfcf01edcbb47d4e186e92b414e71210391` |
| `artifacts/optimizer_discovery/exp18/operational_status_checks/memory_exposure_001.json` | `56894c1322cf5374b8ecd1c8a118cb9174158d2b859bf1f5c865fc4173a31348` |
| `artifacts/optimizer_discovery/exp18/operational_status_checks/pareto_exposure_001.md` | `d1fa6dde2d364459c22e4629d5ab6970351ac4bb88a54c6e5cab7bed3efbe9b4` |
| `artifacts/optimizer_discovery/exp18/pareto_selection.py` | `bb76350204c75a4d96aa4ef46082f347874926ee5b656f00c698d45c270de7b3` |
| `artifacts/optimizer_discovery/exp18/pareto_trainer.py` | `a0c1bd1d452f0a2ae5ff8e2e328a3293b87f7a914b63fff3e9b012d8d258f500` |
| `artifacts/optimizer_discovery/exp18/run/freeze.json` | `9abb465cda1ea55c9cce7e5e784621a9354033c1f6f4c82960f2b3ce917354be` |
| `artifacts/optimizer_discovery/exp18/run/freeze_sha256.json` | `9bb65676d2a9fc9d6675b32d5639f1c0960c6b2f70ad453a014ea6552188ec4d` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/parent_decisions/slot_00.json` | `7907065f67f97e0f3b2af114c9bc09c3172e7d3d57ff14492124dac8cdda634a` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/parent_decisions/slot_01.json` | `b111abcecc96e6c5c776401ea8a66bcd4e03c2f80bf406846aa6344859ddcc3c` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/parent_decisions/slot_02.json` | `0d4c52fc5d43a23f03e186c041adc5d9f7c6a89c5f0ab9e0f21149364e1c9a4f` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/parent_decisions/slot_03.json` | `f60e7396533deca8c5b84941ea1f00b7bdca500c774484a6ef01cf5679447510` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/parent_decisions/slot_04.json` | `fffb99cdf27bfeff187ed5ab6bb1bdcf0f27863662d6370d70c62c54e2571bca` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/parent_decisions/slot_05.json` | `456edd4f4164e84f076bf7ec9a836942eeff63077fcce8ab3c8c5440b0ce15fd` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/parent_decisions/slot_06.json` | `7e07e6a216970f1ce41fb856c971cef8b477844542ed4694d6d9844d2b0a3ea9` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/parent_decisions/slot_07.json` | `5b1a240227112fedcf8db5c1706821b5fdccb9f52c6069ade9966d7c6374e611` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/parent_decisions/slot_08.json` | `9f7416cd92d4ef79863dc170161f87fb5492c69adcc70d8cd4417a88b452800b` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/parent_decisions/slot_09.json` | `53e495dc0e271f344e5a2bf8c064ba724f550703277ce9b292607b0ded50eb21` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/parent_decisions/slot_10.json` | `ae32d605a43b0c4fd1316c2b49562f1a394dc084da70db0780f2ce90a3029780` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/parent_decisions/slot_11.json` | `8335a5ad6ffcd8128acc47aa593d3d9a58f5b0f806808b368906b1c3780528d9` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/parent_decisions/slot_12.json` | `b0675018adbae48ba1b589d418a660073518d6ebef89b8b5aed4d30f9665cf3a` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/parent_decisions/slot_13.json` | `193adc9623c0c5a48a4d6d01f5681f62944eaa6ec56664c97e12d182e1127f4b` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/parent_decisions/slot_14.json` | `1baa62016aebda4d303a94b999cf87d1522aec28e49c93c4d5786973d91d8dc5` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/seed_train_receipt.json` | `73a02ac78a1032c8859279761e32c7056cb1319d35f4a623d60985eb8d906964` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_00/current_context.json` | `11b8b4a50110012659368a2f72f027f5d88d9a1d6b433ea63b264174e9689b79` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_00/request.json` | `1d3c54140cda0fef39ef99de60a988fee16b66373fb9661e6c13a6300c519ccb` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_00/response.json` | `325170c3a5ed306d05d6b2b9670d9f12533791df992454cc7a3990cb0f5ef83a` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_00/train_receipt.json` | `e0ffc84c57ed7199caaba5a3a6409d720c9c2963e60011a3d482a9e859349046` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_01/current_context.json` | `3f257dbaa2461fd47e7ab5fb267667c8673a9a537b55477c3ebe0e0ed33ae4c2` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_01/request.json` | `22dbbf2367941df6eb5b557631261ce69a712bdb6fae9b216238b3478b04085c` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_01/response.json` | `9480ed32c5e1acd05a5b7614f4e1d9ffee15969fee6d3fa8886c598574bfc317` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_01/train_receipt.json` | `9b9c7149d83daed150bc53fef5e65cd80c517878fb62800dc30d746656ba4483` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_02/current_context.json` | `a95cb93781becb0b14f31f5a0f35cab17635bda4fb8c32d39b4b2cd84ca3ef80` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_02/request.json` | `c422d5eafaecec74b18ffb2cf96d22b7f429e76625e6df0b7dc508143824068a` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_02/response.json` | `87b29022dec2bf12878ecd6966ae63079dc66672e91b7e0f345ec6f966b821d6` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_02/train_receipt.json` | `a2c79f0376d7246b4003d1cd2753dbca98efdef486e5131dd1ee71230d4b7c21` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_03/current_context.json` | `fc592ca791795f18cba4b0a971329e6945968ead0d0638d5ed834619ba8e398f` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_03/request.json` | `6e01f2443061a75a9a86a6806fe4f551698d9f99ee498426c4de896fd88374ba` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_03/response.json` | `5492366275f0f37b92d0ec61f64579767bcd1e1bc6d66a6c365d02626284ba1e` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_03/train_receipt.json` | `736e51a54c994dde189c392e6150b72139158e2360017bea686d2fa9343f57d2` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_04/current_context.json` | `678aec021b1b3b5f2a33e0720f33bd7735832bf9995f278131261a3c60fa56b7` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_04/request.json` | `1d9ee914ae61f66ad9dcc8fd86675d2061445055dfb704fb2eb4d88f236983ff` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_04/response.json` | `cc665e2032179ca410e03e1ffe13a57545e5e0e11980e7d999ca7832e4077c0c` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_04/train_receipt.json` | `f838f39bfdb2f08587b19a515b947ef6b7575a2ff5f2e9ba88d5e29a74e7c5e6` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_05/current_context.json` | `6b933fb2a6ed1995997fb50e8dee70baab8b4190343abc4739c85e00ed497e62` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_05/request.json` | `3e72edcf6015b0d47e65fc28f8ee26aaedc6491014bb186b92fc59f6f9069e79` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_05/response.json` | `e3ca83aee5b122249ec53217e21d2f98c5030e63da089b9decf577a20258dec5` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_05/train_receipt.json` | `5c8dc9cdbc26ca23f76f0b2180d5c609ccd6b43dc6dcad0a6cd28d227d9df1f7` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_06/current_context.json` | `80132101224c971bba6e0c89abc1eb88c0bb11dc52e96187bee748eda8e60e18` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_06/request.json` | `687f54efa2eba72b6b551becf9067eeafb896e5b8675d7aa64782490904c71db` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_06/response.json` | `207ed0d6b36cf4c03f3f5a65b2ff2ff2cc2691410d78794f10db16355af911fb` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_06/train_receipt.json` | `d303b4214c6cde163153c41f4997aecefdc4a7dbf342051a5f6416f129dc0e48` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_07/current_context.json` | `3b5e621329eb90a0d868712aef59023b847d644e327eac4b8dfc72602f4f96a3` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_07/request.json` | `88ef4334bc85acc1cc8e23def6b846bb1b0b9036f42510b2fdcf63c57ba6f282` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_07/response.json` | `0e83928303e59c2155cbc50d2631cd90b3ad0c4d386d3fe9876506ed93cceaf2` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_07/train_receipt.json` | `9934c92f5dce0205f3ec0c9a24c47221ee5d365b78878d255973fbbcd624dd58` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_08/current_context.json` | `b4ada33a49e44e90c39f97bc8f2ce10d14a9825defe3a12224cbd6015561d74d` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_08/propagated_feedback.json` | `e81f065fd307ae49bd7ea9955ddf876bb33548a47648eebd76d1aeedcff4e809` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_08/request.json` | `202653386d78dde7f81cfe3c19e02ea0c36ce68ef576ba91e706135a9381276d` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_08/response.json` | `ca4a14387b233fa1b36fe4e84e827576335ab929d6896b1de1c901f6df996a20` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_08/train_receipt.json` | `52eb9f8d880dc712bcb0900fd5c8f593fb2a7f3a863c5aaa3c12376d6f1a4e07` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_09/current_context.json` | `e17ac7fd124d9bc3dfb524321eac3a3b74b18546de57377f7c8226fb4e76cdd5` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_09/propagated_feedback.json` | `279b9b2cf8b45d669af7896c41d79ab3a4073bb7a22323baa6c61057a9ad937d` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_09/request.json` | `9ad96c57c56cfe1a714af8a8ac365e5f6d17e09854aef65cc2a695605e02aad0` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_09/response.json` | `41dc33f05651d40af62f7b41cfeab24dbdd959a62ac78181756191251fb6918b` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_10/current_context.json` | `f6d29fda5d9794699ee9582f2157400954368c9b340076009923c5dcd575df12` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_10/request.json` | `4c59e5a99a91502ac7898cf28d74d2f9e6b513bd3cf16acfb942a5ec19eb0b26` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_10/response.json` | `3bb5d210de8f0e7ea20d2d0a851c82609feee0ecaff246f631eae29c7abc82bc` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_11/current_context.json` | `132e2230ed538f4c731e13b73bd36fe89cdb6ba6c6062181157d7ed1af048510` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_11/request.json` | `f8054af1fc52b242bb62fe4bbb76a5a840223519b8c75c2b375841e680c27b96` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_11/response.json` | `46b5121e3faa9dc5912b9ff753c386bcb6eb83d02ff8289cce22645c1044b270` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_12/current_context.json` | `47c9a2e995cb7eb799f528180f00fa948cc28941c88b65d7e59191e94f8f7616` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_12/request.json` | `8c9fbd29f321e0b9c6fa847bbd88d2de8cb22775e83a976807344a1f473527dc` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_12/response.json` | `b5a9b0a7ad4ba014d61a1c2d4649559d4b689793f7489d8a1f580c68673a233c` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_13/current_context.json` | `2fb77530530dc17ce2cb0c4df7a9bd0721cc403ab85eacd5ce0defab87f2e778` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_13/request.json` | `aae96b4419a670c0ff61257dfbedd998de6ef232e79eac7cc0d92eb16dce97e3` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_13/response.json` | `f2f2114d536b2626ac1eb0a5a9d008fccf677cb1325bd6b125282da133d9d7ff` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_14/current_context.json` | `a4ce6a6f59eec7ac9b53cf83299c5074263f5146595714417f7f7cc94204784f` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_14/request.json` | `8f69cac3aee434ab67bca0ef1b0e8abdce13753bfba7b8821f6782c5511fb86a` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/PM/slot_14/response.json` | `f5d5f37a274ddf7895ce8728a84385cda7019cfee51798d2df63180b793070b6` |
| `artifacts/optimizer_discovery/exp18/study.py` | `22c653204145fe77a2a8619a86182edb0ce1d8f83b886a9b730f06e2fc5670cf` |
| `opto/trainer/objectives.py` | `d08d14618f68d545bac86bf1e5089347edcd58c3d64f0f6e2553ff0f6f669a40` |
