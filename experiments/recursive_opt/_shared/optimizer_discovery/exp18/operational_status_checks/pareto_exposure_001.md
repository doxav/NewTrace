# EXP18 — contrôle opérationnel de l’exposition Pareto 001

Instantané commencé : 2026-09-11T00:17:57.396729+00:00.
Instantané achevé : 2026-09-11T00:17:57.401536+00:00 (2026-09-11T02:17:57.401536+02:00 à Paris).

Périmètre fixé : namespace `EXP18-MECHANISMS-v1`, outer `18011`, bras **P**, slots **00 à 05 inclus**, six réponses terminées. Aucun slot ultérieur ou autre bras n’entre dans ce contrôle.
Gel canonique : `ce88c4b2d9530456e63b497f7bbde7608fbf3d8aca441c13c086d77b6beeea8a`.
Alias de trainer réellement enregistré : `EXP18Pareto_5ee5071850ac6d2baeee5b6f`.

**Le mécanisme a été exercé : 5/6 requêtes ont utilisé un front de plusieurs membres ; 2/6 ont reçu un parent différent du meilleur scalaire.** Ces nombres décrivent des choix effectivement consommés par des requêtes terminées, pas un avantage de performance.

| Slot | Origines du front | Origine choisie | Meilleur scalaire | Choix différent | Sources dans la mémoire de production |
|---:|---|---:|---:|---|---:|
| 00 | [-1] | -1 | -1 | non | 1 |
| 01 | [-1, 0] | 0 | -1 | oui | 2 |
| 02 | [-1, 0] | -1 | -1 | non | 3 |
| 03 | [-1, 0] | -1 | -1 | non | 4 |
| 04 | [-1, 3] | 3 | 3 | non | 5 |
| 05 | [-1, 3] | -1 | 3 | oui | 6 |

L’origine `-1` désigne le seed initial. Les autres nombres sont les indices des propositions antérieures ; ils ne sont ni des scores ni des rangs de performance.

## Vérifications réalisées

- Recalcul indépendant, avec la bibliothèque standard seulement, du front strict de minimisation à partir des seuls reçus TRAIN antérieurs : déduplication par source puis vecteur exact, meilleur scalaire, archive et tirage uniforme SHA256/Random. Tous les champs comparables des six décisions concordent. Les valeurs TRAIN restent hors de ce rapport.
- Concordance des hashes entre source exacte du seed ou de la réponse antérieure, décision de parent, contexte immuable, requête réellement enregistrée et texte propagé par Trace. La seconde section de chaque requête correspond exactement au code du parent et au feedback sauvegardés ; aucun code candidat n’a été exécuté.
- Provenance des reçus : même bras/outer, réponse antérieure correspondante, reçu complet disponible avant l’instantané de la requête, reçu de parent et projection compacte authentifiés. Aucun accès au cache d’évaluation n’a été nécessaire.
- Identité de l’adaptateur : les sept sources listées ci-dessous correspondent aux hashes du gel. Le hook gelé cherche un vrai ModuleCandidate dans la mémoire de production, exige la présence de toutes les sources TRAIN valides, puis retourne l’objet sélectionné. Le callback gelé revalide la décision et préserve son parent dans les messages.
- Les 46 fichiers physiques lus ont été re-hashés en fin d’inspection et sont restés inchangés. Aucune source, preuve de run, configuration ou processus actif n’a été modifié.

## Portée et décisions inutilisées

Les six décisions contrôlées portent `will_generate=true` et chacune est reliée à une réponse terminée. Aucune décision terminale ou inutilisée n’est comptée dans les taux ci-dessus. Le terminal prévu est le slot16 ; il est hors de ce périmètre, comme toutes les décisions après05.

La concordance persistée démontre la source effectivement fournie au modèle via le hook testé et gelé. Elle ne recrée pas l’identité mémoire Python d’un objet après coup. Le contrôle ne réévalue pas les observations brutes sous-jacentes aux reçus ; leur audit numérique complet relève des vérifications finales.

Ce relevé ne mesure aucun gain de regret, aucun avantage inter-bras, aucune réussite de validation ou d’audit. Il ne conclut ni à la qualité d’un spécialiste ni à l’utilité de la diversité. La politique testée combine filtrage par nondominance et échantillonnage uniforme ; ce n’est pas un test isolé du filtrage Pareto.

Aucun résultat de validation/holdout n’a été ouvert. Les tâches de splits contenues dans le gel n’ont pas été utilisées. Aucun appel modèle, objectif ou programme candidat ; aucune reprise ni interaction avec le processus vivant.

## Sources des parents effectivement transmis

| Slot | SHA256 de la source du parent |
|---:|---|
| 00 | `5b74e5a3fe2fc90fcb42a603b058fa7befbf65f38775caa77acd30cdf619b640` |
| 01 | `7fa5eca4a598b91ba019ba1f6988572db7ba334e5ac9b81c1a64a1505d6432b9` |
| 02 | `5b74e5a3fe2fc90fcb42a603b058fa7befbf65f38775caa77acd30cdf619b640` |
| 03 | `5b74e5a3fe2fc90fcb42a603b058fa7befbf65f38775caa77acd30cdf619b640` |
| 04 | `e2cb6e1d49254c1a071c48cc16627dc7d3f5f67c3a4fb576c5d0a50510e341f1` |
| 05 | `5b74e5a3fe2fc90fcb42a603b058fa7befbf65f38775caa77acd30cdf619b640` |

## Empreintes des entrées physiques

Chemins relatifs à la racine du dépôt ; SHA256 sur les octets physiques, avant décodage gzip éventuel. Les documents ci-dessous sont des entrées de lecture, jamais des sorties de ce contrôle.

| Entrée | SHA256 |
|---|---|
| `artifacts/optimizer_discovery/exp17/study.py` | `f16cfd1eac80f1aa27584e6f654ff5c43abe46aeb57265fd67496bc4aeb48b5b` |
| `artifacts/optimizer_discovery/exp18/memory_projection.py` | `0dc1a6d414b60176e737f8d914561cfcf01edcbb47d4e186e92b414e71210391` |
| `artifacts/optimizer_discovery/exp18/pareto_selection.py` | `bb76350204c75a4d96aa4ef46082f347874926ee5b656f00c698d45c270de7b3` |
| `artifacts/optimizer_discovery/exp18/pareto_trainer.py` | `a0c1bd1d452f0a2ae5ff8e2e328a3293b87f7a914b63fff3e9b012d8d258f500` |
| `artifacts/optimizer_discovery/exp18/run/freeze.json` | `9abb465cda1ea55c9cce7e5e784621a9354033c1f6f4c82960f2b3ce917354be` |
| `artifacts/optimizer_discovery/exp18/run/freeze_sha256.json` | `9bb65676d2a9fc9d6675b32d5639f1c0960c6b2f70ad453a014ea6552188ec4d` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/parent_decisions/slot_00.json` | `7907065f67f97e0f3b2af114c9bc09c3172e7d3d57ff14492124dac8cdda634a` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/parent_decisions/slot_01.json` | `a516d0408bf5d7332edfb88bebbef23761e2c4752abb88e968c5b82156293968` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/parent_decisions/slot_02.json` | `3e25990a86d82f0334ba032c36f550c38f1f70dd1cd980d68525e9752227ece6` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/parent_decisions/slot_03.json` | `fa84ef7fe80430d9340c7322a9c536d0fcfb6e96b6efee05ecbedde30bbf17e9` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/parent_decisions/slot_04.json` | `d5feedfd947dc3437e91b6ae7c67872b95887c1b300020570319fb216edd4ed5` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/parent_decisions/slot_05.json` | `fe51a5d07d0845417df74f09f50c8ae2ac49968df12228c15efbaed266bbef00` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/seed_train_receipt.json` | `cd7401da0955c854bebef633feee67bd4ff13dbfd840e7e38a4d87bb15807948` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/slot_00/current_context.json` | `e00029f87f0365ac9e9533710c14e780bb8e20a4766dcf431d7cb425d08ba8c7` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/slot_00/propagated_feedback.json` | `279b9b2cf8b45d669af7896c41d79ab3a4073bb7a22323baa6c61057a9ad937d` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/slot_00/request.json` | `2caa90c3f694f9578fdeeda47bcf7fb33b0a6da421a538fead42849d29637096` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/slot_00/response.json` | `0cce70bbf5c9ec1e988a729bdb66463f0843c06fd4bc52aaff38f81b0d0be870` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/slot_00/train_receipt.json` | `643007b33c873a00817a52126d6a4a772ba9634ff9a9f20ad30f64e079467591` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/slot_01/current_context.json` | `0bdc84744924a1b96e0172ed284758bbcf885f6a2052afb2e55739617b709ab0` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/slot_01/propagated_feedback.json` | `2bc06c12b3d077b44ed80a5e9a72ab673302ce422e22beec5d816c42a3d602f2` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/slot_01/request.json` | `cfe2a6f4744da556e60fd862b8b53379e2566b5165962481c543867ccb2a8762` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/slot_01/response.json` | `cd759aeb9db50cde53c2376fa2380e5b00bb5c6825ed1f1be72ccf07045d9943` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/slot_01/train_receipt.json` | `6d8f761661ef0ed49110862d780248664ff1768abeb2ae2ee91dbaf12a58116c` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/slot_02/current_context.json` | `51d55e1d71b74c491bb48dae272cdb74bd78bb4411c0c237ec4481d4ebd6f456` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/slot_02/propagated_feedback.json` | `279b9b2cf8b45d669af7896c41d79ab3a4073bb7a22323baa6c61057a9ad937d` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/slot_02/request.json` | `8add51dc9dd281465f15fc026169cc57e271721d0658207ebc5aaa98bb0de6fc` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/slot_02/response.json` | `15fb882a0c6441cea83f436d51263e39067d45618cf6d85ca1f3da2aa8ffbd12` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/slot_02/train_receipt.json` | `bfb1b0c08b415770c663e8f159fb73f0268eafec78d195ac367885bbd38160c5` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/slot_03/current_context.json` | `d32a4f923808735045d2ebf8e04926b4cd59369e141208ee9a25f132cdf10f26` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/slot_03/propagated_feedback.json` | `279b9b2cf8b45d669af7896c41d79ab3a4073bb7a22323baa6c61057a9ad937d` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/slot_03/request.json` | `c6004b699cbe92413bab9b069dc4c8fa70cc5826d8c935658acba413d65995ea` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/slot_03/response.json` | `0e8105b659b1038d434de307e6a662fd0582e2da1d49daca86df071a58122eae` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/slot_03/train_receipt.json` | `90d49e55b237554b0c54a44c35e977722a3cd5757605226ae515226be691e615` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/slot_04/current_context.json` | `44c18c3a2e9cdccc5ac486bd2272ea1ea771ae7c21ddd39803c6fcc75e1d0a6e` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/slot_04/propagated_feedback.json` | `883683ead44c8a03fae45ce92e4754775deb6cdeef5946f9fad884ad27eaf2b4` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/slot_04/request.json` | `51c17ad5a31c954371101a095bbcb3b83e74da832865113f9a3c04031a9931bc` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/slot_04/response.json` | `89e9838c75ea60787bd5c02d0f95ba7adc95131d21851e5f6b73ea780b53c185` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/slot_04/train_receipt.json` | `683bff5edebafe87963ef9eab3d678379fc3adbe3926fa7b397f243b04660216` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/slot_05/current_context.json` | `ebd8dd6fd1438f7944e947d676b6d8de2e07081734e1c60ee161b59da9ab5260` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/slot_05/propagated_feedback.json` | `279b9b2cf8b45d669af7896c41d79ab3a4073bb7a22323baa6c61057a9ad937d` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/slot_05/request.json` | `c11f4759f2cd57d11b01cb38083de478eb37e605098878134bb1b20ec658a4a1` |
| `artifacts/optimizer_discovery/exp18/run/raw/18011/P/slot_05/response.json` | `7930e5bf8ebac0d676eac7ab1b9dec04ee7d72e0c8857d21cad3b078689e4042` |
| `artifacts/optimizer_discovery/exp18/run/source_snapshot.json` | `d337b3c1374e3c43b7985c21fe0db8f40c80dfa29f1b7c04563ec016a7cb2688` |
| `artifacts/optimizer_discovery/exp18/study.py` | `22c653204145fe77a2a8619a86182edb0ce1d8f83b886a9b730f06e2fc5670cf` |
| `opto/trainer/algorithms/priority_search.py` | `4d56d26f518bd7baecbf19de91342954c2965da99c85cc8ab48294503a630143` |
| `opto/trainer/objectives.py` | `d08d14618f68d545bac86bf1e5089347edcd58c3d64f0f6e2553ff0f6f669a40` |
