> Historical document, retired from active navigation on 2026-09-29. Original location: `artifacts/RESULTS_INDEX.md`. Use the [canonical experiment index](../../README.md) and [current assessment](../../ASSESSMENT.md). The [original bytes](RESULTS_INDEX.md.original.gz) are preserved; relative links below were rebased for this location.

**EXP21 — bloquée par le crédit du compte, pas terminée** : [tableaux et courbes](../../_shared/o1_learning/EXP21.md). 219/252 réponses, 33/42 mesures ; 961 tests passent. O1/O2 et confirmation restent à exécuter.

Campagne terminée : [EXP-19 — résultats des phases 1 et 2](../../EXP19/RESULTS.md).

**EXP-20 terminée avec Qwen, DeepSeek conservé** : [résultats HotpotQA](../../_shared/o1_learning/EXP20.md). Six graines, 72 réponses d’optimiseur, 60 TRAIN / 24 VALIDATION / 48 TEST. Exactitude TEST finale : initial 28,47 %, standard 47,92 %, curriculum 43,75 %. Gain d’apprentissage observé ; avantage propre au curriculum non établi. Les pilotes et refus historiques restent conservés.
# Où lire, quoi réutiliser, quoi archiver

État documentaire du 13 septembre 2026. **EXP-17 reste suspendue ; EXP-19 a été
autorisée séparément** pour les traces, le curriculum et la convergence O1. Cet index ne
contient pas un second registre d’hypothèses : il renvoie au
[bilan actuel](RESEARCH_LOG.md).

## Trois entrées, selon le besoin

| Besoin | Document à lire | Fonction |
|---|---|---|
| Comprendre le parcours, les UC, les résultats et décider | [RESEARCH_LOG.md](RESEARCH_LOG.md) | Synthèse A–I, hypothèses corrigées, usages et actions ciblées |
| Vérifier une correction | [recursive_opt_assessment.md](../../ASSESSMENT.md), §29 puis la section citée | Audit chronologique ; les verdicts initiaux ne sont pas l’état actuel |
| Présenter le travail à Patrick | [Brief réécrit](../../_shared/optimizer_discovery/exp18/PATRICK_BRIEF.md) | Question, contrat, résultats et limites expliqués ; non envoyé |

## Répertoires, notebooks et code

| Emplacement | Utilité conservée | Statut / ne pas en déduire |
|---|---|---|
| [opto/features/recursive_opt](../../../../opto/features/recursive_opt) | `levels`, `effects`, `memory`, `budget`, `optimize`, `spec` : composants réutilisables | Code de production ; tests de contrats ≠ efficacité universelle |
| [numeric_optimizers.py](../../../../opto/features/recursive_opt/numeric_optimizers.py) | Route vers solveurs numériques pour knobs actifs | Réutilisable ; ne pas attribuer automatiquement le gain à la récursion |
| [traces.py](../../../../opto/features/recursive_opt/traces.py) | Connexion de sources de traces, export interne | OTEL/sysmon/hybrid vérifiés dans EXP-19 ; SDK OTEL dans `humanllm`, absent du venv Phase 0 ; gain d’apprentissage non établi |
| [examples A/B/C/D/E](../../../../examples) | Comprendre les différentes surfaces et déclarations | Démonstrateurs historiques ; ne pas relancer tous les scripts pour « confirmer » les anciens résultats |
| [recursive_opt_phases_V2.ipynb](../../EXP00/notebooks/recursive_opt_phases_V2.ipynb) | Carte d’origine O0/O1/O2a/O2b/O3 | Commentaires « positive claim » historiques, pas conclusions actuelles |
| [recursive_opt_use_cases.ipynb](../../../../examples/recursive_opt_use_cases.ipynb) | Client minimal du control plane | **Deux tests artificiels UC4/UC14**, pas la suite scientifique d’origine |
| [XP_1stattempt/recursive_opt_use_cases.ipynb](../../../../examples/XP_1stattempt/recursive_opt_use_cases.ipynb) | Première suite UC1–6 | Archive de conception ; Git `8c78e1b46` contient la suite ultérieure UC1–13 et three-way |
| [recursive_opt_three_way.py](../../../../examples/recursive_opt_three_way.py) | Comparabilité, construction des bras, comptabilité historique | Résumé de vitesse première graine : à corriger avant réemploi scientifique |
| [PAL curriculum](../../../../examples/OpenTrace_LangGraph_BBEH_boolean_expressions_PAL_curriculum_clean.ipynb) | Exemple de batch courant + succès passés réellement transmis à `backward` | Notebook local non suivi ; aucune performance comparative validée par cet audit ; exécution code en processus local |
| [examples/notebook_outputs/recursive_opt_use_cases](../../../../examples/notebook_outputs/recursive_opt_use_cases) | Courbes, JSON, sources historiques ; accès depuis un claim précis | Archives utilisateur préservées ; succès des notebooks ≠ preuve contrôlée |
| [control_plane_v2](../../_shared/control_plane_v2) | Contrats, normalisation, migration, tests et corrections | **Preuves techniques**, pas validation de tous les UC anciens |
| [probe_2026](../probe_2026) | Données des audits et W2/routage/transfert | Historique ; certains claims retirés, voir registre courant |
| [optimizer_discovery, racine](../../_shared/optimizer_discovery) | Contrat, benchmark commun, Phase 0 et EXP-15 | Contrat/évaluateur réutilisables ; anciens briefs datés |
| [exp15](../../_shared/optimizer_discovery/exp15) | Première comparaison contrôlée, sources choisies, replay | Expérience achevée ; A2−A1 inconclusif ; partitions désormais observées |
| [investigation16/history](../../_shared/optimizer_discovery/investigation16/history) | Replays historiques et limites des mécanismes | Preuves réutilisables et sources exactes ; propositions de suite devenues historiques |
| [investigation16/benchmark](../../_shared/optimizer_discovery/investigation16/benchmark), [selection](../../_shared/optimizer_discovery/investigation16/selection), [throughput](../../_shared/optimizer_discovery/investigation16/throughput) | B1/B2, stabilité S1, débit T1 | Mesures locales utiles ; utiliser T1 **corrigé**, pas les premiers speedups |
| [investigation16/generation](../../_shared/optimizer_discovery/investigation16/generation), [feedback](../../_shared/optimizer_discovery/investigation16/feedback), [runtime](../../_shared/optimizer_discovery/investigation16/runtime) | Diagnostics plafond, transmission et contenu du feedback | Causes confirmées ou hypothèses distinguées dans la matrice ; pas recettes gagnantes |
| [investigation16/production](../../_shared/optimizer_discovery/investigation16/production) | P1 : protocole, inspection, analyse de suite | `FUTURE_DESIGN.md` est une proposition **antérieure**, remplacée comme priorité par le bilan actuel |
| [exp17](../../_shared/optimizer_discovery/exp17) | Protocole C−I, implémentation gelée et preuves partielles | **Suspendu à 545/736**, avant audit ; aucune confirmation disponible |
| [exp18](../../_shared/optimizer_discovery/exp18) | Mémoire/Pareto, résultats complets, programmes et revues | Achevé ; effets mécanistiques inconclusifs ; pas de contrôle indépendant N16 |
| `*/raw`, `*/evaluation_cache`, `*/runtime`, archives `.zip`/`.gz` | Reçus, sorties exactes, clés de cache, provenance | **À conserver**, ne pas lire intégralement et ne pas lancer comme scripts |
| `*/programs`, `*/selected*`, `*/sources` | Artefacts exactement évalués | Réutiliser avec le contrat et le hash ; pas de « best » sélectionné par audit a posteriori |
| `*/presentation`, `*/report_data`, `*/figures` | Visualisation de résultats existants | Présentation secondaire ; données et règles d’analyse restent l’autorité |

Comptage au point de suspension, hors `__pycache__` : investigation16 **87 055
fichiers / 59 Markdown** ; exp17 **221 052 / 10** ; exp18 **179 802 / 33**.
Le volume vient principalement des reçus/records structurés, mais la prolifération
des documents et points d’entrée était également réelle. Cet audit fusionne la
lecture, sans supprimer de preuve scientifique ni déplacer des chemins référencés
par des gels. Une éventuelle archive physique nécessitera hashes et restauration
vérifiée ; aucune suppression massive n’est faite ici.

## Inventaire des Markdown de recherche

Inventaire exhaustif des fichiers `.md`/`.MD` présents sous `control_plane_v2`,
`probe_2026` et `optimizer_discovery`, plus les points d’entrée récursifs de
`artifacts`, `examples` et du module. Les Markdown génériques de Trace et les
rapports répétés des sorties de notebooks sont hors de ce tableau ; ces derniers
sont regroupés dans la ligne d’archives ci-dessus. Les fichiers bruts, gels et
protocoles ne sont pas des documents à supprimer parce qu’ils sont historiques.

Les statuts portent sur **l’usage documentaire**, pas sur un verdict scientifique
nouveau. « Preuve technique » ne certifie pas une efficacité ; « protocole » décrit
une règle enregistrée ; « historique » signifie qu’il ne donne plus la priorité
actuelle. Toutes les preuves de claims doivent être lues avec le bilan corrigé.

**136 fichiers inventoriés.**

| Fichier | Usage actuel |
|---|---|
| [artifacts/EXECUTION_PLAN.md](../reviews/EXECUTION_PLAN.md) | Renvoi — suspension actuelle, aucun lancement |
| [artifacts/RESEARCH_LOG.md](RESEARCH_LOG.md) | Synthèse actuelle — lire en premier |
| [artifacts/RESULTS_INDEX.md](RESULTS_INDEX.md) | Index actuel — navigation |
| [artifacts/control_plane_v2/baseline.md](../../_shared/control_plane_v2/baseline.md) | Contrat / preuve technique datée — pas validation scientifique des UC |
| [artifacts/control_plane_v2/candidate_trajectory_provenance_hotfix.md](../../_shared/control_plane_v2/candidate_trajectory_provenance_hotfix.md) | Contrat / preuve technique datée — pas validation scientifique des UC |
| [artifacts/control_plane_v2/control_plane_v2alpha.md](../../_shared/control_plane_v2/control_plane_v2alpha.md) | Contrat / preuve technique datée — pas validation scientifique des UC |
| [artifacts/control_plane_v2/evidence.md](../../_shared/control_plane_v2/evidence.md) | Contrat / preuve technique datée — pas validation scientifique des UC |
| [artifacts/control_plane_v2/final_hardening_audit.md](../../_shared/control_plane_v2/final_hardening_audit.md) | Contrat / preuve technique datée — pas validation scientifique des UC |
| [artifacts/control_plane_v2/gepa_014_contract_hotfix.md](../../_shared/control_plane_v2/gepa_014_contract_hotfix.md) | Contrat / preuve technique datée — pas validation scientifique des UC |
| [artifacts/control_plane_v2/gepa_reflection_protocol_hotfix.md](../../_shared/control_plane_v2/gepa_reflection_protocol_hotfix.md) | Protocole enregistré — conserver ; ne constitue pas une instruction de relance |
| [artifacts/control_plane_v2/live_transport_resilience_hotfix.md](../../_shared/control_plane_v2/live_transport_resilience_hotfix.md) | Contrat / preuve technique datée — pas validation scientifique des UC |
| [artifacts/control_plane_v2/migration_report.md](../../_shared/control_plane_v2/migration_report.md) | Contrat / preuve technique datée — pas validation scientifique des UC |
| [artifacts/control_plane_v2/optimizer_empty_text_response_hotfix.md](../../_shared/control_plane_v2/optimizer_empty_text_response_hotfix.md) | Contrat / preuve technique datée — pas validation scientifique des UC |
| [artifacts/control_plane_v2/proof.md](../../_shared/control_plane_v2/proof.md) | Contrat / preuve technique datée — pas validation scientifique des UC |
| [artifacts/control_plane_v2/readiness_audit.md](../../_shared/control_plane_v2/readiness_audit.md) | Contrat / preuve technique datée — pas validation scientifique des UC |
| [artifacts/optimizer_discovery/BASELINE.md](../../_shared/optimizer_discovery/BASELINE.md) | Documentation de l’étude — complément, aucune priorité automatique de relance |
| [artifacts/optimizer_discovery/ENGINEERING_SMOKE_SPEC.md](../../_shared/optimizer_discovery/ENGINEERING_SMOKE_SPEC.md) | Protocole enregistré — conserver ; ne constitue pas une instruction de relance |
| [artifacts/optimizer_discovery/EXP15_REPORT.md](../../_shared/optimizer_discovery/EXP15_REPORT.md) | Résultat local de l’étude — valide seulement dans son périmètre, lire les corrections |
| [artifacts/optimizer_discovery/GENERATION_CALIBRATION_SPEC.md](../../_shared/optimizer_discovery/GENERATION_CALIBRATION_SPEC.md) | Protocole enregistré — conserver ; ne constitue pas une instruction de relance |
| [artifacts/optimizer_discovery/OPTIMIZER_PROGRAM_V0.md](../../_shared/optimizer_discovery/OPTIMIZER_PROGRAM_V0.md) | Contrat réutilisable — dépend de son évaluateur |
| [artifacts/optimizer_discovery/PATRICK_BRIEF.md](../../_shared/optimizer_discovery/PATRICK_BRIEF.md) | Brief historique — préférer celui réécrit dans exp18 |
| [artifacts/optimizer_discovery/PHASE0_REPORT.md](../../_shared/optimizer_discovery/PHASE0_REPORT.md) | Résultat local de l’étude — valide seulement dans son périmètre, lire les corrections |
| [artifacts/optimizer_discovery/PHASE0_SPEC.md](../../_shared/optimizer_discovery/PHASE0_SPEC.md) | Protocole enregistré — conserver ; ne constitue pas une instruction de relance |
| [artifacts/optimizer_discovery/PREREG_EXP15.md](../../_shared/optimizer_discovery/PREREG_EXP15.md) | Protocole enregistré — conserver ; ne constitue pas une instruction de relance |
| [artifacts/optimizer_discovery/exp15/PILOT_REPORT.md](../../_shared/optimizer_discovery/exp15/PILOT_REPORT.md) | Résultat local de l’étude — valide seulement dans son périmètre, lire les corrections |
| [artifacts/optimizer_discovery/exp15/PORTABILITY.md](../../_shared/optimizer_discovery/exp15/PORTABILITY.md) | Contrat de remplacement du lanceur — sans sandbox OS |
| [artifacts/optimizer_discovery/exp15/VERIFICATION.md](../../_shared/optimizer_discovery/exp15/VERIFICATION.md) | Vérification datée — détail de preuve, après lecture du rapport associé |
| [artifacts/optimizer_discovery/exp17/ENGINEERING_REPORT.md](../../_shared/optimizer_discovery/exp17/ENGINEERING_REPORT.md) | Preuve d’ingénierie / préparation — étude principale suspendue, sans résultat final |
| [artifacts/optimizer_discovery/exp17/PILOT_PROTOCOL.md](../../_shared/optimizer_discovery/exp17/PILOT_PROTOCOL.md) | Protocole enregistré — conserver ; ne constitue pas une instruction de relance |
| [artifacts/optimizer_discovery/exp17/PREREG_EXP17.md](../../_shared/optimizer_discovery/exp17/PREREG_EXP17.md) | Protocole enregistré — conserver ; ne constitue pas une instruction de relance |
| [artifacts/optimizer_discovery/exp17/RUN_OPERATIONS.md](../../_shared/optimizer_discovery/exp17/RUN_OPERATIONS.md) | Journal / reprise — seulement pour audit opérationnel, pas décision scientifique |
| [artifacts/optimizer_discovery/exp17/WORK_LOG.md](../../_shared/optimizer_discovery/exp17/WORK_LOG.md) | Journal / reprise — seulement pour audit opérationnel, pas décision scientifique |
| [artifacts/optimizer_discovery/exp17/design_review.md](../../_shared/optimizer_discovery/exp17/design_review.md) | Preuve d’ingénierie / préparation — étude principale suspendue, sans résultat final |
| [artifacts/optimizer_discovery/exp17/engineering_diagnostics/timeout_01/PREREG.md](../../_shared/optimizer_discovery/exp17/engineering_diagnostics/timeout_01/PREREG.md) | Protocole enregistré — conserver ; ne constitue pas une instruction de relance |
| [artifacts/optimizer_discovery/exp17/engineering_diagnostics/timeout_01/REPORT.md](../../_shared/optimizer_discovery/exp17/engineering_diagnostics/timeout_01/REPORT.md) | Preuve d’ingénierie / préparation — étude principale suspendue, sans résultat final |
| [artifacts/optimizer_discovery/exp17/protocol_versions/PREREG_EXP17_draft_01.md](../../_shared/optimizer_discovery/exp17/protocol_versions/PREREG_EXP17_draft_01.md) | Protocole enregistré — conserver ; ne constitue pas une instruction de relance |
| [artifacts/optimizer_discovery/exp17/protocol_versions/PREREG_EXP17_pilot_02.md](../../_shared/optimizer_discovery/exp17/protocol_versions/PREREG_EXP17_pilot_02.md) | Protocole enregistré — conserver ; ne constitue pas une instruction de relance |
| [artifacts/optimizer_discovery/exp18/ADAPTER.md](../../_shared/optimizer_discovery/exp18/ADAPTER.md) | Documentation de l’étude — complément, aucune priorité automatique de relance |
| [artifacts/optimizer_discovery/exp18/DOCUMENT_UPDATE_MAP.md](../../_shared/optimizer_discovery/exp18/DOCUMENT_UPDATE_MAP.md) | Proposition historique — aucune reprise autorisée par ce texte |
| [artifacts/optimizer_discovery/exp18/ENGINEERING_REPORT.md](../../_shared/optimizer_discovery/exp18/ENGINEERING_REPORT.md) | Résultat local de l’étude — valide seulement dans son périmètre, lire les corrections |
| [artifacts/optimizer_discovery/exp18/GUIDE_EXPERIENCES.md](../../_shared/optimizer_discovery/exp18/GUIDE_EXPERIENCES.md) | Explication détaillée du benchmark — complément au bilan |
| [artifacts/optimizer_discovery/exp18/INDEPENDENT_RESULT_REVIEW.md](../../_shared/optimizer_discovery/exp18/INDEPENDENT_RESULT_REVIEW.md) | Vérification datée — détail de preuve, après lecture du rapport associé |
| [artifacts/optimizer_discovery/exp18/PATRICK_BRIEF.md](../../_shared/optimizer_discovery/exp18/PATRICK_BRIEF.md) | Brief actuel — réécrit pour un lecteur externe |
| [artifacts/optimizer_discovery/exp18/PILOT_PROTOCOL.md](../../_shared/optimizer_discovery/exp18/PILOT_PROTOCOL.md) | Protocole enregistré — conserver ; ne constitue pas une instruction de relance |
| [artifacts/optimizer_discovery/exp18/PREREG_EXP18.md](../../_shared/optimizer_discovery/exp18/PREREG_EXP18.md) | Protocole enregistré — conserver ; ne constitue pas une instruction de relance |
| [artifacts/optimizer_discovery/exp18/PROGRAM_REVIEW.md](../../_shared/optimizer_discovery/exp18/PROGRAM_REVIEW.md) | Vérification datée — détail de preuve, après lecture du rapport associé |
| [artifacts/optimizer_discovery/exp18/REPORT.md](../../_shared/optimizer_discovery/exp18/REPORT.md) | Résultat local de l’étude — valide seulement dans son périmètre, lire les corrections |
| [artifacts/optimizer_discovery/exp18/RESOURCE_METHOD.md](../../_shared/optimizer_discovery/exp18/RESOURCE_METHOD.md) | Références / mécanismes possibles — aucune preuve locale de gain supplémentaire |
| [artifacts/optimizer_discovery/exp18/RESOURCE_METHOD_initial.md](../../_shared/optimizer_discovery/exp18/RESOURCE_METHOD_initial.md) | Méthode initiale — consulter RESOURCE_METHOD et RESOURCE_REVIEW actuels |
| [artifacts/optimizer_discovery/exp18/RESOURCE_REVIEW.md](../../_shared/optimizer_discovery/exp18/RESOURCE_REVIEW.md) | Vérification datée — détail de preuve, après lecture du rapport associé |
| [artifacts/optimizer_discovery/exp18/figures/README.md](../../_shared/optimizer_discovery/exp18/figures/README.md) | Présentation / calculs secondaires — données originales font autorité |
| [artifacts/optimizer_discovery/exp18/operational_status_checks/client_timeout_audit_01.md](../../_shared/optimizer_discovery/exp18/operational_status_checks/client_timeout_audit_01.md) | Journal / reprise — seulement pour audit opérationnel, pas décision scientifique |
| [artifacts/optimizer_discovery/exp18/operational_status_checks/final_accounting_checklist_001.md](../../_shared/optimizer_discovery/exp18/operational_status_checks/final_accounting_checklist_001.md) | Journal / reprise — seulement pour audit opérationnel, pas décision scientifique |
| [artifacts/optimizer_discovery/exp18/operational_status_checks/generation_integrity_001.md](../../_shared/optimizer_discovery/exp18/operational_status_checks/generation_integrity_001.md) | Journal / reprise — seulement pour audit opérationnel, pas décision scientifique |
| [artifacts/optimizer_discovery/exp18/operational_status_checks/infrastructure_source_read_001.md](../../_shared/optimizer_discovery/exp18/operational_status_checks/infrastructure_source_read_001.md) | Journal / reprise — seulement pour audit opérationnel, pas décision scientifique |
| [artifacts/optimizer_discovery/exp18/operational_status_checks/local_evaluation_resume_audit_001.md](../../_shared/optimizer_discovery/exp18/operational_status_checks/local_evaluation_resume_audit_001.md) | Journal / reprise — seulement pour audit opérationnel, pas décision scientifique |
| [artifacts/optimizer_discovery/exp18/operational_status_checks/pareto_exposure_001.md](../../_shared/optimizer_discovery/exp18/operational_status_checks/pareto_exposure_001.md) | Journal / reprise — seulement pour audit opérationnel, pas décision scientifique |
| [artifacts/optimizer_discovery/exp18/operational_status_checks/pm_joint_exposure_001.md](../../_shared/optimizer_discovery/exp18/operational_status_checks/pm_joint_exposure_001.md) | Journal / reprise — seulement pour audit opérationnel, pas décision scientifique |
| [artifacts/optimizer_discovery/exp18/operational_status_checks/resume_audit_01.md](../../_shared/optimizer_discovery/exp18/operational_status_checks/resume_audit_01.md) | Journal / reprise — seulement pour audit opérationnel, pas décision scientifique |
| [artifacts/optimizer_discovery/exp18/operational_status_checks/resume_audit_02.md](../../_shared/optimizer_discovery/exp18/operational_status_checks/resume_audit_02.md) | Journal / reprise — seulement pour audit opérationnel, pas décision scientifique |
| [artifacts/optimizer_discovery/exp18/operational_status_checks/resume_audit_03.md](../../_shared/optimizer_discovery/exp18/operational_status_checks/resume_audit_03.md) | Journal / reprise — seulement pour audit opérationnel, pas décision scientifique |
| [artifacts/optimizer_discovery/exp18/operational_status_checks/resume_audit_04.md](../../_shared/optimizer_discovery/exp18/operational_status_checks/resume_audit_04.md) | Journal / reprise — seulement pour audit opérationnel, pas décision scientifique |
| [artifacts/optimizer_discovery/exp18/operational_status_checks/resume_audit_05.md](../../_shared/optimizer_discovery/exp18/operational_status_checks/resume_audit_05.md) | Journal / reprise — seulement pour audit opérationnel, pas décision scientifique |
| [artifacts/optimizer_discovery/exp18/operational_status_checks/resume_audit_06.md](../../_shared/optimizer_discovery/exp18/operational_status_checks/resume_audit_06.md) | Journal / reprise — seulement pour audit opérationnel, pas décision scientifique |
| [artifacts/optimizer_discovery/exp18/operational_status_checks/resume_verification_005.md](../../_shared/optimizer_discovery/exp18/operational_status_checks/resume_verification_005.md) | Journal / reprise — seulement pour audit opérationnel, pas décision scientifique |
| [artifacts/optimizer_discovery/exp18/operational_status_checks/resume_verification_006.md](../../_shared/optimizer_discovery/exp18/operational_status_checks/resume_verification_006.md) | Journal / reprise — seulement pour audit opérationnel, pas décision scientifique |
| [artifacts/optimizer_discovery/exp18/operational_status_checks/resume_verification_007.md](../../_shared/optimizer_discovery/exp18/operational_status_checks/resume_verification_007.md) | Journal / reprise — seulement pour audit opérationnel, pas décision scientifique |
| [artifacts/optimizer_discovery/exp18/programs/PROGRAM_INSPECTION.md](../../_shared/optimizer_discovery/exp18/programs/PROGRAM_INSPECTION.md) | Description des sources choisies — ne prouve pas la nouveauté ni la causalité |
| [artifacts/optimizer_discovery/exp18/protocol_versions/PREREG_EXP18_draft_01.md](../../_shared/optimizer_discovery/exp18/protocol_versions/PREREG_EXP18_draft_01.md) | Protocole enregistré — conserver ; ne constitue pas une instruction de relance |
| [artifacts/optimizer_discovery/exp18/readiness_audit_01.md](../../_shared/optimizer_discovery/exp18/readiness_audit_01.md) | Vérification datée — détail de preuve, après lecture du rapport associé |
| [artifacts/optimizer_discovery/investigation16/COMPLETION_AUDIT.md](../../_shared/optimizer_discovery/investigation16/COMPLETION_AUDIT.md) | Vérification datée — détail de preuve, après lecture du rapport associé |
| [artifacts/optimizer_discovery/investigation16/DECISION_MATRIX.md](../../_shared/optimizer_discovery/investigation16/DECISION_MATRIX.md) | Synthèse EXP-16 — mémoire/Pareto actualisés ensuite dans EXP-18 |
| [artifacts/optimizer_discovery/investigation16/PATRICK_BRIEF.md](../../_shared/optimizer_discovery/investigation16/PATRICK_BRIEF.md) | Brief historique — préférer celui réécrit dans exp18 |
| [artifacts/optimizer_discovery/investigation16/PRIMARY_SOURCES.md](../../_shared/optimizer_discovery/investigation16/PRIMARY_SOURCES.md) | Références / mécanismes possibles — aucune preuve locale de gain supplémentaire |
| [artifacts/optimizer_discovery/investigation16/PRODUCTION_PILOT_DRAFT.md](../../_shared/optimizer_discovery/investigation16/PRODUCTION_PILOT_DRAFT.md) | Proposition historique — aucune reprise autorisée par ce texte |
| [artifacts/optimizer_discovery/investigation16/PROTOCOL.md](../../_shared/optimizer_discovery/investigation16/PROTOCOL.md) | Protocole enregistré — conserver ; ne constitue pas une instruction de relance |
| [artifacts/optimizer_discovery/investigation16/REPORT.md](../../_shared/optimizer_discovery/investigation16/REPORT.md) | Résultat local de l’étude — valide seulement dans son périmètre, lire les corrections |
| [artifacts/optimizer_discovery/investigation16/VERIFICATION.md](../../_shared/optimizer_discovery/investigation16/VERIFICATION.md) | Vérification datée — détail de preuve, après lecture du rapport associé |
| [artifacts/optimizer_discovery/investigation16/WORK_LOG.md](../../_shared/optimizer_discovery/investigation16/WORK_LOG.md) | Journal / reprise — seulement pour audit opérationnel, pas décision scientifique |
| [artifacts/optimizer_discovery/investigation16/benchmark/B1_REPORT.md](../../_shared/optimizer_discovery/investigation16/benchmark/B1_REPORT.md) | Résultat local de l’étude — valide seulement dans son périmètre, lire les corrections |
| [artifacts/optimizer_discovery/investigation16/benchmark/B2_PROTOCOL.md](../../_shared/optimizer_discovery/investigation16/benchmark/B2_PROTOCOL.md) | Protocole enregistré — conserver ; ne constitue pas une instruction de relance |
| [artifacts/optimizer_discovery/investigation16/benchmark/B2_REPORT.md](../../_shared/optimizer_discovery/investigation16/benchmark/B2_REPORT.md) | Résultat local de l’étude — valide seulement dans son périmètre, lire les corrections |
| [artifacts/optimizer_discovery/investigation16/benchmark/EXP15_FIRST_POINTS.md](../../_shared/optimizer_discovery/investigation16/benchmark/EXP15_FIRST_POINTS.md) | Documentation de l’étude — complément, aucune priorité automatique de relance |
| [artifacts/optimizer_discovery/investigation16/benchmark/FIRST_POINT_GEOMETRY.md](../../_shared/optimizer_discovery/investigation16/benchmark/FIRST_POINT_GEOMETRY.md) | Documentation de l’étude — complément, aucune priorité automatique de relance |
| [artifacts/optimizer_discovery/investigation16/benchmark/GEOMETRY_REVIEW.md](../../_shared/optimizer_discovery/investigation16/benchmark/GEOMETRY_REVIEW.md) | Vérification datée — détail de preuve, après lecture du rapport associé |
| [artifacts/optimizer_discovery/investigation16/benchmark/PROTOCOL_B1.md](../../_shared/optimizer_discovery/investigation16/benchmark/PROTOCOL_B1.md) | Protocole enregistré — conserver ; ne constitue pas une instruction de relance |
| [artifacts/optimizer_discovery/investigation16/benchmark/b2/independent_review.md](../../_shared/optimizer_discovery/investigation16/benchmark/b2/independent_review.md) | Vérification datée — détail de preuve, après lecture du rapport associé |
| [artifacts/optimizer_discovery/investigation16/budget/S2_PROTOCOL.md](../../_shared/optimizer_discovery/investigation16/budget/S2_PROTOCOL.md) | Protocole enregistré — conserver ; ne constitue pas une instruction de relance |
| [artifacts/optimizer_discovery/investigation16/budget/S2_REPORT.md](../../_shared/optimizer_discovery/investigation16/budget/S2_REPORT.md) | Résultat local de l’étude — valide seulement dans son périmètre, lire les corrections |
| [artifacts/optimizer_discovery/investigation16/feedback/F1_REVIEW.md](../../_shared/optimizer_discovery/investigation16/feedback/F1_REVIEW.md) | Vérification datée — détail de preuve, après lecture du rapport associé |
| [artifacts/optimizer_discovery/investigation16/feedback/PROTOCOL_F1.md](../../_shared/optimizer_discovery/investigation16/feedback/PROTOCOL_F1.md) | Protocole enregistré — conserver ; ne constitue pas une instruction de relance |
| [artifacts/optimizer_discovery/investigation16/feedback/REPORT.md](../../_shared/optimizer_discovery/investigation16/feedback/REPORT.md) | Résultat local de l’étude — valide seulement dans son périmètre, lire les corrections |
| [artifacts/optimizer_discovery/investigation16/feedback_experiment/GENERATION_FAILURES.md](../../_shared/optimizer_discovery/investigation16/feedback_experiment/GENERATION_FAILURES.md) | Documentation de l’étude — complément, aucune priorité automatique de relance |
| [artifacts/optimizer_discovery/investigation16/feedback_experiment/REPORT.md](../../_shared/optimizer_discovery/investigation16/feedback_experiment/REPORT.md) | Résultat local de l’étude — valide seulement dans son périmètre, lire les corrections |
| [artifacts/optimizer_discovery/investigation16/generation/ORDER_NOTE.md](../../_shared/optimizer_discovery/investigation16/generation/ORDER_NOTE.md) | Documentation de l’étude — complément, aucune priorité automatique de relance |
| [artifacts/optimizer_discovery/investigation16/generation/REPORT.md](../../_shared/optimizer_discovery/investigation16/generation/REPORT.md) | Résultat local de l’étude — valide seulement dans son périmètre, lire les corrections |
| [artifacts/optimizer_discovery/investigation16/history/HISTORY_REPORT.md](../../_shared/optimizer_discovery/investigation16/history/HISTORY_REPORT.md) | Résultat local de l’étude — valide seulement dans son périmètre, lire les corrections |
| [artifacts/optimizer_discovery/investigation16/history/P1_SCAFFOLD_REVIEW.md](../../_shared/optimizer_discovery/investigation16/history/P1_SCAFFOLD_REVIEW.md) | Vérification datée — détail de preuve, après lecture du rapport associé |
| [artifacts/optimizer_discovery/investigation16/history/PROTOCOL.md](../../_shared/optimizer_discovery/investigation16/history/PROTOCOL.md) | Protocole enregistré — conserver ; ne constitue pas une instruction de relance |
| [artifacts/optimizer_discovery/investigation16/history/RECOMMENDATIONS.md](../../_shared/optimizer_discovery/investigation16/history/RECOMMENDATIONS.md) | Proposition historique — aucune reprise autorisée par ce texte |
| [artifacts/optimizer_discovery/investigation16/history/REJECTION_MEMORY_DESIGN.md](../../_shared/optimizer_discovery/investigation16/history/REJECTION_MEMORY_DESIGN.md) | Proposition historique — aucune reprise autorisée par ce texte |
| [artifacts/optimizer_discovery/investigation16/history/TRACE_SCHEDULE_REVIEW.md](../../_shared/optimizer_discovery/investigation16/history/TRACE_SCHEDULE_REVIEW.md) | Vérification datée — détail de preuve, après lecture du rapport associé |
| [artifacts/optimizer_discovery/investigation16/production/BASELINE_CONTROL_PROTOCOL.md](../../_shared/optimizer_discovery/investigation16/production/BASELINE_CONTROL_PROTOCOL.md) | Protocole enregistré — conserver ; ne constitue pas une instruction de relance |
| [artifacts/optimizer_discovery/investigation16/production/DESIGN_REVIEW.md](../../_shared/optimizer_discovery/investigation16/production/DESIGN_REVIEW.md) | Vérification datée — détail de preuve, après lecture du rapport associé |
| [artifacts/optimizer_discovery/investigation16/production/DRIVER_REVIEW.md](../../_shared/optimizer_discovery/investigation16/production/DRIVER_REVIEW.md) | Vérification datée — détail de preuve, après lecture du rapport associé |
| [artifacts/optimizer_discovery/investigation16/production/ENGINEERING_REPORT.md](../../_shared/optimizer_discovery/investigation16/production/ENGINEERING_REPORT.md) | Résultat local de l’étude — valide seulement dans son périmètre, lire les corrections |
| [artifacts/optimizer_discovery/investigation16/production/FUTURE_DESIGN.md](../../_shared/optimizer_discovery/investigation16/production/FUTURE_DESIGN.md) | Proposition historique — aucune reprise autorisée par ce texte |
| [artifacts/optimizer_discovery/investigation16/production/INDEPENDENT_FINAL_REVIEW.md](../../_shared/optimizer_discovery/investigation16/production/INDEPENDENT_FINAL_REVIEW.md) | Vérification datée — détail de preuve, après lecture du rapport associé |
| [artifacts/optimizer_discovery/investigation16/production/ORDER_AUDIT.md](../../_shared/optimizer_discovery/investigation16/production/ORDER_AUDIT.md) | Vérification datée — détail de preuve, après lecture du rapport associé |
| [artifacts/optimizer_discovery/investigation16/production/PROGRAM_INSPECTION.md](../../_shared/optimizer_discovery/investigation16/production/PROGRAM_INSPECTION.md) | Description des sources choisies — ne prouve pas la nouveauté ni la causalité |
| [artifacts/optimizer_discovery/investigation16/production/PROTOCOL_P1.md](../../_shared/optimizer_discovery/investigation16/production/PROTOCOL_P1.md) | Protocole enregistré — conserver ; ne constitue pas une instruction de relance |
| [artifacts/optimizer_discovery/investigation16/production/PROTOCOL_P1_E1.md](../../_shared/optimizer_discovery/investigation16/production/PROTOCOL_P1_E1.md) | Protocole enregistré — conserver ; ne constitue pas une instruction de relance |
| [artifacts/optimizer_discovery/investigation16/production/analysis_protocol.md](../../_shared/optimizer_discovery/investigation16/production/analysis_protocol.md) | Protocole enregistré — conserver ; ne constitue pas une instruction de relance |
| [artifacts/optimizer_discovery/investigation16/production/presentation/P1_WORKBOOK_QA.md](../../_shared/optimizer_discovery/investigation16/production/presentation/P1_WORKBOOK_QA.md) | Présentation / calculs secondaires — données originales font autorité |
| [artifacts/optimizer_discovery/investigation16/production_run/preflight_independent_review.md](../../_shared/optimizer_discovery/investigation16/production_run/preflight_independent_review.md) | Vérification datée — détail de preuve, après lecture du rapport associé |
| [artifacts/optimizer_discovery/investigation16/report_data/README.md](../../_shared/optimizer_discovery/investigation16/report_data/README.md) | Présentation / calculs secondaires — données originales font autorité |
| [artifacts/optimizer_discovery/investigation16/report_data/independent_workbook_review.md](../../_shared/optimizer_discovery/investigation16/report_data/independent_workbook_review.md) | Vérification datée — détail de preuve, après lecture du rapport associé |
| [artifacts/optimizer_discovery/investigation16/research/LITERATURE_MECHANISMS.md](../../_shared/optimizer_discovery/investigation16/research/LITERATURE_MECHANISMS.md) | Références / mécanismes possibles — aucune preuve locale de gain supplémentaire |
| [artifacts/optimizer_discovery/investigation16/runtime/MESSAGE_TRANSMISSION_REPORT.md](../../_shared/optimizer_discovery/investigation16/runtime/MESSAGE_TRANSMISSION_REPORT.md) | Résultat local de l’étude — valide seulement dans son périmètre, lire les corrections |
| [artifacts/optimizer_discovery/investigation16/runtime/PROTOCOL.md](../../_shared/optimizer_discovery/investigation16/runtime/PROTOCOL.md) | Protocole enregistré — conserver ; ne constitue pas une instruction de relance |
| [artifacts/optimizer_discovery/investigation16/runtime/ROUTING_OPTIONS.md](../../_shared/optimizer_discovery/investigation16/runtime/ROUTING_OPTIONS.md) | Documentation de l’étude — complément, aucune priorité automatique de relance |
| [artifacts/optimizer_discovery/investigation16/runtime/TIMEOUT_REPORT.md](../../_shared/optimizer_discovery/investigation16/runtime/TIMEOUT_REPORT.md) | Résultat local de l’étude — valide seulement dans son périmètre, lire les corrections |
| [artifacts/optimizer_discovery/investigation16/runtime/VERIFICATION_36_NOTE.md](../../_shared/optimizer_discovery/investigation16/runtime/VERIFICATION_36_NOTE.md) | Vérification datée — détail de preuve, après lecture du rapport associé |
| [artifacts/optimizer_discovery/investigation16/selection/PROTOCOL_S1.md](../../_shared/optimizer_discovery/investigation16/selection/PROTOCOL_S1.md) | Protocole enregistré — conserver ; ne constitue pas une instruction de relance |
| [artifacts/optimizer_discovery/investigation16/selection/REPORT_S1.md](../../_shared/optimizer_discovery/investigation16/selection/REPORT_S1.md) | Résultat local de l’étude — valide seulement dans son périmètre, lire les corrections |
| [artifacts/optimizer_discovery/investigation16/statistics/REPORT.md](../../_shared/optimizer_discovery/investigation16/statistics/REPORT.md) | Résultat local de l’étude — valide seulement dans son périmètre, lire les corrections |
| [artifacts/optimizer_discovery/investigation16/throughput/PROTOCOL_T1.md](../../_shared/optimizer_discovery/investigation16/throughput/PROTOCOL_T1.md) | Protocole enregistré — conserver ; ne constitue pas une instruction de relance |
| [artifacts/optimizer_discovery/investigation16/throughput/REPORT_T1.md](../../_shared/optimizer_discovery/investigation16/throughput/REPORT_T1.md) | Résultat local de l’étude — valide seulement dans son périmètre, lire les corrections |
| [artifacts/optimizer_discovery/investigation16/throughput/TIMING_REPAIR_R1.md](../../_shared/optimizer_discovery/investigation16/throughput/TIMING_REPAIR_R1.md) | Documentation de l’étude — complément, aucune priorité automatique de relance |
| [artifacts/probe_2026/PREREG_W2_routing.md](../probe_2026/PREREG_W2_routing.md) | Protocole enregistré — conserver ; ne constitue pas une instruction de relance |
| [artifacts/probe_2026/README.md](../probe_2026/README.md) | Documentation de l’étude — complément, aucune priorité automatique de relance |
| [artifacts/probe_2026/RESULTS_W2_routing.md](../probe_2026/RESULTS_W2_routing.md) | Résultat local de l’étude — valide seulement dans son périmètre, lire les corrections |
| [artifacts/probe_2026/SPEC_BACKLOG_TRIAGE.md](../probe_2026/SPEC_BACKLOG_TRIAGE.md) | Documentation de l’étude — complément, aucune priorité automatique de relance |
| [artifacts/recursive_opt_assessment.md](../../ASSESSMENT.md) | Audit — §29 courant, sections précédentes datées |
| [examples/recursive_opt_use_cases_CURRENT_LIMITS.MD](../../../../examples/recursive_opt_use_cases_CURRENT_LIMITS.MD) | Historique rétracté en partie — lire la correction, pas les anciennes promotions |
| [opto/features/recursive_opt/README.md](../../../../opto/features/recursive_opt/README.md) | Carte d’architecture — formulations d’efficacité historiques à confronter au bilan |
