# Revue indépendante des gels P1 et contrôle B2

**PASS — aucun blocage constaté au contrôle du 2026-09-09T15:14:40.931197+00:00.** Cette note
transcrit la vérification déjà exécutée (sortie d'outil `613126`) ; aucun log
original autonome n'avait été créé. C'est une provenance non gelée ajoutée après
le contrôle, pas une modification du manifeste, des sources ou des preuves
scientifiques. L'audit n'a pas été relancé pour produire cette note.

Les appels `S.preflight(D.ROOT)` et `C.preflight(C.ROOT)` ont vérifié les sources,
l'environnement, la reconstruction des tâches et des allocations, ainsi que le
lien du contrôle au gel principal. Les digests examinés et attendus étaient :

- Principal : `113f2eb03abfbb3f8e80e8ce5946beecd6b039fef28968b96171475a21d57e8e`.
- Contrôle : `f0aeb8b78d56745c2e54b02462877952bdda15ec4e184fc0346e13d0bb0e288f`.
- ZIP des sources : `f5120ad690b842f6fd2ca1558d00c5c2e84f723984dbbecd336e60c76fc546eb`.

Les **97 entrées uniques** du ZIP correspondent exactement aux hashes du snapshot
et aux octets des fichiers locaux. L'archive couvre les 79 fichiers Python `opto`
et les 20 chemins uniques des deux gels. Les chemins sont relatifs, sans remontée
parent et sans entrée nommée `.env`. Cette vérification n'incluait pas un nouveau
scan du contenu contre une clé réelle.

Les budgets correspondent aux protocoles : **192 réponses**, 15 552 trajectoires
train/validation et 720 d'audit, soit 520 704 appels objectif logiques pour P1.
Le contrôle ajoute 144 trajectoires et 4 608 appels objectif. Les plafonds de
subprocess avec fallback sont respectivement 1 042 560 et 9 504, sous infrastructure
fiable. Les références de normalisation sont comptées séparément : 6 144 pour le
design principal ; les 1 536 références de l'audit du contrôle sont déjà partagées
scientifiquement avec celui-ci. Ce ne sont pas des comptes de reconstructions
physiques à chaque reprise.

Chronologie vérifiée, en nanosecondes d'horloge réelle :

```text
gel principal       1788966650162618709
< gel contrôle      1788966677320233025
< snapshot          1788966703629137738
< lancement         1788966750441347936
< première requête  1788966750854543971
< contrôle          1788966880931196903
```

Au moment précis du contrôle, une requête avait commencé et aucun fichier de
réponse complétée n'existait. Les barrières de fin de génération, de sélection et
d'audit étaient absentes ; le contrôle B2 n'avait ni résultat ni ligne brute.
Cela décrit cet instant historique et ne prétend pas décrire la progression
actuelle du run.

Aucune fitness d'audit ni réponse principale n'a été ouverte. Zéro objectif,
candidat, modèle ou connexion réseau : des gardes locales rejetaient ces appels
pendant la vérification. Les préflights ont seulement reconstruit les identités
des tâches enregistrées. Détails, comptes et portée exacte dans
[preflight_independent_review.json](../../../../EXP16/results/production_run/preflight_independent_review.json).
