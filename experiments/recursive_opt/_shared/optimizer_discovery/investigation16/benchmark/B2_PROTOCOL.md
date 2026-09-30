# B2 — ablation exploratoire du premier point du seed

Ce protocole est écrit avant tout calcul de la variante B2. Les 96 trajectoires
seed B1 et les fixtures B1 sont déjà publiques et ont été inspectées. B2 n'est
donc ni un nouvel holdout ni une confirmation. Aucun appel de modèle ; aucun
changement du seed, des sources ou des protocoles EXP-15, F1, P1 ou P1-E1.

## Intervention unique, fixée

Copier exactement `benchmark.SEED_SOURCE`, puis remplacer l'unique condition
`if not history or rng.random() < 0.25:` par :

```python
    if not history:
        return [(low + high) / 2 for low, high in bounds]
    if rng.random() < 0.25:
```

Tout le reste du texte source reste identique. La signature ne change pas.
À histoire non vide identique, les deux programmes suivent les mêmes branches,
consomment le même aléa et doivent proposer exactement le même point. Les tests
utilisent la frontière subprocess existante avec rejeu déterministe. Aucun
tuning, aucune autre variante, aucune réparation après résultats.

L'intervention modifie la première observation, puis l'histoire et les décisions
ultérieures qui en dépendent. Le contraste mesure cet **effet total sur la
trajectoire**, pas exclusivement la contribution arithmétique du premier terme
de l'AUC. La décomposition premier terme / termes suivants sera descriptive :
les termes suivants mêlent persistance du premier point comme incumbent et
modification des propositions. Elle ne sépare pas ces mécanismes causalement.

## Évaluations et réutilisation fixées

- Réutiliser exactement les tâches `benchmark/freeze.json` de B1 : 12 instances,
  deux par famille/dimension, conditions central et broad appariées ; aucun
  nouveau tirage. Seeds locaux 16201, 16202, 16203, 16204.
- Évaluer uniquement la variante : 12 × 2 × 4 = 96 trajectoires, B=32,
  timeout par proposition 2 secondes, rejeu déterministe existant, sans fallback.
- Réutiliser les 96 contrôles seed B1 sauvegardés, vérifiés par source, tâche,
  seed, budget et hash de ligne. Ne pas les réexécuter.
- Jusqu'à huit workers offline ; aucun LLM. Maximum de recherche additionnel :
  3072 appels objectif et 6144 processus si toutes les propositions sont valides.
  La reconstruction de référence uniforme 24 × 128 = 3072 appels est comptée
  séparément ; conserver la normalisation B1 et vérifier son identité avant
  les évaluations. Les tests d'interface sont aussi hors budget de recherche.
- `b2/freeze.json` fige source exacte et hash, tâches, seeds, paramètres, sources
  d'évaluation, protocole, références de chaque contrôle et leurs hashes avant
  lancement. Écrire chaque nouvelle ligne sans écraser de preuve complétée.
  Reprendre seulement les allocations absentes, après contrôle du gel.

## Analyse fixée

Conserver toutes les lignes, y compris invalides et observations partielles.
Une trajectoire invalide a des métriques nulles ; ne pas l'exclure pour produire
une moyenne favorable. Une agrégation contenant une invalidité restera non
définie. Aucune imputation et aucun remplacement.

Rapporter central et broad séparément : AUC (plus bas = mieux), regret final,
atteinte de cible 0,01 et temps à cible plafonné/censuré à B+1. Agrégation égale
des six strates, puis moyenne des trajectoires dans chaque strate, comme B1.
Publier aussi les deltas appariés variante − seed par strate, seed local et
trajectoire, les courbes moyennes et les comptes de gains/pertes/égalités.
Décomposer le delta AUC en contribution du premier terme et des 31 suivants.
Ne pas estimer un effet R−I, une réplication de génération, un gain futur garanti
ou une supériorité confirmatoire à partir de ce diagnostic.

Les prochaines décisions scientifiques exigeraient une nouvelle expérience
préréglée et des données nouvelles. Le seed de P1 reste strictement inchangé.
