# B1 — diagnostic prospectif du prior central et de la surface

Enregistré avant toute évaluation B1. Aucun appel LLM. Aucun changement d'EXP-15.
La question est : le prior central des optima et la métrique anytime favorisent-ils
une politique sans apprentissage, et existe-t-il de la marge sur des instances
distinctes ? Ce diagnostic ne choisira pas une distribution parce qu'elle favorise A2.

- Douze tâches nouvelles : `generation.fresh_tasks("B1", "train", 2)` ; deux
  instances par strate Sphere/Quadratic/Rosenbrock × dimensions 2/4.
- Conditions appariées : shifts originaux dans [-2,2], puis chaque shift multiplié
  par 2,25 (dans [-4,5,4,5]). Amplitude, échelles, poids et bornes [-5,5] identiques.
  Les optima transformés restent faisables et de valeur zéro.
- Quatre seeds locaux exacts : 16201, 16202, 16203, 16204, partagés entre politiques
  et conditions. Ils ne sont pas des seeds de génération ni de tâches.
- Politiques fixées : seed EXP-15 inchangé ; uniforme déterministe utilisant
  `random.Random(seed + len(history))` ; midpoint constant ; représentant EXP-15
  choisi sur validation (outer 41), exact source/hash, sans réparation.
- B=32 ; même API/subprocess/déterminisme que `benchmark.evaluate`, timeout 2 s,
  pas de fallback. Invalidité conservée et publiée ; pas de remplacement.
- Normalisation identique à EXP-15 : 128 points uniformes, seed déterministe de
  l'identité sémantique. Les constantes changent avec la transformation, ainsi que
  la seed de référence. Les comparaisons entre politiques utilisent toujours la
  même constante au sein d'une tâche. Comparer des AUC entre conditions implique
  donc une autre distribution et une autre normalisation, pas seulement un trajet déplacé.
- Rapport : AUC, regret final, atteinte de cible 0,01, hitting time censuré à B+1,
  fraction de la somme AUC provenant des 1/4/8 premières évaluations, rangs par
  strate et seed ; agrégation moyennée par strate à poids égaux, puis seeds.
  Donner aussi les courbes moyennes. Pas de test de supériorité confirmatoire.
- Budget fixé : 4 politiques × 12 tâches × 4 seeds × 2 conditions = 384 trajectoires,
  12288 appels objectif alloués, jusqu'à 24576 subprocess de déterminisme si tous
  valides. Préparation : 24 × 128 = 3072 références, séparées du budget de recherche.
- Deux workers offline maximum. Aucun accès aux résultats holdout EXP-15 n'est
  nécessaire. Toutes les observations, erreurs, sources, tâches et hashes sont
  conservés avant agrégation ; reprise seulement des trajectoires absentes.

Le fichier `freeze.json` contient l'instance exacte de ce protocole et les sources
avant le lancement. Les tests vérifient pairing, optima, disjonction et aggregation.
