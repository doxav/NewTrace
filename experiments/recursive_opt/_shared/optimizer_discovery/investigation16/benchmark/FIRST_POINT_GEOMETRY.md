# Premier point : rapport théorique pour les quadratiques diagonales

**Sous les hypothèses ci-dessous, le rapport est bien
E[f(centre)] / E[f(uniforme)] = a² / (a² + b²).** Il porte sur les espérances
brutes du premier objectif, dans le modèle probabiliste du générateur. Ce n'est
ni une espérance de ratio normalisé, ni une prédiction de l'AUC de B2.

Écrivons les deux familles diagonales enregistrées sous la forme

`f(x) = Σ_i q_i (x_i − M_i)²`, avec `q_i = A w_i / s_i² > 0`.

Ici A désigne l'amplitude ; a désigne uniquement la demi-largeur de la distribution
des shifts. Supposons `M_i ~ U[−a,a]`, indépendants des coefficients q, et un
premier point `X_i ~ U[−b,b]`, indépendant des shifts et des coefficients.
Les coefficients ont une espérance finie. Ils peuvent être corrélés entre eux,
notamment via l'amplitude commune : cette indépendance supplémentaire n'est pas
nécessaire. Le centre des bornes symétriques est le vecteur zéro.

Conditionnellement à q, les moyennes de M_i et X_i sont nulles, leurs variances
valent a²/3 et b²/3, et le terme croisé a une espérance nulle. Par conséquent :

```text
E[f(0) | q] = (a² / 3) Σ_i q_i
E[f(X) | q] = ((a² + b²) / 3) Σ_i q_i

E[f(0)] / E[f(X)]
  = [(a² / 3) E(Σ_i q_i)] / [((a² + b²) / 3) E(Σ_i q_i)]
  = a² / (a² + b²).
```

Le dénominateur est strictement positif. Le rapport ne dépend ni de la dimension
ni de la distribution des coefficients sous ces hypothèses.

| Demi-largeur des shifts a | Demi-largeur des bornes b | Rapport des espérances brutes |
|---:|---:|---:|
| 2 | 5 | 4/29 ≈ 0,137931 |
| 4,5 | 5 | 81/181 ≈ 0,447514 |

Le centre conserve donc un avantage de premier objectif brut en espérance dans
les deux lois théoriques. Élargir une distribution symétrique des shifts ne
supprime pas automatiquement cet avantage.

La lecture de [benchmark.make_tasks](../../benchmark.py) et de
[generation.fresh_tasks](../generation.py) confirme des appels `rng.uniform`
successifs distincts pour les shifts, les échelles, l'amplitude et, pour
Quadratic, les poids. Aucune valeur de shift n'intervient dans le calcul d'une
échelle, d'un poids ou de l'amplitude ; aucune acceptation ou sélection de ces
paramètres ne dépend des shifts. Le seul embranchement des poids dépend de la
famille. [headroom.task_pairs](headroom.py) multiplie uniquement les shifts par
2,25 pour broad, en conservant les autres coefficients : la loi visée passe
ainsi de U[−2 ; 2] à U[−4,5 ; 4,5] sans couplage supplémentaire aux coefficients.

Il faut toutefois distinguer cette loi visée des tableaux gelés. Le code utilise
un PRNG déterministe commun par instance, pas des variables aléatoires idéales
dont l'indépendance mathématique serait certifiée par la lecture du code. De plus,
`generation.local_seed` dérive un seed de l'identité de tâche : la séparation de
domaines aléatoires ne prouve pas à elle seule une indépendance probabiliste.
B1/B2 emploient quatre seeds locaux fixés directement, communs aux tâches. Le
rapport théorique suppose un premier point uniforme indépendant ; il n'impose
aucune égalité exacte aux moyennes de ces échantillons finis.

Les limites de la formule sont strictes :

- Elle compare `E[f(0)]` à `E[f(X)]`, et non `E[f(0)/f(X)]`.
- Elle n'est pas un calcul de `E[f(0)/normalisation(tâche)]`, ni du rapport des
  regrets normalisés agrégés. La référence de 128 points varie avec la tâche ;
  une division avant moyennage modifie les poids et ne permet pas cette annulation.
- Elle concerne le premier point uniquement. Les minima cumulés, les décisions
  conditionnées par l'histoire, la persistance de l'incumbent et l'AUC nécessitent
  une analyse différente.
- Elle ne s'applique pas à Rosenbrock : les termes couplés et non quadratiques
  changent la dérivation. Elle ne prédit donc pas non plus l'agrégat des trois
  familles du benchmark.
- Elle n'établit aucun avantage chiffré du feedback récursif sur la génération
  indépendante, aucune supériorité terminale et aucune généralisation de B2.

Cette note est une dérivation analytique et une lecture du code local. Aucune
simulation, évaluation d'objectif, exécution de candidat ou donnée F1/P1 n'a été
utilisée ; aucun protocole ou résultat gelé n'a été modifié.
