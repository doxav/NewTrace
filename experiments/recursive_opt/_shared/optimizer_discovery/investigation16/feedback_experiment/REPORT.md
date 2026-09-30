# EXP-16/F1 — instruction anytime et contenu du feedback

**Les 24 réponses et toutes les évaluations prévues sont conservées et passent
l'audit final. F1 ne démontre pas de gain du feedback riche.** Le contraste
riche − sparse donne un signal négatif exploratoire selon la règle enregistrée ;
les trois autres contrastes sont inconclusifs. Ce résultat porte sur une
proposition issue d'un parent fixé, avec fallback commun, et non sur une recherche
itérative complète. Il ne justifie aucun réglage rétroactif pour faire gagner
le feedback riche.

## Protocole exécuté

Le [protocole prospectif](../feedback/PROTOCOL_F1.md) compare quatre conditions :

- **legacy_code** : améliorer le parent avec l'instruction générique initiale,
  sans résultats d'évaluation ;
- **anytime_code** : même requête avec l'objectif anytime explicite ;
- **anytime_sparse** : instruction anytime et feedback sparse EXP-15 du parent ;
- **anytime_rich** : instruction anytime et résumé indexé des trajectoires,
  incumbent, événements d'amélioration et courbe best-so-far du parent.

Les blocs 16301, 16303 et 16305 utilisent le seed inchangé. Les blocs 16302,
16304 et 16306 utilisent le représentant EXP-15 sélectionné antérieurement par
validation, seed externe 41. Chaque bloc partage son parent et ses seeds locaux
entre conditions. Les 24 requêtes étaient fixées avant génération, sans adaptation
aux réponses. Ordre mélangé par RNG 163000, aucun déterminisme LLM supposé.

Modèle demandé : `deepseek/deepseek-v4-flash-0731` via OpenRouter. Tous les reçus
identifient `deepseek/deepseek-v4-flash-20260731`. Réglages communs : température
0,6, top-p 1, limite 32 000, champ natif `reasoning.effort=low`, timeout client
300 s, concurrence 1, cache désactivé et zéro retry de réponse vide. La seed de
requête est celle du bloc ; elle constitue un indice, sans garantie de rejeu LLM.

La limite 32 000 provient de la règle de faisabilité G1 gelée avant F1. Les tâches
F1-R1 comprennent six instances train et six instances validation distinctes :
une pour chaque combinaison Sphere/Quadratic/Rosenbrock × dimension 2/4. B=32,
un seed local par tâche/bloc. Normalisation uniforme indépendante de 128 points
par tâche ; moyenne anytime des regrets normalisés, puis poids égaux des six
strates. Aucun optimum ou constante de normalisation n'entre dans le feedback.

La validation n'est ouverte qu'après les 24 réponses et leurs évaluations train.
Chaque programme est évalué comme politique de déploiement : un échec déclenche
le seed inchangé pour les évaluations restantes, avec l'histoire réellement
observée. Ce test ne sélectionne ni best-of-N ni représentant selon ses résultats.

## Résultats complets

AUC et regret final : plus bas = mieux. La médiane ci-dessous est celle des six
scores de bloc, après agrégation des tâches. Un bloc est l'unité de réplication.

| Condition | AUC moyenne | AUC médiane | Regret final | Cible atteinte | Temps cible plafonné/censuré | Sources éligibles train | Fallback validation |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy_code | 0,164111 | 0,105047 | 0,063648 | 61,11 % | 22,500 | 6/6 | 0/36 |
| anytime_code | 0,122130 | 0,121349 | 0,032739 | 63,89 % | 20,250 | 4/6 | 12/36 |
| anytime_sparse | 0,085002 | 0,088830 | 0,019429 | 63,89 % | 21,222 | 6/6 | 0/36 |
| anytime_rich | 0,134231 | 0,140146 | 0,031395 | 58,33 % | 19,722 | 4/6 | 12/36 |

La cible est un regret normalisé ≤0,01. Les non-atteintes restent `null` comme
temps observé ; 33 est uniquement la convention de moyenne plafonnée/censurée.
Leur fréquence doit être lue avec ce temps moyen.

| Bloc | Parent | legacy_code | anytime_code | anytime_sparse | anytime_rich |
|---|---|---:|---:|---:|---:|
| 16301 | seed | 0,349667 | 0,097074¹ | 0,111199 | 0,097074² |
| 16302 | représentant | 0,045369 | 0,151769³ | 0,012805 | 0,132625 |
| 16303 | seed | 0,130166 | 0,167403 | 0,122349 | 0,147667 |
| 16304 | représentant | 0,079928 | 0,114890 | 0,023636 | 0,176218⁴ |
| 16305 | seed | 0,333280 | 0,127807 | 0,173564 | 0,196337 |
| 16306 | représentant | 0,046258 | 0,073839 | 0,066461 | 0,055464 |

¹ Plusieurs blocs de code, mauvaise API, terminaison `stop` ; ² contenu final
null et terminaison `length` ; ³ plusieurs fragments/mauvaise API et `length` ;
⁴ contenu final null et terminaison fournisseur `error`. Ces quatre scores
proviennent intégralement du fallback commun ; leurs invalidités restent visibles.

## Contrastes enregistrés et incertitude

Chaque delta est traité − contrôle ; un delta négatif favorise la première
condition. Bootstrap apparié des six blocs, 10 000 resamples, RNG 1515,
percentiles interpolés 2,5/97,5. Les quatre comparaisons sont exploratoires,
sans correction de multiplicité et avec une incertitude fragile à n=6.
Ces intervalles sont conditionnels aux deux sources parentes et au panel de
validation fixés ; ils ne couvrent pas de nouvelles familles ou de nouveaux parents.

| Contraste | Moyenne ΔAUC | Médiane ΔAUC | Intervalle bootstrap 95 % | Lecture enregistrée |
|---|---:|---:|---|---|
| anytime_code − legacy_code | −0,041981 | +0,031272 | [−0,149739 ; +0,057924] | inconclusif |
| anytime_sparse − anytime_code | −0,037128 | −0,026216 | [−0,089846 ; +0,011222] | inconclusif |
| anytime_rich − anytime_sparse | +0,049229 | +0,024045 | [+0,000160 ; +0,103587] | signal négatif exploratoire |
| anytime_rich − anytime_code | +0,012100 | −0,009188 | [−0,015993 ; +0,041424] | inconclusif |

Tous les deltas individuels sont conservés dans [analysis_results.json](analysis_results.json).
La borne basse riche − sparse est très proche de zéro. L'intervalle et l'ordre
des moyennes ne démontrent pas une supériorité générale du sparse. L'instruction
anytime présente aussi une moyenne et une médiane de delta de signes opposés.

## Programme parent et portée du mécanisme

Les deux classes de parent étaient prévues ; chacune ne contient que trois
blocs. Les moyennes descriptives sont :

| Parent | Parent inchangé | legacy_code | anytime_code | anytime_sparse | anytime_rich |
|---|---:|---:|---:|---:|---:|
| Seed, n=3 | 0,168653 | 0,271038 | 0,130761 | 0,135704 | 0,147026 |
| Représentant EXP-15, n=3 | 0,044113 | 0,057185 | 0,113499 | 0,034301 | 0,121436 |

Les moyennes descriptives varient selon le parent et les tirages. Dans les blocs du représentant,
le fallback utilise le seed original, comme prévu, et non ce représentant plus
performant : une génération manquante peut donc perdre l'avantage du parent.
Les moyennes de contrôle sur les six blocs sont 0,161305 pour le seed et 0,106383
pour le mélange des parents fixés. Comparer uniquement un programme proposé au
seed pourrait masquer cette différence de point de départ.

F1 teste une intervention de prompt à parent fixé. Il n'évalue pas la capacité
d'un moteur itératif à conserver ses bons incumbents ou à récupérer après un
échec. Le feedback riche contient des valeurs brutes et les trajectoires, mais
aucun score train agrégé normalisé. Six trajectoires ne deviennent pas six
réplications indépendantes du modèle. Ni effet de profondeur, ni nouveauté,
ni amortissement ne sont mesurés. Une recherche production suivant son protocole
propre reste nécessaire ; F1 ne change pas ses choix gelés.

## Invalidité et erreur fournisseur

20/24 sources sont extractibles et éligibles sur tout le train. Ces 20 programmes
complètent aussi leurs 120 trajectoires validation sans erreur. Aucun échec
subprocess, timeout candidat, exception ou non-déterminisme n'est observé parmi
ces programmes. Les quatre autres réponses ont `source_status=missing_source`,
mais leurs causes sont différentes, comme détaillé dans
[GENERATION_FAILURES.md](GENERATION_FAILURES.md).

Les terminaisons comprennent 21 `stop`, deux `length` et une `error`. Les deux
réponses code-only invalides contiennent bien du code fragmenté avec une mauvaise
API ; elles ne sont pas décrites comme absence totale de code. Les deux réponses
rich invalides ont un contenu final null. L'une atteint la limite déclarée ;
l'autre est l'erreur Morph après 4 229,334 s avec 18 186 tokens rapportés et
coût rapporté zéro. Cette dernière est un échec fournisseur/génération, pas une
erreur de syntaxe démontrée. Le slot complété reste consommé conformément au gel.

Le train conserve 24 trajectoires invalides et 120 valides, sans regret imputé.
La validation conserve 144 politiques de déploiement définies, dont 24/144
(16,67 %) utilisent le fallback dès la première proposition. Les compteurs
bruts contiennent 7 680 propositions candidates valides sur 7 728 tentatives
candidates, soit 99,38 %, et 768 propositions valides supplémentaires du fallback.
Ce taux au niveau des points doit être lu avec l'éligibilité de seulement
83,33 % des programmes. Aucun échec n'a donné droit à une réponse supplémentaire.

Le résultat de déploiement inclut la fiabilité des fournisseurs et du canal
de génération. Leur variation, le routage stochastique et les petits effectifs
limitent l'attribution au contenu du feedback. Aucun sous-ensemble de programmes
valides ne remplace l'analyse principale ; aucun résultat négatif n'est supprimé.

## Budgets et ressources réelles

| Condition | Réponses | Tentatives client | Tokens prompt | Completion | Raisonnement rapporté | Total | Coût réponse USD |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy_code | 6 | 6 | 8 439 | 37 462 | 29 532 | 45 901 | 0,018843770 |
| anytime_code | 6 | 6 | 8 895 | 84 049 | 61 265 | 92 944 | 0,021651124 |
| anytime_sparse | 6 | 8 | 15 381 | 51 114 | 44 790 | 66 495 | 0,026292826 |
| anytime_rich | 6 | 6 | 41 909 | 74 442 | 72 546 | 116 351 | 0,012986942 |
| **Total** | **24** | **26** | **74 624** | **247 067** | **208 133** | **321 691** | **0,079774661** |

Les 24 reçus rapportent ensemble 0,079774657 USD, avec une petite différence
d'arrondi par rapport aux réponses. Les compteurs de raisonnement sont conservés
verbatim et ne sont pas ajoutés au total. `16301/rich` rapporte 33 818 tokens de
raisonnement contre 32 000 de completion : cette incohérence est signalée, sans
recalcul inventé ni invalidation du slot. L'égalité porte sur six réponses par
condition et leurs plafonds, pas sur les tokens, coûts ou durées effectivement
dépensés.

Les deux tentatives transport supplémentaires concernent `16302/anytime_sparse`.
Les enregistrements de 702,924 s et 20,036 s sont suivis d'une réponse complétée
à la tentative 3. La deuxième tentative conserve `transient=false` : une
[reproduction du désaccord de classifieurs DNS](../runtime/dns_classifier_reproduction.json)
documente pourquoi l'interruption temporaire a pu être reprise sans changer le
code gelé ni remplacer un slot complété. Les deux tentatives gardent le marqueur
de complétion distante/facturation potentielle inconnue ; leurs coûts ne sont
pas inventés ni inclus comme zéros dans les reçus des réponses connues.

- Allocations candidates train+validation : 288 trajectoires, 9 216 appels
  objectif ; **8 448 appels réellement effectués**, 768 allocations train
  inutilisées après source manquante. Aucune allocation recyclée.
- Préparation des parents train : 36 trajectoires, 1 152 objectifs ; contrôles
  parent+seed validation : 72 trajectoires, 2 304 objectifs.
- Total des trajectoires de politique : 396 allocations, **11 904 objectifs
  effectifs**, **23 808 subprocesses** incluant le rejeu déterministe.
- Référence : 12 tâches ×128 =1 536 évaluations de normalisation distinctes
  scientifiquement ; les reconstructions de ces constantes par différents
  processus sont du travail de préparation/vérification séparé.

Les durées des 24 réponses totalisent 13 912,446 s (3,865 h), médiane 276,014 s,
étendue 17,614–4 229,334 s. Avec les deux tentatives transport, la somme mesurée
est 14 635,406 s (4,065 h). La machine a été suspendue : l'observation d'horloges
conserve notamment 5 577,196 s de suspension additionnelle. Les temps calendrier
ne sont donc pas des latences de fournisseur. Le timeout 300 s est une limite
par opération réseau, pas une échéance absolue de requête ; voir
[TIMEOUT_REPORT.md](../runtime/TIMEOUT_REPORT.md).

Fournisseurs des réponses connues : OpenInference 6, Morph 6, DigitalOcean 2,
DeepInfra 2, Sail Research 2, Reka 2, Relace 2, GMICloud 1, Cloudflare 1. Ces
associations ne sont pas un essai causal entre fournisseurs. Les durées et
préférences de routage ne sont pas modifiées après résultats.

## Sources, intégrité et vérifications

Les sources exactes évaluées restent dans `raw/<bloc>/A<condition>/slot_00/response.json`,
champ `source`, avec hash complet dans chaque réponse et dans l'audit final.
Aucune source n'a été formatée, réparée ou sélectionnée par la validation F1.
Les tailles et comptes de nœuds AST de chaque programme restent descriptifs
dans l'audit ; aucune métrique de sélection n'en dépend.

- Seed : `5b74e5a3fe2fc90fcb42a603b058fa7befbf65f38775caa77acd30cdf619b640`.
- Parent représentant : `1684f91acdc36c0ca6aac70afeb9cc2c4eed7ab847926d5880590e059266abb7`.
- Gel F1 canonique : `9ee3466976a6de41ce885aee3c7b4d081256f46323bec78585011c120f99428c`.
- Résultats primaires canoniques : `9bd11b11d1a7d17ae883ad27aedc41c3c276391bae336eec9a153634fc140798`.
- Audit final canonique : `961de1ab8e726d60ee7f3fdf26fd54dcb886311df4b836552a4158a4ea36a859`.

L'audit indépendant vérifie le gel/configuration, la reconstruction exacte des
24 requêtes, leurs IDs uniques, source/hash et extraction, les six parents,
les 26 tentatives, les 24 reçus correspondants et les 60 fichiers d'évaluation
scellés. Il vérifie aussi les curseurs de budget, le fallback permanent,
les métriques, les allocations et le recalcul exact de toutes les lignes et
des quatre contrastes. Les réponses ont toutes précédé la barrière validation.
La garantie temporelle repose sur le contrôle de flux gelé et les hashes de
barrière ; les fichiers de trajectoire n'ont pas d'horodatage indépendant de
début d'évaluation. La frontière subprocess n'est pas un sandbox OS.

Les annotations nouvelles de terminaison fournisseur et le correctif `len(None)`
concernent uniquement l'analyse dérivée non gelée ; aucun résultat scientifique
ni fichier d'exécution gelé n'a changé. Le protocole conserve aussi l'amendement
antérieur F1-R1, les échecs d'ingénierie et l'ancien namespace F1 non revendiqué
comme inédit. La vérification finale et ses commandes exactes sont enregistrées
dans [final_verification.json](final_verification.json).
