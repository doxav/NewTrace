# Options de routage après F1/E1

**Aucun changement immédiat.** Cette note documentaire prépare une décision après
F1/E1. Les gels, modèles, requêtes et routages existants restent intacts. Aucun
résultat de fitness F1/P1 n'a été consulté ; aucun appel génératif et aucune clé
n'ont été utilisés pour cette recherche.

Consultation du 9 septembre 2026. Deux GET publics du même endpoint ont renvoyé
HTTP 200 et un corps identique ; le second a été conservé à 13:39:43 UTC dans
[ROUTING_PUBLIC_METADATA.json](ROUTING_PUBLIC_METADATA.json), avec champs publics
sélectionnés et SHA256 du corps original.

## Ce que proposent les paramètres officiels

Le routage par défaut privilégie prix et disponibilité. Les options suivantes
appartiennent à l'objet JSON `provider` ; `sort` ou `order` désactivent
l'équilibrage probabiliste habituel. [Provider Selection](https://openrouter.ai/docs/guides/routing/provider-selection)

| Option | Effet documenté |
|---|---|
| `sort: "throughput"` | Essaie d'abord les endpoints au débit supérieur. |
| `sort: "latency"` | Priorise la latence basse. |
| `preferred_min_throughput` / `preferred_max_latency` | Préférences souples, éventuellement par percentile ; les endpoints hors seuil restent des recours. |
| `max_price` | Filtre dur : prix maximum prompt/completion en dollars par million de tokens ; peut rendre la requête impossible. |
| `order`, `only`, `ignore` | Ordre explicite, liste autorisée ou exclusions ; utiliser les slugs exacts des endpoints. |
| `allow_fallbacks: false` | Réduit les recours vers d'autres fournisseurs ; risque de disponibilité accru. |
| `require_parameters: true` | Exclut les endpoints qui ne prennent pas en charge tous les paramètres fournis. |

Les seuils reposent sur des statistiques mobiles ; ils ne garantissent ni SLA ni
durée maximale. `max_tokens` filtre aussi les capacités de sortie compatibles.
[Référence du routage](https://openrouter.ai/docs/guides/routing/provider-selection)

Un fragment possible **à piloter ultérieurement**, avec les autres réglages
explicitement conservés dans une nouvelle spécification :

```json
{
  "model": "deepseek/deepseek-v4-flash-0731",
  "provider": {"sort": "throughput"}
}
```

La documentation distingue l'attente avant le premier token (réseau, file,
préremplissage) et le débit de génération. Pour une réponse longue, diminuer
l'attente initiale seule peut laisser un coût temporel important. Le guide
décrit aussi des seuils de performance sur cinq minutes et l'association d'un
tri par prix à une préférence de débit. **Inférence pour notre pilote futur :**
mesurer séparément premier token, durée totale et volume produit afin de choisir
l'intervention pertinente. [Latency and Performance](https://openrouter.ai/docs/guides/best-practices/latency-and-performance)

## Disponibilité publique du modèle exact et prix

Le GET exact retourne **29 objets endpoint, 28 slugs distincts**. Tous déclarent
`reasoning`, `temperature`, `top_p` et `max_tokens`, et une capacité de sortie
d'au moins 32 768 tokens ; seulement 17 déclarent `seed`. Activer
`require_parameters` avec une seed pourrait donc modifier fortement le pool.
La déclaration de support ne prouve pas un déterminisme inter-fournisseurs.
Tous les champs publics `latency_last_30m` et `throughput_last_30m` sont **null** :
aucun classement de vitesse n'est déductible de cet instantané.
[API publique du modèle](https://openrouter.ai/api/v1/models/deepseek/deepseek-v4-flash-0731/endpoints)

Les tarifs déclarés vont de **0,05 à 0,44 $/million** en entrée et de
**0,16 à 1,32 $/million** en sortie. À titre arithmétique, 32 000 tokens de sortie
facturables représenteraient 0,00512–0,04224 $, hors entrée et autres composantes.
Ce n'est pas une prévision de facture : routage effectif, usage, conventions de
comptage et tarifs peuvent différer. L'instantané contient notamment des
quantifications `fp4`, `fp8`, `bf16` et `unknown` ; un même identifiant demandé
n'impose donc pas une implémentation de service identique.
[Tarifs et attributs par endpoint](https://openrouter.ai/api/v1/models/deepseek/deepseek-v4-flash-0731/endpoints)

Cet accès anonyme ne montre pas les filtres du compte ni une éventuelle capacité
privée. La présence d'un endpoint ne garantit pas sa disponibilité pour la
prochaine requête. Un prix élevé ou un slug contenant `fast` ne prouve aucune
qualité de code ni aucun gain de vitesse sur nos prompts.

## Attention au raccourci `:nitro` et aux tiers

La page générale de routage décrit `:nitro` comme un raccourci du tri par débit.
La page spécialisée actuelle précise cependant qu'il admet aussi les endpoints
`priority` dans le pool. Elle distingue cela d'une demande explicite
`service_tier: "priority"`, qui privilégie ce tier avec possibilité de repli ;
la facturation suit le tier réellement utilisé. Cette nuance interdit de
supposer que raccourci et tri simple ont exactement le même pool ou coût.
[Service Tiers](https://openrouter.ai/docs/guides/features/service-tiers)

Le snapshot consulté ne décrit aucun tier explicite pour le modèle demandé.
Il ne permet donc pas de confirmer une capacité priority utilisable ici.
Conserver le slug exact et tester un champ `provider` explicite serait une
option expérimentalement plus facile à attribuer ; ce choix reste à valider
après F1/E1, sans substituer silencieusement le modèle.

## Identité, tentatives et durée : instrumentation à envisager

Le reçu `GET /generation?id=...` expose notamment `provider_name`, le modèle,
`upstream_id`, les compteurs natifs, les coûts et les temps. C'est une API
authentifiée : **aucun de ces GET n'a été effectué dans cette sous-tâche**.
Pour une expérience future, conserver les reçus avec couverture et valeurs
manquantes, en plus de l'identité demandée. [Generation metadata](https://openrouter.ai/docs/api/api-reference/generations/get-request-&-usage-metadata-for-a-generation)

OpenRouter documente aussi le header opt-in `X-OpenRouter-Metadata: enabled` :
il expose le routage retenu et, lorsqu'elles sont présentes, les tentatives
internes de fallback. Il faut tester sa conservation par notre client avant
un nouveau gel ; ce header n'a pas été ajouté aux expériences en cours.
[Router Metadata](https://openrouter.ai/docs/guides/features/router-metadata)

Ces tentatives internes sont distinctes des retries du client. Les erreurs
429/503 peuvent fournir `Retry-After`. Un échec après début de génération peut
apparaître dans le corps plutôt que par le seul statut HTTP. Une future politique
doit compter séparément appels client, tentatives amont, réponses complétées et
facturation incertaine après interruption. La politique gelée actuelle n'est
pas réécrite ici. [Errors and Debugging](https://openrouter.ai/docs/api_reference/errors-and-debugging)

OpenRouter recommande aux fournisseurs des commentaires SSE de maintien de
connexion pendant un calcul long ; leur absence peut déclencher un fetch
timeout suivi d'un fallback. Le débit public inclut aussi file et délai initial.
Cela explique pourquoi activité réseau, premier token utile et fin de réponse
sont trois événements différents. Cela ne mesure pas la cause de l'appel F1
en cours. [Provider performance metrics](https://openrouter.ai/docs/guides/community/for-providers#11-performance-metrics)

Pour la décision après F1/E1, un pilote symétrique devra fixer les prompts frais,
la politique de routage, les plafonds et les critères de faisabilité avant les
appels. Il devra retenir toutes les réponses, comparer validité et durée active,
mesurer les suspensions de la machine séparément, puis geler la configuration
retenue avant toute nouvelle expérience. Les différences observées entre
fournisseurs dans les appels existants ne constituent pas une estimation
causale de leur qualité. Aucun gain de vitesse ou de fitness n'est projeté ici.
