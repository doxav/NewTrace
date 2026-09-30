# Ce que signifie réellement `timeout=300`

**Les 300 secondes ne sont pas une durée maximale de requête.** Dans le client
installé, cette valeur atteint bien HTTPX et limite séparément les opérations
réseau, dont l'attente de nouvelles données en lecture. Une réponse peut durer
plus de 300 secondes si elle continue à recevoir des données. La valeur n'est
pas silencieusement ignorée.

Ce constat est vérifié par inspection du chemin exécuté et par un serveur HTTP
local traversant les vrais wrappers du projet. Aucun fournisseur de génération
n'a été contacté par cette enquête et aucun processus live, paramètre gelé ou
fichier G/F/I/E/B/S/T n'a été modifié.

## Observation F1 conservée

La première réponse `16301/Aanytime_code/slot_00` a les paramètres enregistrés
suivants : `timeout=300`, `num_retries=0`, limite 32000, raisonnement `low`, modèle
exact `deepseek/deepseek-v4-flash-0731`. Le constructeur live utilise
`max_retries=1`, `request_timeout_s=300`, `allow_env_overrides=False` et zéro retry
de réponse vide.

La réponse conservée indique **538.324706398 secondes**, tentative 1,
`finish_reason=stop`, 760 tokens de prompt, 26594 de completion, dont 18496 de
raisonnement, total 27354, coût fournisseur 0.00676248496 USD. Son code est
imparsable et reste un résultat invalide consommant un slot. Le dépassement de
300 secondes n'est ni une raison de remplacer cette réponse ni une preuve que
le timeout a causé son invalidité.

L'observation de trafic TLS actif pendant l'attente est compatible avec le
mécanisme établi ici. Elle ne révèle pas à elle seule si les octets distants
étaient du contenu de réponse ou des messages de maintien de connexion. Nous
n'avons pas capturé ni déchiffré le flux distant. Cette enquête n'attribue donc
pas avec certitude chaque seconde de cette réponse à un mécanisme côté serveur.

## Chemin installé vérifié

Environnement : **LiteLLM 1.75.0, HTTPX 0.28.1, HTTPcore 1.0.9**.

1. `opto/features/recursive_opt/runmode.py:61` ajoute le timeout par `setdefault` :
   le paramètre explicite de la requête prévaut, sinon le défaut du wrapper vaut
   300.
2. `opto/utils/llm.py:320` appelle directement `litellm.completion` avec ces kwargs,
   dans le retry du projet. Ici `max_retries=1` signifie une seule tentative.
3. LiteLLM `main.py:1124` convertit la valeur en flottant ; sa branche OpenRouter
   `main.py:2523` la passe à `BaseLLMHTTPHandler.completion`.
4. `litellm/llms/custom_httpx/llm_http_handler.py:153` transmet le timeout à
   `HTTPHandler.post`. Pour la requête non streamée, le corps est néanmoins reçu
   en plusieurs lectures réseau.
5. `litellm/llms/custom_httpx/http_handler.py:743` construit la requête HTTPX avec
   `timeout=...` puis appelle `client.send`.
6. HTTPcore `_sync/http11.py:196` réutilise le timeout de lecture pour chaque
   lecture du corps. `_backends/sync.py` l'applique aux lectures du socket. Aucun
   chronomètre absolu de 300 secondes n'entoure tout ce parcours dans le wrapper
   du projet.

Les empreintes des sources réellement inspectées sont dans `results.json`.
La documentation officielle décrit également le timeout de lecture comme une
limite d'attente de chaque fragment de données, et distingue connexion, lecture,
écriture et acquisition d'une connexion dans le pool.
[Documentation HTTPX sur les timeouts](https://www.python-httpx.org/advanced/timeouts/).

## Reproduction locale contrôlée

Le serveur n'écoute que sur IPv4 loopback. Pendant chaque appel, la connexion
réseau est restreinte au port exact de ce serveur. Le client utilise une valeur
d'authentification explicitement factice ; aucun secret local n'est chargé et
aucun en-tête n'est enregistré. La télémétrie LiteLLM est désactivée pour les
appels ; la table locale installée de coûts est utilisée.

| Cas | Timeout observé dans HTTPX | Résultat | Durée monotone | Requêtes HTTP |
|---|---|---|---:|---:|
| Réponse locale immédiate, défaut wrapper 300 | connexion/lecture/écriture/pool : 300 | réponse complète | 0.011795 s | 1 |
| 12 fragments d'espaces JSON à 0.1 s d'intervalle, puis JSON complet | les quatre opérations : 0.5 | réponse complète | **1.203996 s** | 1 |
| En-têtes reçus, puis corps inactif pendant 1 s | les quatre opérations : 0.5 | erreur de lecture | **0.515262 s** | 1 |

Les espaces sont légaux avant un document JSON. Le deuxième cas ne demande
aucun streaming applicatif et reproduit donc directement le point pertinent :
une réponse non streamée peut être reçue progressivement et dépasser la valeur
de timeout en durée totale. Le troisième cas démontre que le timeout est actif,
avec la chaîne typée `TransportRetryError → Timeout → OpenRouterException →
Timeout → ReadTimeout → ReadTimeout → TimeoutError`. Aucune nouvelle requête
HTTP n'est créée par les couches testées après cette erreur.

Le corps local reçu confirme aussi le transfert du modèle, température 0.6,
top-p 1.0, limite 32000 et champ natif `reasoning.effort=low`. Le timeout reste
une configuration client : il ne devient pas une échéance serveur dans le JSON
envoyé. Les transformations natives ajoutent `max_retries=0` au corps, mais le
comptage serveur établit directement l'absence de retry HTTP dans ces cas.

## Formulation à utiliser pour P1

« Timeout client de 300 secondes par opération réseau, incluant l'inactivité
en lecture ; aucune échéance absolue de 300 secondes pour la requête complète.
La génération reste séquentielle. Les retries de transport sont comptés et
bornés séparément ; une réponse complète invalide n'est pas remplacée. »

Le produit nombre d'appels × 300 secondes n'est donc pas une borne supérieure
de durée. Les projections doivent utiliser les latences mesurées, les tokens
effectifs et les pauses identifiées par les horloges du runner. Un éventuel
watchdog absolu appartiendrait à un protocole ultérieur : abandonner l'attente
locale ne prouverait pas l'annulation de la requête distante. Aucun watchdog
n'est ajouté et les règles de l'expérience en cours restent intactes.

## Vérifications

```text
/tmp/phase0-venv/bin/python -m pytest -q \
  artifacts/optimizer_discovery/investigation16/runtime/test_timeout_probe.py
3 passed, 5 warnings in 12.11s

/tmp/phase0-venv/bin/python -m \
  artifacts.optimizer_discovery.investigation16.runtime.timeout_probe
3 cas prévus préservés dans results.json ; zéro appel réel de modèle
```

Les avertissements observés viennent du transport HTTPX installé (`data=` pour
du texte) et de la sérialisation Pydantic du schéma de réponse synthétique.
Ils ne sont pas masqués. Black/Ruff et `git diff --check` sont vérifiés. Les
tests peuvent être relancés ; le script de preuve refuse de remplacer son
`results.json` existant.
