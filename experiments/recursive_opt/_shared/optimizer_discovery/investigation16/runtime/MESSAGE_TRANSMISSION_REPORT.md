# Deux messages user consécutifs : contrôle de transmission

**Les deux messages complets arrivent au transport HTTP local.** Le contrat
initial n'est ni supprimé ni tronqué dans le chemin client testé. Le contrôle
emploie la requête F1 réellement enregistrée :
`feedback_experiment/raw/16301/Aanytime_code/slot_00/request.json`.

Le nouveau `message_probe.py` réutilise `timeout_probe.py` sans modifier ce
dernier. Il conserve le vrai constructeur `make_live_llm`, LiteLLM 1.75.0,
l'adaptateur OpenRouter et HTTPX. Seuls l'endpoint et l'authentification sont ceux
du serveur local préexistant : loopback strict, valeur factice, aucune clé réelle,
aucun appel de modèle. La copie exacte des deux messages et des réglages est
fournie au vrai client ; le serveur capture le corps JSON reçu sans les en-têtes.

| Message | Rôle reçu | Caractères reçus | SHA256 identique à l'entrée |
|---|---|---:|---|
| Contrat, seed commun et objectif | user | 2603 | `74ba5cd766fad5be9ee35c046d080f539897ac4bcec8efa6841510321fc920f6` |
| Current source | user | 592 | `2aa5bca365e96c97cd1feccd2fa45e396dafabc9d22b44c6dfeff3a306441583` |

Le JSON reçu contient toujours deux objets distincts dans leur ordre original.
Le modèle exact, `temperature=0.6`, `top_p=1.0`, `max_tokens=32000`, seed16301 et
le champ natif `reasoning={"effort":"low"}` sont conservés. Le transport effectue
une seule requête locale, sans tentative de connexion externe. Les preuves
complètes et les empreintes des deux scripts sont dans `message_results.json`.

Le test de régression ajoute un cas synthétique plus long avec Unicode,
délimiteurs Python et marqueurs terminaux dans chacun des deux messages ; leur
égalité intégrale est vérifiée côté serveur. Un second test refuse une entrée
hors contrat avant tout transport. Il ne s'agit pas d'une nouvelle génération
ni d'une réparation de candidat F1.

## Ce que dit le sérialiseur publié

Dans le code publié pour DeepSeek-V4-Flash-0731, `merge_tool_messages` regroupe les
messages user consécutifs en blocs de texte et `render_message` joint ces blocs.
Pour les deux messages texte simples testés ici, le contenu du contrat et celui
du current source sont tous deux conservés. Le filtre de raisonnement garde
également les messages user. Ce code publié ne fournit donc pas de mécanisme
de suppression du premier de ces messages.
[Source DeepSeek publiée](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/encoding/encoding_dsv4.py).

Cela ne démontre pas que le fournisseur OpenRouter effectivement choisi utilise
ce même template, cette version ou ces options. Ce contrôle n'observe ni le
prompt tokenisé transmis au modèle distant ni son traitement des instructions.
Une dilution d'attention, un comportement du modèle ou une transformation
distante restent des hypothèses distinctes non testées ici. Les sorties
`choose`/`solve` ne peuvent pas être attribuées à une perte du contrat par le
client local sur la base de ces preuves. Elles restent des réponses invalides
conservées dans l'expérience.

## Commandes

```bash
/tmp/phase0-venv/bin/python -m pytest -q \
  artifacts/optimizer_discovery/investigation16/runtime/test_message_probe.py \
  artifacts/optimizer_discovery/investigation16/runtime/test_timeout_probe.py

/tmp/phase0-venv/bin/python -m \
  artifacts.optimizer_discovery.investigation16.runtime.message_probe \
  artifacts/optimizer_discovery/investigation16/feedback_experiment/raw/16301/Aanytime_code/slot_00/request.json
```

Résultat de la vérification finale : **5 passed, 7 warnings in 5.42s**. Cette
commande regroupe les deux nouveaux tests de transmission et les trois tests
existants de timeout. Sa sortie a été retournée par l'outil d'exécution ; aucun
fichier de log autonome n'a été créé.

Les avertissements HTTPX/Pydantic du serveur synthétique sont les mêmes que dans
le contrôle de timeout et ne sont pas masqués. Aucun fichier gelé ni réglage
live n'est modifié. La commande exacte des 36 tests antérieurs, qui n'ont pas
été relancés, est clarifiée dans `VERIFICATION_36_NOTE.md`.
