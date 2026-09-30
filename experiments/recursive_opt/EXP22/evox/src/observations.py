"""Losslessly share repeated population snapshots in Trace feedback."""

import hashlib
import json
from collections import Counter
from typing import Any

from skydiscover.optimize.search.evox.utils.coevolve_logging import (
    make_json_serializable,
)


def encode_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Encode all stock-normalized evidence using deterministic shared JSON values.

    References replace only repeated values of at least 256 serialized characters.
    Sources, scores, metadata versions, and ordered memberships remain recoverable.
    The stock normalizer converts live Program objects using their to_dict contract;
    existing strings remain opaque, including previously recorded program reprs.
    """
    if not isinstance(observation, dict) or not isinstance(observation.get('active_policy_hash'), str):
        raise TypeError('Observation requires a dictionary with a string active policy hash')
    normalized = make_json_serializable(observation)
    counts: Counter[str] = Counter()
    values: dict[str, Any] = {}

    def identity(value: Any) -> tuple[str, int]:
        """Hash canonical finite JSON independently of dictionary insertion order."""
        serialized = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
        return hashlib.sha256(serialized.encode()).hexdigest(), len(serialized)

    def collect(value: Any) -> None:
        """Count identical large subtrees without changing their contents."""
        key, size = identity(value)
        if isinstance(value, (dict, list, str)) and size >= 256:
            counts[key] += 1
            values[key] = value
        children = value.values() if isinstance(value, dict) else value if isinstance(value, list) else ()
        for child in children:
            collect(child)

    def encode(value: Any, definition: bool = False) -> Any:
        """Share repeated values and escape literal dictionaries resembling markers."""
        key, _ = identity(value)
        if not definition and counts[key] > 1:
            return {'$ref': key}
        if isinstance(value, dict):
            encoded = {name: encode(child) for name, child in sorted(value.items())}
            return {'$literal': encoded} if set(value) in ({'$ref'}, {'$literal'}) else encoded
        if isinstance(value, list):
            return [encode(child) for child in value]
        return value

    collect(normalized)
    return {
        'encoding': 'exp22-lossless-json-v1',
        'active_policy_hash': normalized['active_policy_hash'],
        'instructions': 'root contains the complete observation. Replace each {"$ref": hash} with values[hash], recursively. {"$literal": object} escapes a literal marker-shaped dictionary; decode its values without interpreting that dictionary as a marker. Nothing is omitted.',
        'root': encode(normalized),
        'values': {key: encode(values[key], definition=True) for key in sorted(values) if counts[key] > 1},
    }
