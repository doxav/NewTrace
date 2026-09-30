"""Deterministic toy evaluator shared by the stock and v2 runs."""


def score_source(source):
    namespace = {}
    try:
        exec(source, namespace)
        score = float(namespace['SCORE'])
    except Exception as error:  # noqa: BLE001
        return {'validity': 0, 'combined_score': 0.0, 'error': f'{type(error).__name__}: {error}'}
    return {'combined_score': score}


def evaluate(program_path):
    with open(program_path) as stream:
        return score_source(stream.read())
