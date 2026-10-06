"""Stock EvoX plumbing shared by EXP26's stock scripts. Reuses EXP25's audited harness (imported, not copied),
which also applies its one-hour transport-retry patch on import."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location('exp25_run_evox_stock', Path(__file__).resolve().parents[2] / 'EXP25' / 'scripts' / 'run_evox_stock.py')
E25S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E25S)
W, K, T = E25S.W, E25S.K, E25S.T


def labels_of(controller) -> dict:
    """The controller's labels, flagging stock's silent fallback (empty, or its default templates)."""
    from skydiscover.optimize.search.evox.utils.template import DEFAULT_DIVERGE_TEMPLATE, DEFAULT_REFINE_TEMPLATE
    diverge, refine = controller._diverge_label or '', controller._refine_label or ''
    fallback = not diverge or not refine or (diverge == DEFAULT_DIVERGE_TEMPLATE and refine == DEFAULT_REFINE_TEMPLATE)
    return {'diverge': diverge, 'refine': refine, 'fallback': fallback}


def controller_for(seed: int, output: Path):
    """The controller EXP25's run() builds, without running discovery."""
    config = K.configuration('signal_processing', output, fixed=False)
    config.search.database.random_seed = seed
    database = W.create_database('evox', config.search.database)
    benchmark = W.SKY / W.TASKS['signal_processing']
    return K.AuditController(K.DiscoveryControllerInput(config, str(benchmark / 'evaluator/evaluator.py'), database, output_dir=str(output)))
