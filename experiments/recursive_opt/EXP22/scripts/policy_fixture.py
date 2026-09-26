"""Compare CP-A and CP-B using a real optimizer and a fixed HTTP response."""

import argparse
import asyncio
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--variant', choices=('CP-A', 'CP-B'), required=True)
ARGS = parser.parse_args()
sys.path[:0] = [str(ROOT/'worktrees'/('trace' if ARGS.variant == 'CP-A' else 'trace_cp_b')), str(ROOT)]
import httpx
import httpx2
from scripts.preflight import write_json
from skydiscover.optimize.search.default_discovery_controller import (
    DiscoveryControllerInput,
)
from skydiscover.optimize.search.registry import create_database, get_program
from src.control_plane import register, specification
from src.evaluation import SKY, TASKS
from src.kernel import (
    POLICY,
    PolicyModule,
    TraceMetaCoEvolutionController,
    configuration,
    digest,
)
from src.transport import MODEL, TraceOpenRouter

from opto.features.recursive_opt import spec as S


def run() -> dict[str, Any]:
    """Run the same measured-window, validation and hot-swap fixture in isolation."""
    os.environ['OPENROUTER_API_KEY'] = 'fixture-only-no-network'
    with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temporary:
        directory = Path(temporary)
        register()
        raw = specification('prism', 'TRACE-RECURSIVE', 100, directory, ARGS.variant)
        plan = S.compile_plan(raw)
        profile = plan.spec['levels'][0]['llm_roles']['optimizer']
        module = PolicyModule(POLICY.read_text())
        response_source = POLICY.read_text() + '\n# EXP22 deterministic policy fixture.\n'
        requests = []

        def respond(client: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
            """Return the exact fixture at the final HTTP boundary, never using network."""
            body = json.loads(request.content)
            requests.append(body)
            content = f'<reasoning>Fixture</reasoning><variable><name>{module.policy_source.name}</name><value>{response_source}</value></variable>'
            library = httpx2 if isinstance(request, httpx2.Request) else httpx
            return library.Response(200, request=request, json={'id': 'fixture', 'object': 'chat.completion', 'created': 0, 'model': MODEL, 'provider': 'DeepInfra', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': content}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}})

        with contextlib.ExitStack() as stack:
            for library in (httpx, httpx2):
                stack.enter_context(patch.object(library.Client, 'send', respond))
            client = S._make_guarded_role_client(profile, 'optimizer', TraceOpenRouter if ARGS.variant == 'CP-A' else None, {}, S._BudgetGuard({}))
            config = configuration('prism', directory)
            database = create_database('evox', config.search.database)
            controller = TraceMetaCoEvolutionController(DiscoveryControllerInput(config, str(SKY/TASKS['prism']/'evaluator/evaluator.py'), database, output_dir=temporary), module, client)
            initial = get_program(config, 'def fixture(): return 1\n', 'initial-fixture', {'combined_score': 21.0}, 0)
            database.add(initial, iteration=0)
            database.initial_program_id = initial.id
            database.initial_program_score = 21.0
            observation = {'active_policy_hash': digest(POLICY.read_text()), 'window_metrics': {'combined_score': 0.125, 'search_window_start_score': 21.0, 'search_window_end_score': 22.0}, 'history': [21.0, 22.0]}
            controller.window_observation = observation
            candidate = asyncio.run(controller._trace_proposal(1, 1, False))
            activated = controller._switch_to_new_search_algorithm(candidate)
            result = {'variant': ARGS.variant, 'source_path': S.__file__, 'initial_policy_hash': digest(POLICY.read_text()), 'initial_population': {'id': initial.id, 'solution': initial.solution, 'metrics': initial.metrics}, 'observation': observation, 'requests': requests, 'validation': candidate.child_program_dict['metrics'], 'activated': activated, 'active_policy_hash': digest(controller._active_search_algorithm_code), 'population_after': {'id': controller.database.get(initial.id).id, 'solution': controller.database.get(initial.id).solution, 'metrics': controller.database.get(initial.id).metrics}}
            controller.close()
            controller.search_controller.close()
    return result


if __name__ == '__main__':
    write_json(ROOT/f'artifacts/policy_fixture_{ARGS.variant}.json', run())
