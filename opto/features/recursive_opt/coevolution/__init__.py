"""Online co-evolution for recursive_opt: a solution loop (O0) and a selection-policy loop (O1)
over one shared population, with triggers, deferred window evaluation, hot swapping with
rollback, a population-conditioned operator (context, diffs, retries, variation labels)
and a ``coevolution`` control-plane engine. ``evox_preset()`` reproduces SkyDiscover EvoX.
"""

from .engine import CoevolutionConfig, CoevolutionEngine, evox_preset
from .guides import GuidedEvaluator, format_case_diagnostics
from .projections import CompileCheck, FallbackWrapper, ProjectionError, make_projection, register_projection
from .feedback import META_SYSTEM, POLICY_CONTRACT, FeedbackComposer, population_state
from .operator import DEFAULT_LABELS, OperatorResult, PopulationOperator, apply_search_replace, evaluation_failed, extract_diffs, generate_labels, parse_full_rewrite
from .policy import UNIFORM_POLICY_SOURCE, PolicyContractError, PolicyRuntimeError, PolicySlot, Selection, check_selection, validate_policy_source
from .proposers import LLMRewriteProposer, ProposalResult, TraceProposer
from .scheduling import (ArchiveEntry, DeferredEvaluation, GainScorer, LogWindowScorer, NeverTrigger, PairedScorer, PeriodicTrigger,
                         StagnationTrigger, StrategyArchive, interleave, make_trigger, resolve_patience)
from .state import Candidate, Population, filter_statistics

__all__ = [
    'CompileCheck', 'FallbackWrapper', 'GuidedEvaluator', 'ProjectionError', 'format_case_diagnostics', 'make_projection', 'register_projection',
    'ArchiveEntry', 'Candidate', 'CoevolutionConfig', 'CoevolutionEngine', 'DEFAULT_LABELS', 'DeferredEvaluation', 'FeedbackComposer', 'GainScorer',
    'LLMRewriteProposer', 'LogWindowScorer', 'META_SYSTEM', 'NeverTrigger', 'OperatorResult', 'POLICY_CONTRACT', 'PairedScorer', 'PeriodicTrigger',
    'PolicyContractError', 'PolicyRuntimeError', 'PolicySlot', 'Population', 'PopulationOperator', 'ProposalResult', 'Selection', 'StagnationTrigger',
    'StrategyArchive', 'TraceProposer', 'UNIFORM_POLICY_SOURCE', 'apply_search_replace', 'check_selection', 'evaluation_failed', 'evox_preset',
    'extract_diffs', 'filter_statistics', 'generate_labels', 'interleave', 'make_trigger', 'parse_full_rewrite', 'population_state', 'resolve_patience',
    'validate_policy_source',
]
