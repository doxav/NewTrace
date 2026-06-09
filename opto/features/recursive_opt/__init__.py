"""
opto.features.recursive_opt
===========================
Robust + simple recursive (meta) optimization on Trace / Trace-Bench.

Two trainable surfaces (see levels.py for the full explanation):
  * SELECTION/CONFIG : LevelConfig + MetaLevel        (pick & configure components)
  * CODE/IMPLEMENTATION : ComponentSpec + CodeArtifactLevel  (rewrite component code)

Everything is a ``trace.Module``, so ``opto.trainer.train`` + any
``opto.optimizers`` optimizer drive every recursion level (O0..O3) unchanged.
"""

from .levels import (
    LevelConfig,
    ArtifactLevel,
    MetaLevel,
    FamilyPolicyLevel,
    PriorInductionLevel,
    ComponentSpec,
    CodeArtifactLevel,
    RecursiveGuide,
    best_config_from,
    encode_cfg,
    decode_cfg,
    canonicalize_cfg_text,
    register_config_values,
    validate_config_field,
    validate_level_config,
)
from .memory import MemoryLite, EpisodeTrace, FamilyPrior, ArtifactRecord
from .capabilities import (
    AgenticOptimizer,
    default_optimizer_tools,
    TinkerEnvAdapter,
    HITLGate,
    auto_allow,
)
from .budget import (
    BudgetExceeded,
    RecursiveOptBudget,
    budget_status,
    configure_budget_from_env,
    current_budget,
    reset_budget,
)
from . import traces, tracebench
from .optimize import (
    optimize,
    resolve_trainer,
    current_trainer,
    current_optimizer,
    current_iterations,
    current_num_candidates,
    TRAINER,
    OPTIMIZER,
    ITERATIONS,
    NUM_CANDIDATES,
)

__all__ = [
    "LevelConfig",
    "ArtifactLevel",
    "MetaLevel",
    "FamilyPolicyLevel",
    "PriorInductionLevel",
    "ComponentSpec",
    "CodeArtifactLevel",
    "RecursiveGuide",
    "best_config_from",
    "encode_cfg",
    "decode_cfg",
    "canonicalize_cfg_text",
    "register_config_values",
    "validate_config_field",
    "validate_level_config",
    "MemoryLite",
    "EpisodeTrace",
    "FamilyPrior",
    "ArtifactRecord",
    "AgenticOptimizer",
    "default_optimizer_tools",
    "TinkerEnvAdapter",
    "HITLGate",
    "auto_allow",
    "BudgetExceeded",
    "RecursiveOptBudget",
    "budget_status",
    "configure_budget_from_env",
    "current_budget",
    "reset_budget",
    "traces",
    "tracebench",
    "optimize",
    "resolve_trainer",
    "current_trainer",
    "current_optimizer",
    "current_iterations",
    "current_num_candidates",
    "TRAINER",
    "OPTIMIZER",
    "ITERATIONS",
    "NUM_CANDIDATES",
]
