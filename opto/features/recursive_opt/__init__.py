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
    ComponentSpec,
    CodeArtifactLevel,
    RecursiveGuide,
    best_config_from,
)
from .memory import MemoryLite, EpisodeTrace, FamilyPrior
from .capabilities import (
    AgenticOptimizer,
    default_optimizer_tools,
    TinkerEnvAdapter,
    HITLGate,
    auto_allow,
)
from . import traces, tracebench

__all__ = [
    "LevelConfig",
    "ArtifactLevel",
    "MetaLevel",
    "ComponentSpec",
    "CodeArtifactLevel",
    "RecursiveGuide",
    "best_config_from",
    "MemoryLite",
    "EpisodeTrace",
    "FamilyPrior",
    "AgenticOptimizer",
    "default_optimizer_tools",
    "TinkerEnvAdapter",
    "HITLGate",
    "auto_allow",
    "traces",
    "tracebench",
]
