"""
recursive_opt.memory  —  MemoryLite + thin retrieval  (capability C.1)
======================================================================

The PDF's converged "minimum missing viable base" is **EpisodeTrace +
TieredMemoryLite + thin retrieval**. Markdown-only memory is explicitly
rejected as too weak for promotion / rollback / cross-family comparison.

Tiers (M0->M3):
    M0  raw run scratch            (ephemeral, in-episode)
    M1  EpisodeTrace store         (typed record of one optimization episode)
    M2  Artifact / Experiment store (versioned configs+artifacts, diffs, scores)
    M3  Family-prior library       (promoted, transferable defaults per family)

This is deliberately tiny (dataclasses + JSON files). It is enough to run
O0/O1 recursive experiments *robustly* without committing to full O2/O3 infra.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class EpisodeTrace:  # M1
    episode_id: str
    level: str  # O0 | O1 | O2
    family: str  # llm4ad:online_bin_packing | hf:GSM8K | ...
    cfg: Dict[str, Any]
    score: float
    feedback: str
    metrics: Dict[str, float] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


@dataclass
class FamilyPrior:  # M3
    family: str
    best_cfg: Dict[str, Any]
    best_score: float
    support: int  # how many episodes back this prior
    notes: str = ""


class MemoryLite:
    """Typed, inspectable, ablatable tiered memory with a thin retrieval API."""

    def __init__(self, root: str = "./trace_memory"):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self._episodes: List[EpisodeTrace] = self._load("episodes.jsonl", EpisodeTrace)
        self._priors: Dict[str, FamilyPrior] = {
            p.family: p for p in self._load("priors.jsonl", FamilyPrior)
        }

    # ---- persistence ----------------------------------------------------- #
    def _path(self, name):
        return os.path.join(self.root, name)

    def _load(self, name, cls):
        p = self._path(name)
        if not os.path.exists(p):
            return []
        out = []
        for line in open(p):
            line = line.strip()
            if line:
                out.append(cls(**json.loads(line)))
        return out

    def _append(self, name, obj):
        with open(self._path(name), "a") as f:
            f.write(json.dumps(asdict(obj)) + "\n")

    # ---- M1: record an episode (called by MetaLevel._run_inner) ---------- #
    def record(
        self,
        level: str,
        cfg: Dict[str, Any],
        family: str,
        score: float,
        feedback: str,
        metrics: Optional[Dict] = None,
    ):
        ep = EpisodeTrace(
            episode_id=f"{family}-{len(self._episodes)}-{int(time.time())}",
            level=level,
            family=family,
            cfg=cfg,
            score=float(score),
            feedback=feedback,
            metrics=metrics or {},
        )
        self._episodes.append(ep)
        self._append("episodes.jsonl", ep)
        self._maybe_promote(family)  # M1 -> M3 promotion
        return ep

    # ---- M3: promotion (PromotionEngine, with a support gate) ------------ #
    def _maybe_promote(self, family: str, min_support: int = 3):
        eps = [e for e in self._episodes if e.family == family]
        if len(eps) < min_support:
            return
        best = max(eps, key=lambda e: e.score)
        prior = FamilyPrior(
            family=family,
            best_cfg=best.cfg,
            best_score=best.score,
            support=len(eps),
            notes=f"median={statistics.median(e.score for e in eps):.3f}",
        )
        self._priors[family] = prior
        self._append("priors.jsonl", prior)

    # ---- thin retrieval API ---------------------------------------------- #
    def apply_priors(self, cfg, family: str):
        """Warm-start a config from the family prior (active knowledge building)."""
        prior = self._priors.get(family)
        if prior is None:
            return cfg
        for k, v in prior.best_cfg.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    def similar_failures(
        self, family: Optional[str] = None, k: int = 3
    ) -> List[EpisodeTrace]:
        # ``family=None`` or ``"*"`` searches across *all* families (global).
        # This is what the default agentic ``trace_search`` tool relies on; the
        # previous exact-match-only behaviour made ``family="*"`` always empty.
        pool = (
            self._episodes
            if family in (None, "*")
            else [e for e in self._episodes if e.family == family]
        )
        return sorted(pool, key=lambda e: (e.score, e.ts))[:k]

    def family_prior(self, family: str) -> Optional[FamilyPrior]:
        return self._priors.get(family)

    def summary(self) -> Dict[str, Any]:
        return {
            "episodes": len(self._episodes),
            "families": sorted({e.family for e in self._episodes}),
            "priors": {f: p.best_score for f, p in self._priors.items()},
        }
