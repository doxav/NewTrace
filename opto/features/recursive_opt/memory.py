"""
recursive_opt.memory  —  MemoryLite + thin retrieval  (capability C.1)
======================================================================

The PDF's converged "minimum missing viable base" is **EpisodeTrace +
TieredMemoryLite + thin retrieval**. Markdown-only memory is explicitly
rejected as too weak for promotion / rollback / cross-family comparison.

Tiers (M0->M3) — implemented: M1 (episodes), M2 (artifact/experiment lineage),
M3 (family priors). M0 (in-episode scratch) is intentionally ephemeral/in-memory.
    M0  raw run scratch            (ephemeral, in-episode; not persisted)
    M1  EpisodeTrace store         (typed record of one optimization episode)
    M2  Artifact/Experiment store  (versioned configs/code/capabilities, lineage, scores)
    M3  Family-prior library       (promoted, transferable defaults per family)

This is deliberately tiny (dataclasses + JSON files), enough to run O0..O3
recursive experiments *robustly* with artifact lineage and rollback.
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
class ArtifactRecord:  # M2 — artifact / experiment lineage
    artifact_id: str
    parent_id: Optional[str]
    level: str  # O0 | O1 | O2 | O3 | capability | code
    family: str
    kind: str  # config | code | capability | policy | prior
    content: str
    score: float
    iteration: int
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
        self._artifacts: List[ArtifactRecord] = self._load("artifacts.jsonl", ArtifactRecord)

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

    # ---- M2: artifact / experiment lineage store ------------------------ #
    def record_artifact(
        self,
        level: str,
        family: str,
        kind: str,
        content: str,
        score: float,
        parent_id: Optional[str] = None,
        metrics: Optional[Dict] = None,
    ) -> ArtifactRecord:
        """Append a versioned artifact (config/code/capability/policy/prior).

        ``parent_id`` links a revised artifact to the one it was derived from, so
        ``lineage()`` can reconstruct the initial->final chain with scores/diffs.
        """
        iteration = sum(1 for a in self._artifacts if a.family == family and a.kind == kind)
        rec = ArtifactRecord(
            artifact_id=f"{family}:{kind}:{iteration}:{int(time.time()*1000)%100000}",
            parent_id=parent_id,
            level=level,
            family=family,
            kind=kind,
            content=str(content),
            score=float(score),
            iteration=iteration,
            metrics=metrics or {},
        )
        self._artifacts.append(rec)
        self._append("artifacts.jsonl", rec)
        return rec

    def lineage(self, artifact_id: str) -> List[ArtifactRecord]:
        """Return the chain [root, ..., artifact_id] following parent links."""
        by_id = {a.artifact_id: a for a in self._artifacts}
        chain: List[ArtifactRecord] = []
        cur = by_id.get(artifact_id)
        seen = set()
        while cur is not None and cur.artifact_id not in seen:
            chain.append(cur)
            seen.add(cur.artifact_id)
            cur = by_id.get(cur.parent_id) if cur.parent_id else None
        return list(reversed(chain))

    def artifact_history(
        self, family: Optional[str] = None, kind: Optional[str] = None
    ) -> List[ArtifactRecord]:
        out = [
            a
            for a in self._artifacts
            if (family in (None, "*") or a.family == family)
            and (kind in (None, "*") or a.kind == kind)
        ]
        return sorted(out, key=lambda a: (a.family, a.kind, a.iteration))

    def best_artifact(
        self, family: Optional[str] = None, kind: Optional[str] = None
    ) -> Optional[ArtifactRecord]:
        hist = self.artifact_history(family, kind)
        return max(hist, key=lambda a: a.score) if hist else None

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
            "artifacts": len(self._artifacts),
            "families": sorted({e.family for e in self._episodes}),
            "priors": {f: p.best_score for f, p in self._priors.items()},
        }
