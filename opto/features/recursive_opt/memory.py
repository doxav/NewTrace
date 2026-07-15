"""
recursive_opt.memory  —  MemoryLite + thin retrieval  (capability C.1)
======================================================================

Minimal memory implementation: **EpisodeTrace + TieredMemoryLite + thin retrieval**.
Markdown-only memory is explicitly rejected as too weak for promotion / rollback / cross-family comparison.

Tiers (M0->M3) — implemented:
    M0  raw run scratch            (ephemeral, in-episode; not persisted)
    M1  EpisodeTrace store         (typed record of one optimization episode)
    M2  Artifact/Experiment store  (versioned configs/code/capabilities, lineage, scores)
    M3  Family-prior library       (promoted, transferable defaults per family)

This is deliberately tiny (dataclasses + JSON files), enough for
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
    metrics: Dict[str, Any] = field(default_factory=dict)
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
    metrics: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


@dataclass
class ProgressEvent:
    """Compact, append-only progress event for per-level optimization tracing."""

    event_id: str
    run_id: str
    level_id: str
    level_index: int
    event: str
    level_step: Optional[int] = None
    global_step: Optional[int] = None
    artifact_id: Optional[str] = None
    problem_score: Optional[float] = None
    objective_score: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    task_ids: List[str] = field(default_factory=list)
    budget: Dict[str, Any] = field(default_factory=dict)
    selected_by: str = "objective"
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

    def __init__(
        self,
        root: str = "./trace_memory",
        *,
        promotion_min_support: int = 3,
        promote_priors: bool = True,
        promotion_min_score: Optional[float] = None,
    ) -> None:
        if promotion_min_support <= 0:
            raise ValueError("promotion_min_support must be positive")
        self.root = root
        self._promotion_min_support = promotion_min_support
        self._promote_priors = promote_priors
        self._promotion_min_score = (
            float(promotion_min_score) if promotion_min_score is not None else None
        )
        os.makedirs(root, exist_ok=True)
        self._episodes: List[EpisodeTrace] = self._load("episodes.jsonl", EpisodeTrace)
        self._priors: Dict[str, FamilyPrior] = {
            p.family: p for p in self._load("priors.jsonl", FamilyPrior)
        }
        self._artifacts: List[ArtifactRecord] = self._load("artifacts.jsonl", ArtifactRecord)
        self._progress: List[ProgressEvent] = self._load("events.jsonl", ProgressEvent)

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

    def _refresh(self) -> None:
        """Reload persisted memory written by traced/copy-isolated module calls."""
        self._episodes = self._load("episodes.jsonl", EpisodeTrace)
        self._priors = {p.family: p for p in self._load("priors.jsonl", FamilyPrior)}
        self._artifacts = self._load("artifacts.jsonl", ArtifactRecord)
        self._progress = self._load("events.jsonl", ProgressEvent)

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
        self._artifacts = self._load("artifacts.jsonl", ArtifactRecord)
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

    # ---- progress ledger --------------------------------------------------- #
    def record_progress(
        self,
        *,
        run_id: str,
        level_id: str,
        level_index: int,
        event: str,
        level_step: Optional[int] = None,
        global_step: Optional[int] = None,
        artifact_id: Optional[str] = None,
        problem_score: Optional[float] = None,
        objective_score: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
        task_ids: Optional[List[str]] = None,
        budget: Optional[Dict[str, Any]] = None,
        selected_by: str = "objective",
    ) -> ProgressEvent:
        """Append one optimization progress event to ``events.jsonl``."""
        if not run_id:
            raise ValueError("run_id must be non-empty")
        if not level_id:
            raise ValueError("level_id must be non-empty")
        if level_index < 0:
            raise ValueError("level_index must be non-negative")
        if not event:
            raise ValueError("event must be non-empty")
        rec = ProgressEvent(
            event_id=f"{run_id}:{level_id}:{event}:{len(self._progress)}",
            run_id=str(run_id),
            level_id=str(level_id),
            level_index=int(level_index),
            event=str(event),
            level_step=level_step,
            global_step=global_step,
            artifact_id=artifact_id,
            problem_score=problem_score,
            objective_score=objective_score,
            metrics=metrics or {},
            task_ids=[str(task) for task in (task_ids or [])],
            budget=budget or {},
            selected_by=str(selected_by),
        )
        self._progress.append(rec)
        self._append("events.jsonl", rec)
        return rec

    def progress_events(
        self,
        *,
        level_id: Optional[str] = None,
        event: Optional[str] = None,
    ) -> List[ProgressEvent]:
        """Return persisted progress events, optionally filtered by level/event."""
        self._progress = self._load("events.jsonl", ProgressEvent)
        return [
            rec
            for rec in self._progress
            if (level_id is None or rec.level_id == level_id)
            and (event is None or rec.event == event)
        ]

    def write_run_summary(self, summary: Dict[str, Any]) -> None:
        """Persist the compact run summary used by notebooks and reports."""
        if not isinstance(summary, dict):
            raise TypeError("summary must be a dictionary")
        with open(self._path("summary.json"), "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")

    def load_run_summary(self) -> Optional[Dict[str, Any]]:
        """Return ``summary.json`` when present, otherwise ``None``."""
        path = self._path("summary.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("summary.json must contain a JSON object")
        return data

    # ---- M3: promotion (PromotionEngine, with a support gate) ------------ #
    def _maybe_promote(self, family: str):
        if not self._promote_priors:
            return
        self.reconsolidate_family(family)

    def reconsolidate_family(self, family: str) -> Optional[FamilyPrior]:
        """Re-derive the family prior from ALL episodes (support + score gated).

        Public so campaigns can re-consolidate after pruning, after new evidence,
        or after changing promotion gates — promotion is not a one-shot
        side effect of record(). Returns the promoted prior, or None when the
        evidence does not clear the gates (insufficient support / flat scores).
        """
        eps = [e for e in self._load("episodes.jsonl", EpisodeTrace) if e.family == family]
        if len(eps) < self._promotion_min_support:
            return None
        best = max(eps, key=lambda e: e.score)
        if self._promotion_min_score is not None and best.score < self._promotion_min_score:
            return None  # score gate: never promote priors learned from flat/failed runs
        prior = FamilyPrior(
            family=family,
            best_cfg=best.cfg,
            best_score=best.score,
            support=len(eps),
            notes=f"median={statistics.median(e.score for e in eps):.3f}; episodes={len(eps)}",
        )
        self._priors[family] = prior
        self._append("priors.jsonl", prior)
        return prior

    # ---- thin retrieval API ---------------------------------------------- #
    def retrieve(self, family: Optional[str] = None, *, level: Optional[str] = None,
                 kind: Optional[str] = None, min_score: Optional[float] = None,
                 topk: int = 5, sort: str = "best") -> Dict[str, Any]:
        """Filtered retrieval over M1 episodes + M2 artifacts (+ the M3 prior).

        ``sort``: "best" (score desc) or "recent" (timestamp desc). ``family``
        None/"*" means all families (then ``prior`` is None — priors are
        family-scoped by definition).
        """
        self._refresh()
        fam = None if family in (None, "*") else family
        def _keep(x, with_kind: bool) -> bool:
            return ((fam is None or x.family == fam)
                    and (level is None or x.level == level)
                    and (not with_kind or kind is None or x.kind == kind)
                    and (min_score is None or x.score >= min_score))
        key = (lambda x: x.score) if sort == "best" else (lambda x: x.ts)
        eps = sorted((e for e in self._episodes if _keep(e, False)), key=key, reverse=True)
        arts = sorted((a for a in self._artifacts if _keep(a, True)), key=key, reverse=True)
        return {"episodes": eps[:topk], "artifacts": arts[:topk],
                "prior": self._priors.get(fam) if fam else None}

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
        self._priors = {p.family: p for p in self._load("priors.jsonl", FamilyPrior)}
        return self._priors.get(family)

    def summary(self) -> Dict[str, Any]:
        self._refresh()
        return {
            "episodes": len(self._episodes),
            "artifacts": len(self._artifacts),
            "progress_events": len(self._progress),
            "families": sorted({e.family for e in self._episodes}),
            "priors": {f: p.best_score for f, p in self._priors.items()},
        }
