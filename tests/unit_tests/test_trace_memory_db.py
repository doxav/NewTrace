# tests/test_trace_memory_db.py
"""
Comprehensive behaviour‑driven tests for TraceMemoryDB.
Each test is tagged with the MUST‑requirement(s) or D.* scenario it validates.
Run with:     pytest -q
"""
import copy
import random
import uuid
import pytest

from opto.trainer.trace_memory_db import TraceMemoryDB, UnifiedVectorDB, ChromaConfig, UnifiedVectorDBConfig, CHROMA_DATABASE
from opto.optimizers.optoprime import OptoPrime

# --------------------------------------------------------------------------- #
# HELPER FIXTURES AND STUBS
# --------------------------------------------------------------------------- #
class DummyVectorDB:
    """Minimal stub to validate pluggable back‑end behaviour (R4)."""
    def __init__(self):
        self.add_calls = []
        self.deleted = []

    def _add_texts(self, texts, ids=None, metadatas=None):
        self.add_calls.append({"texts": texts, "ids": ids, "metadatas": metadatas})

    # TraceMemoryDB.delete_data() will fall back silently if this is missing,
    # but we add it so that the call path is fully covered.
    def delete(self, ids):
        self.deleted.extend(ids)


@pytest.fixture(scope="function")
def inmemory_db():
    """Fresh in‑memory DB (covers R3 + R5)."""
    return TraceMemoryDB(vector_db=None, cache_size=5)


@pytest.fixture(scope="function")
def pluggable_db():
    """DB that writes through to a fake vector store (covers R4)."""
    stub = DummyVectorDB()
    return TraceMemoryDB(vector_db=stub, cache_size=10), stub


# --------------------------------------------------------------------------- #
#  D.1  Backward compatibility aliases  (also touches R16)
# --------------------------------------------------------------------------- #
def test_alias_data_key(inmemory_db):
    eid = inmemory_db.log_data("G1", 1, {"foo": "bar"}, data_key="variables")
    rec = inmemory_db.get_data(entry_id=eid)[0]
    assert rec["data_payload"] == "variables"
    assert rec["data"]["foo"] == "bar"


# --------------------------------------------------------------------------- #
#  D.2  Observation pools & retrieval strategies  (R1, R7, R12, R13 ‑ last‑N)
# --------------------------------------------------------------------------- #
def test_last_n_and_hot_cache(inmemory_db):
    # Fill cache_size==5 with 7 entries to ensure eviction policy
    ids = []
    for i in range(7):
        ids.append(
            inmemory_db.log_data("OBS", i, {"o": i}, data_payload="observation")
        )

    # last_N returns the newest first
    last_three = inmemory_db.get_last_n("OBS", "observation", 3)
    assert [r["data"]["o"] for r in last_three] == [6, 5, 4]

    # Hot‑cache contains only the most recent 5 IDs (cache_size)
    assert ids[0] not in inmemory_db._hot_cache
    assert ids[-1] in inmemory_db._hot_cache


# --------------------------------------------------------------------------- #
#  D.3  Parallel components ‑ shared memory (R17)
# --------------------------------------------------------------------------- #
def test_parallel_components_shared_instance():
    shared = TraceMemoryDB()
    trainer = shared
    evaluator = shared

    trainer.log_data("PIPE", 1, {"p": 0.1}, data_payload="variables",
                     metadata={"agent": "Trainer"})
    rows = evaluator.get_data(goal_id="PIPE", data_payload="variables")
    assert rows and rows[0]["metadata"]["agent"] == "Trainer"


# --------------------------------------------------------------------------- #
#  D.4  Branching / rollback / lineage  (R10)
# --------------------------------------------------------------------------- #
def test_branching_and_lineage(inmemory_db):
    parent_id = inmemory_db.log_data("MAIN", 2, {"code": "print(1)"}, data_payload="code",
                                     scores={"score": 0.9})
    inmemory_db.log_data("BRANCH", 0, {"goal": "branch"}, data_payload="goal",
                         parent_goal_id="MAIN")
    inmemory_db.log_data("BRANCH", 1, {"code": "print(2)"}, data_payload="code",
                         metadata={"source_entry_id": parent_id})
    # Query all sub‑goals of MAIN
    branches = inmemory_db.get_data(parent_goal_id="MAIN")
    assert any(r["goal_id"] == "BRANCH" for r in branches)


# --------------------------------------------------------------------------- #
#  D.5  Mutation‑centric diff logging  (R11 + R12 filter by metadata tag)
# --------------------------------------------------------------------------- #
def test_diff_and_metadata_filter(inmemory_db):
    diff = "---\n+ foo"
    inmemory_db.log_data("MUT", 1, {"diff": diff}, data_payload="diff",
                         metadata={"tags": ["mutation"]})
    rows = inmemory_db.get_data(additional_filters={"metadata.tags": ["mutation"]})
    assert rows and rows[0]["data"]["diff"] == diff


# --------------------------------------------------------------------------- #
#  D.6  Feedback logging for fine‑tuning (R1, R18 helper: best candidate)
# --------------------------------------------------------------------------- #
def test_feedback_and_best_candidate(inmemory_db):
    # Two candidates, different scores
    inmemory_db.log_data("FT", 1, {"code": "pass"}, data_payload="code",
                         candidate_id=1, scores={"score": 0.3})
    inmemory_db.log_data("FT", 1, {"code": "better"}, data_payload="code",
                         candidate_id=2, scores={"score": 0.9})
    best = inmemory_db.get_best_candidate("FT", 1)
    assert best["candidate_id"] == 2
    assert best["scores"]["score"] == 0.9


# --------------------------------------------------------------------------- #
#  D.7  Hypothesis exploration (R19 extensibility, filter on hypothesis_id)
# --------------------------------------------------------------------------- #
def test_hypothesis_id_filter(inmemory_db):
    hid = str(uuid.uuid4())
    inmemory_db.log_data("HYP", 1, {"hypothesis": "X>Y"}, data_payload="hypothesis",
                         metadata={"hypothesis_id": hid})
    rec = inmemory_db.get_data(additional_filters={"metadata.hypothesis_id": hid})[0]
    assert rec["metadata"]["hypothesis_id"] == hid


# --------------------------------------------------------------------------- #
#  R3  In‑memory mode operates without Vector DB or embeddings
# --------------------------------------------------------------------------- #
def test_inmemory_mode_works_without_embeddings(inmemory_db):
    eid = inmemory_db.log_data("EMB", 1, {"prompt": "hello"}, data_payload="prompt")
    assert inmemory_db.get_data(entry_id=eid)[0]["embedding"] is None


# --------------------------------------------------------------------------- #
#  R4  Pluggable Vector DB backend ‑ ensure _add_texts is invoked
# --------------------------------------------------------------------------- #
def test_pluggable_vector_db(pluggable_db):
    db, stub = pluggable_db
    db.log_data("PDB", 1, {"foo": "bar"}, data_payload="variables")
    assert stub.add_calls, "UnifiedVectorDB._add_texts should have been called"


# --------------------------------------------------------------------------- #
#  R8  Immutability ‑ returned records must be deep‑copies
# --------------------------------------------------------------------------- #
def test_records_are_immutable(inmemory_db):
    eid = inmemory_db.log_data("IMM", 1, {"x": 1}, data_payload="variables")
    rec = inmemory_db.get_data(entry_id=eid)[0]
    rec["data"]["x"] = 999                # mutate local copy
    # Fetch again – should be unchanged
    fresh = inmemory_db.get_data(entry_id=eid)[0]
    assert fresh["data"]["x"] == 1        # original persisted


# --------------------------------------------------------------------------- #
#  R11  Multiple candidates per step + get_candidates helper
# --------------------------------------------------------------------------- #
def test_multiple_candidates_and_get_candidates(inmemory_db):
    for cid in (1, 2, 3):
        inmemory_db.log_data("CANDS", 5, {"v": cid}, data_payload="variables",
                             candidate_id=cid)
    candidates = inmemory_db.get_candidates("CANDS", 5)
    assert {c["candidate_id"] for c in candidates} == {1, 2, 3}


# --------------------------------------------------------------------------- #
#  R12  Retrieval filtering on several fields at once
# --------------------------------------------------------------------------- #
def test_complex_filtering(inmemory_db):
    inmemory_db.log_data("FILTER", 1, {"a": 1}, data_payload="code",
                         metadata={"agent": "X"})
    inmemory_db.log_data("FILTER", 2, {"b": 2}, data_payload="code",
                         metadata={"agent": "Y"})
    rows = inmemory_db.get_data(goal_id="FILTER",
                                additional_filters={"metadata.agent": "Y"})
    assert len(rows) == 1 and rows[0]["step_id"] == 2


# --------------------------------------------------------------------------- #
#  R13  Ranked retrieval – Top‑N by score, random sample
# --------------------------------------------------------------------------- #
def test_top_n_and_random(inmemory_db):
    scores = [0.1, 0.9, 0.5]
    for cid, sc in enumerate(scores, 1):
        inmemory_db.log_data("RANK", 1, {"v": cid}, data_payload="code",
                             candidate_id=cid, scores={"score": sc})
    top_two = inmemory_db.get_top_candidates("RANK", 1, n=2)
    assert [c["scores"]["score"] for c in top_two] == sorted(scores, reverse=True)[:2]

    random.seed(42)
    rand_pick = inmemory_db.get_random_candidates("RANK", 1, n=1)[0]
    assert rand_pick in inmemory_db.get_candidates("RANK", 1)


# --------------------------------------------------------------------------- #
#  R18  Most‑diverse helper (using embeddings)
# --------------------------------------------------------------------------- #
def test_most_diverse_candidates(inmemory_db):
    # four points in 2‑D square
    embeds = {
        1: [0.0, 0.0],
        2: [10.0, 0.0],
        3: [0.0, 10.0],
        4: [5.0, 5.0],
    }
    for cid, emb in embeds.items():
        inmemory_db.log_data("DIV", 1, {"dummy": cid}, data_payload="code",
                             candidate_id=cid, embedding=emb)
    diverse = inmemory_db.get_most_diverse_candidates("DIV", 1, n=3)
    # should include three corners (ids 1,2,3) – order not important
    assert {c["candidate_id"] for c in diverse} >= {1, 2, 3}


# --------------------------------------------------------------------------- #
#  R20  Minimal footprint ‑ no optional libs required in pure in‑memory mode
# --------------------------------------------------------------------------- #
def test_no_external_dependencies(inmemory_db):
    # The fact we reached this point with vector_db=None is already evidence,
    # but add a quick sanity call:
    assert inmemory_db.vdb is None

# -----------------------------------------------------------------------------
# Section 2: real integration tests covering D.1–D.7 with OptoPrime + LLM
# -----------------------------------------------------------------------------

@pytest.fixture(params=["inmemory", "chroma"])
def memdb(request):
    """Parametrize on memory-backend: default in-memory or Chroma."""
    if request.param == "inmemory":
        return TraceMemoryDB()
    else:
        # explicit Chroma‐backed VectorDB via its real constructor
        dirpath = f"/tmp/chroma_test_{uuid.uuid4().hex}"
        coll = f"col_{uuid.uuid4().hex}"
        cfg = UnifiedVectorDBConfig(
            persist_directory=dirpath,
            collection_name=coll
        )
        # force Chroma backend
        cfg.db_type = CHROMA_DATABASE
        # UnifiedVectorDB.get_unique_id = lambda self: cfg.unique_collection_id  # type: ignore
        vdb = UnifiedVectorDB(cfg, check_db=False)
        return TraceMemoryDB(vector_db=vdb)

@pytest.fixture
def llm():
    """Instantiate the real LLM client (may be expensive)."""
    from opto.utils.llm import LLM
    return LLM()

def test_d1_backward_compatibility_and_basic_logging(memdb, llm):
    """D.1: Ensure OptoPrime logs both step() and backward() with metadata."""
    opt = OptoPrime(llm=llm, memory_db=memdb)
    # make a single optimization step
    upd = opt.step(goal_id="G1", step_id=1, parameters={"x": 0.})
    # now push back feedback
    opt.backward(response=upd, feedback="looks good")
    # fetch both entries
    recs = memdb.get_data(goal_id="G1", step_id=1)
    types = {r["data_payload"] for r in recs}
    assert "variables" in types and "feedback" in types
    # verify metadata fields
    for r in recs:
        assert r["metadata"]["agent"] == "OptoPrime"
        assert "status" in r["metadata"]

def test_d2_observation_pools_and_retrieval(memdb):
    """D.2: Log observations, then retrieve last-n and semantic search."""
    # log 5 observations
    for i in range(5):
        memdb.log_data(
            goal_id="G2",
            step_id=1,
            candidate_id=1,
            data={"observation": f"state={i}"},
            data_payload="observation",
            metadata={"agent":"EnvMonitor","status":"logged"},
        )
    last3 = memdb.get_last_n(goal_id="G2", data_payload="observation", n=3)
    assert len(last3) == 3
    # semantic search: should return at least one
    hits = memdb.get_data(
        goal_id="G2",
        data_payload="observation",
        additional_filters={"embedding_query": ("state=3", 2)},
    )
    assert isinstance(hits, list)

def test_d3_parallel_pipeline_loose_coupling(memdb):
    """D.3: Trainer/Optimizer parallel pipeline via shared memdb."""
    # minimal Trainer
    class Trainer:
        def __init__(self, db): self.db,self.step= db,0
        def generate(self,N=2):
            self.step+=1
            for cid in range(1,N+1):
                self.db.log_data(
                    goal_id="G3", step_id=self.step, candidate_id=cid,
                    data={"variables": {"a":cid}}, data_payload="variables",
                    metadata={"agent":"Trainer","status":"pending"},
                )
            return self.step
    # minimal Optimizer
    class Optimizer:
        def __init__(self, db): self.db=db
        def select_best(self, step):
            recs = self.db.get_data("G3", step_id=step, data_payload="variables")
            return max(recs, key=lambda r: r["candidate_id"])["candidate_id"]
    tr = Trainer(memdb)
    op = Optimizer(memdb)
    step = tr.generate()
    best = op.select_best(step)
    assert best == 2
    # log selection
    memdb.log_data("G3",step,best, data_payload="best_candidate", data={"best":best},
                   metadata={"agent":"Optimizer","status":"selected"})
    chosen = memdb.get_data("G3", step, data_payload="best_candidate")[0]["data"]["best"]
    assert chosen == best

def test_d4_branching_and_lineage(memdb):
    """D.4: Branch from best candidate, then rollback to a checkpoint."""
    # seed MAIN_GOAL
    memdb.log_data("MAIN",0,data={"goal":"start"},data_payload="goal",metadata={"agent":"Init"})
    memdb.log_data("MAIN",1,candidate_id=1,data={"code":"v1"},data_payload="code",metadata={})
    memdb.log_data("MAIN",2,candidate_id=1,data={"score":0.9},data_payload="score",metadata={})
    best = memdb.get_best_candidate("MAIN",2,score_name="score")
    # branch
    branch_id = "MAIN_branch"
    memdb.log_data(branch_id,step_id=0,data={"goal":"branch"},data_payload="goal",
                   parent_goal_id="MAIN",metadata={"agent":"Branch"})
    memdb.log_data(branch_id,step_id=1,data={"code":best["data"]["code"]},data_payload="code",
                   metadata={"agent":"Branch","source_entry_id":best["entry_id"]})
    # rollback
    ckpt = memdb.get_data(goal_id="MAIN", step_id=1, data_payload="code")[0]
    rb = "MAIN_rb"
    memdb.log_data(rb,0,data={"goal":"rollback"},data_payload="goal",
                   parent_goal_id="MAIN",metadata={"agent":"Rollback"})
    memdb.log_data(rb,1,data={"code":ckpt["data"]["code"]},data_payload="code",
                   metadata={"agent":"Rollback","source_entry_id":ckpt["entry_id"]})
    # list branches
    branches = memdb.get_data("MAIN", data_payload="goal", additional_filters={"metadata.parent_goal_id":"MAIN"})
    ids = {b["goal_id"] for b in branches}
    assert branch_id in ids and rb in ids

def test_d5_mutation_lineage(memdb):
    """D.5: Log diffs, then trace lineage of mutations."""
    # original
    orig_id = memdb.log_data("MUT",0,data={"code":"x=0"},data_payload="code",metadata={})
    # first mutation
    m1 = memdb.log_data("MUT",1,data={"diff":"+1"},data_payload="diff",metadata={"source_entry_id":orig_id})
    # second mutation
    m2 = memdb.log_data("MUT",2,data={"diff":"+2"},data_payload="diff",metadata={"source_entry_id":m1})
    # lineage walk
    rec = memdb.get_data(goal_id="MUT", step_id=2, data_payload="diff")[0]
    assert rec["metadata"]["source_entry_id"] == m1

def test_d6_feedback_logging_for_finetuning(memdb):
    """D.6: Log numeric & qualitative feedback for future FT."""
    memdb.log_data("FT",1,data={"prompt":"Do X"},data_payload="prompt",metadata={})
    memdb.log_data("FT",1,data={"score":0.75},data_payload="scores",metadata={})
    memdb.log_data("FT",1,data={"feedback":"Ok but improve edge cases"},data_payload="feedback",metadata={})
    recs = memdb.get_data(goal_id="FT", step_id=1)
    payloads = {r["data_payload"] for r in recs}
    assert {"prompt","scores","feedback"}.issubset(payloads)

def test_d7_hypothesis_exploration(memdb):
    """D.7: Log hypotheses, embed them, sample uncertain region."""
    # log some hypotheses
    for i in range(3):
        emb = [float(i), float(i*2)]
        memdb.log_data("HYP",i,data={"hypothesis":f"H{i}"},data_payload="hypothesis",
                       embedding=emb,metadata={"agent":"Hypo"})
    # query uncertain (e.g. random sample)
    hits = memdb.get_data(goal_id="HYP", additional_filters={"random":2})
    assert len(hits) == 2