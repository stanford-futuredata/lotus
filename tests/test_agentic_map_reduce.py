"""Offline tests for agentic map-reduce (no network, no Docker).

The model is faked via a scripted ``Completer`` so the full pipeline — tool-calling
loop, corpus sharding, parallel map, and reduce — is exercised deterministically.
"""

from __future__ import annotations

import pandas as pd
import pytest

from lotus.agentic import Plan, normalize_ops, run_agent
from lotus.agentic.loop import AgentStep, ToolCall
from lotus.agentic.pipeline import _parse_verdict, run_pipeline
from lotus.corpus import Corpus
from lotus.tools import PythonREPLTool, tool


# --------------------------------------------------------------------------- fakes
class ScriptedCompleter:
    """Returns pre-scripted AgentSteps in order, per instance."""

    def __init__(self, steps: list[AgentStep]):
        self._steps = list(steps)
        self.calls = 0
        self.last_tools_enabled: bool | None = None

    def __call__(self, messages, *, tools_enabled: bool = True):
        self.calls += 1
        self.last_tools_enabled = tools_enabled
        if self._steps:
            return self._steps.pop(0)
        return AgentStep(content="(no more scripted steps)")


# --------------------------------------------------------------------------- tools
def test_tool_decorator_schema_and_run():
    @tool(description="Add two integers.")
    def add(a: int, b: int) -> str:
        return str(a + b)

    schema = add.to_openai_schema()
    assert schema["function"]["name"] == "add"
    assert set(schema["function"]["parameters"]["properties"]) == {"a", "b"}
    assert add.run(a=13, b=16) == "29"


def test_local_repl_executes_and_captures_error():
    repl = PythonREPLTool()  # LocalSandbox default — no Docker
    assert repl.run("print(6 * 7)").strip() == "42"
    assert "ValueError" in repl.run("raise ValueError('boom')")


# ---------------------------------------------------------------------- agent loop
def test_run_agent_executes_tool_then_finalizes():
    @tool(description="Add two integers.")
    def add(a: int, b: int) -> str:
        return str(a + b)

    completer = ScriptedCompleter(
        [
            AgentStep(tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})]),
            AgentStep(content="The answer is 5."),
        ]
    )
    res = run_agent(completer, [add], system_prompt="sys", user_content="add 2 and 3", max_steps=5)
    assert res.output == "The answer is 5."
    assert res.truncated is False
    assert res.trace == [{"tool": "add", "arguments": {"a": 2, "b": 3}, "result": "5"}]
    assert res.steps == 2


def test_run_agent_truncates_at_max_steps():
    @tool(description="Loop forever.")
    def noop() -> str:
        return "ok"

    # Always asks for a tool call -> never finalizes on its own.
    step = AgentStep(tool_calls=[ToolCall(id="c", name="noop", arguments={})])
    completer = ScriptedCompleter([step, step, AgentStep(content="forced final")])
    res = run_agent(completer, [noop], system_prompt="s", user_content="u", max_steps=2)
    assert res.truncated is True
    assert res.output == "forced final"
    # The forced-final turn must disable tools so the model must produce text.
    assert completer.last_tools_enabled is False


def test_run_agent_handles_unknown_and_failing_tools():
    @tool(description="Boom.")
    def boom() -> str:
        raise RuntimeError("kaboom")

    completer = ScriptedCompleter(
        [
            AgentStep(tool_calls=[ToolCall(id="1", name="ghost", arguments={})]),
            AgentStep(tool_calls=[ToolCall(id="2", name="boom", arguments={})]),
            AgentStep(content="done"),
        ]
    )
    res = run_agent(completer, [boom], system_prompt="s", user_content="u", max_steps=5)
    assert "unknown tool 'ghost'" in res.trace[0]["result"]
    assert "kaboom" in res.trace[1]["result"]
    assert res.output == "done"


# --------------------------------------------------------------------------- corpus
def test_corpus_loaders_and_sharding():
    assert [len(s) for s in Corpus.from_documents(["a", "b", "c"]).shard(2)] == [2, 1]
    df = pd.DataFrame({"x": [1, 2], "y": ["p", "q"]})
    corpus = Corpus.from_dataframe(df)
    assert len(corpus) == 2
    assert "x: 1" in corpus.units[0].content
    assert Corpus.from_text("abcdefg", chunk_chars=3).shard(1) == [
        [u] for u in Corpus.from_text("abcdefg", chunk_chars=3).units
    ]


# ------------------------------------------------------------------- full pipeline
class StatelessFakeCompleter:
    """Deterministic, thread-safe fake (mirrors the stateless LiteLLMCompleter).

    It reads the last user message: reduce prompts (which contain "PER-SHARD FINDINGS")
    produce a combined answer; map prompts echo their shard's unit id.
    """

    def __call__(self, messages, *, tools_enabled: bool = True):
        last = messages[-1]["content"]
        if "PER-SHARD FINDINGS" in last:
            return AgentStep(content="REDUCED: combined all shards")
        # map prompt: extract the unit marker so each shard yields a distinct finding
        marker = last.split("[unit ", 1)[1].split("]", 1)[0] if "[unit " in last else "?"
        return AgentStep(content=f"finding for {marker}")


def test_map_reduce_pipeline_end_to_end():
    """plan (explicit) -> shard -> parallel map -> reduce, via a stateless fake."""
    corpus = Corpus.from_documents(["alpha", "beta", "gamma"], ids=["A", "B", "C"])
    plan = Plan(
        ops=["map", "reduce"],
        instructions={"map": "summarize", "reduce": "combine"},
        shard_size=1,
        parallelism=3,
    )

    res = run_pipeline(
        corpus,
        task="dummy task",
        ops=["map", "reduce"],
        plan=plan,
        completer_factory=lambda tools: StatelessFakeCompleter(),
        lm=object(),  # unused: completer_factory is injected
    )
    assert len(res.findings) == 3
    assert set(res.findings) == {"finding for A", "finding for B", "finding for C"}
    assert res.output == "REDUCED: combined all shards"
    assert res.plan is plan


def test_pipeline_respects_parallelism_cap():
    corpus = Corpus.from_documents(["a", "b"])
    plan = Plan(
        ops=["map", "reduce"],
        instructions={"map": "m", "reduce": "r"},
        shard_size=1,
        parallelism=100,
    )

    res = run_pipeline(
        corpus,
        task="t",
        ops=["map", "reduce"],
        plan=plan,
        max_parallelism=4,
        completer_factory=lambda tools: StatelessFakeCompleter(),
        lm=object(),
    )
    assert res.plan.parallelism == 4


# --------------------------------------------------------------------- ops model
def test_normalize_ops_valid_invalid_and_defaults():
    assert normalize_ops(None) == ["map", "reduce"]
    assert normalize_ops("map") == ["map"]
    assert normalize_ops(["FILTER", "Map"]) == ["filter", "map"]  # case-insensitive, order kept

    with pytest.raises(ValueError):
        normalize_ops([])  # empty
    with pytest.raises(ValueError):
        normalize_ops(["map", "map"])  # duplicate
    with pytest.raises(ValueError):
        normalize_ops(["reduce", "map"])  # terminal op must be last
    with pytest.raises(ValueError):
        normalize_ops(["frobnicate"])  # unknown op


def test_parse_verdict_keep_drop_and_default():
    assert _parse_verdict("Looks relevant. VERDICT: KEEP") is True
    assert _parse_verdict("not relevant\nVERDICT: drop") is False
    assert _parse_verdict("I would keep this one") is True  # bare token
    assert _parse_verdict("we should drop it") is False
    assert _parse_verdict("no clear verdict here") is True  # ambiguous -> keep (never silently drop)


def test_parse_confidence_and_proxy_keep_score():
    from lotus.agentic.cascade import parse_confidence, proxy_keep_score

    assert parse_confidence("VERDICT: KEEP\nCONFIDENCE: 0.9") == 0.9
    assert parse_confidence("no conf here") == 0.5
    assert proxy_keep_score("VERDICT: KEEP\nCONFIDENCE: 0.8", _parse_verdict) == 0.8
    assert proxy_keep_score("VERDICT: DROP\nCONFIDENCE: 0.8", _parse_verdict) == pytest.approx(0.2)


def test_agentic_filter_cascade_with_fixed_thresholds():
    """High-confidence proxy decisions skip the oracle; mid-band units escalate."""
    from lotus.agentic import AgenticCascadeArgs

    class CascadeAwareFake:
        def __call__(self, messages, *, tools_enabled: bool = True):
            last = messages[-1]["content"]
            marker = last.split("[unit ", 1)[1].split("]", 1)[0] if "[unit " in last else "?"
            # Proxy path asks for CONFIDENCE; oracle path does not.
            if "CONFIDENCE:" in last:
                if marker == "A":
                    return AgentStep(content="VERDICT: KEEP\nCONFIDENCE: 0.95")
                if marker == "B":
                    return AgentStep(content="VERDICT: DROP\nCONFIDENCE: 0.95")
                return AgentStep(content="VERDICT: KEEP\nCONFIDENCE: 0.5")  # mid → escalate
            # Oracle: keep C
            return AgentStep(content=f"Assessed {marker}. VERDICT: KEEP")

    corpus = Corpus.from_documents(["a", "b", "c"], ids=["A", "B", "C"])
    cascade = AgenticCascadeArgs(
        filter_pos_cascade_threshold=0.8,
        filter_neg_cascade_threshold=0.2,
    )
    res = run_pipeline(
        corpus,
        "t",
        ops=["filter"],
        plan=_plan(["filter"]),
        completer_factory=lambda tools: CascadeAwareFake(),
        lm=object(),
        cascade_args=cascade,
    )
    assert [u.id for u in res.corpus.units] == ["A", "C"]
    assert res.cascade_stats is not None
    assert res.cascade_stats.proxy_calls == 3
    assert res.cascade_stats.oracle_calls == 1  # only C escalated
    assert res.cascade_stats.accepted_by_proxy == 2


# ------------------------------------------------------------- composable ops e2e
class OpAwareFake:
    """Op-aware stateless fake: detects the op from the system prompt.

    - reduce -> "REDUCED"; filter -> KEEP/DROP verdict (DROP for ids in ``drop_ids``);
    - map -> "mapped:<unit-id>". Thread-safe (only reads ``drop_ids``).
    """

    def __init__(self, drop_ids=()):
        self.drop_ids = set(drop_ids)

    def __call__(self, messages, *, tools_enabled: bool = True):
        system = messages[0]["content"]
        last = messages[-1]["content"]
        if "reducer" in system:
            return AgentStep(content="REDUCED")
        marker = last.split("[unit ", 1)[1].split("]", 1)[0] if "[unit " in last else "?"
        if "agentic filter" in system:
            verdict = "DROP" if marker in self.drop_ids else "KEEP"
            return AgentStep(content=f"Assessed {marker}. VERDICT: {verdict}")
        return AgentStep(content=f"mapped:{marker}")


def _plan(ops, **kw):
    return Plan(ops=ops, instructions={op: op for op in ops}, shard_size=1, parallelism=3, **kw)


def test_agent_map_only_returns_corpus():
    corpus = Corpus.from_documents(["a", "b", "c"], ids=["A", "B", "C"])
    res = run_pipeline(
        corpus, "t", ops=["map"], plan=_plan(["map"]),
        completer_factory=lambda tools: OpAwareFake(), lm=object(),
    )
    assert res.output is None
    assert res.findings == ["mapped:A", "mapped:B", "mapped:C"]
    assert res.corpus is not None
    assert [u.content for u in res.corpus.units] == ["mapped:A", "mapped:B", "mapped:C"]


def test_agent_filter_only_returns_subset():
    corpus = Corpus.from_documents(["a", "b", "c"], ids=["A", "B", "C"])
    res = run_pipeline(
        corpus, "t", ops=["filter"], plan=_plan(["filter"]),
        completer_factory=lambda tools: OpAwareFake(drop_ids={"A", "C"}), lm=object(),
    )
    assert res.output is None
    assert res.findings is None
    assert [u.id for u in res.corpus.units] == ["B"]


def test_agent_filter_map_reduce_chains():
    corpus = Corpus.from_documents(["a", "b", "c"], ids=["A", "B", "C"])
    res = run_pipeline(
        corpus, "t", ops=["filter", "map", "reduce"], plan=_plan(["filter", "map", "reduce"]),
        completer_factory=lambda tools: OpAwareFake(drop_ids={"B"}), lm=object(),
    )
    # B is filtered out; A and C are mapped; then reduced.
    assert res.findings == ["mapped:A", "mapped:C"]
    assert res.output == "REDUCED"
    assert res.corpus is None


# ----------------------------------------------------------------- op strategies
class BatchAwareFake:
    """Op- and strategy-aware fake: handles single and batched shards, map and filter.

    A shard with >1 unit (batched) returns a per-unit JSON array; a single-unit shard
    returns the plain map output or a VERDICT line. Records whether shared context was
    injected. Thread-safe (only reads ``drop_ids``).
    """

    def __init__(self, drop_ids=(), seen_context=None):
        self.drop_ids = set(drop_ids)
        self.seen_context = seen_context  # a set() to record shared-context injection

    def __call__(self, messages, *, tools_enabled: bool = True):
        import json
        import re

        system = messages[0]["content"]
        last = messages[-1]["content"]
        if self.seen_context is not None and "SHARED CONTEXT" in last:
            self.seen_context.add(True)
        if "reducer" in system:
            return AgentStep(content="REDUCED")
        ids = re.findall(r"\[unit ([^\]]+)\]", last)
        is_filter = "agentic filter" in system
        if len(ids) > 1:  # batched -> per-unit JSON
            if is_filter:
                arr = [{"id": i, "keep": i not in self.drop_ids} for i in ids]
            else:
                arr = [{"id": i, "output": f"mapped:{i}"} for i in ids]
            return AgentStep(content="Result:\n" + json.dumps(arr))
        i = ids[0] if ids else "?"
        if is_filter:
            return AgentStep(content=f"VERDICT: {'DROP' if i in self.drop_ids else 'KEEP'}")
        return AgentStep(content=f"mapped:{i}")


def _plan_strat(ops, strategies=None, contexts=None, shard_size=1):
    return Plan(
        ops=ops,
        instructions={op: op for op in ops},
        strategies=strategies or {},
        contexts=contexts or {},
        shard_size=shard_size,
        parallelism=4,
    )


def test_map_batched_yields_per_unit_outputs():
    corpus = Corpus.from_documents(["a", "b", "c", "d"], ids=["A", "B", "C", "D"])
    res = run_pipeline(
        corpus, "t", ops=["map"],
        plan=_plan_strat(["map"], strategies={"map": "batched"}, shard_size=2),
        completer_factory=lambda tools: BatchAwareFake(), lm=object(),
    )
    # 2 shards of 2 units; batched still produces one output *per unit*, in order.
    assert res.findings == ["mapped:A", "mapped:B", "mapped:C", "mapped:D"]
    assert [u.id for u in res.corpus.units] == ["A", "B", "C", "D"]


def test_filter_batched_selects_per_unit():
    corpus = Corpus.from_documents(["a", "b", "c", "d"], ids=["A", "B", "C", "D"])
    res = run_pipeline(
        corpus, "t", ops=["filter"],
        plan=_plan_strat(["filter"], strategies={"filter": "batched"}, shard_size=2),
        completer_factory=lambda tools: BatchAwareFake(drop_ids={"B", "D"}), lm=object(),
    )
    assert [u.id for u in res.corpus.units] == ["A", "C"]


def test_shared_context_is_injected():
    corpus = Corpus.from_documents(["a", "b"], ids=["A", "B"])
    seen = set()
    res = run_pipeline(
        corpus, "t", ops=["map"],
        plan=_plan_strat(["map"], strategies={"map": "shared_context"}, contexts={"map": "REF"}),
        completer_factory=lambda tools: BatchAwareFake(seen_context=seen), lm=object(),
    )
    assert seen == {True}  # every agent saw the injected shared context
    assert res.findings == ["mapped:A", "mapped:B"]


def test_strategy_can_be_overridden_via_param():
    corpus = Corpus.from_documents(["a", "b"], ids=["A", "B"])
    res = run_pipeline(
        corpus, "t", ops=["map"],
        plan=_plan_strat(["map"], shard_size=2),  # plan defaults to per_unit
        strategies={"map": "batched"},            # ...overridden at call time
        completer_factory=lambda tools: BatchAwareFake(), lm=object(),
    )
    assert res.findings == ["mapped:A", "mapped:B"]


def test_parse_batched_map_and_filter():
    from lotus.agentic.pipeline import _parse_batched

    txt = 'analysis... [{"id": "x", "output": "hi"}, {"id": "y", "output": "yo"}]'
    assert _parse_batched(txt, "map") == {"x": "hi", "y": "yo"}

    ftxt = 'ok [{"id": "x", "keep": true}, {"id": "y", "keep": false}]'
    assert _parse_batched(ftxt, "filter") == {"x": "VERDICT: KEEP", "y": "VERDICT: DROP"}

    assert _parse_batched("no json array here", "map") == {}
