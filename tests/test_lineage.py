"""Offline unit tests for sem_lineage (no live LLM required)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pandas as pd

import lotus
from lotus.sem_ops.postprocessors import lineage_postprocess
from lotus.sem_ops.sem_lineage import events_from_agent_traces, sem_lineage
from lotus.types import LMOutput


def test_events_from_agent_traces_flattens_tool_calls():
    traces = [
        [
            {"tool": "search", "arguments": {"q": "lotus"}, "result": "LOTUS is a semantic query engine"},
            {"tool": "python", "arguments": {"code": "1+1"}, "result": "2"},
        ],
        [
            {"tool": "search", "arguments": {"q": "unrelated"}, "result": "weather in SF"},
        ],
    ]
    events = events_from_agent_traces(traces, session_ids=["s0", "s1"])
    assert list(events.columns) == [
        "event_id",
        "session_id",
        "unit_id",
        "step",
        "event_type",
        "tool",
        "arguments",
        "result",
    ]
    assert len(events) == 3
    assert events.iloc[0]["event_id"] == "s0:0"
    assert events.iloc[0]["tool"] == "search"
    assert events.iloc[2]["session_id"] == "s1"


def test_lineage_postprocess_parses_json():
    answers = [
        json.dumps(
            {
                "supported": True,
                "evidence_event_ids": ["s0:0"],
                "lineage_path": ["s0:0"],
                "rationale": "search result states the claim",
            }
        ),
        "not-json",
    ]
    model = MagicMock()
    parsed, explanations = lineage_postprocess(answers, model, cot_reasoning=False)
    assert parsed[0]["supported"] is True
    assert explanations[0] == "search result states the claim"
    assert parsed[1] == {}


def test_sem_lineage_core_with_mocked_lm():
    claims = [
        "LOTUS is a semantic query engine",
        "The capital of Mars is Olympus",
    ]
    events0 = pd.DataFrame(
        [
            {
                "event_id": "s0:0",
                "session_id": "s0",
                "tool": "search",
                "arguments": '{"q": "lotus"}',
                "result": "LOTUS is a semantic query engine from Berkeley/Stanford",
            },
            {
                "event_id": "s0:1",
                "session_id": "s0",
                "tool": "python",
                "arguments": "{}",
                "result": "ok",
            },
        ]
    )
    events1 = pd.DataFrame(
        [
            {
                "event_id": "s1:0",
                "session_id": "s1",
                "tool": "search",
                "arguments": '{"q": "mars"}',
                "result": "Mars has no capital city",
            }
        ]
    )

    responses = [
        json.dumps(
            {
                "supported": True,
                "evidence_event_ids": ["s0:0"],
                "lineage_path": ["search", "s0:0"],
                "rationale": "direct support",
            }
        ),
        json.dumps(
            {
                "supported": False,
                "evidence_event_ids": [],
                "lineage_path": [],
                "rationale": "events refute the claim",
            }
        ),
    ]

    model = MagicMock()
    model.count_tokens.return_value = 1
    model.is_deepseek.return_value = False
    model.side_effect = lambda inputs, **kwargs: LMOutput(outputs=responses[: len(inputs)])

    out = sem_lineage(claims, [events0, events1], model)
    assert out.supported == [True, False]
    assert out.evidence_event_ids[0] == ["s0:0"]
    assert out.evidence_event_ids[1] == []
    assert "search" in out.lineage_paths[0] or "s0:0" in out.lineage_paths[0]
    assert out.stats is not None
    assert out.stats["n_with_evidence"] == 1


def test_sem_lineage_accessor_scopes_by_session():
    claims = pd.DataFrame(
        {
            "session_id": ["s0", "s1"],
            "claim": [
                "LOTUS is a semantic query engine",
                "The capital of Mars is Olympus",
            ],
        }
    )
    events = events_from_agent_traces(
        [
            [{"tool": "search", "arguments": {"q": "lotus"}, "result": "LOTUS is a semantic query engine"}],
            [{"tool": "search", "arguments": {"q": "mars"}, "result": "Mars has no capital city"}],
        ],
        session_ids=["s0", "s1"],
    )

    responses = [
        json.dumps(
            {"supported": True, "evidence_event_ids": ["s0:0"], "lineage_path": ["s0:0"], "rationale": "ok"}
        ),
        json.dumps(
            {"supported": False, "evidence_event_ids": [], "lineage_path": [], "rationale": "unsupported"}
        ),
    ]

    model = MagicMock()
    model.count_tokens.return_value = 1
    model.is_deepseek.return_value = False
    model.side_effect = lambda inputs, **kwargs: LMOutput(outputs=responses[: len(inputs)])
    lotus.settings.configure(lm=model, enable_cache=False)

    result = claims.sem_lineage(events, claim_col="claim", session_col="session_id")
    assert result["supported"].tolist() == [True, False]
    assert result["evidence_event_ids"].iloc[0] == ["s0:0"]
    assert result["evidence_event_ids"].iloc[1] == []


def test_offline_lineage_eval_precision_lift():
    """Naive all-events baseline should be less precise than lineage-style selection."""
    from examples.op_examples.lineage_eval import main

    metrics = main()
    assert metrics["lineage_precision"] > metrics["naive_precision"]
    assert metrics["relative_precision_improvement"] > 0
