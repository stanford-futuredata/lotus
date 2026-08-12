"""Offline tests for sem_group_by helpers and classification path (no network)."""

from __future__ import annotations

import pandas as pd
import pytest

import lotus
from lotus.sem_ops.sem_group_by import _match_label, _parse_label_list
from lotus.types import LMOutput


class StubLM:
    """Minimal LM stand-in: returns scripted string outputs in call order."""

    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.calls: list[list] = []

    def __call__(self, inputs, progress_bar_desc: str = "", **kwargs):
        self.calls.append(inputs)
        outs = []
        for _ in inputs:
            if not self.outputs:
                raise AssertionError("StubLM ran out of scripted outputs")
            outs.append(self.outputs.pop(0))
        return LMOutput(outputs=outs)


def test_parse_label_list_json_and_numbered():
    assert _parse_label_list('Here: ["DB", "IR", "Security"]', expected=3) == ["DB", "IR", "Security"]
    assert _parse_label_list("1. Topic A\n2. Topic B\n3. Topic C", expected=3) == [
        "Topic A",
        "Topic B",
        "Topic C",
    ]


def test_match_label_exact_and_fuzzy():
    labels = ["Databases", "Information Retrieval", "Security"]
    assert _match_label("Databases", labels) == "Databases"
    assert _match_label("Answer: security", labels) == "Security"
    assert _match_label("This is about Information Retrieval systems", labels) == "Information Retrieval"


def test_sem_group_by_with_fixed_labels(monkeypatch):
    import lotus.sem_ops.sem_group_by  # noqa: F401 — register accessor

    df = pd.DataFrame(
        {
            "paper": [
                "A survey of vector databases",
                "Neural information retrieval",
                "Malware detection with transformers",
                "Query optimization in OLAP",
            ]
        }
    )
    labels = ["Databases", "Information Retrieval", "Security"]
    # One scripted answer per row (cycles if needed).
    stub = StubLM(
        [
            "Databases",
            "Information Retrieval",
            "Security",
            "Databases",
        ]
    )
    monkeypatch.setattr(lotus.settings, "lm", stub)

    out = df.sem_group_by("the topic of each {paper}", labels=labels, use_clustering=False)
    assert list(out["_group"]) == labels[:3] + ["Databases"]
    assert set(out["_group_id"]) == {0, 1, 2}
    assert out.attrs["group_labels"] == labels
    assert len(stub.calls) == 1  # one batched classify call
    assert len(stub.calls[0]) == 4


def test_sem_group_by_discovers_then_assigns(monkeypatch):
    import lotus.sem_ops.sem_group_by  # noqa: F401

    df = pd.DataFrame({"paper": ["doc a", "doc b", "doc c", "doc d"]})
    # First call = discovery (1 prompt); second = classify (4 prompts).
    stub = StubLM(['["Alpha", "Beta"]', "Alpha", "Beta", "Alpha", "Beta"])
    monkeypatch.setattr(lotus.settings, "lm", stub)

    out = df.sem_group_by(
        "the topic of each {paper}",
        n=2,
        use_clustering=False,
        reassign=True,
    )
    assert out.attrs["group_labels"] == ["Alpha", "Beta"]
    assert set(out["_group"]) == {"Alpha", "Beta"}
    assert len(stub.calls) == 2


def test_sem_group_by_requires_n_or_labels():
    import lotus.sem_ops.sem_group_by  # noqa: F401

    df = pd.DataFrame({"paper": ["x"]})
    lotus.settings.configure(lm=StubLM(["x"]))
    with pytest.raises(ValueError, match="labels=|n="):
        df.sem_group_by("the topic of each {paper}")
