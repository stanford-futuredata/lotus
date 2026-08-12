"""Tests for operator fusion and adaptive agentic batching."""

from __future__ import annotations

from lotus.ast import LazyFrame
from lotus.ast.nodes import SemFilterNode, SemMapNode
from lotus.ast.optimizer import OperatorFusionOptimizer
from lotus.agentic.planner import suggest_adaptive_batching
from lotus.corpus import Corpus
from lotus.types import CascadeArgs


def test_fuses_consecutive_filters():
    lf = (
        LazyFrame()
        .sem_filter("{text} is about sports")
        .sem_filter("{text} mentions a team")
        .sem_map("Summarize {text}")
    )
    opt = OperatorFusionOptimizer()
    out = lf.optimize([opt], auto_include_default_optimizers=False)
    filters = [n for n in out._nodes if isinstance(n, SemFilterNode)]
    maps = [n for n in out._nodes if isinstance(n, SemMapNode)]
    assert len(filters) == 1
    assert "AND" in filters[0].user_instruction
    assert len(maps) == 1
    assert opt.fusions_applied == 1


def test_does_not_fuse_cascaded_filters():
    lf = (
        LazyFrame()
        .sem_filter("{text} a", cascade_args=CascadeArgs())
        .sem_filter("{text} b")
    )
    opt = OperatorFusionOptimizer()
    out = lf.optimize([opt], auto_include_default_optimizers=False)
    filters = [n for n in out._nodes if isinstance(n, SemFilterNode)]
    assert len(filters) == 2
    assert opt.fusions_applied == 0


def test_adaptive_batching_for_tiny_units():
    corpus = Corpus.from_documents(["a", "b", "c", "d", "e", "f"])
    strategy, size = suggest_adaptive_batching(corpus)
    assert strategy == "batched"
    assert size >= 2


def test_adaptive_batching_skips_long_units():
    corpus = Corpus.from_documents(["x" * 2000, "y" * 2000, "z" * 2000, "w" * 2000])
    strategy, size = suggest_adaptive_batching(corpus)
    assert strategy == "per_unit"
    assert size == 1
