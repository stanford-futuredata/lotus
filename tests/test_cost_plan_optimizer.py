"""Offline tests for CostBasedPlanOptimizer (no LM calls)."""

from __future__ import annotations

from lotus.ast import LazyFrame
from lotus.ast.nodes import SemFilterNode, SemMapNode
from lotus.ast.optimizer import CostBasedPlanOptimizer, CostModel
from lotus.types import CascadeArgs


def test_enables_cascade_when_cheaper_than_full_lm():
    lf = LazyFrame().sem_filter("{text} is about sports")
    opt = CostBasedPlanOptimizer(
        estimated_rows=10_000,
        enable_cascades=True,
        reorder_filters=False,
        cost_model=CostModel(primary_cost_per_call=1.0, helper_cost_per_call=0.05, cascade_resolve_rate=0.6),
    )
    out = lf.optimize([opt], auto_include_default_optimizers=False)
    node = out._nodes[-1]
    assert isinstance(node, SemFilterNode)
    assert node.cascade_args is not None
    assert isinstance(node.cascade_args, CascadeArgs)
    assert opt.last_estimate is not None
    assert opt.last_baseline_estimate is not None
    assert opt.last_estimate.total_cost < opt.last_baseline_estimate.total_cost


def test_skips_cascade_when_dataset_too_small():
    lf = LazyFrame().sem_filter("{text} is about sports")
    opt = CostBasedPlanOptimizer(
        estimated_rows=10,
        min_rows_for_cascade=50,
        enable_cascades=True,
    )
    out = lf.optimize([opt], auto_include_default_optimizers=False)
    node = out._nodes[-1]
    assert isinstance(node, SemFilterNode)
    assert node.cascade_args is None


def test_reorders_filters_cheapest_first():
    # Attach cascade manually on the second filter only; optimizer should move it first.
    lf = (
        LazyFrame()
        .sem_filter("{text} is long and expensive full scan", cascade_args=None)
        .sem_filter(
            "{text} is cheap cascade",
            cascade_args=CascadeArgs(recall_target=0.8, precision_target=0.8),
        )
    )
    # First node has no cascade; second has cascade. Reorder with cascades disabled
    # so we don't rewrite args — only sort by unit cost.
    opt = CostBasedPlanOptimizer(
        estimated_rows=1000,
        enable_cascades=False,
        reorder_filters=True,
    )
    out = lf.optimize([opt], auto_include_default_optimizers=False)
    filters = [n for n in out._nodes if isinstance(n, SemFilterNode)]
    assert len(filters) == 2
    assert filters[0].cascade_args is not None
    assert filters[1].cascade_args is None
    assert "cheap cascade" in filters[0].user_instruction


def test_estimate_plan_cost_counts_map_and_filter():
    nodes = [
        SemFilterNode(user_instruction="{t} ok"),
        SemMapNode(user_instruction="summarize {t}"),
    ]
    opt = CostBasedPlanOptimizer(estimated_rows=100, enable_cascades=False)
    est = opt.estimate_plan_cost(nodes, 100)
    # filter: 100 primary, then map on 50 (default selectivity 0.5)
    assert est.primary_calls == 100 + 50
    assert est.helper_calls == 0
    assert est.total_cost == est.primary_calls * opt.cost_model.primary_cost_per_call
