"""Cost-based physical plan optimizer for LOTUS LazyFrames.

Chooses cheap physical implementations for multi-operator pipelines under a
simple analytical cost model (Abacus-style v1):

* Optionally attach ``CascadeArgs`` to ``SemFilterNode`` / ``SemJoinNode`` when
  a helper cascade is estimated cheaper than a full primary-LM scan at the
  configured recall/precision targets.
* Reorder consecutive ``SemFilterNode`` blocks so cheaper (cascaded) filters
  run first, reducing rows seen by expensive full-LM filters.

This optimizer does **not** call models; it only rewrites the plan. Use
``CascadeOptimizer`` afterward if you want thresholds learned on train data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

import lotus
from lotus.types import CascadeArgs

from ..nodes import BaseNode, SemFilterNode, SemJoinNode, SemMapNode
from .base import BaseOptimizer

if TYPE_CHECKING:
    from ..lazyframe import LazyFrame


@dataclass
class CostModel:
    """Per-call dollar (or abstract unit) costs for physical alternatives."""

    primary_cost_per_call: float = 1.0
    helper_cost_per_call: float = 0.1
    # Fraction of rows a cascade is assumed to resolve without the primary LM
    # after thresholds are learned (conservative default).
    cascade_resolve_rate: float = 0.5
    # Default selectivity applied after each filter when projecting row counts.
    default_selectivity: float = 0.5


@dataclass
class PlanCostEstimate:
    """Estimated cost of a physical plan."""

    total_cost: float
    primary_calls: float
    helper_calls: float
    details: list[dict[str, Any]] = field(default_factory=list)


class CostBasedPlanOptimizer(BaseOptimizer):
    """Search cheap physical implementations under a cost model.

    Example::

        from lotus.ast import LazyFrame
        from lotus.ast.optimizer import CostBasedPlanOptimizer

        lf = (
            LazyFrame()
            .sem_filter("{text} is about sports")
            .sem_filter("{text} mentions a specific team")
            .sem_map("Summarize {text}")
        )
        opt = CostBasedPlanOptimizer(estimated_rows=10_000, enable_cascades=True)
        plan = lf.optimize([opt], auto_include_default_optimizers=False)
        print(opt.last_estimate)
    """

    requires_train_data: bool = False

    def __init__(
        self,
        *,
        cost_model: CostModel | None = None,
        recall_target: float = 0.8,
        precision_target: float = 0.8,
        sampling_percentage: float = 0.1,
        failure_probability: float = 0.2,
        enable_cascades: bool = True,
        reorder_filters: bool = True,
        estimated_rows: int | None = None,
        min_rows_for_cascade: int = 50,
    ) -> None:
        self.cost_model = cost_model or CostModel()
        self.recall_target = recall_target
        self.precision_target = precision_target
        self.sampling_percentage = sampling_percentage
        self.failure_probability = failure_probability
        self.enable_cascades = enable_cascades
        self.reorder_filters = reorder_filters
        self.estimated_rows = estimated_rows
        self.min_rows_for_cascade = min_rows_for_cascade
        self.last_estimate: PlanCostEstimate | None = None
        self.last_baseline_estimate: PlanCostEstimate | None = None

    def optimize(
        self,
        nodes: list[BaseNode],
        train_data: dict["LazyFrame", pd.DataFrame] | pd.DataFrame | None = None,
    ) -> list[BaseNode]:
        n_rows = self._resolve_row_count(train_data)
        nodes = [n.model_copy(deep=True) if hasattr(n, "model_copy") else n for n in nodes]

        self.last_baseline_estimate = self.estimate_plan_cost(nodes, n_rows)

        if self.enable_cascades:
            nodes = self._maybe_attach_cascades(nodes, n_rows)

        if self.reorder_filters:
            nodes = self._reorder_filter_blocks(nodes)

        self.last_estimate = self.estimate_plan_cost(nodes, n_rows)
        lotus.logger.debug(
            "CostBasedPlanOptimizer: baseline=%.3f optimized=%.3f (rows≈%s)",
            self.last_baseline_estimate.total_cost,
            self.last_estimate.total_cost,
            n_rows,
        )
        return nodes

    def estimate_plan_cost(self, nodes: list[BaseNode], n_rows: int) -> PlanCostEstimate:
        """Walk ``nodes`` and sum analytical primary/helper call costs."""
        rows = float(max(n_rows, 0))
        primary = 0.0
        helper = 0.0
        details: list[dict[str, Any]] = []
        cm = self.cost_model

        for node in nodes:
            if isinstance(node, SemFilterNode):
                if node.cascade_args is not None:
                    # Helper scores all rows; primary only unresolved + calibration sample.
                    unresolved = max(0.0, 1.0 - cm.cascade_resolve_rate)
                    sample = node.cascade_args.sampling_percentage
                    h_calls = rows
                    p_calls = rows * (unresolved + sample)
                    helper += h_calls
                    primary += p_calls
                    details.append(
                        {
                            "op": "sem_filter",
                            "physical": "cascade",
                            "rows_in": rows,
                            "primary_calls": p_calls,
                            "helper_calls": h_calls,
                        }
                    )
                    rows *= cm.default_selectivity
                else:
                    primary += rows
                    details.append(
                        {
                            "op": "sem_filter",
                            "physical": "full_lm",
                            "rows_in": rows,
                            "primary_calls": rows,
                            "helper_calls": 0.0,
                        }
                    )
                    rows *= cm.default_selectivity
            elif isinstance(node, SemMapNode):
                primary += rows
                details.append(
                    {
                        "op": "sem_map",
                        "physical": "full_lm",
                        "rows_in": rows,
                        "primary_calls": rows,
                        "helper_calls": 0.0,
                    }
                )
            elif isinstance(node, SemJoinNode):
                # Treat join as quadratic in left rows as a coarse upper bound.
                join_pairs = rows * rows
                if node.cascade_args is not None:
                    unresolved = max(0.0, 1.0 - cm.cascade_resolve_rate)
                    sample = node.cascade_args.sampling_percentage
                    h_calls = join_pairs
                    p_calls = join_pairs * (unresolved + sample)
                    helper += h_calls
                    primary += p_calls
                    details.append(
                        {
                            "op": "sem_join",
                            "physical": "cascade",
                            "rows_in": rows,
                            "primary_calls": p_calls,
                            "helper_calls": h_calls,
                        }
                    )
                else:
                    primary += join_pairs
                    details.append(
                        {
                            "op": "sem_join",
                            "physical": "full_lm",
                            "rows_in": rows,
                            "primary_calls": join_pairs,
                            "helper_calls": 0.0,
                        }
                    )
                rows *= cm.default_selectivity

        total = primary * cm.primary_cost_per_call + helper * cm.helper_cost_per_call
        return PlanCostEstimate(
            total_cost=total,
            primary_calls=primary,
            helper_calls=helper,
            details=details,
        )

    def _resolve_row_count(
        self,
        train_data: dict["LazyFrame", pd.DataFrame] | pd.DataFrame | None,
    ) -> int:
        if self.estimated_rows is not None:
            return int(self.estimated_rows)
        if isinstance(train_data, pd.DataFrame):
            return len(train_data)
        if isinstance(train_data, dict) and train_data:
            return max(len(df) for df in train_data.values())
        return 1000  # abstract default for planning without data

    def _default_cascade_args(self) -> CascadeArgs:
        return CascadeArgs(
            recall_target=self.recall_target,
            precision_target=self.precision_target,
            sampling_percentage=self.sampling_percentage,
            failure_probability=self.failure_probability,
        )

    def _maybe_attach_cascades(self, nodes: list[BaseNode], n_rows: int) -> list[BaseNode]:
        out: list[BaseNode] = []
        rows = float(n_rows)
        cm = self.cost_model

        for node in nodes:
            if isinstance(node, SemFilterNode) and node.cascade_args is None and rows >= self.min_rows_for_cascade:
                full_cost = rows * cm.primary_cost_per_call
                unresolved = max(0.0, 1.0 - cm.cascade_resolve_rate)
                sample = self.sampling_percentage
                cascade_cost = (
                    rows * cm.helper_cost_per_call
                    + rows * (unresolved + sample) * cm.primary_cost_per_call
                )
                if cascade_cost < full_cost:
                    node = node.model_copy(update={"cascade_args": self._default_cascade_args()})
                    lotus.logger.debug(
                        "CostBasedPlanOptimizer: enabling cascade on filter (%s) "
                        "(est cascade %.3f < full %.3f @ ~%.0f rows)",
                        node.user_instruction[:60],
                        cascade_cost,
                        full_cost,
                        rows,
                    )
                rows *= cm.default_selectivity
            elif isinstance(node, SemJoinNode) and node.cascade_args is None and rows >= self.min_rows_for_cascade:
                pairs = rows * rows
                full_cost = pairs * cm.primary_cost_per_call
                unresolved = max(0.0, 1.0 - cm.cascade_resolve_rate)
                sample = self.sampling_percentage
                cascade_cost = (
                    pairs * cm.helper_cost_per_call
                    + pairs * (unresolved + sample) * cm.primary_cost_per_call
                )
                if cascade_cost < full_cost:
                    node = node.model_copy(update={"cascade_args": self._default_cascade_args()})
                rows *= cm.default_selectivity
            elif isinstance(node, SemFilterNode):
                rows *= cm.default_selectivity
            out.append(node)
        return out

    def _filter_unit_cost(self, node: SemFilterNode) -> float:
        """Approximate per-row cost used for filter reordering."""
        cm = self.cost_model
        if node.cascade_args is not None:
            unresolved = max(0.0, 1.0 - cm.cascade_resolve_rate)
            sample = node.cascade_args.sampling_percentage
            return cm.helper_cost_per_call + (unresolved + sample) * cm.primary_cost_per_call
        return cm.primary_cost_per_call

    def _reorder_filter_blocks(self, nodes: list[BaseNode]) -> list[BaseNode]:
        result: list[BaseNode] = []
        i = 0
        while i < len(nodes):
            if isinstance(nodes[i], SemFilterNode):
                j = i
                while j < len(nodes) and isinstance(nodes[j], SemFilterNode):
                    j += 1
                block = nodes[i:j]
                if len(block) > 1:
                    block = sorted(block, key=self._filter_unit_cost)
                    lotus.logger.debug(
                        "CostBasedPlanOptimizer: reordered %d consecutive filters by unit cost",
                        len(block),
                    )
                result.extend(block)
                i = j
            else:
                result.append(nodes[i])
                i += 1
        return result
