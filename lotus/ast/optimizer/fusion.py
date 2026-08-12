"""Operator fusion optimizer: collapse consecutive LM ops into fewer calls.

v1 fuses adjacent ``SemFilterNode`` predicates into a single conjunctive filter
when safe (no cascades, examples, or conflicting options). This cuts LM calls
roughly in half for filter chains without changing semantics of AND-able
predicates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

import lotus

from ..nodes import BaseNode, SemFilterNode
from .base import BaseOptimizer

if TYPE_CHECKING:
    from ..lazyframe import LazyFrame


def _filters_fusible(a: SemFilterNode, b: SemFilterNode) -> bool:
    """Return True when ``a`` and ``b`` can be safely AND-fused into one call."""
    if a.examples is not None or b.examples is not None:
        return False
    if a.helper_examples is not None or b.helper_examples is not None:
        return False
    if a.cascade_args is not None or b.cascade_args is not None:
        return False
    if a.strategy != b.strategy:
        return False
    if a.system_prompt != b.system_prompt:
        return False
    if a.output_tokens != b.output_tokens:
        return False
    if a.default != b.default:
        return False
    return True


def _fuse_filter_pair(a: SemFilterNode, b: SemFilterNode) -> SemFilterNode:
    instruction = f"({a.user_instruction}) AND ({b.user_instruction})"
    return a.model_copy(
        update={
            "user_instruction": instruction,
            "progress_bar_desc": "Filtering (fused)",
        }
    )


class OperatorFusionOptimizer(BaseOptimizer):
    """Fuse consecutive semantic filters into fewer LM calls.

    Example::

        from lotus.ast import LazyFrame
        from lotus.ast.optimizer import OperatorFusionOptimizer

        lf = (
            LazyFrame()
            .sem_filter("{text} is about sports")
            .sem_filter("{text} mentions a team")
        )
        fused = lf.optimize([OperatorFusionOptimizer()], auto_include_default_optimizers=False)
        # -> one sem_filter with AND-ed predicates
    """

    requires_train_data: bool = False

    def __init__(self, *, fuse_filters: bool = True) -> None:
        self.fuse_filters = fuse_filters
        self.fusions_applied: int = 0

    def optimize(
        self,
        nodes: list[BaseNode],
        train_data: dict["LazyFrame", pd.DataFrame] | pd.DataFrame | None = None,
    ) -> list[BaseNode]:
        del train_data
        self.fusions_applied = 0
        if not self.fuse_filters:
            return list(nodes)

        out: list[BaseNode] = []
        i = 0
        while i < len(nodes):
            node = nodes[i]
            if isinstance(node, SemFilterNode):
                fused = node
                j = i + 1
                while j < len(nodes) and isinstance(nodes[j], SemFilterNode) and _filters_fusible(fused, nodes[j]):
                    fused = _fuse_filter_pair(fused, nodes[j])
                    self.fusions_applied += 1
                    j += 1
                out.append(fused)
                i = j
            else:
                out.append(node)
                i += 1

        if self.fusions_applied:
            lotus.logger.debug(
                "OperatorFusionOptimizer: fused %d filter edge(s); %d -> %d nodes",
                self.fusions_applied,
                len(nodes),
                len(out),
            )
        return out
