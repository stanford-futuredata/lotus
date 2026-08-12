"""Optimizer module for LOTUS LazyFrames."""

from .base import BaseOptimizer
from .cascade import CascadeOptimizer
from .cost_based import CostBasedPlanOptimizer, CostModel, PlanCostEstimate
from .gepa_optimizer import GEPAOptimizer
from .predicate_pushdown import PredicatePushdownOptimizer

DEFAULT_OPTIMIZERS: list[BaseOptimizer] = [PredicatePushdownOptimizer()]

__all__ = [
    "BaseOptimizer",
    "CascadeOptimizer",
    "CostBasedPlanOptimizer",
    "CostModel",
    "DEFAULT_OPTIMIZERS",
    "GEPAOptimizer",
    "PlanCostEstimate",
    "PredicatePushdownOptimizer",
]
