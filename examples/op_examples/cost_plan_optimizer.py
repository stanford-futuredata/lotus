"""Example: cost-based plan optimizer chooses cascades + filter order.

This does not call an LM — it only rewrites the LazyFrame plan. Pair with
``CascadeOptimizer`` + train data if you want thresholds learned before execute.
"""

from lotus.ast import LazyFrame
from lotus.ast.optimizer import CostBasedPlanOptimizer, CostModel

lf = (
    LazyFrame()
    .sem_filter("{text} is about machine learning")
    .sem_filter("{text} proposes a new method")
    .sem_map("One-sentence summary of {text}")
)

opt = CostBasedPlanOptimizer(
    estimated_rows=50_000,
    enable_cascades=True,
    reorder_filters=True,
    cost_model=CostModel(
        primary_cost_per_call=1.0,
        helper_cost_per_call=0.05,
        cascade_resolve_rate=0.55,
    ),
)
plan = lf.optimize([opt], auto_include_default_optimizers=False)

print("Tree:\n", plan.show())
print("Baseline cost:", opt.last_baseline_estimate)
print("Optimized cost:", opt.last_estimate)
