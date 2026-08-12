"""Example: fuse consecutive sem_filters into one AND-ed LM call."""

from lotus.ast import LazyFrame
from lotus.ast.optimizer import OperatorFusionOptimizer

lf = (
    LazyFrame()
    .sem_filter("{text} is about machine learning")
    .sem_filter("{text} proposes a new method")
    .sem_map("One-sentence summary of {text}")
)

print("Before:\n", lf.show())
fused = lf.optimize([OperatorFusionOptimizer()], auto_include_default_optimizers=False)
print("\nAfter fusion:\n", fused.show())
