"""Agentic operators for LOTUS: composable agent ops (map / filter / reduce) over a corpus."""

from .cascade import AgenticCascadeArgs, AgenticCascadeStats, parse_confidence, proxy_keep_score
from .loop import AgentResult, AgentStep, LiteLLMCompleter, ToolCall, run_agent
from .ops import CORPUS_OPS, DEFAULT_OPS, FILTER, MAP, OPS, REDUCE, TERMINAL_OPS, normalize_ops
from .pipeline import Result, run_pipeline
from .planner import Plan, derive_plan

__all__ = [
    "run_pipeline",
    "Result",
    "Plan",
    "derive_plan",
    "normalize_ops",
    "MAP",
    "FILTER",
    "REDUCE",
    "OPS",
    "CORPUS_OPS",
    "TERMINAL_OPS",
    "DEFAULT_OPS",
    "run_agent",
    "AgentResult",
    "AgentStep",
    "ToolCall",
    "LiteLLMCompleter",
    "AgenticCascadeArgs",
    "AgenticCascadeStats",
    "parse_confidence",
    "proxy_keep_score",
]
