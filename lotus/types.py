from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Literal

import pandas as pd
from litellm.types.utils import ChatCompletionTokenLogprob
from pydantic import BaseModel, model_validator


################################################################################
# LM related
################################################################################
@dataclass
class LMOutput:
    outputs: list[str]
    logprobs: list[list[ChatCompletionTokenLogprob]] | None = None


@dataclass
class LMStats:
    @dataclass
    class TotalUsage:
        prompt_tokens: int = 0
        completion_tokens: int = 0
        total_tokens: int = 0
        total_cost: float = 0.0
        # Cached tokens (prompt cache hits) - charged at lower rate
        cached_prompt_tokens: int = 0
        # Tokens used to create the cache (prompt cache writes) - one-time cost
        cache_creation_tokens: int = 0

        def __sub__(self, other: "LMStats.TotalUsage") -> "LMStats.TotalUsage":
            return LMStats.TotalUsage(
                prompt_tokens=self.prompt_tokens - other.prompt_tokens,
                completion_tokens=self.completion_tokens - other.completion_tokens,
                total_tokens=self.total_tokens - other.total_tokens,
                total_cost=self.total_cost - other.total_cost,
                cached_prompt_tokens=self.cached_prompt_tokens - other.cached_prompt_tokens,
                cache_creation_tokens=self.cache_creation_tokens - other.cache_creation_tokens,
            )

        def __add__(self, other: "LMStats.TotalUsage") -> "LMStats.TotalUsage":
            return LMStats.TotalUsage(
                prompt_tokens=self.prompt_tokens + other.prompt_tokens,
                completion_tokens=self.completion_tokens + other.completion_tokens,
                total_tokens=self.total_tokens + other.total_tokens,
                total_cost=self.total_cost + other.total_cost,
                cached_prompt_tokens=self.cached_prompt_tokens + other.cached_prompt_tokens,
                cache_creation_tokens=self.cache_creation_tokens + other.cache_creation_tokens,
            )

    # Usage stats if there was no caching
    virtual_usage: TotalUsage = field(default_factory=TotalUsage)
    # Actual usage with caching applied
    physical_usage: TotalUsage = field(default_factory=TotalUsage)

    cache_hits: int = 0
    operator_cache_hits: int = 0

    def __add__(self, other: "LMStats") -> "LMStats":
        return LMStats(
            virtual_usage=self.virtual_usage + other.virtual_usage,
            physical_usage=self.physical_usage + other.physical_usage,
            cache_hits=self.cache_hits + other.cache_hits,
            operator_cache_hits=self.operator_cache_hits + other.operator_cache_hits,
        )


@dataclass
class LogprobsForCascade:
    tokens: list[list[str]]
    confidences: list[list[float]]


@dataclass
class LogprobsForFilterCascade:
    positive_probs: list[float]
    tokens: list[list[str]]
    confidences: list[list[float]]

    # @deprecated("Use positive_probs instead")
    @property
    def true_probs(self) -> list[float]:
        import lotus

        lotus.logger.warning("true_probs is deprecated. Use positive_probs instead")
        return self.positive_probs


################################################################################
# Semantic operation outputs
################################################################################
@dataclass
class SemanticMapPostprocessOutput:
    raw_outputs: list[str]
    outputs: list[str]
    explanations: list[str | None]


@dataclass
class SemanticMapOutput:
    raw_outputs: list[str]
    outputs: list[str]
    explanations: list[str | None]


@dataclass
class SemanticExtractPostprocessOutput:
    raw_outputs: list[str]
    outputs: list[dict[str, str]]
    explanations: list[str | None]


@dataclass
class SemanticExtractOutput:
    raw_outputs: list[str]
    outputs: list[dict[str, str]]
    explanations: list[str | None]


@dataclass
class SemanticFilterPostprocessOutput:
    raw_outputs: list[str]
    outputs: list[bool]
    explanations: list[str | None]


@dataclass
class SemanticFilterOutput:
    raw_outputs: list[str]
    outputs: list[bool]
    explanations: list[str | None]
    stats: dict[str, Any] | None = None
    logprobs: list[list[ChatCompletionTokenLogprob]] | None = None


@dataclass
class SemanticAggOutput:
    outputs: list[str]


class LongContextStrategy(Enum):
    """Enumeration of available document long_context strategies for semantic aggregation."""

    TRUNCATE = auto()
    CHUNK = auto()


@dataclass
class SemanticJoinOutput:
    join_results: list[tuple[int, int, str | None]]
    filter_outputs: list[bool]
    all_raw_outputs: list[str]
    all_explanations: list[str | None]
    stats: dict[str, Any] | None = None


@dataclass
class SemanticLineageOutput:
    """Per-claim provenance results from ``sem_lineage``.

    Each parallel list is aligned with the input claim rows.
    """

    supported: list[bool]
    evidence_event_ids: list[list[str]]
    evidence_texts: list[list[str]]
    lineage_paths: list[list[str]]
    raw_outputs: list[str]
    explanations: list[str | None]
    stats: dict[str, Any] | None = None


class ProxyModel(Enum):
    HELPER_LM = "helper_lm"
    EMBEDDING_MODEL = "embedding_model"


class CascadeArgs(BaseModel):
    recall_target: float = 0.8
    precision_target: float = 0.8
    sampling_percentage: float = 0.1
    failure_probability: float = 0.2
    map_instruction: str | None = None
    map_examples: pd.DataFrame | None = None
    proxy_model: ProxyModel = ProxyModel.HELPER_LM

    # Filter cascade args
    # Optional helper-model instruction for sem_filter cascades. If unset,
    # helper-model calls default to the main sem_filter user instruction.
    helper_filter_instruction: str | None = None
    cascade_IS_weight: float = 0.9
    cascade_num_calibration_quantiles: int = 50
    filter_pos_cascade_threshold: float | None = None
    filter_neg_cascade_threshold: float | None = None

    # Join cascade args
    min_join_cascade_size: int = 100
    cascade_IS_max_sample_range: int = 200
    cascade_IS_random_seed: int | None = None
    join_cascade_strategy: Literal["search_filter", "map_search_filter"] | None = None
    join_cascade_pos_threshold: float | None = None
    join_cascade_neg_threshold: float | None = None

    # to enable pandas
    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="after")
    def check_filter_cascade_thresholds(self):
        # Both must be set together or both unset
        if (self.filter_pos_cascade_threshold is None) != (self.filter_neg_cascade_threshold is None):
            raise ValueError(
                "Both filter_cascade_threshold_high and filter_cascade_threshold_low must be provided together."
            )
        # If both are set, high must be >= low
        if (
            self.filter_pos_cascade_threshold is not None
            and self.filter_neg_cascade_threshold is not None
            and self.filter_pos_cascade_threshold < self.filter_neg_cascade_threshold
        ):
            raise ValueError("filter_cascade_threshold_high must be >= filter_cascade_threshold_low.")
        return self

    @model_validator(mode="after")
    def check_join_cascade_thresholds(self):
        if self.join_cascade_strategy is not None:
            if (self.join_cascade_pos_threshold is None) or (self.join_cascade_neg_threshold is None):
                raise ValueError(
                    "join_cascade_strategy is provided but join_cascade_threshold_pos or join_cascade_threshold_neg are not provided."
                )
            if self.join_cascade_pos_threshold < self.join_cascade_neg_threshold:
                raise ValueError("join_cascade_threshold_pos must be >= join_cascade_threshold_neg.")
        return self


@dataclass
class SemanticTopKOutput:
    indexes: list[int]
    stats: dict[str, Any] | None = None


################################################################################
# RM related
################################################################################


@dataclass
class RMOutput:
    distances: list[list[float]]
    indices: list[list[int]]


################################################################################
# Reranker related
################################################################################
@dataclass
class RerankerOutput:
    indices: list[int]


################################################################################
# Serialization related
################################################################################
class SerializationFormat(Enum):
    JSON = "json"
    XML = "xml"
    DEFAULT = "default"


################################################################################
# Utility
################################################################################
@dataclass
class UsageLimit:
    prompt_tokens_limit: float = float("inf")
    completion_tokens_limit: float = float("inf")
    total_tokens_limit: float = float("inf")
    total_cost_limit: float = float("inf")


################################################################################
# Exception related
################################################################################
class LotusException(Exception):
    """Base class for all Lotus exceptions."""

    pass


class LotusUsageLimitException(LotusException):
    """Exception raised when the usage limit is exceeded."""

    pass


################################################################################
# Reasoning Strategy
################################################################################
class ReasoningStrategy(Enum):
    DEFAULT = auto()
    COT = auto()
    ZS_COT = auto()
    FEW_SHOT = auto()
