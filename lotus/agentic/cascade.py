"""Accuracy-guaranteed cascades for agentic filter ops.

Mirrors ``sem_filter`` cascades: a cheap proxy agent (no tools, few steps) scores
each unit; thresholds learned against a sample of full tool-using "oracle" agent
runs decide which units can skip the expensive path while targeting recall /
precision under a failure probability (via ``learn_cascade_thresholds``).
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from lotus.sem_ops.cascade_utils import importance_sampling, learn_cascade_thresholds
from lotus.types import CascadeArgs

from .loop import Completer, run_agent
from .ops import FILTER

if TYPE_CHECKING:
    from lotus.corpus import Corpus, Unit
    from lotus.tools.base import Tool

_CONF_RE = re.compile(r"CONFIDENCE\s*:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)

_FILTER_CASCADE_SUFFIX = (
    "\n\nEnd your reply with two lines:\n"
    "VERDICT: KEEP|DROP\n"
    "CONFIDENCE: <float between 0 and 1>"
)


@dataclass
class AgenticCascadeArgs:
    """Targets and knobs for an agentic filter cascade."""

    recall_target: float = 0.8
    precision_target: float = 0.8
    sampling_percentage: float = 0.25
    failure_probability: float = 0.2
    cascade_IS_weight: float = 0.9
    cascade_IS_max_sample_range: int = 200
    cascade_IS_random_seed: int | None = 0
    cascade_num_calibration_quantiles: int = 50
    proxy_max_steps: int = 1
    # Learned / fixed thresholds on P(keep). Both None → learn on a sample.
    filter_pos_cascade_threshold: float | None = None
    filter_neg_cascade_threshold: float | None = None

    def to_cascade_args(self) -> CascadeArgs:
        return CascadeArgs(
            recall_target=self.recall_target,
            precision_target=self.precision_target,
            sampling_percentage=self.sampling_percentage,
            failure_probability=self.failure_probability,
            cascade_IS_weight=self.cascade_IS_weight,
            cascade_IS_max_sample_range=self.cascade_IS_max_sample_range,
            cascade_IS_random_seed=self.cascade_IS_random_seed,
            cascade_num_calibration_quantiles=self.cascade_num_calibration_quantiles,
            filter_pos_cascade_threshold=self.filter_pos_cascade_threshold,
            filter_neg_cascade_threshold=self.filter_neg_cascade_threshold,
        )


@dataclass
class AgenticCascadeStats:
    proxy_calls: int = 0
    oracle_calls: int = 0
    accepted_by_proxy: int = 0
    pos_threshold: float | None = None
    neg_threshold: float | None = None
    details: list[dict[str, Any]] = field(default_factory=list)


def parse_confidence(text: str, default: float = 0.5) -> float:
    """Extract CONFIDENCE in [0, 1] from an agent reply."""
    m = _CONF_RE.search(text or "")
    if not m:
        return default
    try:
        return float(np.clip(float(m.group(1)), 0.0, 1.0))
    except ValueError:
        return default


def proxy_keep_score(text: str, parse_verdict: Callable[[str], bool]) -> float:
    """Map proxy verdict+confidence to P(keep) in [0, 1]."""
    conf = parse_confidence(text)
    return conf if parse_verdict(text) else 1.0 - conf


def _proxy_user_content(instruction: str, shard: list["Unit"], context: str | None) -> str:
    from .pipeline import _shard_content

    parts = [f"INSTRUCTION:\n{instruction}"]
    if context:
        parts.append(f"SHARED CONTEXT:\n{context}")
    parts.append(f"SHARD:\n{_shard_content(shard)}")
    return "\n\n".join(parts) + _FILTER_CASCADE_SUFFIX


def run_agentic_filter_cascade(
    corpus: "Corpus",
    instruction: str,
    *,
    cascade_args: AgenticCascadeArgs,
    context: str | None,
    completer: Completer,
    tools: list["Tool"],
    system: str,
    parallelism: int,
    max_steps: int,
    usage: dict[str, int],
    parse_verdict: Callable[[str], bool],
    merge_usage: Callable[[dict[str, int], dict[str, int]], None],
) -> tuple["Corpus", AgenticCascadeStats]:
    """Filter ``corpus`` with a proxy→oracle cascade.

    Proxy agents run with **no tools** and ``proxy_max_steps``. Uncertain units
    (scores between learned neg/pos thresholds) are escalated to the full
    tool-using agent at ``max_steps``.
    """
    from lotus.corpus import Corpus

    units = list(corpus.units)
    stats = AgenticCascadeStats(proxy_calls=len(units))
    if not units:
        return Corpus([]), stats

    # --- 1) Proxy pass (always per-unit, no tools) ---
    def _proxy_one(u: "Unit") -> tuple[str, dict[str, int]]:
        res = run_agent(
            completer,
            [],  # no tools — cheap proxy
            system_prompt=system,
            user_content=_proxy_user_content(instruction, [u], context),
            max_steps=cascade_args.proxy_max_steps,
        )
        return res.output, res.usage

    with ThreadPoolExecutor(max_workers=max(1, parallelism)) as ex:
        proxy_outs = list(ex.map(_proxy_one, units))

    proxy_texts: list[str] = []
    for text, u in proxy_outs:
        proxy_texts.append(text)
        merge_usage(usage, u)

    proxy_scores = [proxy_keep_score(t, parse_verdict) for t in proxy_texts]

    # --- 2) Learn thresholds if needed ---
    ca = cascade_args.to_cascade_args()
    if cascade_args.filter_pos_cascade_threshold is None:
        sample_indices, correction_factors = importance_sampling(proxy_scores, ca)
        sample_indices_list = [int(i) for i in sample_indices]
        sample_proxy = [proxy_scores[i] for i in sample_indices_list]
        sample_correction = correction_factors[sample_indices]

        def _oracle_one(u: "Unit") -> tuple[bool, dict[str, int]]:
            from .pipeline import _op_user_content

            res = run_agent(
                completer,
                tools,
                system_prompt=system,
                user_content=_op_user_content(FILTER, instruction, [u], context, batched=False),
                max_steps=max_steps,
            )
            return parse_verdict(res.output), res.usage

        sample_units = [units[i] for i in sample_indices_list]
        with ThreadPoolExecutor(max_workers=max(1, parallelism)) as ex:
            oracle_raw = list(ex.map(_oracle_one, sample_units))
        oracle_keeps: list[bool] = []
        for keep, u in oracle_raw:
            oracle_keeps.append(keep)
            merge_usage(usage, u)
            stats.oracle_calls += 1

        (tau_pos, tau_neg), _ = learn_cascade_thresholds(
            sample_proxy,
            oracle_keeps,
            np.asarray(sample_correction, dtype=np.float64),
            ca,
        )
        cascade_args.filter_pos_cascade_threshold = float(tau_pos)
        cascade_args.filter_neg_cascade_threshold = float(tau_neg)

    tau_pos = float(cascade_args.filter_pos_cascade_threshold or 1.0)
    tau_neg = float(cascade_args.filter_neg_cascade_threshold or 0.0)
    stats.pos_threshold = tau_pos
    stats.neg_threshold = tau_neg

    # --- 3) Accept or escalate ---
    kept: list["Unit"] = []
    escalate_idx = [
        i for i, s in enumerate(proxy_scores) if not (s >= tau_pos or s <= tau_neg)
    ]

    for i, (u, score, text) in enumerate(zip(units, proxy_scores, proxy_texts)):
        if score >= tau_pos:
            kept.append(u)
            stats.accepted_by_proxy += 1
            stats.details.append({"unit_id": u.id, "path": "proxy_keep", "score": score})
        elif score <= tau_neg:
            stats.accepted_by_proxy += 1
            stats.details.append({"unit_id": u.id, "path": "proxy_drop", "score": score})
        else:
            stats.details.append({"unit_id": u.id, "path": "escalate", "score": score, "proxy": text[:200]})

    if escalate_idx:
        from .pipeline import _op_user_content

        def _oracle_esc(u: "Unit") -> tuple["Unit", bool, dict[str, int]]:
            res = run_agent(
                completer,
                tools,
                system_prompt=system,
                user_content=_op_user_content(FILTER, instruction, [u], context, batched=False),
                max_steps=max_steps,
            )
            return u, parse_verdict(res.output), res.usage

        esc_units = [units[i] for i in escalate_idx]
        with ThreadPoolExecutor(max_workers=max(1, parallelism)) as ex:
            esc_outs = list(ex.map(_oracle_esc, esc_units))
        for u, keep, u_usage in esc_outs:
            merge_usage(usage, u_usage)
            stats.oracle_calls += 1
            if keep:
                kept.append(u)

    # Preserve original corpus order among kept units.
    kept_ids = {u.id for u in kept}
    ordered = [u for u in units if u.id in kept_ids]
    return Corpus(ordered), stats
