"""Semantic lineage: claim provenance over agent event streams.

``sem_lineage`` judges whether each claim is supported by a scoped set of agent
events (tool calls / observations / finals) and returns the supporting event
ids, evidence texts, and an ordered lineage path.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

import lotus
from lotus.cache import operator_cache
from lotus.models import LM
from lotus.sem_ops.postprocessors import lineage_postprocess
from lotus.templates import task_instructions
from lotus.types import ReasoningStrategy, SemanticLineageOutput
from lotus.utils import show_safe_mode


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return default


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        # Try JSON list first; otherwise treat as comma-separated.
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed]
            except json.JSONDecodeError:
                pass
        return [part.strip() for part in text.split(",") if part.strip()]
    return [str(value)]


def _format_events_blob(events: pd.DataFrame, event_id_col: str, event_cols: list[str]) -> str:
    lines: list[str] = []
    for _, row in events.iterrows():
        eid = str(row[event_id_col])
        parts = [f"event_id={eid}"]
        for col in event_cols:
            if col == event_id_col or col not in events.columns:
                continue
            parts.append(f"{col}={row[col]}")
        lines.append("- " + " | ".join(parts))
    return "\n".join(lines) if lines else "(no events)"


def events_from_agent_traces(
    traces: list[list[dict[str, Any]]] | pd.Series,
    *,
    session_ids: list[Any] | pd.Series | None = None,
    unit_ids: list[Any] | pd.Series | None = None,
) -> pd.DataFrame:
    """Flatten LOTUS ``AgentResult.trace`` lists into an events DataFrame.

    Each tool-call dict in a trace is expected to look like::

        {"tool": name, "arguments": {...}, "result": ...}

    Returns columns: ``event_id``, ``session_id``, ``unit_id``, ``step``,
    ``event_type``, ``tool``, ``arguments``, ``result``.
    """
    if isinstance(traces, pd.Series):
        trace_list = traces.tolist()
    else:
        trace_list = list(traces)

    if session_ids is None:
        session_vals: list[Any] = list(range(len(trace_list)))
    elif isinstance(session_ids, pd.Series):
        session_vals = session_ids.tolist()
    else:
        session_vals = list(session_ids)

    if unit_ids is None:
        unit_vals: list[Any] = session_vals
    elif isinstance(unit_ids, pd.Series):
        unit_vals = unit_ids.tolist()
    else:
        unit_vals = list(unit_ids)

    rows: list[dict[str, Any]] = []
    for i, trace in enumerate(trace_list):
        sid = session_vals[i] if i < len(session_vals) else i
        uid = unit_vals[i] if i < len(unit_vals) else sid
        if not isinstance(trace, list):
            continue
        for step, event in enumerate(trace):
            if not isinstance(event, dict):
                continue
            rows.append(
                {
                    "event_id": f"{sid}:{step}",
                    "session_id": sid,
                    "unit_id": uid,
                    "step": step,
                    "event_type": "tool_call",
                    "tool": event.get("tool"),
                    "arguments": json.dumps(event.get("arguments", {}), default=str)
                    if not isinstance(event.get("arguments"), str)
                    else event.get("arguments"),
                    "result": event.get("result"),
                }
            )
    return pd.DataFrame(rows)


def sem_lineage(
    claims: list[str],
    event_groups: list[pd.DataFrame],
    model: LM,
    *,
    instruction: str | None = None,
    event_id_col: str = "event_id",
    event_text_cols: list[str] | None = None,
    max_evidence: int = 5,
    safe_mode: bool = False,
    progress_bar_desc: str = "Tracing lineage",
    return_explanations: bool = False,
    strategy: ReasoningStrategy | None = None,
) -> SemanticLineageOutput:
    """Run claim-provenance judgments against per-claim event groups.

    Args:
        claims: Claim texts, one per row.
        event_groups: Parallel list of event DataFrames scoped to each claim.
        model: Configured language model.
        instruction: Optional override for the provenance judging task.
        event_id_col: Column holding stable event identifiers.
        event_text_cols: Columns to show the model for each event. Defaults to
            common agent-trace fields when present.
        max_evidence: Cap on returned evidence event ids.
        safe_mode: Estimate token cost before calling the model.
        progress_bar_desc: Progress bar label.
        return_explanations: Unused directly; rationale is always parsed when present.
        strategy: Optional CoT reasoning strategy.
    """
    del return_explanations  # rationale is always extracted when present in JSON

    default_cols = ["event_type", "tool", "arguments", "result", "content", "step"]
    text_cols = event_text_cols or default_cols

    inputs = []
    for claim, events in zip(claims, event_groups):
        present_cols = [c for c in text_cols if c in events.columns]
        blob = _format_events_blob(events, event_id_col, present_cols)
        prompt = task_instructions.lineage_formatter(
            model,
            claim_text=claim,
            events_blob=blob,
            instruction=instruction,
            max_evidence=max_evidence,
            strategy=strategy,
        )
        lotus.logger.debug(f"lineage prompt: {prompt}")
        inputs.append(prompt)

    if safe_mode:
        estimated_cost = sum(model.count_tokens(inp) for inp in inputs)
        show_safe_mode(estimated_cost, len(claims))

    cot = strategy in (ReasoningStrategy.COT, ReasoningStrategy.ZS_COT)
    if cot:
        lm_output = model(inputs, progress_bar_desc=progress_bar_desc)
    else:
        lm_output = model(
            inputs,
            response_format={"type": "json_object"},
            progress_bar_desc=progress_bar_desc,
        )

    parsed, explanations = lineage_postprocess(lm_output.outputs, model, cot_reasoning=cot)

    supported: list[bool] = []
    evidence_ids: list[list[str]] = []
    evidence_texts: list[list[str]] = []
    lineage_paths: list[list[str]] = []

    for claim_events, obj in zip(event_groups, parsed):
        ids = _as_str_list(obj.get("evidence_event_ids"))[:max_evidence]
        # Keep only ids that exist in the scoped events.
        if event_id_col in claim_events.columns and len(claim_events):
            valid = set(claim_events[event_id_col].astype(str))
            ids = [eid for eid in ids if eid in valid]
        path = _as_str_list(obj.get("lineage_path"))
        if not path:
            path = ids

        texts: list[str] = []
        if event_id_col in claim_events.columns and ids:
            id_to_row = {str(r[event_id_col]): r for _, r in claim_events.iterrows()}
            for eid in ids:
                row = id_to_row.get(eid)
                if row is None:
                    continue
                present_cols = [c for c in text_cols if c in claim_events.columns]
                texts.append(" | ".join(f"{c}={row[c]}" for c in present_cols))

        supported.append(_as_bool(obj.get("supported"), default=bool(ids)))
        evidence_ids.append(ids)
        evidence_texts.append(texts)
        lineage_paths.append(path)

    if safe_mode:
        model.print_total_usage()

    return SemanticLineageOutput(
        supported=supported,
        evidence_event_ids=evidence_ids,
        evidence_texts=evidence_texts,
        lineage_paths=lineage_paths,
        raw_outputs=lm_output.outputs,
        explanations=explanations,
        stats={"n_claims": len(claims), "n_with_evidence": sum(1 for e in evidence_ids if e)},
    )


@pd.api.extensions.register_dataframe_accessor("sem_lineage")
class SemLineageDataFrame:
    """Attach claim provenance from an events DataFrame onto claim rows.

    Example::

        claim_df.sem_lineage(
            events_df,
            claim_col="claim",
            session_col="session_id",
        )
    """

    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj

    @operator_cache
    def __call__(
        self,
        events: pd.DataFrame,
        claim_col: str = "claim",
        event_id_col: str = "event_id",
        event_text_cols: list[str] | None = None,
        session_col: str | None = "session_id",
        *,
        instruction: str | None = None,
        max_evidence: int = 5,
        return_explanations: bool = False,
        return_raw_outputs: bool = False,
        safe_mode: bool = False,
        progress_bar_desc: str = "Tracing lineage",
        strategy: ReasoningStrategy | None = None,
        suffix: str = "",
    ) -> pd.DataFrame:
        if lotus.settings.lm is None:
            raise ValueError(
                "The language model must be an instance of LM. "
                "Please configure a valid language model using lotus.settings.configure()"
            )

        if claim_col not in self._obj.columns:
            raise ValueError(f"Column {claim_col} not found in claim DataFrame")
        if event_id_col not in events.columns:
            raise ValueError(f"Column {event_id_col} not found in events DataFrame")

        claims = self._obj[claim_col].astype(str).tolist()
        event_groups: list[pd.DataFrame] = []

        use_session = (
            session_col is not None and session_col in self._obj.columns and session_col in events.columns
        )
        for _, claim_row in self._obj.iterrows():
            if use_session:
                sid = claim_row[session_col]
                scoped = events[events[session_col] == sid]
            else:
                scoped = events
            event_groups.append(scoped.reset_index(drop=True))

        out = sem_lineage(
            claims=claims,
            event_groups=event_groups,
            model=lotus.settings.lm,
            instruction=instruction,
            event_id_col=event_id_col,
            event_text_cols=event_text_cols,
            max_evidence=max_evidence,
            safe_mode=safe_mode,
            progress_bar_desc=progress_bar_desc,
            strategy=strategy,
        )

        new_df = self._obj.copy()
        supported_col = f"supported{suffix}"
        evidence_ids_col = f"evidence_event_ids{suffix}"
        evidence_texts_col = f"evidence_texts{suffix}"
        lineage_path_col = f"lineage_path{suffix}"

        new_df[supported_col] = out.supported
        new_df[evidence_ids_col] = out.evidence_event_ids
        new_df[evidence_texts_col] = out.evidence_texts
        new_df[lineage_path_col] = out.lineage_paths

        if return_raw_outputs:
            new_df[f"raw_output{suffix}"] = out.raw_outputs
        if return_explanations:
            new_df[f"explanation{suffix}"] = out.explanations

        return new_df
