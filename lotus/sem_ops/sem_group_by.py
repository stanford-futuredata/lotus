"""Semantic group-by: discover / assign natural-language group labels.

Implements the paper operator ``sem_group_by(langex, C=n)`` (arXiv:2407.11418):
group rows by a natural-language criterion, optionally discovering ``n`` labels
or classifying into user-provided ``labels``.

Gold path (when ``n`` is set and no ``labels``):
  1. Cluster (embedding k-means) or sample documents to propose candidate groups
  2. Ask the LM for a short label per group
  3. Assign every row to one of the labels (by cluster membership, or LM reassignment)

Classification path (when ``labels`` is set): skip discovery and map each row to
the best matching label.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

import lotus
from lotus.cache import operator_cache
from lotus.templates import task_instructions
from lotus.types import LMOutput, SemanticMapOutput


def _parse_label_list(text: str, expected: int | None = None) -> list[str]:
    """Extract a list of group labels from an LM response."""
    text = text.strip()
    # Prefer a JSON array anywhere in the response.
    match = re.search(r"\[.*?\]", text, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                labels = [x.strip() for x in parsed if x.strip()]
                if labels:
                    return labels[:expected] if expected else labels
        except json.JSONDecodeError:
            pass

    labels: list[str] = []
    for line in text.splitlines():
        line = line.strip().strip("-*").strip()
        line = re.sub(r"^\d+[.)]\s*", "", line)
        if line:
            labels.append(line)
    if expected is not None and len(labels) >= expected:
        return labels[:expected]
    return labels


def _match_label(raw: str, labels: list[str], default: str | None = None) -> str:
    """Map a free-form LM answer onto one of ``labels``."""
    cleaned = raw.strip().strip('"').strip("'")
    # Drop "Answer:" prefix if present.
    if "Answer:" in cleaned:
        cleaned = cleaned.split("Answer:")[-1].strip()

    by_lower = {lab.lower(): lab for lab in labels}
    if cleaned.lower() in by_lower:
        return by_lower[cleaned.lower()]

    for lab in labels:
        if lab.lower() in cleaned.lower() or cleaned.lower() in lab.lower():
            return lab

    return default if default is not None else (labels[0] if labels else cleaned)


def discover_labels_formatter(
    grouping_instruction: str,
    sample_texts: list[str],
    n: int,
) -> list[dict[str, str]]:
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(sample_texts))
    return [
        {
            "role": "system",
            "content": (
                "You invent short, distinct group labels for documents.\n"
                f"Propose exactly {n} labels that cover the samples under the given "
                "grouping criteria. Reply with a JSON array of strings only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Grouping criteria:\n{grouping_instruction}\n\n"
                f"Sample documents:\n{numbered}\n\n"
                f"Return a JSON array with exactly {n} short label strings."
            ),
        },
    ]


def classify_label_formatter(
    multimodal_data: dict[str, Any],
    grouping_instruction: str,
    labels: list[str],
) -> list[dict[str, Any]]:
    label_list = ", ".join(f'"{lab}"' for lab in labels)
    return [
        {
            "role": "system",
            "content": (
                "Assign the document to exactly one group label from the provided list.\n"
                'Reply with only the chosen label (optionally as "Answer: <label>").'
            ),
        },
        task_instructions.user_message_formatter(
            multimodal_data,
            (
                f"Grouping criteria: {grouping_instruction}\n"
                f"Allowed labels: [{label_list}]\n"
                "Which single label best fits this document?"
            ),
        ),
    ]


def _discover_labels_from_samples(
    model: lotus.models.LM,
    grouping_instruction: str,
    sample_texts: list[str],
    n: int,
) -> list[str]:
    prompt = discover_labels_formatter(grouping_instruction, sample_texts, n)
    lm_output: LMOutput = model([prompt], progress_bar_desc="Discovering group labels")
    labels = _parse_label_list(lm_output.outputs[0], expected=n)
    if len(labels) < n:
        # Pad with generic labels so assignment can proceed.
        labels.extend(f"Group {i + 1}" for i in range(len(labels), n))
    return labels[:n]


def _classify_rows(
    docs: list[dict[str, Any]],
    model: lotus.models.LM,
    grouping_instruction: str,
    labels: list[str],
    progress_bar_desc: str = "Assigning groups",
) -> SemanticMapOutput:
    inputs = [classify_label_formatter(doc, grouping_instruction, labels) for doc in docs]
    lm_output: LMOutput = model(inputs, progress_bar_desc=progress_bar_desc)
    outputs = [_match_label(raw, labels) for raw in lm_output.outputs]
    return SemanticMapOutput(raw_outputs=lm_output.outputs, outputs=outputs, explanations=[None] * len(outputs))


def _sample_texts_for_discovery(
    df: pd.DataFrame,
    col_li: list[str],
    n: int,
    samples_per_group: int,
    cluster_ids: list[int] | None,
) -> list[str]:
    """Pick representative texts: per-cluster samples when clustered, else a global sample."""
    multimodal = task_instructions.df2multimodal_info(df, col_li)
    texts = [row["text"] for row in multimodal]

    if cluster_ids is not None:
        by_c: dict[int, list[str]] = {}
        for cid, text in zip(cluster_ids, texts):
            by_c.setdefault(int(cid), []).append(text)
        samples: list[str] = []
        for cid in sorted(by_c):
            samples.extend(by_c[cid][:samples_per_group])
        return samples

    # Uniform sample across the frame (cap to keep the discovery prompt short).
    cap = max(n * samples_per_group, n)
    if len(texts) <= cap:
        return texts
    step = max(1, len(texts) // cap)
    return texts[::step][:cap]


def sem_group_by(
    docs: list[dict[str, Any]],
    model: lotus.models.LM,
    user_instruction: str,
    labels: list[str],
    progress_bar_desc: str = "Assigning groups",
) -> SemanticMapOutput:
    """Classify each document into one of ``labels`` under ``user_instruction``."""
    return _classify_rows(docs, model, user_instruction, labels, progress_bar_desc=progress_bar_desc)


@pd.api.extensions.register_dataframe_accessor("sem_group_by")
class SemGroupByDataframe:
    """
    Group rows by a natural-language criterion (paper ``sem_group_by``).

    Args:
        user_instruction: Langex describing how to group rows (may reference
            columns with ``{col}``).
        n: Target number of groups to discover when ``labels`` is omitted.
        labels: Optional fixed label set (classification / supervised grouping).
        cluster_col: Column to embed-cluster when discovering labels (defaults to
            the first column referenced in ``user_instruction``). Requires a
            semantic index on that column when clustering is used.
        use_clustering: If True (default) and ``n`` is set without ``labels``,
            discover labels via embedding clusters. Falls back to sampling when
            no retrieval model / index is available.
        reassign: If True, after discovering labels, LM-classify every row into
            those labels (paper gold stage 2). If False, map each cluster id to
            its discovered label (cheaper).
        samples_per_group: Max docs shown to the LM per cluster when naming groups.
        suffix: Output column for the assigned label (default ``"_group"``).
        group_id_suffix: Optional column for numeric group ids (default
            ``"_group_id"``); set to ``None`` to omit.
        safe_mode: Reserved; currently unused.
        progress_bar_desc: Progress bar label for classification calls.

    Returns:
        DataFrame with original columns plus the group label column.

    Example:
        >>> df.sem_group_by("the topic of each {paper}", n=5)
        >>> df.sem_group_by("the topic of each {paper}", labels=["DB", "IR", "Security"])
    """

    def __init__(self, pandas_obj: Any) -> None:
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: Any) -> None:
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    @operator_cache
    def __call__(
        self,
        user_instruction: str,
        n: int | None = None,
        labels: list[str] | None = None,
        cluster_col: str | None = None,
        use_clustering: bool = True,
        reassign: bool = True,
        samples_per_group: int = 3,
        suffix: str = "_group",
        group_id_suffix: str | None = "_group_id",
        safe_mode: bool = False,
        progress_bar_desc: str = "Assigning groups",
        **model_kwargs: Any,
    ) -> pd.DataFrame:
        del safe_mode, model_kwargs  # reserved for parity with other ops
        if lotus.settings.lm is None:
            raise ValueError(
                "The language model must be an instance of LM. "
                "Please configure a valid language model using lotus.settings.configure()"
            )

        if labels is None and n is None:
            raise ValueError("Provide either labels=... or n=... (target number of groups)")
        if labels is not None and len(labels) == 0:
            raise ValueError("labels must be a non-empty list when provided")
        if n is not None and n < 1:
            raise ValueError("n must be >= 1")

        col_li = lotus.nl_expression.parse_cols(user_instruction)
        for column in col_li:
            if column not in self._obj.columns:
                raise ValueError(f"Column {column} not found in DataFrame")
        if not col_li:
            raise ValueError("user_instruction must reference at least one column with {col}")

        formatted_instr = lotus.nl_expression.nle2str(user_instruction, col_li)
        multimodal_data = task_instructions.df2multimodal_info(self._obj, col_li)
        model = lotus.settings.lm

        discovered = list(labels) if labels is not None else None
        cluster_ids: list[int] | None = None

        if discovered is None:
            assert n is not None
            if len(self._obj) == 0:
                out = self._obj.copy()
                out[suffix] = pd.Series(dtype=str)
                if group_id_suffix:
                    out[group_id_suffix] = pd.Series(dtype=int)
                return out

            cluster_col = cluster_col or col_li[0]
            if use_clustering:
                try:
                    clustered = self._obj.sem_cluster_by(cluster_col, n)
                    cluster_ids = clustered["cluster_id"].astype(int).tolist()
                except (ValueError, AttributeError, KeyError) as exc:
                    lotus.logger.warning(
                        "sem_group_by: clustering unavailable (%s); falling back to sampling.",
                        exc,
                    )
                    cluster_ids = None

            sample_texts = _sample_texts_for_discovery(
                self._obj, col_li, n, samples_per_group, cluster_ids
            )
            discovered = _discover_labels_from_samples(model, formatted_instr, sample_texts, n)

            # Prefer one label per cluster when we clustered successfully.
            if cluster_ids is not None and not reassign:
                # Re-discover with one naming call that sees per-cluster samples —
                # already done; map cluster id -> label by sorted unique ids.
                unique_cids = sorted(set(cluster_ids))
                # If we somehow have fewer labels, pad; if more, truncate.
                while len(discovered) < len(unique_cids):
                    discovered.append(f"Group {len(discovered) + 1}")
                cid_to_label = {cid: discovered[i] for i, cid in enumerate(unique_cids)}
                assigned = [cid_to_label[cid] for cid in cluster_ids]
                new_df = self._obj.copy()
                new_df[suffix] = assigned
                if group_id_suffix:
                    label_to_id = {lab: i for i, lab in enumerate(discovered[: len(unique_cids)])}
                    new_df[group_id_suffix] = [label_to_id[lab] for lab in assigned]
                new_df.attrs["group_labels"] = discovered[: len(unique_cids)]
                return new_df

        assert discovered is not None
        output = sem_group_by(
            multimodal_data,
            model,
            formatted_instr,
            discovered,
            progress_bar_desc=progress_bar_desc,
        )

        new_df = self._obj.copy()
        new_df[suffix] = output.outputs
        if group_id_suffix:
            label_to_id = {lab: i for i, lab in enumerate(discovered)}
            new_df[group_id_suffix] = [label_to_id.get(lab, -1) for lab in output.outputs]
        new_df.attrs["group_labels"] = discovered
        return new_df
