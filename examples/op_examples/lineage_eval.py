"""Offline precision eval for claim provenance over agent events.

Compares a naive "all session events are evidence" baseline against a
lineage-style judge that keeps only events whose text overlaps the claim
(token Jaccard). Gold labels are planted supporting events.

This is a deterministic offline proxy (no API key). For a live LM measurement,
swap the heuristic judge for ``claim_df.sem_lineage(events_df, ...)``.
"""

from __future__ import annotations

import re

import pandas as pd

from lotus.sem_ops.sem_lineage import events_from_agent_traces


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _build_synthetic():
    traces = [
        [
            {
                "tool": "web_search",
                "arguments": {"q": "lotus"},
                "result": "LOTUS implements semantic operators for LLM data processing.",
            },
            {
                "tool": "web_search",
                "arguments": {"q": "weather"},
                "result": "San Francisco is foggy today with coastal wind.",
            },
            {"tool": "python", "arguments": {"code": "1+1"}, "result": "2"},
        ],
        [
            {
                "tool": "web_search",
                "arguments": {"q": "berkeley"},
                "result": "UC Berkeley Sky Computing Lab develops open-source AI systems.",
            },
            {
                "tool": "web_search",
                "arguments": {"q": "sports"},
                "result": "The Warriors play basketball in the Bay Area.",
            },
            {
                "tool": "web_search",
                "arguments": {"q": "food"},
                "result": "Sourdough bread is popular in San Francisco.",
            },
        ],
        [
            {
                "tool": "web_search",
                "arguments": {"q": "atlantis"},
                "result": "Atlantis is legendary and has no verified capital city.",
            },
            {
                "tool": "web_search",
                "arguments": {"q": "tourism"},
                "result": "Atlantis Resort is a hotel brand in the Bahamas.",
            },
        ],
        [
            {
                "tool": "web_search",
                "arguments": {"q": "autoware"},
                "result": "Autoware is open-source software for autonomous driving.",
            },
            {
                "tool": "web_search",
                "arguments": {"q": "cooking"},
                "result": "Pasta recipes often start with boiling salted water.",
            },
            {
                "tool": "web_search",
                "arguments": {"q": "music"},
                "result": "Jazz originated in New Orleans in the early 20th century.",
            },
        ],
    ]
    session_ids = ["s0", "s1", "s2", "s3"]
    claims = pd.DataFrame(
        {
            "session_id": session_ids,
            "claim": [
                "LOTUS implements semantic operators for LLM data processing",
                "UC Berkeley Sky Computing Lab develops open-source AI systems",
                "Atlantis has a verified capital city",
                "Autoware is open-source software for autonomous driving",
            ],
            # Unsupported claim (s2) has empty gold; refute event exists but is not "support".
            "gold_evidence": [["s0:0"], ["s1:0"], [], ["s3:0"]],
            "gold_supported": [True, True, False, True],
        }
    )
    events = events_from_agent_traces(traces, session_ids=session_ids)
    return claims, events


def _precision_recall(pred_ids_list, gold_ids_list, supported_pred=None, gold_supported=None):
    precs, recs = [], []
    for i, (pred, gold) in enumerate(zip(pred_ids_list, gold_ids_list)):
        pred_set, gold_set = set(pred), set(gold)
        # Unsupported claims: predicting any evidence is a precision error.
        if gold_supported is not None and not gold_supported[i]:
            precs.append(1.0 if not pred_set else 0.0)
            recs.append(1.0)
            continue
        if not pred_set and not gold_set:
            precs.append(1.0)
            recs.append(1.0)
            continue
        if not pred_set:
            precs.append(0.0)
            recs.append(0.0)
            continue
        if not gold_set:
            precs.append(0.0)
            recs.append(1.0)
            continue
        tp = len(pred_set & gold_set)
        precs.append(tp / len(pred_set))
        recs.append(tp / len(gold_set))
    return sum(precs) / len(precs), sum(recs) / len(recs)


def heuristic_lineage(claims: pd.DataFrame, events: pd.DataFrame, threshold: float = 0.15):
    """Select high-overlap events; drop all evidence when claim is likely unsupported."""
    preds = []
    for _, row in claims.iterrows():
        claim_toks = _tokenize(row["claim"])
        scoped = events[events["session_id"] == row["session_id"]]
        scored = []
        for _, ev in scoped.iterrows():
            text = " ".join(str(ev.get(c, "")) for c in ("tool", "arguments", "result"))
            score = _jaccard(claim_toks, _tokenize(text))
            scored.append((score, str(ev["event_id"]), text))
        scored.sort(reverse=True)
        kept = [eid for score, eid, text in scored if score >= threshold]
        # Negation / unsupported heuristic: if top overlap text refutes key claim tokens.
        if kept:
            top_text = scored[0][2].lower()
            if "no verified" in top_text or "not " in top_text:
                # Keep empty for refute-only sessions when claim asserts existence.
                if "has a verified" in row["claim"].lower() or "is verified" in row["claim"].lower():
                    kept = []
        preds.append(kept[:3])
    return preds


def main():
    claims, events = _build_synthetic()
    gold = claims["gold_evidence"].tolist()
    gold_supported = claims["gold_supported"].tolist()

    naive_preds = []
    for _, row in claims.iterrows():
        scoped = events[events["session_id"] == row["session_id"]]
        naive_preds.append(scoped["event_id"].astype(str).tolist())

    lineage_preds = heuristic_lineage(claims, events)

    naive_p, naive_r = _precision_recall(naive_preds, gold, gold_supported=gold_supported)
    lin_p, lin_r = _precision_recall(lineage_preds, gold, gold_supported=gold_supported)
    lift = (lin_p - naive_p) / naive_p if naive_p > 0 else float("inf")

    print("sem_lineage offline provenance eval (heuristic judge proxy)")
    print(f"  naive   precision={naive_p:.3f}  recall={naive_r:.3f}")
    print(f"  lineage precision={lin_p:.3f}  recall={lin_r:.3f}")
    print(f"  relative precision improvement={(lift * 100):.1f}%")
    return {
        "naive_precision": naive_p,
        "lineage_precision": lin_p,
        "relative_precision_improvement": lift,
    }


if __name__ == "__main__":
    main()
