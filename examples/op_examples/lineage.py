"""Example: claim provenance over agent event streams with sem_lineage."""

import pandas as pd

import lotus
from lotus.models import LM
from lotus.sem_ops.sem_lineage import events_from_agent_traces

# Requires OPENAI_API_KEY (or configure another LiteLLM model).
lm = LM(model="gpt-4o-mini")
lotus.settings.configure(lm=lm)

# Simulated agent traces (same shape as AgentResult.trace from lotus.agentic.loop).
traces = [
    [
        {
            "tool": "web_search",
            "arguments": {"query": "What is LOTUS semantic operators?"},
            "result": "LOTUS is a query engine that implements semantic operators over tables of unstructured data.",
        },
        {
            "tool": "python",
            "arguments": {"code": "print(2+2)"},
            "result": "4",
        },
    ],
    [
        {
            "tool": "web_search",
            "arguments": {"query": "capital of Atlantis"},
            "result": "Atlantis is a legendary island; it has no verified capital.",
        }
    ],
]

events_df = events_from_agent_traces(traces, session_ids=["agent-0", "agent-1"])

claims_df = pd.DataFrame(
    {
        "session_id": ["agent-0", "agent-1"],
        "claim": [
            "LOTUS implements semantic operators over unstructured data",
            "Atlantis has a verified capital city",
        ],
    }
)

result = claims_df.sem_lineage(
    events_df,
    claim_col="claim",
    session_col="session_id",
    return_explanations=True,
)
print(result[["claim", "supported", "evidence_event_ids", "lineage_path", "explanation"]])
