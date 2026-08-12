"""Example: semantic group-by with label discovery (paper sem_group_by).

Requires an LM. For the clustering-based discovery path, also configure an RM +
vector store and index the text column first.
"""

import pandas as pd

import lotus
from lotus.models import LM

lm = LM(model="gpt-4o-mini")
lotus.settings.configure(lm=lm)

df = pd.DataFrame(
    {
        "paper": [
            "We present a learned index for disk-based key-value stores.",
            "Dense retrieval with late interaction improves IR quality.",
            "A novel ransomware detection method using graph neural networks.",
            "Cost-based query optimization for cloud data warehouses.",
            "Contrastive learning for passage ranking in search engines.",
            "Side-channel attacks on trusted execution environments.",
        ]
    }
)

# Classification into known labels (no retrieval model required).
labeled = df.sem_group_by(
    "the research topic of each {paper}",
    labels=["Databases", "Information Retrieval", "Security"],
)
print("Fixed labels:\n", labeled[["paper", "_group"]])

# Unsupervised discovery of n=3 labels via sampling + LM assignment.
discovered = df.sem_group_by(
    "the research topic of each {paper}",
    n=3,
    use_clustering=False,
)
print("\nDiscovered labels:", discovered.attrs.get("group_labels"))
print(discovered[["paper", "_group"]])
