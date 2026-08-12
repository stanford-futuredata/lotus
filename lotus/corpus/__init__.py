"""The ``Corpus`` abstraction — the high-level input to agentic map-reduce.

A corpus is any body of work to process, normalized into a stream of ``Unit``s that can
be sharded into bounded batches for parallel agentic mapping. Loaders cover several
input forms; more can be added without touching the pipeline.

    corpus = Corpus.from_files("repo/**/*.py")
    result = corpus.agent(task="Find every use of foo() and rank by risk.", ops=["map", "reduce"])
"""

from __future__ import annotations

import glob as _glob
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    import pandas as pd

    from lotus.agentic.pipeline import Result


@dataclass
class Unit:
    """One atomic segment of a corpus."""

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Corpus:
    """A body of work, as a list of :class:`Unit`s, that can be sharded."""

    def __init__(self, units: Sequence[Unit]):
        self.units: list[Unit] = list(units)

    def __len__(self) -> int:
        return len(self.units)

    # ------------------------------------------------------------------ loaders
    @classmethod
    def from_documents(cls, docs: Sequence[str], ids: Sequence[str] | None = None) -> "Corpus":
        ids = list(ids) if ids is not None else [str(i) for i in range(len(docs))]
        return cls([Unit(id=i, content=d) for i, d in zip(ids, docs)])

    @classmethod
    def from_dataframe(cls, df: "pd.DataFrame", content_cols: Sequence[str] | None = None) -> "Corpus":
        cols = list(content_cols) if content_cols is not None else list(df.columns)
        units = []
        for i, (_, row) in enumerate(df.iterrows()):
            content = "\n".join(f"{c}: {row[c]}" for c in cols)
            units.append(Unit(id=str(i), content=content, metadata={"row": i}))
        return cls(units)

    @classmethod
    def from_files(cls, pattern: str, encoding: str = "utf-8", recursive: bool = True) -> "Corpus":
        """Load each file matching a glob as a unit (id = path, content = file text)."""
        paths = sorted(p for p in _glob.glob(pattern, recursive=recursive) if os.path.isfile(p))
        units = []
        for path in paths:
            try:
                with open(path, encoding=encoding, errors="replace") as f:
                    content = f.read()
            except OSError as e:
                content = f"<unreadable: {e}>"
            units.append(Unit(id=path, content=content, metadata={"path": path}))
        return cls(units)

    @classmethod
    def from_text(cls, text: str, chunk_chars: int = 4000) -> "Corpus":
        """Split one large document into fixed-size chunks."""
        chunks = [text[i : i + chunk_chars] for i in range(0, len(text), chunk_chars)] or [""]
        return cls([Unit(id=str(i), content=c, metadata={"chunk": i}) for i, c in enumerate(chunks)])

    # ------------------------------------------------------------------ sharding
    def sample(self, n: int = 3) -> list[Unit]:
        return self.units[:n]

    def shard(self, shard_size: int | None = 1) -> list[list[Unit]]:
        """Group units into bounded batches (the shard step)."""
        size = max(1, shard_size or 1)
        return [self.units[i : i + size] for i in range(0, len(self.units), size)] or [[]]

    # ------------------------------------------------------------------ pipeline
    def agent(self, task: str, *, ops: "str | list[str] | None" = None, **kwargs: Any) -> "Result":
        """Run an ordered pipeline of agent ops over this corpus.

        ``ops`` is a list of op names — ``"map"``, ``"filter"``, ``"reduce"`` — run in
        order (default ``["map", "reduce"]``). ``map``/``filter`` are Corpus -> Corpus
        (chainable); ``reduce`` collapses the corpus to one answer and must be last.
        See ``pipeline.run_pipeline``.

        The returned ``Result`` retains a flattened agent event log in ``events``
        (tool calls + finals per shard/op); use ``result.events_dataframe()`` for a table.
        """
        from lotus.agentic.pipeline import run_pipeline

        return run_pipeline(self, task, ops=ops, **kwargs)


__all__ = ["Unit", "Corpus"]
