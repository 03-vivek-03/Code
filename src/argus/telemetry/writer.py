"""Trace persistence as JSONL.

JSONL rather than a database or a framework-specific format, for one reason: the traces
are a released research artefact. Anyone should be able to read them with the standard
library, and the analysis must survive any library churn in the agent framework.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from argus.telemetry.spans import Trace


class TraceWriter:
    """Append-only JSONL writer, with optional gzip.

    Flushes on every write. A grid run that dies at 80 percent should keep the 80
    percent already produced, otherwise a rate limit costs you the whole run.
    """

    def __init__(self, path: str | Path, compress: bool = False, append: bool = True) -> None:
        self.path = Path(path)
        if compress and self.path.suffix != ".gz":
            self.path = self.path.with_suffix(self.path.suffix + ".gz")
        self.compress = compress or self.path.suffix == ".gz"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "at" if append else "wt"
        self._fh = (
            gzip.open(self.path, mode, encoding="utf-8")
            if self.compress
            else open(self.path, mode, encoding="utf-8")
        )
        self._n = 0

    def write(self, trace: Trace) -> None:
        self._fh.write(trace.to_json() + "\n")
        self._fh.flush()
        self._n += 1

    def write_many(self, traces: list[Trace]) -> None:
        for t in traces:
            self.write(t)

    @property
    def count(self) -> int:
        return self._n

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class TraceReader:
    """Streaming JSONL reader.

    Iterating rather than loading everything keeps memory flat, which matters once the
    corpus of traces reaches tens of thousands of runs.

    Corrupt lines are skipped rather than raised, and counted in :attr:`n_corrupt`. A
    run killed mid-write leaves a truncated final line, and one such line in the first
    grid made every downstream tool that touched that file raise `JSONDecodeError`. The
    analysis should report the damage, not refuse to run; the *runner* is where an
    incomplete cell gets re-executed.
    """

    def __init__(self, path: str | Path, strict: bool = False) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"no trace file at {self.path}")
        self.compress = self.path.suffix == ".gz"
        self.strict = strict
        self.n_corrupt = 0

    def _open(self):
        return (
            gzip.open(self.path, "rt", encoding="utf-8")
            if self.compress
            else open(self.path, encoding="utf-8")
        )

    def _records(self) -> Iterator[dict[str, Any]]:
        self.n_corrupt = 0
        with self._open() as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    self.n_corrupt += 1
                    if self.strict:
                        raise ValueError(
                            f"{self.path.name} line {lineno} is not valid JSON: {exc}"
                        ) from exc

    def __iter__(self) -> Iterator[Trace]:
        for rec in self._records():
            yield Trace.from_dict(rec)

    def read_all(self) -> list[Trace]:
        return list(self)

    def read_dicts(self) -> Iterator[dict[str, Any]]:
        """Raw dicts, including the precomputed `outcomes` block."""
        return self._records()

    def count(self) -> int:
        return sum(1 for _ in self._records())

    @staticmethod
    def find(directory: str | Path, pattern: str = "*.jsonl*") -> list[Path]:
        return sorted(Path(directory).glob(pattern))

    @classmethod
    def read_dir(cls, directory: str | Path, pattern: str = "*.jsonl*") -> Iterator[Trace]:
        for path in cls.find(directory, pattern):
            yield from cls(path)
