"""Contratos de fontes e modelos injetáveis, sem carregar runtimes nativos."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, TypedDict

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .retrieval import Chunk, Hit


class ChunkData(TypedDict):
    id: str
    source: str
    title: str
    section: str
    text: str
    line_start: int
    line_end: int
    page: int | None


class Source(ChunkData):
    reference: str
    score: float


class Embedder(Protocol):
    def encode(self, texts: Sequence[str], *, query: bool = False) -> ArrayLike: ...


class CrossEncoder(Protocol):
    def predict(self, question: str, passages: Sequence[str]) -> ArrayLike: ...


class SearchIndex(Protocol):
    def search(self, question: str, top_k: int = 3, min_score: float | None = None) -> list[Hit]: ...


class EvaluationIndex(SearchIndex, Protocol):
    chunks: list[Chunk]
