"""P2-04: vector store numpy + interface VectorStore + filtro por metadata (R-08, R-21 ponto de extensão)."""

from __future__ import annotations

import math
import time
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from app.domain import Chunk
from app.embeddings import HashEmbeddingProvider
from app.retrieval import VectorIndex, cosine_similarity
from app.store import NumpyVectorStore, VectorStore


def _chunks(n: int) -> list[Chunk]:
    return [
        Chunk(
            id=f"doc{i % 3}.pdf:p1:c{i}",
            text=f"texto {i}",
            source=f"doc{i % 3}.pdf",
            locator={"page": i % 2 + 1},
            section="A" if i % 2 else "B",
        )
        for i in range(n)
    ]


def _store(n: int = 6, d: int = 4, seed: int = 0) -> NumpyVectorStore:
    rng = np.random.default_rng(seed)
    return NumpyVectorStore(_chunks(n), rng.normal(size=(n, d)))


def test_store_matches_pure_python_cosine_and_orders_desc() -> None:
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(50, 16))
    store = _store(50, 16, seed=1)
    query = rng.normal(size=16)
    expected = [cosine_similarity(list(query), list(row)) for row in vectors]
    assert store.scores(query) == pytest.approx(expected, abs=1e-5)
    hits = store.search(query, k=5)
    assert len(hits) == 5
    assert [index for index, _ in hits] == list(np.argsort(-np.asarray(expected))[:5])
    assert all(a >= b for (_, a), (_, b) in pairwise(hits))


def test_store_validates_shapes_and_query_dimension() -> None:
    with pytest.raises(ValueError, match="pelo menos um chunk"):
        NumpyVectorStore([], np.zeros((0, 3)))
    with pytest.raises(RuntimeError, match="diferente da quantidade"):
        NumpyVectorStore(_chunks(2), np.zeros((3, 3)))
    with pytest.raises(RuntimeError, match="dimensão zero"):
        NumpyVectorStore(_chunks(1), np.zeros((1, 0)))
    store = _store(3, 4)
    with pytest.raises(RuntimeError, match="dimensão 5"):
        store.search([0.0] * 5, k=1)
    assert store.dimension == 4 and len(store) == 3


def test_zero_vectors_do_not_produce_nan() -> None:
    store = NumpyVectorStore(_chunks(2), np.array([[0.0, 0.0], [1.0, 0.0]]))
    scores = store.scores([0.0, 0.0])
    assert all(not math.isnan(value) for value in scores)
    assert store.search([1.0, 0.0], k=2)[0][0] == 1


def test_k_larger_than_corpus_and_k_zero() -> None:
    store = _store(3, 4)
    assert len(store.search([1.0, 0.0, 0.0, 0.0], k=10)) == 3
    assert len(store.search([1.0, 0.0, 0.0, 0.0], k=0)) == 1


def test_filter_by_source_section_locator_and_callable() -> None:
    store = _store(12, 4)
    query = [1.0, 0.0, 0.0, 0.0]
    by_source = store.search(query, k=12, filter={"source": "doc1.pdf"})
    assert by_source and all(store.chunks[index].source == "doc1.pdf" for index, _ in by_source)
    by_list = store.search(query, k=12, filter={"source": ["doc0.pdf", "doc2.pdf"]})
    assert {store.chunks[index].source for index, _ in by_list} == {"doc0.pdf", "doc2.pdf"}
    by_section_page = store.search(query, k=12, filter={"section": "A", "page": 2})
    assert by_section_page and all(
        store.chunks[i].section == "A" and store.chunks[i].locator["page"] == 2 for i, _ in by_section_page
    )
    by_callable = store.search(query, k=12, filter=lambda chunk: chunk.id.endswith("c5"))
    assert [store.chunks[index].id for index, _ in by_callable] == ["doc2.pdf:p1:c5"]
    assert store.search(query, k=3, filter={"source": "inexistente.pdf"}) == []


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    store = _store(5, 3)
    store.save(tmp_path / "idx")
    loaded = NumpyVectorStore.load(tmp_path / "idx")
    assert loaded.chunks == store.chunks
    assert np.allclose(loaded.matrix, store.matrix)
    original, reloaded = store.search([1.0, 0.0, 0.0], k=2), loaded.search([1.0, 0.0, 0.0], k=2)
    assert [index for index, _ in reloaded] == [index for index, _ in original]
    assert [score for _, score in reloaded] == pytest.approx([score for _, score in original], abs=1e-6)


def test_vector_index_uses_store_and_supports_filter_and_from_store() -> None:
    chunks = [
        Chunk(id="a.csv:r2", text="prazo de devolução 10 dias", source="a.csv", locator={"row": 2}),
        Chunk(id="b.csv:r2", text="pagamento cartão pix", source="b.csv", locator={"row": 2}),
    ]
    index = VectorIndex(chunks, HashEmbeddingProvider())
    assert isinstance(index.store, NumpyVectorStore) and index.dimension == 384
    assert index.search("devolução", k=1)[0].chunk.id == "a.csv:r2"
    assert index.search("devolução", k=1, filter={"source": "b.csv"})[0].chunk.id == "b.csv:r2"
    rebuilt = VectorIndex.from_store(index.store, HashEmbeddingProvider())
    assert rebuilt.chunks == chunks and rebuilt.scores("pagamento") == index.scores("pagamento")


def test_store_satisfies_protocol() -> None:
    store: VectorStore = _store(2, 2)
    assert store.dimension == 2 and len(store.chunks) == 2


def test_search_is_fast_at_ten_thousand_chunks() -> None:
    rng = np.random.default_rng(3)
    chunks = [Chunk(id=f"d.pdf:p1:c{i}", text="t", source="d.pdf", locator={}) for i in range(10_000)]
    store = NumpyVectorStore(chunks, rng.normal(size=(10_000, 768)).astype(np.float32))
    query = rng.normal(size=768)
    store.search(query, k=10)  # aquecimento
    started = time.perf_counter()
    for _ in range(20):
        store.search(query, k=10)
    per_query_ms = (time.perf_counter() - started) / 20 * 1000
    assert per_query_ms < 50, per_query_ms  # meta do plano: <= 5 ms; folga generosa para CI compartilhada
