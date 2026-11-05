from unittest.mock import Mock

import numpy as np
import pytest

from docora.assistant import answer_question
from docora import embeddings
from docora.ranking import BM25, rrf, tokenize
from docora.retrieval import KnowledgeIndex, chunks_from_text, demo_chunks


def test_bm25_normalizes_portuguese_and_ranks_relevant_document():
    corpus = ["Pagamento por Pix dá desconto no valor.", "A entrega leva cinco dias úteis.", "Atendimento em horário comercial."]
    bm25 = BM25(corpus)
    scores = bm25.scores("PAGAMENTO por pix")
    assert scores[0] > 0
    assert list(scores[1:]) == [0, 0]
    assert tokenize("DEVOLUÇÃO e proteção") == ["devolucao", "protecao"]
    assert not bm25.scores("astrofísica").any()
    np.testing.assert_allclose(scores, bm25.scores("pagamento pagamento pix"))


def test_rrf_combines_ranks_and_ignores_duplicate_votes():
    result = rrf([["a", "b"], ["b", "c"]])
    assert [item for item, _ in result] == ["b", "a", "c"]
    assert result[0][1] == pytest.approx(1 / 62 + 1 / 61)
    assert rrf([["a", "a", "b"]]) == rrf([["a", "b"]])
    assert rrf([[], []]) == []
    with pytest.raises(ValueError):
        rrf([], k=-1)


@pytest.mark.parametrize("retriever", ["tfidf", "bm25"])
def test_lexical_retrievers_do_not_load_encoder(retriever, monkeypatch):
    loader = Mock(side_effect=AssertionError("Não carregar modelo neste caminho"))
    monkeypatch.setattr(embeddings, "load_embedder", loader)
    index = KnowledgeIndex(demo_chunks(), retriever=retriever)
    assert answer_question("Prazo de entrega para São Paulo?", index).sources[0]["section"] == "Regiões atendidas e prazo de entrega"
    assert answer_question("Como compilar o kernel Linux?", index).status == "not_found"
    loader.assert_not_called()


def test_unavailable_encoder_falls_back_with_identical_tfidf_results(monkeypatch):
    monkeypatch.setattr(embeddings, "load_embedder", lambda: (None, "Modelo indisponível"))
    index = KnowledgeIndex(demo_chunks(), retriever="hybrid")
    baseline = KnowledgeIndex(demo_chunks(), retriever="tfidf")
    assert index.requested_retriever == "hybrid"
    assert index.active_retriever == "tfidf"
    assert index.fallback_reason == "Modelo indisponível"
    for question in ["Desconto no Pix?", "Qual o prazo para São Paulo?", "Como compilar o kernel Linux?"]:
        assert index.search(question) == baseline.search(question)


@pytest.mark.parametrize("exception", [ImportError, OSError, MemoryError])
def test_load_failures_are_cached_without_retries(exception, monkeypatch, tmp_path):
    monkeypatch.delenv("DOCORA_DISABLE_EMBEDDINGS")
    monkeypatch.setenv("DOCORA_MODEL_CACHE", str(tmp_path))
    factory = Mock(side_effect=exception("Falha simulada"))
    monkeypatch.setattr(embeddings, "LocalEmbedder", factory)
    embeddings.load_embedder.cache_clear()
    try:
        for _ in range(2):
            model, reason = embeddings.load_embedder()
            assert model is None
            assert exception.__name__ in reason
        factory.assert_called_once_with(tmp_path, allow_download=False)
    finally:
        embeddings.load_embedder.cache_clear()


@pytest.mark.parametrize("exception", [OSError, MemoryError])
@pytest.mark.parametrize("stage", ["corpus", "query"])
def test_corpus_and_query_encoding_failures_preserve_demo(exception, stage):
    chunks = demo_chunks()
    model = Mock()
    if stage == "corpus":
        model.encode.side_effect = exception("Falha simulada")
    else:
        model.encode.side_effect = [np.eye(len(chunks)), exception("Falha simulada")]
    index = KnowledgeIndex(chunks, retriever="hybrid", embedder=model)
    result = answer_question("Qual o prazo de entrega para São Paulo?", index)
    assert index.active_retriever == "tfidf"
    assert exception.__name__ in index.fallback_reason
    assert result.status == "evidence_found"
    assert result.sources[0]["section"] == "Regiões atendidas e prazo de entrega"


def tiny_index(cosines):
    chunks = chunks_from_text("## Prazo\nEntrega em cinco dias úteis.\n## Pix\nDesconto de cinco por cento no Pix.\n## Suporte\nAtendimento humano em horário comercial.", "base.md")
    model = Mock()
    model.encode.side_effect = [np.array([[c, np.sqrt(1 - c * c)] for c in cosines]), np.array([[1, 0]])]
    return KnowledgeIndex(chunks, retriever="hybrid", embedder=model)


def test_absolute_semantic_cut_rejects_even_strong_lexical_match():
    index = tiny_index([0.849, 0.7, 0.6])
    assert answer_question("Entrega em cinco dias úteis?", index).status == "not_found"
    assert index.active_retriever == "hybrid"


def test_rrf_cannot_revive_candidates_outside_semantic_window():
    index = tiny_index([0.92, 0.88, 0.7])
    # Pix tem a evidência lexical mais forte, mas está fora da janela de 0,03.
    result = answer_question("Desconto Pix", index)
    assert [source["section"] for source in result.sources] == ["Prazo"]


def test_dense_evidence_can_retrieve_without_shared_words():
    index = tiny_index([0.92, 0.7, 0.6])
    result = answer_question("Quando recebo minha encomenda?", index)
    assert result.status == "evidence_found"
    assert result.sources[0]["section"] == "Prazo"


def test_hybrid_uses_bm25_to_resolve_close_dense_candidates():
    index = tiny_index([0.92, 0.91, 0.7])
    hits = index.search("Desconto Pix", top_k=5)
    assert [hit.chunk.section for hit in hits] == ["Pix", "Prazo"]


def test_invalid_vectors_fall_back_and_invalid_selection_is_rejected(monkeypatch):
    model = Mock()
    model.encode.return_value = np.full((20, 3), np.nan)
    index = KnowledgeIndex(demo_chunks(), retriever="hybrid", embedder=model)
    assert index.active_retriever == "tfidf"
    monkeypatch.setenv("DOCORA_RETRIEVER", "bm25")
    assert KnowledgeIndex(demo_chunks()).active_retriever == "bm25"
    with pytest.raises(ValueError, match="Recuperador"):
        KnowledgeIndex(demo_chunks(), retriever="inválido")
    with pytest.raises(ValueError, match="positivo"):
        index.search("Qual o prazo?", top_k=0)
