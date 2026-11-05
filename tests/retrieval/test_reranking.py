"""Ordenação e abstenção independentes da escala de escores do recuperador."""

import pytest

from docora.reranking import rules_rerank
from docora.retrieval import Hit, KnowledgeIndex, chunks_from_text, demo_chunks


def test_rules_promote_heading_without_changing_evidence_or_scores():
    chunks = chunks_from_text(
        "## Diversos\nSaiba mais sobre políticas comerciais da loja.\n"
        "## Prazo de entrega\nAs encomendas chegam em cinco dias úteis.",
        "base.md",
    )
    hits = [Hit(c, 0.2) for c in chunks]
    result = rules_rerank("prazo de entrega", hits)
    assert result[0].chunk.section == "Prazo de entrega"
    assert set(result) == set(hits)
    assert result[0].score == 0.2


@pytest.mark.parametrize("retriever", ["tfidf", "bm25", "hybrid"])
def test_rules_keep_retrieval_filters_and_prefix_consistency(retriever):
    base = KnowledgeIndex(demo_chunks(), retriever=retriever, embedder=None, reranker="none")
    rules = KnowledgeIndex(demo_chunks(), retriever=retriever, embedder=None, reranker="rules")
    assert rules.search("receita de bolo de cenoura") == []
    question = "pagamento e desconto no Pix"
    assert set(base.search(question, top_k=10)) == set(rules.search(question, top_k=10))
    assert rules.search(question, top_k=1) == rules.search(question, top_k=5)[:1]
    with pytest.raises(ValueError):
        rules.search(question, top_k=0)


def test_invalid_reranker_is_explicit():
    with pytest.raises(ValueError, match="Reranker"):
        KnowledgeIndex(demo_chunks(), reranker="outro")
