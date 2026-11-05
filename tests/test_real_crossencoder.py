"""Integração optativa ONNX; exige ambos os modelos já presentes no cache."""
import json
from pathlib import Path

import pytest

from docora.assistant import answer_question
from docora.crossencoder import load_crossencoder
from docora.embeddings import load_embedder
from docora.evaluation import evaluate
from docora.retrieval import KnowledgeIndex, demo_chunks


@pytest.mark.slow
def test_real_crossencoder_both_suites(monkeypatch, tmp_path):
    monkeypatch.delenv("DOCORA_DISABLE_EMBEDDINGS")
    monkeypatch.delenv("DOCORA_DISABLE_RERANKER")
    load_crossencoder.cache_clear()
    load_embedder.cache_clear()
    try:
        index = KnowledgeIndex(demo_chunks(), retriever="hybrid", reranker="crossencoder")
        assert index.active_retriever == "hybrid", index.fallback_reason
        assert index.active_reranker == "crossencoder", index.reranker_fallback_reason
        root = Path(__file__).resolve().parents[1]
        for suite, file in [("original", "questions.json"), ("heldout", "heldout_questions.json")]:
            report = evaluate(json.loads((root / "evals" / file).read_text()), index, suite)
            assert report["faithfulness_pct"] == 100
            assert report["answer_checks"]["inline_citation_pct"] == 100
            if suite == "original":
                assert report["passed"] == report["total"] == 28
            (tmp_path / f"real_ce_{suite}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        answer = answer_question("Em quanto tempo o pedido chega na capital paulista?", index)
        assert answer.sources[0]["section"] == "Regiões atendidas e prazo de entrega"
        assert answer_question("receita de bolo de cenoura", index).status == "not_found"
    finally:
        load_crossencoder.cache_clear()
        load_embedder.cache_clear()
