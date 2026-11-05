"""Integração optativa: exige encoder já disponível no cache local."""
import json
from unittest.mock import Mock

import pytest

from docora.assistant import answer_question
from docora.embeddings import load_embedder
from docora.evaluation import evaluate
from docora.retrieval import KnowledgeIndex, demo_chunks


@pytest.mark.slow
@pytest.mark.model
def test_real_model_on_both_suites_and_streamlit_without_key(monkeypatch, tmp_path, project_root):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("DOCORA_DISABLE_EMBEDDINGS")
    monkeypatch.setenv("DOCORA_RETRIEVER", "hybrid")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    provider = Mock(side_effect=AssertionError("Nenhuma chamada à OpenAI é permitida"))
    monkeypatch.setattr("openai.OpenAI", provider)
    load_embedder.cache_clear()
    model, reason = load_embedder()
    # Invocação explícita deve falhar se o modelo estiver ausente, sem encobrir
    # uma suposta execução real com skip ou retorno silencioso ao TF-IDF.
    assert model is not None, reason
    index = KnowledgeIndex(demo_chunks(), retriever="hybrid", embedder=model)
    root = project_root
    for suite, file in [("original", "questions.json"), ("heldout", "heldout_questions.json")]:
        report = evaluate(json.loads((root / "evals" / file).read_text()), index, suite)
        assert report["retriever_active"] == "hybrid"
        assert report["faithfulness_pct"] == 100
        assert report["source_recall_at_3_pct"] > 0
        (tmp_path / f"real_{suite}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    question = "Em quanto tempo o pedido chega na capital paulista?"
    result = answer_question(question, index)
    assert result.sources[0]["section"] == "Regiões atendidas e prazo de entrega"
    st.cache_resource.clear()
    try:
        app = AppTest.from_file(str(root / "app.py"), default_timeout=60).run()
        assert not app.exception
        assert any(c.value == "Busca ativa: Busca híbrida" for c in app.caption)
        app.chat_input[0].set_value(question).run()
        assert not app.exception
        assert app.session_state["history"][-1]["answer"]["sources"][0]["section"] == "Regiões atendidas e prazo de entrega"
        assert any("3 a 5 dias úteis" in m.value and "[F1]" in m.value for m in app.markdown)
        provider.assert_not_called()
    finally:
        st.cache_resource.clear()
        load_embedder.cache_clear()
