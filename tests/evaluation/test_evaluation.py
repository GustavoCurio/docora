from types import SimpleNamespace

import pytest

from docora.assistant import Answer
from docora.evaluation import evaluate, faithfulness
from docora.retrieval import Hit, chunks_from_text


def test_ranking_abstention_denominators_and_multisection():
    chunks = chunks_from_text(
        "## Uma\nEntrega em cinco dias úteis.\n## Duas\nTroca em trinta dias corridos.", "base.md"
    )
    index = SimpleNamespace(
        chunks=chunks,
        search=lambda question, top_k=3: [Hit(c, 1) for c in chunks][:top_k] if question != "ausente" else [],
    )
    cases = [
        {"question": "entrega", "expected_status": "evidence_found", "source": "base.md", "section": "Duas"},
        {
            "question": "alternativa",
            "expected_status": "evidence_found",
            "acceptable_sources": [{"source": "base.md", "section": "Uma"}],
        },
        {"question": "ausente", "expected_status": "evidence_found", "source": "base.md", "section": "Duas"},
        {"question": "ausente", "expected_status": "not_found"},
        {"question": "outro assunto", "expected_status": "not_found"},
    ]
    report = evaluate(cases, index)
    assert report["source_recall_at_1_pct"] == 33.33
    assert report["source_recall_at_3_pct"] == report["source_recall_at_5_pct"] == 66.67
    assert report["mrr_at_5"] == 0.5
    assert report["not_found_precision_pct"] == 50
    assert report["ood_false_positive_rate_pct"] == 50
    assert report["faithfulness_pct"] == 100


def test_faithfulness_measures_message_against_its_cited_chunk():
    source = {"reference": "F1", "text": "Entrega em cinco dias úteis."}
    answer = Answer(
        "evidence_found",
        "**Prazo** [F1]\n\nEntrega em cinco dias úteis.\nUnicórnios voam alegremente.",
        "extractive",
        [source],
    )
    assert faithfulness(answer) == {"supported_sentences": 1, "sentences": 2, "score": 0.5}
    answer.message = "**Prazo** [F99]\n\nEntrega em cinco dias úteis."
    assert faithfulness(answer)["score"] == 0
    answer.message = "Entrega em cinco dias úteis."
    assert faithfulness(answer)["score"] == 0
    assert faithfulness(Answer("not_found", "Sem informação", "extractive"))["score"] is None


def test_empty_suite_is_rejected():
    with pytest.raises(ValueError, match="conjunto"):
        evaluate([], None)


def test_inline_citation_cannot_support_a_sentence_from_another_source():
    sources = [
        {"reference": "F1", "text": "Entrega em cinco dias úteis."},
        {"reference": "F2", "text": "Pagamento em três parcelas."},
    ]
    answer = Answer(
        "evidence_found", "Entrega em cinco dias úteis. [F2]\n\nPagamento em três parcelas. [F2]", "extractive", sources
    )
    assert faithfulness(answer)["score"] == 0.5


def test_evaluation_reuses_one_ranking_for_answer_and_metrics():
    from unittest.mock import Mock

    chunks = chunks_from_text("## Prazo\nEntrega em cinco dias úteis.", "base.md")
    search = Mock(return_value=[Hit(chunks[0], 1)])
    index = SimpleNamespace(chunks=chunks, search=search)
    report = evaluate(
        [{"question": "entrega", "expected_status": "evidence_found", "source": "base.md", "section": "Prazo"}], index
    )
    search.assert_called_once_with("entrega", top_k=5)
    assert report["passed"] == 1
    assert report["answer_checks"]["gold_cited_in_body_pct"] == 100
