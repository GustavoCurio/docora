"""Regressões da seleção de sentenças e da sua proveniência."""
from types import SimpleNamespace

from docora.assistant import answer_question
from docora.evaluation import answer_checks, faithfulness
from docora.extractive import near_duplicate, select_sentences, split_sentences
from docora.retrieval import Hit, chunks_from_text


def test_portuguese_boundaries_and_source_fragments():
    text = "A Dra. Ana atende na Av. Paulista. Custa R$ 20,50 e mede 2.5 cm.\n- Entrega em SP! Outra opção?"
    sentences = split_sentences(text)
    assert sentences == ["A Dra. Ana atende na Av. Paulista.", "Custa R$ 20,50 e mede 2.5 cm.",
                         "- Entrega em SP!", "Outra opção?"]
    assert all(sentence in text for sentence in sentences)


def test_deduplication_budget_and_position():
    repeated = "A entrega acontece em cinco dias úteis."
    source = {"reference": "F1", "section": "Entrega", "text": repeated + " " + repeated + " " +
              " ".join(f"A entrega da região {n} custa {n} reais e leva nove dias." for n in range(20))}
    selected = select_sentences("entrega região", [source])
    assert len(selected) <= 6
    assert sum(len(text.split()) for text, _ in selected) <= 120
    assert sum(text == repeated for text, _ in selected) <= 1
    positions = [source["text"].index(text) for text, _ in selected]
    assert positions == sorted(positions)
    assert near_duplicate(repeated, "A entrega acontece em cinco dias úteis!")
    assert not near_duplicate(repeated, "O pagamento aceita cartão de crédito.")


def test_each_sentence_cites_its_actual_chunk_and_avoids_unused_sources():
    chunks = chunks_from_text("## Prazo\nA entrega leva cinco dias úteis. O pagamento aceita cartão.\n"
                              "## Frete\nO frete custa vinte reais. A entrega leva cinco dias úteis.", "base.md")
    index = SimpleNamespace(search=lambda question: [Hit(c, 1) for c in chunks])
    result = answer_question("prazo e frete", index)
    assert result.status == "evidence_found"
    assert faithfulness(result)["score"] == 1
    checks = answer_checks(result, [{"source": "base.md", "section": "Frete"}])
    assert checks["gold_cited_in_body"]
    assert checks["exact_source_sentences"] == checks["inline_cited_sentences"] == checks["sentences"]
    assert all(f"[{s['reference']}]" in result.message for s in result.sources)
    assert result.message.count("A entrega leva cinco dias úteis.") == 1


def test_guard_and_abstention_do_not_select_sentences():
    index = SimpleNamespace(search=lambda question: [])
    assert answer_question("algo ausente", index).status == "not_found"
    index.search = lambda question: (_ for _ in ()).throw(AssertionError("Busca indevida"))
    assert answer_question("pedido #123456", index).status == "needs_human"


def test_oversized_sentence_is_never_truncated_into_an_unsupported_claim():
    assert select_sentences("entrega", [{"reference": "F1", "section": "Entrega", "text": "entrega " * 121}]) == []
