from io import BytesIO
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from docora.assistant import ProviderError, answer_question
from docora.retrieval import KnowledgeIndex, MAX_BYTES, chunks_from_text, ingest_bytes

def test_reference_questions(reference_case, index):
    case = reference_case
    result = answer_question(case["question"], index)
    assert result.status == case["expected_status"]
    if case.get("source"):
        assert any(source["source"] == case["source"] and source["section"] == case["section"] for source in result.sources)
        assert all(f"[{source['reference']}]" in result.message for source in result.sources)
    else:
        assert result.sources == []


def test_uploaded_text_preserves_source_section_and_lines():
    chunks = ingest_bytes("../politica.md", b"# Atendimento\n\n## Horario\nO suporte atende das 10h as 17h.\n")
    assert chunks[0].source == "politica.md"
    assert chunks[0].section == "Horario"
    assert chunks[0].line_start == 4
    assert chunks[0].line_end == 4
    index = KnowledgeIndex(chunks)
    assert "10h" in answer_question("Qual o horario do suporte?", index).message


def test_documents_are_isolated_from_demo():
    own = KnowledgeIndex(chunks_from_text("# Empresa\n## Expediente\nO expediente termina as 17h e comeca as 10h.", "empresa.txt"))
    assert answer_question("Qual o desconto no Pix?", own).status == "not_found"
    assert {chunk.source for chunk in own.chunks} == {"empresa.txt"}


def test_ingestion_limits_and_encoding():
    with pytest.raises(ValueError, match="2 MB"):
        ingest_bytes("texto.txt", b"x" * (MAX_BYTES + 1))
    with pytest.raises(ValueError, match="Markdown"):
        ingest_bytes("arquivo.exe", b"abc")
    with pytest.raises(ValueError, match="UTF-8"):
        ingest_bytes("texto.txt", b"\xff\xfe")
    with pytest.raises(ValueError, match="pelo menos"):
        KnowledgeIndex([])
    long_text = "atendimento horario documento " * 200
    chunks = chunks_from_text(long_text, "longo.txt")
    assert len(chunks) > 1
    assert max(len(chunk.text) for chunk in chunks) <= 1000
    assert len(KnowledgeIndex(chunks + chunks).chunks) == len(chunks)


def pdf_with_text():
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
    writer = PdfWriter()
    for text in ["Atendimento das 10h as 17h.", "Trocas em ate 20 dias corridos."]:
        page = writer.add_blank_page(width=595, height=842)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        content = DecodedStreamObject()
        content.set_data(f"BT /F1 12 Tf 50 780 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(content)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def test_pdf_ingestion_tracks_real_page_numbers():
    chunks = ingest_bytes("manual.pdf", pdf_with_text())
    assert {chunk.page for chunk in chunks} == {1, 2}
    result = answer_question("Qual o prazo de trocas?", KnowledgeIndex(chunks))
    assert result.sources[0]["page"] == 2
    assert "20 dias" in result.message


def test_blank_pdf_is_not_silently_indexed():
    from pypdf import PdfWriter
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    output = BytesIO()
    writer.write(output)
    with pytest.raises(ValueError, match="OCR"):
        ingest_bytes("digitalizado.pdf", output.getvalue())


def fake_client(payload):
    return SimpleNamespace(responses=SimpleNamespace(create=Mock(return_value=SimpleNamespace(output_text=json.dumps(payload)))))


def test_structured_generation_uses_only_retrieved_evidence(index):
    client = fake_client({"answer": "O Pix tem 5% de desconto sobre produtos. [F1]", "answerable": True, "citations": ["F1"]})
    answer = answer_question("Existe desconto no Pix?", index, mode="generative", client=client, model="modelo-de-teste")
    arguments = client.responses.create.call_args.kwargs
    assert answer.status == "answered"
    assert len(answer.sources) == 1
    assert arguments["store"] is False
    assert arguments["text"]["format"]["strict"] is True
    assert len(json.loads(arguments["input"])["evidence"]) <= 3
    assert "OPENAI_API_KEY" not in arguments["input"]


@pytest.mark.parametrize("payload", [
    {"answer": "Uma oferta inventada. [F99]", "answerable": True, "citations": ["F99"]},
    {"answer": "Resposta sem fonte", "answerable": True, "citations": []},
    {"answer": "Resposta. [F1]", "answerable": True, "citations": ["F2"]},
    {"answer": "", "answerable": True, "citations": ["F1"]},
])
def test_invalid_generated_citations_rejected(index, payload):
    with pytest.raises(ProviderError, match="referências válidas"):
        answer_question("Desconto no Pix?", index, mode="generative", client=fake_client(payload), model="modelo-de-teste")


def test_guarded_question_never_reaches_provider(index):
    client = fake_client({})
    answer = answer_question("Consulte o pedido 12345", index, mode="generative", client=client, model="modelo-de-teste")
    assert answer.status == "needs_human"
    client.responses.create.assert_not_called()


def test_missing_configuration_and_invalid_question(index, monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    with pytest.raises(ValueError, match="OPENAI_MODEL"):
        answer_question("Desconto no Pix?", index, mode="generative")
    for question in [" ", "a" * 1501]:
        with pytest.raises(ValueError, match="1 a 1.500"):
            answer_question(question, index)
