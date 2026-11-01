from pathlib import Path
from streamlit.testing.v1 import AppTest
ROOT = Path(__file__).resolve().parents[1]

def test_demo_question_and_source_evidence(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not app.exception
    assert len(app.metric) == 0
    next(b for b in app.button if b.key == "Prazo de entrega").click().run()
    assert not app.exception
    assert any("3 a 5 dias úteis" in item.value for item in app.markdown)
    assert any("[F1]" in item.value for item in app.markdown)
    app.chat_input[0].set_value("Quais formas de pagamento são aceitas?").run()
    assert not app.exception
    assert len(app.chat_message) == 4
    assert len(next(s for s in app.selectbox if s.label == "Ver fontes de qual pergunta?").options) == 2
    next(b for b in app.button if b.label == "Nova conversa").click().run()
    assert not app.exception
    assert len(app.chat_message) == 0

def test_provider_configuration_and_empty_upload_base(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    next(r for r in app.radio if r.label == "Formato da resposta").set_value("Síntese com IA").run()
    assert not app.exception
    assert app.chat_input[0].disabled
    next(r for r in app.radio if r.label == "Documentos usados na consulta").set_value("Meus documentos").run()
    assert not app.exception
    assert app.chat_input[0].disabled
    assert any("Envie documentos" in item.value for item in app.info)