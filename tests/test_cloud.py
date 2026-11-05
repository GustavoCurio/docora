"""Regressões do Community Cloud, sem credenciais reais nem chamadas externas."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def cloud_without_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    # Reproduz o acesso aos segredos quando nenhum arquivo está configurado.
    missing_secrets = Mock()
    missing_secrets.get.side_effect = StreamlitSecretNotFoundError("Nenhum segredo configurado")
    monkeypatch.setattr(st, "secrets", missing_secrets)
    factory = Mock(side_effect=AssertionError("Cliente externo não autorizado neste teste"))
    monkeypatch.setattr("openai.OpenAI", factory)
    return factory


def start_app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()


def test_cloud_demo_without_environment_or_secrets(cloud_without_credentials):
    app = start_app()
    assert not app.exception
    assert not app.error
    assert not any(r.label == "Formato da resposta" for r in app.radio)
    assert not app.checkbox
    assert not app.chat_input[0].disabled

    app.button(key="Prazo de entrega").click().run()
    assert not app.exception
    assert any("3 a 5 dias úteis" in item.value and "[F1]" in item.value for item in app.markdown)
    assert any(item.value == "Fontes da conversa" for item in app.subheader)
    assert any("**[F1]" in item.value for item in app.markdown)
    assert "OPENAI_API_KEY" not in os.environ
    assert "OPENAI_MODEL" not in os.environ
    cloud_without_credentials.assert_not_called()


def test_load_example_is_idempotent_and_preserves_demo_history():
    app = start_app()
    app.button(key="Prazo de entrega").click().run()
    history = app.session_state["history"][:]
    for _ in range(2):
        next(b for b in app.button if b.label == "Carregar exemplo").click().run()
        assert not app.exception
        assert any("Exemplo carregado: Lume Casa" in item.value for item in app.success)
        assert any("3 documentos disponíveis · 20 trechos" == item.value for item in app.caption)
        assert app.session_state["history"] == history
        assert len(app.chat_message) == 2

    app.radio(key="collection").set_value("Meus documentos").run()
    assert app.chat_input[0].disabled
    next(b for b in app.button if b.label == "Carregar exemplo").click().run()
    assert not app.exception
    assert app.radio(key="collection").value == "Loja demonstrativa"
    assert not app.chat_input[0].disabled
    assert any("Exemplo carregado" in item.value for item in app.success)


@pytest.mark.parametrize("configuration", ["ambiente", "segredos", "ambiente_prioritario"])
def test_optional_generation_receives_server_settings_and_requires_consent(monkeypatch, configuration):
    if configuration in {"ambiente", "ambiente_prioritario"}:
        monkeypatch.setenv("OPENAI_API_KEY", "chave-ficticia-ambiente")
        monkeypatch.setenv("OPENAI_MODEL", "modelo-ficticio-ambiente")
    if configuration in {"segredos", "ambiente_prioritario"}:
        monkeypatch.setattr(st, "secrets", {
            "OPENAI_API_KEY": "chave-ficticia-segredos",
            "OPENAI_MODEL": "modelo-ficticio-segredos",
        })
    expected = "segredos" if configuration == "segredos" else "ambiente"
    create = Mock(return_value=SimpleNamespace(output_text=json.dumps({
        "answer": "O prazo é de 3 a 5 dias úteis. [F1]",
        "answerable": True,
        "citations": ["F1"],
    })))
    factory = Mock(return_value=SimpleNamespace(responses=SimpleNamespace(create=create)))
    monkeypatch.setattr("openai.OpenAI", factory)
    app = start_app()
    assert not app.exception
    mode = next(r for r in app.radio if r.label == "Formato da resposta")
    assert mode.value == "Trechos dos documentos"
    mode.set_value("Síntese com IA").run()
    assert not app.exception
    assert app.chat_input[0].disabled
    factory.assert_not_called()

    app.checkbox[0].check().run()
    assert not app.chat_input[0].disabled
    app.button(key="Prazo de entrega").click().run()
    assert not app.exception
    assert any("3 a 5 dias úteis. [F1]" in item.value for item in app.markdown)
    factory.assert_called_once_with(api_key=f"chave-ficticia-{expected}", timeout=25.0, max_retries=1)
    assert create.call_args.kwargs["model"] == f"modelo-ficticio-{expected}"
    assert create.call_args.kwargs["store"] is False


def test_key_without_model_keeps_extractive_default_and_blocks_generation(monkeypatch, cloud_without_credentials):
    monkeypatch.setenv("OPENAI_API_KEY", "chave-ficticia")
    app = start_app()
    assert not app.exception
    assert not app.chat_input[0].disabled
    next(r for r in app.radio if r.label == "Formato da resposta").set_value("Síntese com IA").run()
    assert not app.exception
    assert app.chat_input[0].disabled
    assert any("modelo não foi configurado" in item.value for item in app.info)
    cloud_without_credentials.assert_not_called()


def test_evaluation_default_output_uses_temporary_directory(monkeypatch, tmp_path):
    import docora.__main__ as cli

    original_report = (ROOT / "reports/evaluation.json").read_bytes()
    monkeypatch.setattr(cli.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr("sys.argv", ["docora", "evaluate"])
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == 0
    report = json.loads((tmp_path / "docora/evaluation.json").read_text("utf-8"))
    assert report["passed"] == report["total"] == 28
    assert (ROOT / "reports/evaluation.json").read_bytes() == original_report
