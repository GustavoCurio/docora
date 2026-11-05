"""A suíte rápida não carrega modelos reais nem autoriza downloads."""
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import streamlit as st
from fastapi.testclient import TestClient
from streamlit.errors import StreamlitSecretNotFoundError
from streamlit.testing.v1 import AppTest

from docora.api import app, index as api_index
from docora.crossencoder import load_crossencoder
from docora.embeddings import load_embedder
from docora.retrieval import KnowledgeIndex, demo_chunks

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT


@pytest.fixture(autouse=True)
def local_retrieval_by_default(monkeypatch):
    monkeypatch.setenv("DOCORA_RETRIEVER", "tfidf")
    monkeypatch.setenv("DOCORA_DISABLE_EMBEDDINGS", "1")
    monkeypatch.setenv("DOCORA_ALLOW_MODEL_DOWNLOAD", "0")
    monkeypatch.setenv("DOCORA_DISABLE_RERANKER", "1")
    monkeypatch.setenv("DOCORA_RERANKER", "none")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setattr("openai.OpenAI", Mock(side_effect=AssertionError("Cliente externo não autorizado nos testes")))
    for clear in (load_embedder.cache_clear, load_crossencoder.cache_clear, api_index.cache_clear, st.cache_resource.clear):
        clear()
    yield
    for clear in (load_embedder.cache_clear, load_crossencoder.cache_clear, api_index.cache_clear, st.cache_resource.clear):
        clear()


@pytest.fixture
def index():
    return KnowledgeIndex(demo_chunks())


@pytest.fixture(
    params=json.loads((PROJECT_ROOT / "evals/questions.json").read_text(encoding="utf-8")),
    ids=lambda case: case["question"],
)
def reference_case(request):
    return request.param


@pytest.fixture
def api_client():
    with TestClient(app) as client:
        yield client


@pytest.fixture
def start_app(project_root):
    def start():
        return AppTest.from_file(str(project_root / "app.py"), default_timeout=60).run()

    return start


@pytest.fixture
def cloud_without_credentials(monkeypatch):
    # Reproduz o acesso aos segredos quando nenhum arquivo está configurado.
    missing_secrets = Mock()
    missing_secrets.get.side_effect = StreamlitSecretNotFoundError("Nenhum segredo configurado")
    monkeypatch.setattr(st, "secrets", missing_secrets)
    factory = Mock(side_effect=AssertionError("Cliente externo não autorizado neste teste"))
    monkeypatch.setattr("openai.OpenAI", factory)
    return factory
