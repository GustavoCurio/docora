"""A suíte rápida não carrega modelos reais nem autoriza downloads."""
import pytest


@pytest.fixture(autouse=True)
def local_retrieval_by_default(monkeypatch):
    monkeypatch.setenv("DOCORA_RETRIEVER", "tfidf")
    monkeypatch.setenv("DOCORA_DISABLE_EMBEDDINGS", "1")
    monkeypatch.setenv("DOCORA_ALLOW_MODEL_DOWNLOAD", "0")
    monkeypatch.setenv("DOCORA_DISABLE_RERANKER", "1")
    monkeypatch.setenv("DOCORA_RERANKER", "none")
