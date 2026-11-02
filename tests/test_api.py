from fastapi.testclient import TestClient

from docora.api import app

client = TestClient(app)


def test_health_sources_and_question():
    assert client.get("/health").json()["status"] == "ok"
    assert len(client.get("/sources").json()["documents"]) == 3
    response = client.post("/ask", json={"question": "Qual é o prazo de entrega para São Paulo?"})
    assert response.status_code == 200
    assert response.json()["sources"][0]["source"] == "entregas.md"
    assert "3 a 5" in response.json()["message"]


def test_api_validates_request_and_external_processing():
    assert client.post("/ask", json={}).status_code == 422
    assert client.post("/ask", json={"question": "a" * 1501}).status_code == 422
    assert client.post("/ask", json={"question": " "}).status_code == 400
    response = client.post("/ask", json={"question": "Troca de produto", "mode": "generative"})
    assert response.status_code == 403


def test_api_refuses_unknown_facts_and_private_order_lookup():
    assert client.post("/ask", json={"question": "Qual é a previsão do tempo?"}).json()["status"] == "not_found"
    assert client.post("/ask", json={"question": "Onde está o pedido 98765?"}).json()["status"] == "needs_human"
