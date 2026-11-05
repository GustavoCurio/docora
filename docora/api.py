"""API HTTP para consultar a base demonstrativa do Docora."""
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .assistant import ProviderError, answer_question
from .retrieval import KnowledgeIndex, demo_chunks

app = FastAPI(title="Docora API", version="1.0.0", description="Atendimento baseado em documentos com fontes rastreáveis.")


@lru_cache(maxsize=1)
def index():
    """Inicializa e reutiliza o índice demonstrativo no processo que atende à API."""
    return KnowledgeIndex(demo_chunks())


# Contrato de entrada: a geração externa depende de autorização explícita.
class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1500)
    mode: Literal["extractive", "generative"] = "extractive"
    approve_external_processing: bool = False


@app.get("/health")
def health():
    """Retorna disponibilidade e número de trechos, inicializando a base se necessário."""
    return {"status": "ok", "chunks": len(index().chunks), "retriever_active": index().active_retriever}


@app.get("/sources")
def sources():
    """Lista os documentos e a quantidade de trechos da base demonstrativa do servidor."""
    return {"documents": sorted({chunk.source for chunk in index().chunks}), "chunks": len(index().chunks)}


@app.post("/ask")
def ask(body: AskRequest):
    """Executa uma consulta na biblioteca demonstrativa e devolve o contrato Answer.

    Args:
        body: Pergunta, modo e autorização explícita de processamento externo.

    Returns:
        Dicionário com status, mensagem, fontes, modo e duração.

    Raises:
        HTTPException: 403 para modo generativo sem autorização; 400 para
            argumentos/configuração inválidos; 502 para falha do provedor.

    Uploads da interface não alteram esta base. A validação do modelo de
    entrada do FastAPI é anterior à execução deste endpoint.
    """
    if body.mode == "generative" and not body.approve_external_processing:
        raise HTTPException(403, "A síntese generativa exige autorização para enviar a pergunta e os trechos ao provedor.")
    try:
        return answer_question(body.question, index(), body.mode).to_dict()
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    except ProviderError as error:
        raise HTTPException(502, str(error)) from error
