"""Ingestão e índices locais TF-IDF, BM25 e híbrido com RRF."""

from __future__ import annotations

import os
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import Enum, auto
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from .contracts import ChunkData, CrossEncoder, Embedder
from .ranking import BM25

BASE_DIR = Path(__file__).resolve().parents[1] / "data" / "knowledge_base"
MAX_BYTES = 2 * 1024 * 1024
MAX_CHARS = 150_000
STOP_WORDS = set(
    [
        "a",
        "ao",
        "aos",
        "as",
        "o",
        "os",
        "de",
        "da",
        "das",
        "do",
        "dos",
        "em",
        "um",
        "uma",
        "uns",
        "umas",
        "e",
        "ou",
        "se",
        "que",
        "qual",
        "quais",
        "como",
        "quando",
        "onde",
        "por",
        "para",
        "com",
        "sem",
        "sobre",
        "no",
        "na",
        "nos",
        "nas",
        "eu",
        "meu",
        "minha",
        "voce",
        "voces",
        "me",
        "posso",
        "pode",
        "podem",
        "poderia",
        "quero",
        "saber",
        "funciona",
        "tem",
        "tenho",
        "faz",
        "fazer",
        "e",
        "ela",
        "ele",
        "isso",
        "esse",
        "essa",
        "este",
        "esta",
        "isto",
        "sao",
        "ser",
        "foi",
        "muito",
        "por",
        "favor",
        "quanto",
        "quantos",
        "ate",
    ]
)
ALIASES = {
    "devolver": "devolucao troca",
    "devolucoes": "devolucao troca",
    "trocar": "troca",
    "rastrear": "rastreamento",
    "rastreio": "rastreamento",
    "chegar": "entrega",
    "demora": "prazo",
    "atrasou": "atraso",
    "gratuito": "gratis frete",
    "gratuita": "gratis frete",
    "parcelar": "parcelamento cartao",
    "pagar": "pagamento",
    "quebrado": "avaria",
    "quebrada": "avaria",
    "danificado": "avaria",
    "lavar": "lavagem limpeza conservacao",
    "tamanho": "medidas capacidade",
    "cancelar": "cancelamento",
    "falar": "contato atendimento",
}


class _AutoModel(Enum):
    EMBEDDER = auto()
    RERANKER = auto()


# Calibrados somente em questions.json + development_questions.json.
HYBRID_MIN_COSINE = 0.85
HYBRID_MAX_GAP = 0.03


def normalize(text: str) -> str:
    """Normaliza caixa e remove marcas de acento para a comparação lexical."""
    return "".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c))


def terms(text: str) -> set[str]:
    """Retorna termos únicos com ao menos dois caracteres, excluindo as stopwords configuradas."""
    return {token for token in re.findall(r"\b\w{2,}\b", normalize(text)) if token not in STOP_WORDS}


def expanded_query(question: str) -> str:
    """Acrescenta aliases explícitos do domínio à pergunta normalizada, preservando o texto base."""
    original = normalize(question)
    return original + " " + " ".join(ALIASES[token] for token in terms(question) if token in ALIASES)


@dataclass(frozen=True)
class Chunk:
    """Trecho imutável com conteúdo e posição na fonte.

    id identifica a combinação de conteúdo e origem; source guarda o nome do
    arquivo. line_start/line_end usam posições de base um. Em PDF, page informa
    a página e as linhas se referem ao texto extraído daquela página.
    """

    id: str
    source: str
    title: str
    section: str
    text: str
    line_start: int
    line_end: int
    page: int | None = None

    def to_dict(self) -> ChunkData:
        """Retorna uma cópia dos campos do trecho em dicionário serializável."""
        return cast(ChunkData, asdict(self))


@dataclass(frozen=True)
class Hit:
    """Trecho e escore do recuperador ativo; não é confiança calibrada."""

    chunk: Chunk
    score: float


def chunks_from_text(text: str, source: str, page: int | None = None) -> list[Chunk]:
    """Segmenta texto por títulos e tamanho, preservando posição e identificação.

    Args:
        text: Texto já decodificado, com até MAX_CHARS caracteres.
        source: Nome do documento usado na identificação dos trechos.
        page: Página do PDF em base um, ou None para arquivos textuais.

    Returns:
        Lista de Chunk. Texto vazio ou blocos sem três termos úteis não
        produzem trechos. Títulos Markdown de níveis 1–3 delimitam seções.

    Raises:
        ValueError: O texto excede o limite de caracteres.

    Linhas longas são fragmentadas em partes de 900 caracteres e agregadas
    com limiar de cerca de 1.000 caracteres, mantendo sua linha de origem.
    """
    if len(text) > MAX_CHARS:
        raise ValueError("Documento com texto muito longo. Limite: 150.000 caracteres por arquivo.")
    if not text.strip():
        return []
    result: list[Chunk] = []
    buffer: list[tuple[int, str]] = []
    title, section = Path(source).stem, "Conteúdo"

    def flush() -> None:
        """Emite o bloco acumulado quando há termos úteis e esvazia o buffer da seção."""
        if not buffer:
            return
        body = "\n".join(line for _, line in buffer).strip()
        if len(terms(body)) >= 3:
            identifier = sha256(f"{source}|{page}|{section}|{buffer[0][0]}|{len(result)}|{body}".encode()).hexdigest()[
                :12
            ]
            result.append(Chunk(identifier, source, title, section, body, buffer[0][0], buffer[-1][0], page))
        buffer.clear()

    for number, line in enumerate(text.splitlines(), 1):
        heading = re.match(r"^(#{1,3})\s+(.+)", line)
        if heading:
            flush()
            if len(heading[1]) == 1:
                title = heading[2].strip()
            section = heading[2].strip()
        else:
            for offset in range(0, max(len(line), 1), 900):
                piece = line[offset : offset + 900]
                if buffer and sum(len(value) for _, value in buffer) + len(piece) > 1000:
                    flush()
                if piece.strip():
                    buffer.append((number, piece))
    flush()
    return result


def ingest_bytes(name: str, payload: bytes) -> list[Chunk]:
    """Lê um documento em memória e converte seu texto em trechos rastreáveis.

    Args:
        name: Nome do arquivo; componentes de diretório são descartados.
        payload: Bytes de Markdown/TXT em UTF-8 ou de PDF com texto selecionável.

    Returns:
        Lista de Chunk, com página para PDF e faixa de linhas para cada trecho.

    Raises:
        ValueError: Formato, tamanho ou codificação inválidos; PDF protegido,
            com mais de 30 páginas, ilegível ou sem texto selecionável.

    Limita o arquivo a 2 MiB e o texto extraído a 150.000 caracteres. Não
    executa OCR, não grava uploads e não acessa o caminho recebido em name.
    """
    filename = name.replace("\\", "/").split("/")[-1]
    extension = Path(filename).suffix.lower()
    if len(payload) > MAX_BYTES:
        raise ValueError("Cada documento deve ter no máximo 2 MB.")
    if extension not in {".md", ".txt", ".pdf"}:
        raise ValueError("Use documentos Markdown, TXT ou PDF com texto selecionável.")
    if extension != ".pdf":
        try:
            return chunks_from_text(payload.decode("utf-8-sig"), filename)
        except UnicodeDecodeError as error:
            raise ValueError("O arquivo de texto deve estar em UTF-8.") from error
    from pypdf import PdfReader

    try:
        reader = PdfReader(BytesIO(payload))
        if reader.is_encrypted:
            raise ValueError("PDF protegido por senha. Use uma cópia sem senha.")
        if len(reader.pages) > 30:
            raise ValueError("Use um PDF com até 30 páginas.")
        result, total = [], 0
        for number, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            total += len(text)
            if total > MAX_CHARS:
                raise ValueError("O PDF excede 150.000 caracteres extraídos.")
            result.extend(chunks_from_text(text, filename, page=number))
        if not result:
            raise ValueError(
                "Não foi encontrado texto selecionável no PDF. Arquivos digitalizados precisam de OCR antes do envio."
            )
        return result
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("Não foi possível ler o PDF. Confira se o arquivo é válido.") from error


def demo_chunks() -> list[Chunk]:
    """Lê os Markdown da demonstração em ordem de nome e retorna seus trechos."""
    return [chunk for path in sorted(BASE_DIR.glob("*.md")) for chunk in ingest_bytes(path.name, path.read_bytes())]


class KnowledgeIndex:
    """Índice em memória com seleção de recuperador e fallback lexical.

    A base é deduplicada pelo ID do trecho e os vetorizadores aprendem apenas
    o corpus recebido. Escores TF-IDF, BM25 e RRF têm escalas diferentes.
    """

    def __init__(
        self,
        chunks: Iterable[Chunk],
        retriever: str | None = None,
        embedder: Embedder | Literal[_AutoModel.EMBEDDER] | None = _AutoModel.EMBEDDER,
        reranker: str | None = None,
        crossencoder: CrossEncoder | Literal[_AutoModel.RERANKER] | None = _AutoModel.RERANKER,
    ) -> None:
        """Prepara TF-IDF, BM25 e, se disponível, vetores locais do corpus.

        Args:
            chunks: Trechos a indexar; IDs repetidos conservam a última ocorrência.
            retriever: tfidf, bm25 ou hybrid; None consulta DOCORA_RETRIEVER.
            embedder: Encoder injetável com encode; None força fallback lexical.
            reranker: none (padrão), rules ou crossencoder; lê DOCORA_RERANKER.
            crossencoder: Pontuador injetável com predict; None força regras.

        Raises:
            ValueError: A base fica vazia, supera 600 trechos/300.000 caracteres
                ou não permite formar o vocabulário dos vetorizadores.

        As matrizes permanecem nesta instância; nenhuma fonte é enviada à rede.
        """
        started = perf_counter()
        unique = {chunk.id: chunk for chunk in chunks}
        self.chunks = list(unique.values())
        if not self.chunks:
            raise ValueError("A base precisa de pelo menos um trecho de texto legível.")
        if len(self.chunks) > 600 or sum(len(chunk.text) for chunk in self.chunks) > 300_000:
            raise ValueError("A base excede o limite de 600 trechos ou 300.000 caracteres. Reduza os documentos.")
        corpus = [f"{chunk.section}\n{chunk.text}" for chunk in self.chunks]
        self.word = TfidfVectorizer(
            strip_accents="unicode", ngram_range=(1, 2), stop_words=sorted(STOP_WORDS), max_features=20000
        )
        self.char = TfidfVectorizer(strip_accents="unicode", analyzer="char_wb", ngram_range=(3, 5), max_features=30000)
        self.word_matrix = cast(csr_matrix, self.word.fit_transform(corpus))
        self.char_matrix = cast(csr_matrix, self.char.fit_transform(corpus))
        self.requested_retriever = retriever or os.environ.get("DOCORA_RETRIEVER", "hybrid")
        self.requested_reranker = reranker or os.environ.get("DOCORA_RERANKER", "none")
        if self.requested_reranker not in {"none", "rules", "crossencoder"}:
            raise ValueError("Reranker inválido. Use none, rules ou crossencoder.")
        self.active_reranker = self.requested_reranker
        self.reranker_fallback_reason: str | None = None
        self.crossencoder: CrossEncoder | None = None
        self.last_rerank_ms: float = 0
        if self.requested_reranker == "crossencoder":
            if crossencoder is _AutoModel.RERANKER:
                from .crossencoder import load_crossencoder

                crossencoder, self.reranker_fallback_reason = load_crossencoder()
            self.crossencoder = crossencoder
            if crossencoder is None:
                self.active_reranker = "rules"
                self.reranker_fallback_reason = (
                    self.reranker_fallback_reason or "Cross-encoder ausente; usando regras locais."
                )
        if self.requested_retriever not in {"tfidf", "bm25", "hybrid"}:
            raise ValueError("Recuperador inválido. Use tfidf, bm25 ou hybrid.")
        self.active_retriever = self.requested_retriever
        self.fallback_reason: str | None = None
        self.embedder: Embedder | None = None
        self.embeddings: NDArray[np.float32] | None = None
        self.bm25: BM25 | None = None
        if self.requested_retriever in {"bm25", "hybrid"}:
            self.bm25 = BM25(corpus)
        if self.requested_retriever == "hybrid":
            if embedder is _AutoModel.EMBEDDER:
                from .embeddings import load_embedder

                embedder, self.fallback_reason = load_embedder()
            if embedder is not None:
                try:
                    self.embeddings = self._vectors(embedder.encode(corpus), len(corpus))
                    self.embedder = embedder
                except Exception as error:
                    self.fallback_reason = (
                        f"Não foi possível indexar com embeddings ({type(error).__name__}); usando TF-IDF."
                    )
            if self.embedder is None:
                self.active_retriever = "tfidf"
                self.fallback_reason = self.fallback_reason or "Encoder local indisponível; usando TF-IDF."
        self.build_ms = round((perf_counter() - started) * 1000, 2)

    @staticmethod
    def _vectors(vectors: ArrayLike, rows: int) -> NDArray[np.float32]:
        """Valida e normaliza também encoders injetados nos testes."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != rows or not vectors.shape[1] or not np.isfinite(vectors).all():
            raise ValueError("Embeddings inválidos.")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norms < 1e-12).any():
            raise ValueError("Embeddings vazios.")
        return vectors / norms

    def search(self, question: str, top_k: int = 3, min_score: float | None = None) -> list[Hit]:
        """Aplica reranking ao top-10 após os filtros, antes do corte de saída."""
        if top_k < 1:
            raise ValueError("top_k deve ser positivo.")
        self.last_rerank_ms = 0
        if self.active_reranker == "none":
            return self._retrieve(question, top_k, min_score)
        from .reranking import TOP_N, rules_rerank

        candidates = self._retrieve(question, max(TOP_N, top_k), min_score)
        if not candidates:
            return []
        started = perf_counter()
        if self.active_reranker == "crossencoder":
            try:
                passages = [f"{hit.chunk.title}\n{hit.chunk.section}\n{hit.chunk.text}" for hit in candidates[:TOP_N]]
                scores = np.asarray(cast(CrossEncoder, self.crossencoder).predict(question, passages), dtype=np.float32)
                if scores.shape != (len(passages),) or not np.isfinite(scores).all():
                    raise ValueError("Escores inválidos do reranker.")
                ordered = [candidates[int(i)] for i in np.argsort(-scores, kind="stable")]
            except Exception as error:
                self.active_reranker = "rules"
                self.crossencoder = None
                self.reranker_fallback_reason = (
                    f"Falha no cross-encoder ({type(error).__name__}); usando regras locais."
                )
                ordered = rules_rerank(question, candidates[:TOP_N])
        else:
            ordered = rules_rerank(question, candidates[:TOP_N])
        self.last_rerank_ms = round((perf_counter() - started) * 1000, 3)
        return (ordered + candidates[TOP_N:])[:top_k]

    def _retrieve(self, question: str, top_k: int = 3, min_score: float | None = None) -> list[Hit]:
        """Busca com o recuperador ativo; falhas do encoder voltam ao TF-IDF.

        BM25 conserva os filtros lexicais do TF-IDF e muda a ordenação. O
        híbrido filtra por cosseno absoluto e distância do melhor candidato
        antes de fundir BM25 e embeddings por RRF. O corte RRF nunca decide
        se existe evidência. min_score sobrescreve apenas o corte lexical.
        """
        if top_k < 1:
            raise ValueError("top_k deve ser positivo.")
        lexical_cut = 0.14 if min_score is None else min_score
        if self.active_retriever == "tfidf":
            return self._search_tfidf(question, top_k, lexical_cut)
        from .ranking import rrf

        if self.active_retriever == "bm25":
            eligible = {hit.chunk.id for hit in self._search_tfidf(question, len(self.chunks), lexical_cut)}
            scores = cast(BM25, self.bm25).scores(expanded_query(question))
            return [
                Hit(self.chunks[int(i)], round(float(scores[i]), 5))
                for i in np.argsort(-scores, kind="stable")
                if scores[i] > 0 and self.chunks[int(i)].id in eligible
            ][:top_k]
        try:
            query = self._vectors(cast(Embedder, self.embedder).encode([question], query=True), 1)
            semantic = (cast(NDArray[np.float32], self.embeddings) @ query.T).ravel()
        except Exception as error:
            self.active_retriever = "tfidf"
            self.fallback_reason = f"Falha na consulta ao encoder ({type(error).__name__}); usando TF-IDF."
            self.embedder = self.embeddings = None
            return self._search_tfidf(question, top_k, lexical_cut)
        semantic_eligible = {
            int(i)
            for i in range(len(self.chunks))
            if semantic[i] >= HYBRID_MIN_COSINE and semantic.max() - semantic[i] <= HYBRID_MAX_GAP
        }
        dense_rank = [int(i) for i in np.argsort(-semantic, kind="stable")[:10] if int(i) in semantic_eligible]
        lexical = cast(BM25, self.bm25).scores(expanded_query(question))
        sparse_rank = [
            int(i) for i in np.argsort(-lexical, kind="stable")[:10] if lexical[i] > 0 and int(i) in semantic_eligible
        ]
        return [Hit(self.chunks[i], round(score, 6)) for i, score in rrf([dense_rank, sparse_rank])[:top_k]]

    def _search_tfidf(self, question: str, top_k: int = 3, min_score: float = 0.14) -> list[Hit]:
        """Classifica trechos por relação lexical e filtra evidências pouco relacionadas.

        Args:
            question: Pergunta a normalizar e expandir com aliases do domínio.
            top_k: Limite positivo de fontes pretendido; o padrão é três.
            min_score: Corte aplicado à soma ponderada dos escores TF-IDF.

        Returns:
            Lista de Hit em ordem decrescente de escore, possivelmente vazia.
            Combina palavras com peso 0,8 e caracteres com peso 0,2. Também exige
            sobreposição não vazia e cobertura de 20% dos termos expandidos.

        O corte padrão 0,14 é um critério da demonstração, não uma probabilidade
        de resposta correta. Empates preservam a ordem dos trechos no índice.
        """
        query = expanded_query(question)
        query_terms = terms(query)
        word_scores = (self.word_matrix @ cast(csr_matrix, self.word.transform([query])).T).toarray().ravel()
        char_scores = (self.char_matrix @ cast(csr_matrix, self.char.transform([query])).T).toarray().ravel()
        # Caracteres ajudam em variações de escrita; palavras continuam dominando o ranking.
        scores = 0.8 * word_scores + 0.2 * char_scores
        hits = []
        for position in np.argsort(-scores, kind="stable"):
            chunk = self.chunks[int(position)]
            # O escore isolado pode aproximar textos sem cobrir o assunto perguntado.
            overlap = query_terms & terms(chunk.section + " " + chunk.text)
            if float(scores[position]) >= min_score and overlap and len(overlap) / max(len(query_terms), 1) >= 0.2:
                hits.append(Hit(chunk, round(float(scores[position]), 5)))
            if len(hits) == top_k:
                break
        return hits
