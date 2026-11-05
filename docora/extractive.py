"""Seleção de sentenças em português, reutilizando o encoder quando disponível."""

from difflib import SequenceMatcher
import re

import numpy as np

from .retrieval import expanded_query, normalize, terms

MAX_SENTENCES = 6
MAX_WORDS = 120


def split_sentences(text):
    """Separa parágrafos/listas e pontuação, preservando decimais e abreviações.

    É uma heurística para português, sem dependência de modelos linguísticos.
    Não corta uma sentença no meio; o texto permanece um fragmento da fonte.
    """
    result = []
    for line in text.splitlines():
        start = 0
        for match in re.finditer(r"[.!?]+(?=\s+|$)", line):
            end = match.end()
            prefix = line[start:end]
            if match[0] == "." and re.search(r"\b(?:sr|sra|srta|dr|dra|prof|profa|av|art|etc|pág|ex)\.$", prefix, re.I):
                continue
            if match[0] == "." and re.search(r"\b[A-ZÁÉÍÓÚ]\.$", prefix):
                continue
            sentence = line[start:end].strip()
            if terms(sentence):
                result.append(sentence)
            start = end
        remaining = line[start:].strip()
        if terms(remaining):
            result.append(remaining)
    return result


def near_duplicate(left, right):
    """Remove repetição exata, Jaccard >=0,85 ou semelhança textual >=0,92."""
    # Textos quase iguais podem expressar políticas incompatíveis. Preserve
    # números, negações e condições diferentes para inspeção nas fontes.
    signature = lambda text: re.findall(
        r"\b(?:\d+(?:[.,]\d+)*|não|nao|nunca|sem|antes|após|apos|até|ate|exceto|somente)\b", normalize(text)
    )
    if signature(left) != signature(right):
        return False
    a, b = terms(left), terms(right)
    return (
        normalize(left) == normalize(right)
        or bool(a | b)
        and len(a & b) / len(a | b) >= 0.85
        or SequenceMatcher(None, normalize(left), normalize(right), autojunk=False).ratio() >= 0.92
    )


def select_sentences(question, sources, embedder=None):
    """Seleciona até seis sentenças/120 palavras, cada uma com sua referência.

    Overlap normalizado e uma prioridade 1/(rank+1) ordenam as candidatas.
    Quando fornecido, o encoder já carregado adiciona similaridade por cosseno;
    falhas de vetorização preservam a seleção lexical, sem baixar outro modelo.
    Uma rodada por fonte recuperada favorece perguntas com vários assuntos;
    as vagas restantes exigem overlap e seguem o escore. A apresentação agrupa fontes pelo
    ranking e preserva a posição original das sentenças em cada trecho.
    """
    query = terms(expanded_query(question))
    candidates = []
    for rank, source in enumerate(sources):
        heading_overlap = len(query & terms(source["section"])) / max(len(query), 1)
        for position, sentence in enumerate(split_sentences(source["text"])):
            overlap = len(query & terms(sentence)) / max(len(query), 1)
            score = overlap + 0.15 * heading_overlap + 0.2 / (rank + 1)
            candidates.append((score, rank, position, sentence, source["reference"]))
    if candidates and embedder is not None:
        try:
            vectors = np.asarray(embedder.encode([c[3] for c in candidates]), dtype=np.float32)
            query_vector = np.asarray(embedder.encode([question], query=True), dtype=np.float32)
            if (
                vectors.ndim != 2
                or query_vector.shape != (1, vectors.shape[1])
                or vectors.shape[0] != len(candidates)
                or not np.isfinite(vectors).all()
                or not np.isfinite(query_vector).all()
            ):
                raise ValueError("Vetores de sentenças inválidos.")
            norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector)
            if (norms < 1e-12).any():
                raise ValueError("Vetor de sentença vazio.")
            semantic = (vectors @ query_vector.T).ravel() / norms
            candidates = [(c[0] + float(similarity), *c[1:]) for c, similarity in zip(candidates, semantic)]
        except Exception:
            # Uma falha opcional na seleção nunca impede a resposta lexical.
            pass
    ordered = sorted(candidates, key=lambda c: (-c[0], c[1], c[2]))
    selected, words = [], 0

    def add(candidate):
        nonlocal words
        count = len(candidate[3].split())
        if len(selected) >= MAX_SENTENCES or words + count > MAX_WORDS:
            return
        if any(near_duplicate(candidate[3], old[3]) for old in selected):
            return
        selected.append(candidate)
        words += count

    for rank in range(len(sources)):
        for candidate in ordered:
            if candidate[1] == rank:
                before = len(selected)
                add(candidate)
                if len(selected) > before:
                    break
    for candidate in ordered:
        if query & terms(candidate[3]):
            add(candidate)
    selected.sort(key=lambda c: (c[1], c[2]))
    return [(sentence, reference) for _, _, _, sentence, reference in selected]
