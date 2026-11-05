"""Métricas locais de recuperação, abstenção e suporte lexical das respostas."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import re

from .assistant import answer_question
from .retrieval import terms


def answer_units(answer):
    """Lê referências inline novas e cabeçalhos herdados do extrativo anterior."""
    body = answer.message.removeprefix("Encontrei estas informações nos documentos:\n\n")
    current = None
    units = []
    for block in body.splitlines():
        heading = re.fullmatch(r"\*\*[^\n]+\*\* \[(F\d+)\]", block.strip())
        if heading:
            current = heading[1]
            continue
        if not terms(block):
            continue
        inline = re.fullmatch(r"(.+?)\s*\[(F\d+)\]", block.strip())
        if inline:
            units.append((inline[1], inline[2], True))
        else:
            units.extend((sentence, current, False) for sentence in re.split(r"(?<=[.!?])\s+", block)
                         if terms(sentence))
    return units


def answer_checks(answer, gold):
    """Checagens da resposta: cópia, referência, fonte esperada, limite e repetição.

    Não são um julgamento semântico: uma sentença copiada pode omitir ressalvas.
    Os limites de seis sentenças e 120 palavras são fixos para ambas as suítes.
    """
    units = answer_units(answer) if answer.status == "evidence_found" else []
    sources = {s["reference"]: s for s in answer.sources}
    compact = lambda text: " ".join(text.split())
    exact = sum(ref in sources and compact(text) in compact(sources[ref]["text"])
                for text, ref, _ in units)
    words = sum(len(text.split()) for text, _, _ in units)
    cited_gold = any(ref in sources and any(sources[ref]["source"] == g["source"] and
                     sources[ref]["section"] == g["section"] for g in gold) for _, ref, _ in units)
    unique = len({compact(text).casefold() for text, _, _ in units}) == len(units)
    return {"sentences": len(units), "words": words, "exact_source_sentences": exact,
            "inline_cited_sentences": sum(inline and ref in sources for _, ref, inline in units),
            "within_budget": len(units) <= 6 and words <= 120, "no_exact_duplicates": unique,
            "gold_cited_in_body": cited_gold if gold else None}


def expected_sources(case):
    """Aceita uma fonte única ou uma lista de alternativas igualmente válidas."""
    if case.get("acceptable_sources"):
        return case["acceptable_sources"]
    if case.get("source"):
        return [{"source": case["source"], "section": case["section"]}]
    return []


def faithfulness(answer):
    """Conta sentenças com >=80% dos termos presentes no trecho citado.

    Remove somente a introdução fixa, cabeçalhos com referência e divisórias.
    Cada bloco herda a referência de seu cabeçalho; texto sem referência tem
    suporte zero. Não avalia implicação semântica nem pertinência à pergunta.
    """
    if answer.status != "evidence_found":
        return {"supported_sentences": 0, "sentences": 0, "score": None}
    by_ref = {source["reference"]: terms(source["text"]) for source in answer.sources}
    supported = total = 0
    for sentence, reference, _ in answer_units(answer):
        tokens = terms(sentence)
        if not tokens:
            continue
        total += 1
        overlap = len(tokens & by_ref.get(reference, set())) / len(tokens)
        supported += int(reference in by_ref and overlap >= 0.8)
    return {"supported_sentences": supported, "sentences": total,
            "score": supported / total if total else None}


def evaluate(cases, index, suite="original"):
    """Mede ranking filtrado @1/3/5, MRR@5, status e suporte lexical.

    O ranking tem os mesmos filtros da aplicação. Encaminhamentos não fazem
    busca. O critério por caso continua sendo status + fonte entre as três
    fontes efetivamente apresentadas na resposta extrativa.
    """
    if not cases:
        raise ValueError("O conjunto de avaliação precisa conter casos.")
    results = []
    recall = {k: 0 for k in (1, 3, 5)}
    answerable = reciprocal = supported = sentences = 0
    for case in cases:
        answer = answer_question(case["question"], index)
        ranking = index.search(case["question"], top_k=5) if answer.status != "needs_human" else []
        gold = expected_sources(case)
        ranks = [rank for rank, hit in enumerate(ranking, 1)
                 if any(hit.chunk.source == item["source"] and hit.chunk.section == item["section"] for item in gold)]
        first = min(ranks, default=None)
        if gold:
            answerable += 1
            for k in recall:
                recall[k] += int(first is not None and first <= k)
            reciprocal += 1 / first if first else 0
        source_ok = not gold or any(
            source["source"] == item["source"] and source["section"] == item["section"]
            for source in answer.sources for item in gold)
        support = faithfulness(answer)
        checks = answer_checks(answer, gold)
        supported += support["supported_sentences"]
        sentences += support["sentences"]
        results.append({**case, "actual_status": answer.status,
                        "retriever_active": getattr(index, "active_retriever", "tfidf"),
                        "pass": answer.status == case["expected_status"] and source_ok,
                        "first_relevant_rank": first,
                        "retrieved_sections": [s["section"] for s in answer.sources],
                        "ranking_at_5": [{"source": h.chunk.source, "section": h.chunk.section, "score": h.score} for h in ranking],
                        "faithfulness": support, "answer_checks": checks,
                        "answer_text": answer.message, "elapsed_ms": answer.elapsed_ms})
    predicted_nf = sum(row["actual_status"] == "not_found" for row in results)
    correct_nf = sum(row["actual_status"] == row["expected_status"] == "not_found" for row in results)
    ood = sum(row["expected_status"] == "not_found" for row in results)
    false_positive = sum(row["expected_status"] == "not_found" and row["actual_status"] == "evidence_found" for row in results)
    passed = sum(row["pass"] for row in results)

    def percent(numerator, denominator):
        return round(100 * numerator / denominator, 2) if denominator else None

    metadata = {}
    if getattr(index, "active_retriever", "tfidf") == "hybrid":
        from .embeddings import MODEL_REPO, MODEL_REVISION
        from .retrieval import HYBRID_MIN_COSINE, HYBRID_MAX_GAP
        metadata = {"embedding_model": MODEL_REPO, "model_revision": MODEL_REVISION,
                    "min_cosine": HYBRID_MIN_COSINE, "max_cosine_gap": HYBRID_MAX_GAP, "rrf_k": 60}
    return {
        "executed_utc": datetime.now(timezone.utc).isoformat(), "mode": "extractive", "suite": suite,
        "retriever_requested": getattr(index, "requested_retriever", "tfidf"),
        "retriever_active": getattr(index, "active_retriever", "tfidf"),
        "fallback_reason": getattr(index, "fallback_reason", None),
        "reranker_requested": getattr(index, "requested_reranker", "none"),
        "reranker_active": getattr(index, "active_reranker", "none"),
        "reranker_fallback_reason": getattr(index, "reranker_fallback_reason", None),
        "retrieval_parameters": metadata,
        "index_build_ms": getattr(index, "build_ms", None),
        "cases_sha256": sha256(json.dumps(cases, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        "corpus_sha256": sha256(json.dumps([c.to_dict() for c in index.chunks], ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        "total": len(cases), "answerable": answerable, "ood": ood,
        "passed": passed, "pass_rate_pct": percent(passed, len(cases)),
        **{f"source_recall_at_{k}_pct": percent(recall[k], answerable) for k in recall},
        "mrr_at_5": round(reciprocal / answerable, 6) if answerable else None,
        "not_found_precision_pct": percent(correct_nf, predicted_nf),
        "not_found_predicted": predicted_nf, "not_found_correct": correct_nf,
        "ood_false_positive_rate_pct": percent(false_positive, ood), "ood_false_positives": false_positive,
        "faithfulness_pct": percent(supported, sentences), "faithfulness_sentences": sentences,
        "answer_checks": {
            "inline_citation_pct": percent(sum(r["answer_checks"]["inline_cited_sentences"] for r in results), sentences),
            "exact_source_pct": percent(sum(r["answer_checks"]["exact_source_sentences"] for r in results), sentences),
            "gold_cited_in_body_pct": percent(sum(r["answer_checks"]["gold_cited_in_body"] is True for r in results), answerable),
            "answers_within_budget": sum(r["answer_checks"]["within_budget"] for r in results),
            "answers_without_exact_duplicates": sum(r["answer_checks"]["no_exact_duplicates"] for r in results),
            "mean_words_evidence": round(sum(r["answer_checks"]["words"] for r in results) /
                                         max(sum(r["actual_status"] == "evidence_found" for r in results), 1), 2),
        },
        "scope": "Base fictícia Lume Casa; ranking após filtros; MRR limitado a cinco; suporte lexical não mede relevância nem implicação semântica.",
        "results": results,
    }
