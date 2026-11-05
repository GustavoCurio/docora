"""Métricas locais de recuperação, abstenção e suporte lexical das respostas."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import re

from .assistant import answer_question
from .retrieval import terms


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
    body = answer.message.removeprefix("Encontrei estas informações nos documentos:\n\n")
    blocks = re.split(r"\*\*[^\n]+\*\* \[(F\d+)\]\s*", body)
    current = None
    supported = total = 0
    for position, block in enumerate(blocks):
        if position % 2:
            current = block
            continue
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", block):
            tokens = terms(sentence)
            if not tokens:
                continue
            total += 1
            overlap = len(tokens & by_ref.get(current, set())) / len(tokens)
            supported += int(current in by_ref and overlap >= 0.8)
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
        supported += support["supported_sentences"]
        sentences += support["sentences"]
        results.append({**case, "actual_status": answer.status,
                        "pass": answer.status == case["expected_status"] and source_ok,
                        "first_relevant_rank": first,
                        "retrieved_sections": [s["section"] for s in answer.sources],
                        "ranking_at_5": [{"source": h.chunk.source, "section": h.chunk.section, "score": h.score} for h in ranking],
                        "faithfulness": support, "elapsed_ms": answer.elapsed_ms})
    predicted_nf = sum(row["actual_status"] == "not_found" for row in results)
    correct_nf = sum(row["actual_status"] == row["expected_status"] == "not_found" for row in results)
    ood = sum(row["expected_status"] == "not_found" for row in results)
    false_positive = sum(row["expected_status"] == "not_found" and row["actual_status"] == "evidence_found" for row in results)
    passed = sum(row["pass"] for row in results)

    def percent(numerator, denominator):
        return round(100 * numerator / denominator, 2) if denominator else None

    return {
        "executed_utc": datetime.now(timezone.utc).isoformat(), "mode": "extractive", "suite": suite,
        "retriever_requested": getattr(index, "requested_retriever", "tfidf"),
        "retriever_active": getattr(index, "active_retriever", "tfidf"),
        "fallback_reason": getattr(index, "fallback_reason", None),
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
        "scope": "Base fictícia Lume Casa; ranking após filtros; MRR limitado a cinco; suporte lexical não mede relevância nem implicação semântica.",
        "results": results,
    }
