"""Avaliação reproduzível e consultas do AtendeDoc pelo terminal."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .assistant import answer_question
from .retrieval import KnowledgeIndex, demo_chunks

ROOT = Path(__file__).resolve().parents[1]


def evaluate(cases, index):
    """Verifica status e recuperação de fontes nos casos de referência.

    Args:
        cases: Casos com question/expected_status e, quando aplicável, source/section.
        index: Biblioteca sobre a qual todas as perguntas serão consultadas.

    Returns:
        Relatório com data UTC, critérios atendidos, recall de fonte @3 e
        resultado individual dos casos. A lista de casos deve ser não vazia.

    Executa somente a consulta extrativa. Recall considera os casos com
    fonte esperada e não mede a redação de um modelo generativo.
    """
    results = []
    answerable = hits = 0
    for case in cases:
        answer = answer_question(case["question"], index)
        status_ok = answer.status == case["expected_status"]
        source_ok = True
        if case.get("source"):
            answerable += 1
            source_ok = any(source["source"] == case["source"] and case["section"] == source["section"] for source in answer.sources)
            hits += int(source_ok)
        results.append({**case, "actual_status": answer.status, "pass": status_ok and source_ok,
                        "retrieved_sections": [source["section"] for source in answer.sources], "elapsed_ms": answer.elapsed_ms})
    return {"executed_utc": datetime.now(timezone.utc).isoformat(), "mode": "extractive", "total": len(cases),
            "passed": sum(row["pass"] for row in results),
            "pass_rate_pct": round(100 * sum(row["pass"] for row in results) / len(results), 2),
            "source_recall_at_3_pct": round(100 * hits / answerable, 2) if answerable else None,
            "scope": "Perguntas de referência da base fictícia Lume Casa; não é avaliação externa nem teste de geração por LLM.",
            "results": results}


def main():
    """Oferece os subcomandos ask e evaluate sobre a biblioteca demonstrativa.

    ask imprime a resposta em JSON; evaluate grava o relatório no caminho
    escolhido e encerra com código 1 se algum caso não cumprir seu critério.
    O código de sucesso é zero. Falhas de leitura ou gravação são propagadas.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    assessment = sub.add_parser("evaluate")
    assessment.add_argument("--output", type=Path, default=ROOT / "reports" / "evaluation.json")
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    args = parser.parse_args()
    index = KnowledgeIndex(demo_chunks())
    if args.command == "ask":
        print(json.dumps(answer_question(args.question, index).to_dict(), ensure_ascii=False, indent=2))
    else:
        cases = json.loads((ROOT / "evals" / "questions.json").read_text(encoding="utf-8"))
        report = evaluate(cases, index)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{report['passed']}/{report['total']} critérios atendidos; relatório em {args.output}")
        raise SystemExit(0 if report["passed"] == report["total"] else 1)


if __name__ == "__main__":
    main()
