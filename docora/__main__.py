"""Avaliação reproduzível e consultas do Docora pelo terminal."""
import argparse
import json
import os
from pathlib import Path
import tempfile

from .assistant import answer_question
from .evaluation import evaluate
from .retrieval import KnowledgeIndex, demo_chunks

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Oferece os subcomandos ask e evaluate sobre a biblioteca demonstrativa.

    ask imprime a resposta em JSON; evaluate grava o relatório no caminho
    escolhido e encerra com código 1 se algum caso não cumprir seu critério.
    O código de sucesso é zero. Falhas de leitura ou gravação são propagadas.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    assessment = sub.add_parser("evaluate")
    assessment.add_argument("--output", type=Path, default=Path(tempfile.gettempdir()) / "docora" / "evaluation.json")
    assessment.add_argument("--suite", choices=["original", "heldout", "all"], default="original")
    assessment.add_argument("--allow-failures", action="store_true", help="Grava medições incompletas sem encerrar com código 1.")
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    for command in (ask, assessment):
        command.add_argument("--retriever", choices=["tfidf", "bm25", "hybrid"], default=None,
                             help="Padrão: DOCORA_RETRIEVER ou hybrid, com fallback TF-IDF.")
    sub.add_parser("prepare-model", help="Baixa o encoder público para o cache local; não faz chamadas à OpenAI.")
    args = parser.parse_args()
    if args.command == "prepare-model":
        from .embeddings import load_embedder
        os.environ["DOCORA_ALLOW_MODEL_DOWNLOAD"] = "1"
        model, reason = load_embedder()
        print("Modelo local pronto no cache." if model else reason)
        raise SystemExit(0 if model else 1)
    index = KnowledgeIndex(demo_chunks(), retriever=args.retriever)
    if args.command == "ask":
        result = answer_question(args.question, index).to_dict()
        result.update(retriever_requested=index.requested_retriever, retriever_active=index.active_retriever,
                      fallback_reason=index.fallback_reason)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        suites = ["original", "heldout"] if args.suite == "all" else [args.suite]
        reports = []
        for suite in suites:
            name = "questions.json" if suite == "original" else "heldout_questions.json"
            cases = json.loads((ROOT / "evals" / name).read_text(encoding="utf-8"))
            reports.append(evaluate(cases, index, suite))
        report = reports[0] if len(reports) == 1 else {"suites": reports}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        for item in reports:
            print(f"{item['suite']} · {item['retriever_active']}: {item['passed']}/{item['total']} critérios; R@1/3/5 {item['source_recall_at_1_pct']}/{item['source_recall_at_3_pct']}/{item['source_recall_at_5_pct']}; relatório em {args.output}")
        raise SystemExit(0 if args.allow_failures or all(r["passed"] == r["total"] for r in reports) else 1)


if __name__ == "__main__":
    main()
