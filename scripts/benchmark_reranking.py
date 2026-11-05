"""Mede recursos e latência do reranking em processo Linux, sem OpenAI."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def rss_mib():
    """RSS corrente do processo Linux; o pico usa getrusage separadamente."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return round(int(line.split()[1]) / 1024, 2)
    raise RuntimeError("RSS indisponível neste sistema.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reranker", choices=["none", "rules", "crossencoder"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat deve ser positivo")
    cache = Path(os.environ.get("DOCORA_MODEL_CACHE", "/tmp/docora-models"))
    preexisting = {
        "e5": any(cache.glob("models--Xenova--multilingual-e5-small/snapshots/*/onnx/*.onnx")),
        "crossencoder": any(cache.glob("models--cross-encoder--mmarco*/snapshots/*/onnx/*.onnx")),
    }
    started = perf_counter()
    from docora.assistant import answer_question, guard_question
    from docora.retrieval import KnowledgeIndex, demo_chunks
    from docora.crossencoder import MODEL_REPO, MODEL_REVISION, MODEL_FILE
    import numpy as np

    # Mesma base E5 para isolar a memória adicional do reranker.
    index = KnowledgeIndex(demo_chunks(), retriever="hybrid", reranker=args.reranker)
    assert index.active_retriever == "hybrid", index.fallback_reason
    assert index.active_reranker == args.reranker, index.reranker_fallback_reason
    question = "Em quanto tempo o pedido chega na capital paulista?"
    answer = answer_question(question, index)
    first_answer_ms = (perf_counter() - started) * 1000
    assert answer.sources[0]["section"] == "Regiões atendidas e prazo de entrega"
    assert answer_question("Qual a receita de bolo de cenoura?", index).status == "not_found"
    rss_index = rss_mib()

    import streamlit as st
    from streamlit.testing.v1 import AppTest

    provider = Mock(side_effect=AssertionError("Chamada à OpenAI proibida"))
    with (
        patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "", "OPENAI_MODEL": "", "DOCORA_RETRIEVER": "hybrid", "DOCORA_RERANKER": args.reranker},
        ),
        patch("openai.OpenAI", provider),
    ):
        st.cache_resource.clear()
        ui_started = perf_counter()
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
        assert not app.exception and not app.error
        app.chat_input[0].set_value(question).run()
        ui_first_answer_ms = (perf_counter() - ui_started) * 1000
        assert not app.exception
        ui_answer = app.session_state["history"][-1]["answer"]
        assert ui_answer["sources"][0]["section"] == "Regiões atendidas e prazo de entrega"
        provider.assert_not_called()
    rss_ui = rss_mib()

    cases = json.loads((ROOT / "evals/heldout_questions.json").read_text())
    rows = []
    for repeat in range(args.repeat):
        for case in cases:
            if guard_question(case["question"]):
                continue
            now = perf_counter()
            hits = index.search(case["question"])
            search_ms = (perf_counter() - now) * 1000
            rerank_ms = index.last_rerank_ms
            response = answer_question(case["question"], index)
            rows.append(
                {
                    "repeat": repeat,
                    "id": case["id"],
                    "candidates_returned": len(hits),
                    "search_ms": round(search_ms, 3),
                    "rerank_ms": rerank_ms,
                    "answer_ms": response.elapsed_ms,
                }
            )
    searches = [r["search_ms"] for r in rows]
    reranks = [r["rerank_ms"] for r in rows if r["candidates_returned"]]
    # Exercita dez pares mesmo quando os filtros da demo retornam menos.
    from docora.reranking import rules_rerank
    from docora.retrieval import Hit

    top10_times = []
    for _ in range(3):
        now = perf_counter()
        if args.reranker == "crossencoder":
            index.crossencoder.predict(question, [f"{c.title}\n{c.section}\n{c.text}" for c in index.chunks[:10]])
        elif args.reranker == "rules":
            rules_rerank(question, [Hit(c, 1) for c in index.chunks[:10]])
        top10_times.append((perf_counter() - now) * 1000)
    files = list(cache.glob("models--cross-encoder--mmarco*/snapshots/*/onnx/*.onnx"))
    report = {
        "executed_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "retriever": index.active_retriever,
        "reranker": index.active_reranker,
        "model": MODEL_REPO if args.reranker == "crossencoder" else None,
        "revision": MODEL_REVISION if args.reranker == "crossencoder" else None,
        "model_file": MODEL_FILE if args.reranker == "crossencoder" else None,
        "model_bytes": files[0].stat().st_size if files and args.reranker == "crossencoder" else 0,
        "cache_preexisting": preexisting,
        "download_allowed": os.environ.get("DOCORA_ALLOW_MODEL_DOWNLOAD") == "1",
        "index_build_ms": index.build_ms,
        "cold_process_first_answer_ms": round(first_answer_ms, 2),
        "streamlit_first_answer_cached_model_ms": round(ui_first_answer_ms, 2),
        "rss_after_index_mib": rss_index,
        "rss_including_streamlit_mib": rss_ui,
        "peak_rss_including_streamlit_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
        "search_mean_ms": round(float(np.mean(searches)), 3),
        "search_p50_ms": round(float(np.median(searches)), 3),
        "search_p95_ms": round(float(np.percentile(searches, 95)), 3),
        "answer_mean_ms": round(float(np.mean([r["answer_ms"] for r in rows])), 3),
        "answer_p95_ms": round(float(np.percentile([r["answer_ms"] for r in rows], 95)), 3),
        "rerank_nonempty_mean_ms": round(float(np.mean(reranks)), 3),
        "rerank_nonempty_p95_ms": round(float(np.percentile(reranks, 95)), 3),
        "ten_pairs_rerank_mean_ms": round(float(np.mean(top10_times)), 3),
        "queries": len(rows),
        "openai_calls": provider.call_count,
        "measurements": rows,
        "scope": "CPU Linux local, processos novos por opção; inclui E5, Streamlit AppTest e dois índices compartilhando modelos. Latência de busca inclui OOD sem guardrails; rerank só candidatos aceitos (até dez). Não mede Cloud nem concorrência.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "measurements"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
