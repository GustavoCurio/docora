"""Mede boot, memória e smoke extrativo real, sem chamar a OpenAI (Linux)."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import sys
import tempfile
from time import perf_counter
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cache = Path(os.environ.get("DOCORA_MODEL_CACHE", str(Path(tempfile.gettempdir()) / "docora-models")))
    cache_preexisting = cache.exists() and any(cache.rglob("*.onnx"))
    started = perf_counter()
    from docora.assistant import answer_question
    from docora.embeddings import load_embedder, MODEL_REPO, MODEL_REVISION
    from docora.retrieval import demo_chunks, KnowledgeIndex
    imported_ms = (perf_counter() - started) * 1000
    index = KnowledgeIndex(demo_chunks(), retriever="hybrid")
    assert index.active_retriever == "hybrid", index.fallback_reason
    question = "Em quanto tempo o pedido chega na capital paulista?"
    answer = answer_question(question, index)
    first_answer_ms = (perf_counter() - started) * 1000
    second = KnowledgeIndex(demo_chunks(), retriever="hybrid")
    assert second.embedder is index.embedder
    assert answer.sources[0]["section"] == "Regiões atendidas e prazo de entrega"

    import streamlit as st
    from streamlit.testing.v1 import AppTest
    provider = Mock(side_effect=AssertionError("Chamada à OpenAI proibida nesta medição"))
    with patch.dict(os.environ, {"OPENAI_API_KEY": "", "OPENAI_MODEL": "", "DOCORA_RETRIEVER": "hybrid"}), patch("openai.OpenAI", provider):
        st.cache_resource.clear()
        boot = perf_counter()
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        streamlit_boot_ms = (perf_counter() - boot) * 1000
        assert not app.exception
        assert not app.error
        app.chat_input[0].set_value(question).run()
        assert not app.exception
        ui_answer = app.session_state["history"][-1]["answer"]
        assert ui_answer["status"] == "evidence_found"
        assert ui_answer["sources"][0]["section"] == "Regiões atendidas e prazo de entrega"
        provider.assert_not_called()
    report = {
        "executed_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "model": MODEL_REPO, "revision": MODEL_REVISION,
        "provider": index.embedder.session.get_providers(),
        "cache_preexisting": bool(cache_preexisting),
        "download_allowed": os.environ.get("DOCORA_ALLOW_MODEL_DOWNLOAD") == "1",
        "cache_bytes": sum(path.stat().st_size for path in cache.rglob("*") if path.is_file() and not path.is_symlink()),
        "imports_ms": round(imported_ms, 2), "first_index_build_ms": index.build_ms,
        "cached_model_index_build_ms": second.build_ms,
        "first_answer_including_imports_ms": round(first_answer_ms, 2),
        "streamlit_boot_with_cached_model_ms": round(streamlit_boot_ms, 2),
        "peak_rss_mib_including_streamlit": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
        "scope": "Processo Linux local, CPU, AppTest sem chave; modelo compartilhado com a interface. Não é medição dentro do Community Cloud nem teste de carga concorrente.",
        "openai_calls": provider.call_count,
        "question": question, "cli_answer": answer.to_dict(), "streamlit_answer": ui_answer,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"cli_answer", "streamlit_answer"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
