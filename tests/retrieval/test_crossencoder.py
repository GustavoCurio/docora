"""Cross-encoder: apenas mocks na suíte rápida, sem pesos nem rede."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from docora import crossencoder as ce
from docora.assistant import answer_question
from docora.retrieval import Hit, KnowledgeIndex, chunks_from_text, demo_chunks


@pytest.mark.parametrize("allowed", [False, True])
def test_download_gate_and_shared_loader(allowed, monkeypatch, tmp_path):
    monkeypatch.delenv("DOCORA_DISABLE_RERANKER")
    monkeypatch.setenv("DOCORA_ALLOW_MODEL_DOWNLOAD", "1" if allowed else "0")
    monkeypatch.setenv("DOCORA_MODEL_CACHE", str(tmp_path))
    factory = Mock()
    monkeypatch.setattr(ce, "LocalCrossEncoder", factory)
    assert ce.load_crossencoder()[0] is ce.load_crossencoder()[0]
    factory.assert_called_once_with(tmp_path, allow_download=allowed)


@pytest.mark.parametrize("failure", [ImportError, OSError, MemoryError])
def test_cached_load_failures_fall_back_cleanly(failure, monkeypatch):
    monkeypatch.delenv("DOCORA_DISABLE_RERANKER")
    factory = Mock(side_effect=failure("Falha simulada"))
    monkeypatch.setattr(ce, "LocalCrossEncoder", factory)
    for _ in range(2):
        index = KnowledgeIndex(demo_chunks(), reranker="crossencoder")
        assert index.active_reranker == "rules"
        assert failure.__name__ in index.reranker_fallback_reason
        assert answer_question("Prazo de entrega?", index).status == "evidence_found"
    factory.assert_called_once()


def test_disabled_and_default_modes_never_load_model(monkeypatch):
    factory = Mock(side_effect=AssertionError("Modelo indevido"))
    monkeypatch.setattr(ce, "LocalCrossEncoder", factory)
    assert ce.load_crossencoder()[0] is None
    for mode in ["none", "rules"]:
        KnowledgeIndex(demo_chunks(), reranker=mode).search("Desconto Pix?")
    factory.assert_not_called()


def test_only_top_ten_eligible_pairs_are_scored_and_prefix_is_stable(monkeypatch):
    chunks = chunks_from_text(
        "\n".join(f"## Seção {i}\nA entrega da loja tem prazo de {i} dias." for i in range(12)), "base.md"
    )
    model = SimpleNamespace(predict=Mock(return_value=list(range(10))))
    index = KnowledgeIndex(chunks, reranker="crossencoder", crossencoder=model)
    hits = [Hit(c, 1) for c in chunks]
    monkeypatch.setattr(index, "_retrieve", lambda *args: hits)
    assert index.search("entrega", 1) == [hits[9]]
    assert index.search("entrega", 5) == list(reversed(hits[:10]))[:5]
    assert index.search("entrega", 12) == list(reversed(hits[:10])) + hits[10:]
    assert len(model.predict.call_args.args[1]) == 10
    assert "Seção 0" in model.predict.call_args.args[1][0]
    assert index.active_reranker == "crossencoder"


@pytest.mark.parametrize("bad", [None, [np.nan], [np.inf], [[1]], [], [1, 2]])
def test_malformed_scores_fall_back_without_losing_evidence(bad, monkeypatch):
    index = KnowledgeIndex(demo_chunks(), reranker="crossencoder", crossencoder=Mock(predict=Mock(return_value=bad)))
    hit = Hit(index.chunks[1], 1)
    monkeypatch.setattr(index, "_retrieve", lambda *args: [hit])
    assert index.search("entrega") == [hit]
    assert index.active_reranker == "rules"
    assert index.reranker_fallback_reason


@pytest.mark.parametrize("failure", [RuntimeError, MemoryError])
def test_query_failure_is_not_retried(failure):
    model = Mock(predict=Mock(side_effect=failure("Falha simulada")))
    index = KnowledgeIndex(demo_chunks(), reranker="crossencoder", crossencoder=model)
    for _ in range(2):
        assert answer_question("Desconto Pix?", index).status == "evidence_found"
    assert index.active_reranker == "rules"
    model.predict.assert_called_once()


def test_ood_and_guard_never_call_predict():
    model = Mock()
    index = KnowledgeIndex(demo_chunks(), reranker="crossencoder", crossencoder=model)
    assert answer_question("receita de bolo de cenoura", index).status == "not_found"
    assert answer_question("Consulte pedido #123456", index).status == "needs_human"
    model.predict.assert_not_called()


def test_onnx_adapter_uses_pairs_masks_batches_and_no_torch(monkeypatch, tmp_path):
    downloader = Mock(side_effect=lambda repo, file, **kw: str(tmp_path / file))
    token = SimpleNamespace(ids=[0, 7, 2], attention_mask=[1, 1, 1], type_ids=[0, 0, 0])
    tokenizer = Mock()
    tokenizer.encode_batch.side_effect = lambda pairs: [token for _ in pairs]
    session = Mock()
    session.get_inputs.return_value = [SimpleNamespace(name=n) for n in ["input_ids", "attention_mask"]]
    session.run.side_effect = lambda _, inputs: [np.ones((len(inputs["input_ids"]), 1), dtype=np.float32)]
    runtime = SimpleNamespace(
        SessionOptions=Mock(),
        InferenceSession=Mock(return_value=session),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_BASIC=1),
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=downloader))
    monkeypatch.setitem(
        sys.modules, "tokenizers", SimpleNamespace(Tokenizer=SimpleNamespace(from_file=Mock(return_value=tokenizer)))
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    model = ce.LocalCrossEncoder(tmp_path)
    assert model.predict("pergunta", ["um", "dois", "três"]) == [1, 1, 1]
    assert tokenizer.encode_batch.call_args_list[0].args[0] == [("pergunta", "um"), ("pergunta", "dois")]
    assert session.run.call_count == 2
    assert "token_type_ids" not in session.run.call_args.args[1]
    assert session.run.call_args.args[1]["input_ids"].dtype == np.int64
    assert all(c.kwargs["local_files_only"] for c in downloader.call_args_list)
    assert runtime.InferenceSession.call_args.kwargs["providers"] == ["CPUExecutionProvider"]
    tokenizer.enable_truncation.assert_called_once_with(max_length=256, strategy="longest_first")
    session.run.return_value = [np.array([[np.nan]])]
    session.run.side_effect = None
    with pytest.raises(ValueError, match="Escores"):
        model.predict("pergunta", ["texto"])
