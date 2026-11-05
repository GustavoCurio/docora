"""Encoder multilíngue ONNX quantizado, local e carregado sob demanda."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from threading import RLock

import numpy as np
from numpy.typing import NDArray

MODEL_REPO = "Xenova/multilingual-e5-small"
MODEL_REVISION = "761b726dd34fb83930e26aab4e9ac3899aa1fa78"
MODEL_FILE = "onnx/model_quantized.onnx"


class LocalEmbedder:
    """Pooling médio com máscara, normalização L2 e prefixos do E5.

    Importações pesadas ocorrem somente na criação. Lotes de quatro, limite de
    256 tokens, duas threads e arena desativada reduzem o pico no Cloud. O lock
    protege o tokenizador compartilhado por sessões do Streamlit.
    """

    def __init__(self, cache_dir: str | Path, allow_download: bool = False) -> None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from huggingface_hub import hf_hub_download

        paths = {}
        for file in ("tokenizer.json", MODEL_FILE):
            paths[file] = hf_hub_download(
                MODEL_REPO,
                file,
                revision=MODEL_REVISION,
                cache_dir=cache_dir,
                local_files_only=not allow_download,
                token=False,
                etag_timeout=5,
            )
        # Cache ausente não precisa importar os runtimes nativos no cold start.
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(paths["tokenizer.json"])
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        self.session = ort.InferenceSession(paths[MODEL_FILE], sess_options=options, providers=["CPUExecutionProvider"])
        self.inputs = {item.name for item in self.session.get_inputs()}
        self.lock = RLock()

    def encode(self, texts: Sequence[str], *, query: bool = False) -> NDArray[np.float32]:
        """Retorna float32 (n, 384); nunca envia textos a serviços remotos."""
        prefix = "query: " if query else "passage: "
        vectors = []
        with self.lock:
            for offset in range(0, len(texts), 4):
                encoded = self.tokenizer.encode_batch([prefix + text for text in texts[offset : offset + 4]])
                mask = np.asarray([item.attention_mask for item in encoded], dtype=np.int64)
                inputs = {
                    "input_ids": np.asarray([item.ids for item in encoded], dtype=np.int64),
                    "attention_mask": mask,
                    "token_type_ids": np.asarray([item.type_ids for item in encoded], dtype=np.int64),
                }
                output = self.session.run(None, {key: value for key, value in inputs.items() if key in self.inputs})[0]
                pooled = (output * mask[..., None]).sum(axis=1) / np.maximum(mask.sum(axis=1, keepdims=True), 1)
                pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
                vectors.append(pooled.astype(np.float32))
        return np.concatenate(vectors) if vectors else np.empty((0, 384), dtype=np.float32)


@lru_cache(maxsize=1)
def load_embedder() -> tuple[LocalEmbedder | None, str | None]:
    """Reutiliza modelo ou falha; sem download implícito no boot da demo.

    DOCORA_ALLOW_MODEL_DOWNLOAD=1 autoriza baixar apenas pesos/tokenizador
    públicos; a inferência é local. Por padrão só usa o cache. Falhas são
    memorizadas até reiniciar o processo, evitando tentativas a cada consulta.
    """
    if os.environ.get("DOCORA_DISABLE_EMBEDDINGS") == "1":
        return None, "Embeddings desativados pela configuração."
    cache_dir = Path(os.environ.get("DOCORA_MODEL_CACHE", str(Path(tempfile.gettempdir()) / "docora-models")))
    try:
        model = LocalEmbedder(cache_dir, allow_download=os.environ.get("DOCORA_ALLOW_MODEL_DOWNLOAD") == "1")
        return model, None
    except Exception as error:
        # Inclui ImportError, MemoryError, cache ausente e erros nativos do ONNX.
        # OOM com encerramento pelo sistema operacional não é capturável.
        return None, f"Encoder local indisponível ({type(error).__name__}); usando TF-IDF."
