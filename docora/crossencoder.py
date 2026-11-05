"""Cross-encoder mMARCO multilíngue em ONNX INT8, inteiramente local."""
from functools import lru_cache
import os
from pathlib import Path
import tempfile
from threading import RLock

import numpy as np

MODEL_REPO = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MODEL_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"
MODEL_FILE = "onnx/model_quint8_avx2.onnx"


class LocalCrossEncoder:
    """Pontua pares pergunta/trecho; lotes de dois e limite de 256 tokens.

    Tokenizer e sessão são compartilhados sob lock, com duas threads CPU,
    arena desativada e sem Torch, transformers ou código remoto do modelo.
    """
    def __init__(self, cache_dir, allow_download=False):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from huggingface_hub import hf_hub_download

        paths = {file: hf_hub_download(
            MODEL_REPO, file, revision=MODEL_REVISION, cache_dir=cache_dir,
            local_files_only=not allow_download, token=False, etag_timeout=5,
        ) for file in ("tokenizer.json", MODEL_FILE)}
        # Ausência do cache não importa os runtimes nativos.
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(paths["tokenizer.json"])
        self.tokenizer.enable_truncation(max_length=256, strategy="longest_first")
        self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        self.session = ort.InferenceSession(paths[MODEL_FILE], sess_options=options, providers=["CPUExecutionProvider"])
        self.inputs = {item.name for item in self.session.get_inputs()}
        self.lock = RLock()

    def predict(self, question, passages):
        """Retorna um logit por par; escores servem só para ordenar, sem corte OOD."""
        scores = []
        with self.lock:
            for offset in range(0, len(passages), 2):
                batch = passages[offset:offset + 2]
                encoded = self.tokenizer.encode_batch([(question, text) for text in batch])
                inputs = {
                    "input_ids": np.asarray([item.ids for item in encoded], dtype=np.int64),
                    "attention_mask": np.asarray([item.attention_mask for item in encoded], dtype=np.int64),
                    "token_type_ids": np.asarray([item.type_ids for item in encoded], dtype=np.int64),
                }
                logits = np.asarray(self.session.run(None, {k: v for k, v in inputs.items() if k in self.inputs})[0])
                if logits.shape != (len(batch), 1) or not np.isfinite(logits).all():
                    raise ValueError("Escores inválidos do cross-encoder.")
                scores.extend(logits[:, 0].tolist())
        return scores


@lru_cache(maxsize=1)
def load_crossencoder():
    """Somente cache, salvo DOCORA_ALLOW_MODEL_DOWNLOAD=1; falha memorizada.

    DOCORA_RERANKER=crossencoder é uma escolha explícita. O padrão none
    nunca chama este loader, mesmo com o download de embeddings autorizado.
    """
    if os.environ.get("DOCORA_DISABLE_RERANKER") == "1":
        return None, "Cross-encoder desativado; usando regras locais."
    cache = Path(os.environ.get("DOCORA_MODEL_CACHE", str(Path(tempfile.gettempdir()) / "docora-models")))
    try:
        return LocalCrossEncoder(cache, allow_download=os.environ.get("DOCORA_ALLOW_MODEL_DOWNLOAD") == "1"), None
    except Exception as error:
        return None, f"Cross-encoder indisponível ({type(error).__name__}); usando regras locais."
