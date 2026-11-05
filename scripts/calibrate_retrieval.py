"""Mede cortes no original e negativos de desenvolvimento; não lê held-out."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from docora.embeddings import load_embedder, MODEL_REPO, MODEL_REVISION
from docora.retrieval import demo_chunks, HYBRID_MIN_COSINE, HYBRID_MAX_GAP


def main():
    root = Path(__file__).resolve().parents[1]
    cases = json.loads((root / "evals/questions.json").read_text()) + json.loads(
        (root / "evals/development_questions.json").read_text()
    )
    cases = [case for case in cases if case["expected_status"] != "needs_human"]
    model, reason = load_embedder()
    if model is None:
        raise RuntimeError(reason)
    chunks = demo_chunks()
    matrix = model.encode([c.section + "\n" + c.text for c in chunks])
    # Quantização dinâmica pode variar conforme o padding do lote. A aplicação
    # consulta uma pergunta por vez; a calibração reproduz esse mesmo caminho.
    queries = np.concatenate([model.encode([c["question"]], query=True) for c in cases])
    maximum = (queries @ matrix.T).max(axis=1)
    sweep = []
    for cut in (0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88):
        sweep.append(
            {
                "cut": cut,
                "answerable_accepted": sum(
                    int(score >= cut)
                    for case, score in zip(cases, maximum)
                    if case["expected_status"] == "evidence_found"
                ),
                "ood_accepted": sum(
                    int(score >= cut) for case, score in zip(cases, maximum) if case["expected_status"] == "not_found"
                ),
            }
        )
    result = {
        "model": MODEL_REPO,
        "revision": MODEL_REVISION,
        "scope": "Somente original e negativos de desenvolvimento; nenhum acesso ao held-out.",
        "selected_min_cosine": HYBRID_MIN_COSINE,
        "selected_max_gap": HYBRID_MAX_GAP,
        "selection_rule": "Maior corte da grade que preserva os 20 positivos originais. Negativos de desenvolvimento medem o custo em falsos positivos. Janela fixa de 0,03 para limitar a cauda antes de RRF.",
        "sweep": sweep,
        "results": [{**case, "max_cosine": round(float(score), 6)} for case, score in zip(cases, maximum)],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
