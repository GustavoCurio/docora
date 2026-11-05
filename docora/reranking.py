"""Reranking de candidatos já aceitos, com regras sem modelo adicional."""

from .retrieval import expanded_query, normalize, terms

TOP_N = 10


def rules_rerank(question, hits):
    """Combina prioridade do ranking com overlap de seção, título e corpo.

    Pesos fixos, sem aliases adicionais: 0,5/(rank+1), seção 0,3, título
    0,1, corpo 0,2 e frase da seção 0,1. Não cria nem elimina evidências.
    Mantém os escores originais; a posição expressa a nova ordenação.
    """
    query = terms(expanded_query(question))
    denominator = max(len(query), 1)
    plain = normalize(question)

    def score(item):
        rank, hit = item
        chunk = hit.chunk
        return (
            0.5 / (rank + 1)
            + 0.3 * len(query & terms(chunk.section)) / denominator
            + 0.1 * len(query & terms(chunk.title)) / denominator
            + 0.2 * len(query & terms(chunk.text)) / denominator
            + 0.1 * (normalize(chunk.section) in plain)
        )

    return [hit for _, hit in sorted(enumerate(hits), key=lambda item: -score(item))]
