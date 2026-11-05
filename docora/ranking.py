"""BM25 e fusão por posição, sem dependências de modelos ou rede."""

from collections import Counter
import math

import numpy as np

from .textnorm import tokenize


class BM25:
    """BM25 Okapi com IDF positivo e parâmetros usuais k1=1,5 e b=0,75."""

    def __init__(self, corpus, k1=1.5, b=0.75):
        self.documents = [Counter(tokenize(text)) for text in corpus]
        self.lengths = np.asarray([sum(doc.values()) for doc in self.documents], dtype=np.float64)
        self.average_length = max(float(self.lengths.mean()) if len(self.lengths) else 1.0, 1.0)
        frequency = Counter(token for doc in self.documents for token in doc)
        self.idf = {
            token: math.log(1 + (len(corpus) - count + 0.5) / (count + 0.5)) for token, count in frequency.items()
        }
        self.k1, self.b = k1, b

    def scores(self, question):
        """Soma IDF × saturação TF; termos repetidos da consulta contam uma vez."""
        result = np.zeros(len(self.documents), dtype=np.float64)
        for token in set(tokenize(question)):
            frequencies = np.asarray([doc[token] for doc in self.documents], dtype=np.float64)
            denominator = frequencies + self.k1 * (1 - self.b + self.b * self.lengths / self.average_length)
            result += self.idf.get(token, 0.0) * frequencies * (self.k1 + 1) / np.maximum(denominator, 1e-12)
        return result


def rrf(rankings, k=60):
    """Soma 1/(k+posição) por lista; empates preservam a primeira aparição.

    Posições começam em um. Duplicatas numa mesma lista não somam duas vezes.
    Os escores de BM25 e cosseno não são comparados nem tratados como confiança.
    """
    if k < 0:
        raise ValueError("A constante RRF deve ser não negativa.")
    scores = {}
    for ranking in rankings:
        for position, item in enumerate(dict.fromkeys(ranking), 1):
            scores[item] = scores.get(item, 0) + 1 / (k + position)
    return sorted(scores.items(), key=lambda pair: -pair[1])
