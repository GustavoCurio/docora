"""Normalização e stopwords compartilhadas, sem import circular."""

import re
import unicodedata

STOP_WORDS = set(
    "a ao aos as o os de da das do dos em um uma uns umas e ou se que qual quais como quando onde por para com sem sobre no na nos nas eu meu minha voce voces me posso pode podem poderia quero saber funciona tem tenho faz fazer e ela ele isso esse essa este esta isto sao ser foi muito por favor quanto quantos ate".split()
)


def normalize(text):
    """Normaliza caixa e remove marcas de acento para a comparação lexical."""
    return "".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c))


def tokenize(text):
    """Tokens com ao menos dois caracteres, excluindo stopwords."""
    return [token for token in re.findall(r"\b\w{2,}\b", normalize(text)) if token not in STOP_WORDS]
