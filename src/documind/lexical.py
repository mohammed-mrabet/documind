"""Recherche lexicale BM25, complémentaire de la recherche sémantique.

Les embeddings comprennent le sens général d'une question, mais ratent parfois un mot
précis (« mot de passe », « USB », « 4444 »). BM25 retrouve justement les extraits qui
contiennent les mêmes mots, en pondérant les mots rares plus fortement que les mots courants.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

STOP_WORDS = set(
    "les des une est que qui quoi quel quelle quels quelles pour par dans sur avec sans pas plus sont ont aux ces cette "
    "son sa ses leur leurs nous vous ils elles elle lui eux mon ton mes tes faut peut doit combien comment quand "
    "dois peux etre avoir fait faire tout tous toute toutes aussi apres avant entre chez donc car mais "
    "the and document atlas industrie".split()
)


def tokenize(text: str) -> list[str]:
    """Minuscules, sans accents, sans mots grammaticaux ; pluriel simple retiré (« jours » -> « jour »)."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    tokens = []
    for token in re.findall(r"[a-z0-9]+", text):
        if token in STOP_WORDS or (len(token) < 3 and not token.isdigit()):
            continue
        if len(token) > 4 and token.endswith(("s", "x")):
            token = token[:-1]
        tokens.append(token)
    return tokens


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_length = (sum(self.lengths) / len(self.lengths)) if self.docs else 0.0
        df: Counter[str] = Counter()
        for doc in self.docs:
            df.update(doc.keys())
        n = len(self.docs)
        self.idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        result = []
        for doc, length in zip(self.docs, self.lengths, strict=True):
            score = 0.0
            for term in terms:
                tf = doc.get(term, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * length / (self.avg_length or 1))
                    score += self.idf[term] * tf * (self.k1 + 1) / norm
            result.append(score)
        return result
