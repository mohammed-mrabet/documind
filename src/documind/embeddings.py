"""Choix du modèle d'embeddings."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata

from langchain_core.embeddings import Embeddings

from .config import Settings

# Mots grammaticaux ignorés par l'embedding lexical.
STOP_WORDS = set(
    "les des une est que qui quoi quel quelle quels quelles pour par dans sur avec sans pas plus sont ont aux ces cette "
    "son sa ses leur leurs nous vous ils elles elle lui eux mon ton mes tes faut peut doit combien comment quand "
    "dois peux etre avoir fait faire tout tous toute toutes aussi apres avant entre chez donc car mais".split()
)


class HashingEmbeddings(Embeddings):
    """Embeddings lexicaux déterministes (sac de mots haché), sans téléchargement de modèle.

    Utilisés pour les tests automatiques et la CI : ils vérifient toute la chaîne RAG
    hors ligne. Ils ne comprennent pas les synonymes : en usage réel, on utilise le
    modèle HuggingFace multilingue.
    """

    def __init__(self, dimensions: int = 2048):
        self.dimensions = dimensions

    @staticmethod
    def _tokens(text: str) -> list[str]:
        text = unicodedata.normalize("NFD", text.lower())
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        return [t for t in re.findall(r"[a-z0-9]+", text) if len(t) > 2 and t not in STOP_WORDS]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in self._tokens(text):
            index = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding_backend == "hashing":
        return HashingEmbeddings()

    from langchain_huggingface import HuggingFaceEmbeddings  # import tardif : charge PyTorch

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},  # vecteurs normés : similarité cosinus
    )
