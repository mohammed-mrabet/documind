"""Choix du modèle d'embeddings."""

from __future__ import annotations

import hashlib
import math

from langchain_core.embeddings import Embeddings

from .config import Settings
from .lexical import tokenize


class HashingEmbeddings(Embeddings):
    """Embeddings lexicaux déterministes (sac de mots haché), sans téléchargement de modèle.

    Utilisés pour les tests automatiques et la CI : ils vérifient toute la chaîne RAG
    hors ligne. Ils ne comprennent pas les synonymes : en usage réel, on utilise le
    modèle HuggingFace multilingue.
    """

    def __init__(self, dimensions: int = 2048):
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
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
