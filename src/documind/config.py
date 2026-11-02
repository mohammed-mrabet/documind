"""Configuration par variables d'environnement (valeurs par défaut pour un usage local)."""

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    docs_dir: Path = field(default_factory=lambda: Path(_env("DOCUMIND_DOCS_DIR", str(ROOT / "data" / "docs"))))
    chroma_dir: Path = field(default_factory=lambda: Path(_env("DOCUMIND_CHROMA_DIR", str(ROOT / "data" / "chroma"))))
    collection: str = field(default_factory=lambda: _env("DOCUMIND_COLLECTION", "documind"))

    # "huggingface" (modèle sémantique multilingue) ou "hashing" (lexical, hors ligne : tests et CI).
    embedding_backend: str = field(default_factory=lambda: _env("DOCUMIND_EMBEDDINGS", "huggingface"))
    embedding_model: str = field(
        default_factory=lambda: _env("DOCUMIND_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    )

    ollama_base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"))
    llm_model: str = field(default_factory=lambda: _env("DOCUMIND_LLM_MODEL", "qwen2.5:3b"))

    chunk_size: int = field(default_factory=lambda: int(_env("DOCUMIND_CHUNK_SIZE", "700")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("DOCUMIND_CHUNK_OVERLAP", "100")))
    top_k: int = field(default_factory=lambda: int(_env("DOCUMIND_TOP_K", "4")))
    # En dessous de ce score de pertinence (0 à 1), un extrait est jugé hors sujet.
    min_relevance: float = field(default_factory=lambda: float(_env("DOCUMIND_MIN_RELEVANCE", "0.3")))


def get_settings() -> Settings:
    return Settings()
