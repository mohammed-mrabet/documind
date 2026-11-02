"""Fixtures partagées : pipeline 100 % hors ligne (embeddings lexicaux + LLM simulé)."""

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402

from documind.config import Settings  # noqa: E402
from documind.embeddings import HashingEmbeddings  # noqa: E402
from documind.ingest import index_documents, open_vectorstore  # noqa: E402
from documind.loader import load_directory  # noqa: E402
from documind.rag import RAGPipeline  # noqa: E402

SAMPLE_DOCS = ROOT / "data" / "docs"


@pytest.fixture
def settings(tmp_path) -> Settings:
    docs = tmp_path / "docs"
    shutil.copytree(SAMPLE_DOCS, docs)
    return Settings(
        docs_dir=docs,
        chroma_dir=tmp_path / "chroma",
        collection="test",
        embedding_backend="hashing",
        min_relevance=0.15,
    )


@pytest.fixture
def store(settings):
    vectorstore = open_vectorstore(settings, HashingEmbeddings())
    index_documents(vectorstore, load_directory(settings.docs_dir), settings)
    return vectorstore


def make_pipeline(store, settings, responses):
    return RAGPipeline(store, FakeListChatModel(responses=responses), settings)
