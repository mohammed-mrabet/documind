"""Indexation : PDF -> morceaux de texte -> embeddings -> base vectorielle Chroma.

Usage : python -m documind.ingest   (indexe tous les PDF de data/docs)
"""

from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings, get_settings
from .embeddings import get_embeddings
from .loader import load_directory, load_pdf


def split_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Découpe en morceaux d'environ `chunk_size` caractères qui se chevauchent de `chunk_overlap`.
    Le chevauchement évite de couper une information en deux entre deux morceaux.
    Les séparateurs sont essayés dans l'ordre : paragraphe, ligne, phrase, mot."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        # En-tête de contexte : un extrait « 2. Horaires… » seul est ambigu ; avec le titre du
        # document (« Guide d'accueil… »), la recherche sait de quels horaires il s'agit.
        title = chunk.metadata.get("title")
        if title:
            chunk.page_content = f"{title}\n{chunk.page_content}"
    counters: dict[tuple[str, int], int] = {}
    for chunk in chunks:
        key = (chunk.metadata["source"], chunk.metadata["page"])
        counters[key] = counters.get(key, 0) + 1
        chunk.metadata["chunk"] = counters[key]
    return chunks


def chunk_id(chunk: Document) -> str:
    """Identifiant stable : réindexer un fichier remplace ses morceaux au lieu de les dupliquer."""
    m = chunk.metadata
    return f"{m['source']}:p{m['page']}:c{m['chunk']}"


def open_vectorstore(settings: Settings, embeddings: Embeddings | None = None) -> Chroma:
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.collection,
        embedding_function=embeddings or get_embeddings(settings),
        persist_directory=str(settings.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},  # pertinence = 1 - distance cosinus
    )


def remove_source(store: Chroma, source: str) -> int:
    ids = store.get(where={"source": source})["ids"]
    if ids:
        store.delete(ids=ids)
    return len(ids)


def index_documents(store: Chroma, documents: list[Document], settings: Settings) -> int:
    chunks = split_documents(documents, settings.chunk_size, settings.chunk_overlap)
    for source in {c.metadata["source"] for c in chunks}:
        remove_source(store, source)
    if chunks:
        store.add_documents(chunks, ids=[chunk_id(c) for c in chunks])
    return len(chunks)


def ingest_file(store: Chroma, path: Path, settings: Settings) -> int:
    return index_documents(store, load_pdf(path), settings)


def list_sources(store: Chroma) -> list[dict]:
    """Documents indexés et nombre de morceaux de chacun."""
    counts: dict[str, int] = {}
    for meta in store.get(include=["metadatas"])["metadatas"]:
        counts[meta["source"]] = counts.get(meta["source"], 0) + 1
    return [{"source": s, "chunks": n} for s, n in sorted(counts.items())]


def main() -> None:
    settings = get_settings()
    store = open_vectorstore(settings)
    documents = load_directory(settings.docs_dir)
    n = index_documents(store, documents, settings)
    print(f"{len(documents)} pages lues, {n} morceaux indexés dans {settings.chroma_dir}")
    for item in list_sources(store):
        print(f"  {item['source']:<35} {item['chunks']} morceaux")


if __name__ == "__main__":
    main()
