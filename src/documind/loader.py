"""Extraction du texte des PDF, page par page, et nettoyage des en-têtes et pieds de page."""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

_PAGE_NUMBER = re.compile(r"^\s*page\s+\d+(\s*/\s*\d+)?\s*$", re.IGNORECASE)
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")


def clean_page(text: str, repeated_lines: set[str] = frozenset()) -> str:
    """Supprime numéros de page et lignes répétées sur chaque page (pied de page),
    recolle les mots coupés en fin de ligne et normalise les espaces."""
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not _PAGE_NUMBER.match(line) and line.strip() not in repeated_lines
    ]
    return re.sub(r"[ \t]+", " ", "\n".join(lines)).strip()


def load_pdf(path: Path) -> list[Document]:
    """Un Document par page, avec la source (nom du fichier) et le numéro de page en métadonnées."""
    reader = PdfReader(str(path))
    title = (reader.metadata.title if reader.metadata and reader.metadata.title else path.stem.replace("_", " ")).strip()
    raw_pages = [page.extract_text() or "" for page in reader.pages]

    # Une ligne présente sur toutes les pages d'un document de plusieurs pages = en-tête ou pied de page.
    repeated: set[str] = set()
    if len(raw_pages) > 1:
        line_sets = [{line.strip() for line in page.splitlines() if line.strip()} for page in raw_pages]
        repeated = set.intersection(*line_sets)

    documents = []
    for number, raw in enumerate(raw_pages, start=1):
        text = clean_page(raw, repeated)
        if text:
            documents.append(Document(page_content=text, metadata={"source": path.name, "page": number, "title": title}))
    return documents


def load_directory(directory: Path) -> list[Document]:
    documents: list[Document] = []
    for pdf in sorted(directory.glob("*.pdf")):
        documents.extend(load_pdf(pdf))
    return documents
