"""API REST de DocuMind.

Lancement : uvicorn documind.api:app --reload    Documentation : http://localhost:8000/docs
"""

from __future__ import annotations

import shutil
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

from . import __version__
from .config import Settings, get_settings
from .ingest import ingest_file, list_sources
from .rag import RAGPipeline, build_pipeline

app = FastAPI(
    title="DocuMind API",
    version=__version__,
    description="Questions-réponses sourcées sur des documents d'entreprise (RAG : Chroma + LLM local via Ollama).",
)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    return build_pipeline(get_settings())


def get_config() -> Settings:
    return get_settings()


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=1000, examples=["Quel est le plafond de remboursement d'un repas ?"])
    top_k: int | None = Field(default=None, ge=1, le=10)


class SourceOut(BaseModel):
    source: str
    page: int
    score: float
    excerpt: str
    cited: bool


class AnswerOut(BaseModel):
    question: str
    answer: str
    found: bool = Field(description="Faux si l'information n'a pas été trouvée dans les documents")
    sources: list[SourceOut]
    latency_ms: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/ask", response_model=AnswerOut)
def ask(payload: Question, pipeline: RAGPipeline = Depends(get_pipeline)) -> AnswerOut:
    result = pipeline.ask(payload.question, payload.top_k)
    return AnswerOut(**asdict(result))


@app.get("/documents")
def documents(pipeline: RAGPipeline = Depends(get_pipeline)) -> list[dict]:
    return list_sources(pipeline.vectorstore)


@app.post("/documents", status_code=201)
def upload(
    file: UploadFile,
    pipeline: RAGPipeline = Depends(get_pipeline),
    settings: Settings = Depends(get_config),
) -> dict:
    """Ajoute un PDF et l'indexe immédiatement. Un fichier du même nom est remplacé."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Seuls les fichiers PDF sont acceptés.")
    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Fichier trop volumineux (20 Mo maximum).")

    settings.docs_dir.mkdir(parents=True, exist_ok=True)
    target = settings.docs_dir / Path(file.filename).name  # Path(...).name : empêche « ../../ »
    with target.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        chunks = ingest_file(pipeline.vectorstore, target, settings)
    except Exception as exc:  # PDF illisible ou protégé
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"PDF illisible : {exc}") from exc
    if chunks == 0:
        raise HTTPException(status_code=422, detail="Aucun texte extrait (PDF scanné sans OCR ?).")
    return {"source": target.name, "chunks": chunks}
