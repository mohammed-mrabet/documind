"""Chaîne RAG : recherche des extraits pertinents puis génération d'une réponse sourcée."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore

from .config import Settings

NOT_FOUND = "Je ne trouve pas cette information dans les documents fournis."

SYSTEM_PROMPT = f"""Tu es DocuMind, l'assistant documentaire d'une entreprise.
Règles :
1. Réponds uniquement à partir des extraits numérotés ci-dessous, jamais de tes connaissances générales.
2. Si les extraits ne contiennent pas la réponse, réponds exactement : « {NOT_FOUND} »
3. Cite chaque information utilisée avec le numéro de son extrait entre crochets, par exemple [1] ou [2].
4. Réponds en français, de façon précise et concise (trois phrases au maximum), en reprenant les chiffres exacts.

Extraits :
{{context}}"""

PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])
_CITATION = re.compile(r"\[(\d+)\]")


@dataclass
class Source:
    source: str
    page: int
    score: float
    excerpt: str
    cited: bool = False


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[Source] = field(default_factory=list)
    found: bool = True
    latency_ms: int = 0

    @property
    def cited_sources(self) -> list[Source]:
        return [s for s in self.sources if s.cited]


def format_context(documents: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] ({d.metadata['source']}, page {d.metadata['page']})\n{d.page_content}" for i, d in enumerate(documents, start=1)
    )


def cited_numbers(answer: str) -> set[int]:
    return {int(n) for n in _CITATION.findall(answer)}


class RAGPipeline:
    def __init__(self, vectorstore: VectorStore, llm: BaseChatModel | None, settings: Settings):
        self.vectorstore = vectorstore
        self.settings = settings
        # llm=None : recherche seule (évaluation de la récupération sans LLM).
        self.chain = PROMPT | llm | StrOutputParser() if llm is not None else None

    def retrieve(self, question: str, top_k: int | None = None) -> list[tuple[Document, float]]:
        """Les k extraits les plus proches, filtrés par score de pertinence minimal."""
        results = self.vectorstore.similarity_search_with_relevance_scores(question, k=top_k or self.settings.top_k)
        return [(doc, score) for doc, score in results if score >= self.settings.min_relevance]

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        start = time.perf_counter()
        retrieved = self.retrieve(question, top_k)

        # Aucun extrait pertinent : on répond sans appeler le LLM, ce qui évite toute invention.
        if not retrieved:
            return Answer(question, NOT_FOUND, found=False, latency_ms=int((time.perf_counter() - start) * 1000))

        if self.chain is None:
            raise RuntimeError("Aucun LLM configuré : impossible de générer une réponse.")
        documents = [doc for doc, _ in retrieved]
        text = self.chain.invoke({"context": format_context(documents), "question": question}).strip()
        cited = cited_numbers(text)
        sources = [
            Source(
                source=doc.metadata["source"],
                page=doc.metadata["page"],
                score=round(score, 3),
                excerpt=doc.page_content[:300],
                cited=i in cited,
            )
            for i, (doc, score) in enumerate(retrieved, start=1)
        ]
        return Answer(
            question=question,
            answer=text,
            sources=sources,
            found=NOT_FOUND.rstrip(".").lower() not in text.lower(),
            latency_ms=int((time.perf_counter() - start) * 1000),
        )


def build_pipeline(settings: Settings) -> RAGPipeline:
    """Pipeline de production : embeddings HuggingFace, Chroma et LLM local servi par Ollama."""
    from langchain_ollama import ChatOllama

    from .ingest import open_vectorstore

    llm = ChatOllama(model=settings.llm_model, base_url=settings.ollama_base_url, temperature=0)
    return RAGPipeline(open_vectorstore(settings), llm, settings)
