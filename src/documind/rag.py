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
from .lexical import BM25

NOT_FOUND = "Je ne trouve pas cette information dans les documents fournis."

SYSTEM_PROMPT = f"""Tu es DocuMind, l'assistant documentaire d'une entreprise.
Règles :
1. Réponds uniquement à partir des extraits numérotés ci-dessous, jamais de tes connaissances générales.
2. Lis attentivement tous les extraits avant de répondre : la réponse peut se trouver dans n'importe lequel.
3. Si la question comporte plusieurs parties, réponds à chacune.
4. Écris des phrases complètes qui reprennent les chiffres exacts (montants, délais, numéros), puis cite
   l'extrait utilisé avec sa lettre entre crochets à la fin de la phrase, par exemple : « Le plafond est de 150 dirhams [A]. »
5. Réponds en français, de façon précise et concise (trois phrases au maximum).
6. Seulement si aucun extrait ne contient la réponse, écris uniquement : « {NOT_FOUND} »

Extraits :
{{context}}"""

PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])
# Lettres plutôt que chiffres : un petit LLM confond facilement « [2] » avec une valeur numérique.
_CITATION = re.compile(r"\[([A-Z])\]")
_NOT_FOUND = re.compile(r"je ne (trouve|dispose)|ne figure pas|pas (cette|d'|de cette) information", re.IGNORECASE)


def letter(i: int) -> str:
    """1 -> A, 2 -> B..."""
    return chr(ord("A") + i - 1)


def is_not_found(answer: str) -> bool:
    """Détecte une abstention, même formulée un peu différemment (« Je ne trouve cette information… »)."""
    return bool(_NOT_FOUND.search(answer))


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
        f"[{letter(i)}] ({d.metadata['source']}, page {d.metadata['page']})\n{d.page_content}" for i, d in enumerate(documents, start=1)
    )


def cited_numbers(answer: str) -> set[int]:
    """Numéros (1, 2...) des extraits cités sous forme de lettres ([A], [B]...)."""
    return {ord(c) - ord("A") + 1 for c in _CITATION.findall(answer)}


class RAGPipeline:
    """Recherche hybride (sémantique + BM25, fusionnées par Reciprocal Rank Fusion) puis génération."""

    RRF_K = 60  # constante standard de la Reciprocal Rank Fusion
    CANDIDATES = 10  # extraits candidats retenus par chaque méthode avant fusion

    def __init__(self, vectorstore: VectorStore, llm: BaseChatModel | None, settings: Settings):
        self.vectorstore = vectorstore
        self.settings = settings
        # llm=None : recherche seule (évaluation de la récupération sans LLM).
        self.chain = PROMPT | llm | StrOutputParser() if llm is not None else None
        self._bm25: BM25 | None = None
        self._corpus: list[Document] = []
        self._corpus_ids: list[str] = []

    def _lexical_index(self) -> BM25:
        """Index BM25 de tous les extraits, reconstruit si des documents ont été ajoutés ou retirés."""
        data = self.vectorstore.get(include=["documents", "metadatas"])
        if self._bm25 is None or data["ids"] != self._corpus_ids:
            self._corpus_ids = data["ids"]
            self._corpus = [Document(page_content=t, metadata=m) for t, m in zip(data["documents"], data["metadatas"], strict=True)]
            self._bm25 = BM25([d.page_content for d in self._corpus])
        return self._bm25

    def rank(self, question: str, top_k: int | None = None) -> list[tuple[Document, float, float]]:
        """Classement hybride sans filtrage : (extrait, pertinence sémantique, score BM25)."""
        top_k = top_k or self.settings.top_k
        key = lambda d: (d.metadata["source"], d.metadata["page"], d.metadata.get("chunk"))  # noqa: E731

        semantic = self.vectorstore.similarity_search_with_relevance_scores(question, k=self.CANDIDATES)
        bm25 = self._lexical_index()
        lexical_scores = bm25.scores(question)
        lexical = sorted(zip(self._corpus, lexical_scores, strict=True), key=lambda x: x[1], reverse=True)[: self.CANDIDATES]
        lexical = [(doc, score) for doc, score in lexical if score > 0]

        fused: dict[tuple, dict] = {}
        for rank, (doc, score) in enumerate(semantic, start=1):
            entry = fused.setdefault(key(doc), {"doc": doc, "semantic": score, "bm25": 0.0, "rrf": 0.0})
            entry["rrf"] += 1 / (self.RRF_K + rank)
        for rank, (doc, score) in enumerate(lexical, start=1):
            entry = fused.setdefault(key(doc), {"doc": doc, "semantic": 0.0, "bm25": 0.0, "rrf": 0.0})
            entry["bm25"] = score
            entry["rrf"] += 1 / (self.RRF_K + rank)

        best = sorted(fused.values(), key=lambda e: e["rrf"], reverse=True)[:top_k]
        return [(e["doc"], e["semantic"], e["bm25"]) for e in best]

    def retrieve(self, question: str, top_k: int | None = None) -> list[tuple[Document, float]]:
        """Extraits à transmettre au LLM, ou liste vide si la question est hors sujet :
        ni proximité sémantique suffisante, ni mot-clé important en commun."""
        ranked = self.rank(question, top_k)
        best_semantic = max((s for _, s, _ in ranked), default=0.0)
        best_lexical = max((b for _, _, b in ranked), default=0.0)
        if best_semantic < self.settings.min_relevance and best_lexical < self.settings.min_bm25:
            return []
        return [(doc, semantic) for doc, semantic, _ in ranked]

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
            found=not is_not_found(text),
            latency_ms=int((time.perf_counter() - start) * 1000),
        )


def build_pipeline(settings: Settings) -> RAGPipeline:
    """Pipeline de production : embeddings HuggingFace, Chroma et LLM local servi par Ollama."""
    from langchain_ollama import ChatOllama

    from .ingest import open_vectorstore

    llm = ChatOllama(model=settings.llm_model, base_url=settings.ollama_base_url, temperature=0)
    return RAGPipeline(open_vectorstore(settings), llm, settings)
