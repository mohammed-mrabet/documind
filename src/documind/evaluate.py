"""Évaluation de DocuMind sur un jeu de questions dont la réponse est connue (eval/questions.json).

Mesures :
- Recherche  : hit@k (le bon document est-il parmi les k extraits ?) et MRR (à quel rang arrive-t-il ?)
- Réponse    : exactitude (les informations attendues, ex. « 150 », sont-elles dans la réponse ?)
- Citation   : la réponse cite-t-elle le bon document ?
- Abstention : pour les questions hors sujet, l'assistant dit-il qu'il ne sait pas au lieu d'inventer ?

Usage : python -m documind.evaluate            (recherche + réponses, nécessite Ollama)
        python -m documind.evaluate --no-llm   (recherche seule, rapide)
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import unicodedata
from datetime import datetime
from pathlib import Path

from .config import ROOT, get_settings
from .ingest import open_vectorstore
from .rag import RAGPipeline

QUESTIONS = ROOT / "eval" / "questions.json"
REPORT = ROOT / "reports" / "eval.json"


def normalize(text: str) -> str:
    """Minuscules, sans accents, espaces des nombres supprimés (« 5 000 » -> « 5000 », « 9 h 30 » -> « 9h30 »)."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
    text = re.sub(r"(?<=\d)\s*h\s*(?=\d)", "h", text)
    return text


def keywords_found(answer: str, keywords: list[str]) -> bool:
    """Chaque mot-clé attendu doit apparaître ; « a|b » accepte l'une ou l'autre forme."""
    text = normalize(answer)
    return all(any(normalize(alt) in text for alt in kw.split("|")) for kw in keywords)


def retrieval_metrics(pipeline: RAGPipeline, items: list[dict], k: int) -> dict:
    hits, reciprocal_ranks = [], []
    for item in items:
        results = pipeline.vectorstore.similarity_search_with_relevance_scores(item["question"], k=k)
        sources = [doc.metadata["source"] for doc, _ in results]
        rank = sources.index(item["expected_source"]) + 1 if item["expected_source"] in sources else None
        hits.append(rank is not None)
        reciprocal_ranks.append(1 / rank if rank else 0.0)
    return {f"hit_at_{k}": round(sum(hits) / len(hits), 3), "mrr": round(statistics.mean(reciprocal_ranks), 3)}


def answer_metrics(pipeline: RAGPipeline, in_scope: list[dict], out_of_scope: list[dict]) -> tuple[dict, list[dict]]:
    details, correct, cited_ok, false_abstentions, latencies = [], 0, 0, 0, []
    for item in in_scope:
        result = pipeline.ask(item["question"])
        ok = result.found and keywords_found(result.answer, item["keywords"])
        cites = item["expected_source"] in {s.source for s in result.cited_sources}
        correct += ok
        cited_ok += cites
        false_abstentions += not result.found
        latencies.append(result.latency_ms)
        details.append({"question": item["question"], "answer": result.answer, "correct": ok, "cites_expected_source": cites})

    abstained = 0
    for item in out_of_scope:
        result = pipeline.ask(item["question"])
        abstained += not result.found
        latencies.append(result.latency_ms)
        details.append({"question": item["question"], "answer": result.answer, "correct_abstention": not result.found})

    n = len(in_scope)
    metrics = {
        "answer_accuracy": round(correct / n, 3),
        "citation_accuracy": round(cited_ok / n, 3),
        "false_abstention_rate": round(false_abstentions / n, 3),
        "out_of_scope_abstention": round(abstained / len(out_of_scope), 3) if out_of_scope else None,
        "median_latency_s": round(statistics.median(latencies) / 1000, 2),
    }
    return metrics, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true", help="évaluer seulement la recherche")
    args = parser.parse_args()

    settings = get_settings()
    items = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    in_scope = [i for i in items if i["expected_source"]]
    out_of_scope = [i for i in items if not i["expected_source"]]

    if args.no_llm:
        pipeline = RAGPipeline(open_vectorstore(settings), llm=None, settings=settings)  # type: ignore[arg-type]
    else:
        from .rag import build_pipeline

        pipeline = build_pipeline(settings)

    report = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "embeddings": settings.embedding_model if settings.embedding_backend == "huggingface" else "hashing",
        "llm": None if args.no_llm else settings.llm_model,
        "questions": {"in_scope": len(in_scope), "out_of_scope": len(out_of_scope)},
        "retrieval": {**retrieval_metrics(pipeline, in_scope, 1), **retrieval_metrics(pipeline, in_scope, settings.top_k)},
    }
    print(f"Recherche ({len(in_scope)} questions) : {report['retrieval']}")

    if not args.no_llm:
        metrics, details = answer_metrics(pipeline, in_scope, out_of_scope)
        report["generation"] = metrics
        report["details"] = details
        print(f"Réponses : {metrics}")
        for d in details:
            if not d.get("correct", d.get("correct_abstention")):
                print(f"  ✗ {d['question']}\n    -> {d['answer'][:200]}")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Rapport : {Path(REPORT).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
