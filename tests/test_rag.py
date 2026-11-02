from conftest import make_pipeline
from documind.rag import NOT_FOUND, cited_numbers, format_context


def test_recherche_trouve_le_bon_document(store, settings):
    pipeline = make_pipeline(store, settings, [])
    docs = pipeline.retrieve("Quel est le plafond de remboursement pour un repas ?")
    assert docs[0][0].metadata["source"] == "procedure_notes_de_frais.pdf"


def test_reponse_sourcee(store, settings):
    pipeline = make_pipeline(store, settings, ["Un repas est remboursé dans la limite de 150 dirhams [1]."])
    answer = pipeline.ask("Quel est le plafond de remboursement pour un repas ?")
    assert answer.found
    assert [s.source for s in answer.cited_sources] == [answer.sources[0].source]
    assert answer.sources[0].page >= 1


def test_aucun_extrait_pertinent_pas_d_appel_au_llm(store, settings):
    # Liste de réponses vide : un appel au LLM lèverait une erreur.
    pipeline = make_pipeline(store, settings, [])
    answer = pipeline.ask("Zorglub xylophone quantique ?")
    assert answer.answer == NOT_FOUND
    assert not answer.found and answer.sources == []


def test_le_llm_peut_aussi_s_abstenir(store, settings):
    pipeline = make_pipeline(store, settings, [NOT_FOUND])
    assert not pipeline.ask("Quelle est la politique de télétravail pour les stagiaires étrangers ?").found


def test_format_context_et_citations():
    from langchain_core.documents import Document

    ctx = format_context([Document(page_content="Texte", metadata={"source": "a.pdf", "page": 2})])
    assert ctx == "[1] (a.pdf, page 2)\nTexte"
    assert cited_numbers("Selon [1] et [3], puis [1].") == {1, 3}
