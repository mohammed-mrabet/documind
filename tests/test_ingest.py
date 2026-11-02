from langchain_core.documents import Document

from documind.ingest import chunk_id, index_documents, list_sources, split_documents


def test_split_respecte_la_taille_et_numerote_les_morceaux():
    doc = Document(page_content="Phrase numéro un. " * 200, metadata={"source": "a.pdf", "page": 1})
    chunks = split_documents([doc], chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 300 for c in chunks)
    assert [c.metadata["chunk"] for c in chunks] == list(range(1, len(chunks) + 1))
    assert chunk_id(chunks[0]) == "a.pdf:p1:c1"


def test_les_5_documents_sont_indexes(store):
    sources = list_sources(store)
    assert len(sources) == 5
    assert all(s["chunks"] >= 2 for s in sources)


def test_reindexer_ne_cree_pas_de_doublons(store, settings):
    before = sum(s["chunks"] for s in list_sources(store))
    doc = Document(page_content="Nouveau contenu du guide.", metadata={"source": "guide_accueil.pdf", "page": 1})
    index_documents(store, [doc], settings)
    sources = {s["source"]: s["chunks"] for s in list_sources(store)}
    assert sources["guide_accueil.pdf"] == 1  # les anciens morceaux du guide ont été remplacés
    assert sum(sources.values()) < before
