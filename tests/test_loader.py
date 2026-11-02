from documind.loader import clean_page, load_pdf


def test_clean_page_supprime_numeros_et_pieds_de_page():
    raw = "Pied de page commun\nPage 2\n4. Circuit de valida-\ntion\nTexte utile"
    assert clean_page(raw, {"Pied de page commun"}) == "4. Circuit de validation\nTexte utile"


def test_load_pdf_une_entree_par_page_avec_metadonnees(settings):
    pages = load_pdf(settings.docs_dir / "procedure_notes_de_frais.pdf")
    assert [p.metadata["page"] for p in pages] == [1, 2]
    assert all(p.metadata["source"] == "procedure_notes_de_frais.pdf" for p in pages)
    # Le pied de page répété sur chaque page a été retiré.
    assert all("document de démonstration" not in p.page_content for p in pages)
    assert "150 dirhams" in pages[0].page_content
