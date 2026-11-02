import json

from documind.evaluate import QUESTIONS, keywords_found, normalize


def test_normalize_nombres_et_heures():
    assert normalize("5 000 Dirhams, de 9 h 30 à 16 h 30") == "5000 dirhams, de 9h30 a 16h30"


def test_keywords_found_avec_alternatives():
    assert keywords_found("Deux jours par semaine.", ["deux|2"])
    assert keywords_found("Il faut 12 caractères.", ["12"])
    assert not keywords_found("Trois jours.", ["deux|2"])


def test_jeu_d_evaluation_coherent():
    items = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    assert len(items) >= 20
    assert all(i["keywords"] for i in items if i["expected_source"])
    assert any(i["expected_source"] is None for i in items)  # questions hors sujet présentes
