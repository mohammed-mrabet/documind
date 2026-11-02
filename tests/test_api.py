import io

import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from conftest import make_pipeline
from documind.api import app, get_config, get_pipeline


@pytest.fixture
def client(store, settings):
    pipeline = make_pipeline(store, settings, ["Deux jours par semaine maximum [1]."] * 5)
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    app.dependency_overrides[get_config] = lambda: settings
    yield TestClient(app)
    app.dependency_overrides.clear()


def make_pdf(text: str) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer)
    c.drawString(72, 750, text)
    c.save()
    return buffer.getvalue()


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_ask(client):
    r = client.post("/ask", json={"question": "Combien de jours de télétravail par semaine ?"})
    assert r.status_code == 200
    data = r.json()
    assert data["found"] and "[1]" in data["answer"]
    assert data["sources"][0]["source"] == "politique_teletravail.pdf"
    assert data["sources"][0]["cited"]


def test_question_trop_courte(client):
    assert client.post("/ask", json={"question": "a"}).status_code == 422


def test_liste_des_documents(client):
    assert len(client.get("/documents").json()) == 5


def test_upload_pdf(client, settings):
    pdf = make_pdf("Le parking visiteurs est situé derrière le bâtiment B.")
    r = client.post("/documents", files={"file": ("parking.pdf", pdf, "application/pdf")})
    assert r.status_code == 201 and r.json()["chunks"] == 1
    assert (settings.docs_dir / "parking.pdf").exists()
    assert len(client.get("/documents").json()) == 6


def test_upload_refuse_les_autres_formats(client):
    r = client.post("/documents", files={"file": ("notes.txt", b"bonjour", "text/plain")})
    assert r.status_code == 415


def test_upload_empeche_la_traversee_de_repertoire(client, settings):
    r = client.post("/documents", files={"file": ("../../evil.pdf", make_pdf("test"), "application/pdf")})
    assert r.status_code == 201
    assert (settings.docs_dir / "evil.pdf").exists()
    assert not (settings.docs_dir.parent.parent / "evil.pdf").exists()
