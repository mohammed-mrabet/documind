"""Interface de discussion Streamlit (cliente de l'API).

Lancement : streamlit run src/documind/app.py
"""

import os

import requests
import streamlit as st

API_URL = os.getenv("DOCUMIND_API_URL", "http://localhost:8000")

st.set_page_config(page_title="DocuMind", page_icon="📚", layout="wide")
st.title("📚 DocuMind")
st.caption("Posez une question sur les documents internes : chaque réponse cite ses sources.")

with st.sidebar:
    st.header("Documents indexés")
    try:
        for doc in requests.get(f"{API_URL}/documents", timeout=10).json():
            st.write(f"📄 {doc['source']} · {doc['chunks']} extraits")
    except requests.RequestException:
        st.error(f"API injoignable ({API_URL}).")

    st.divider()
    uploaded = st.file_uploader("Ajouter un PDF", type="pdf")
    if uploaded and st.button("Indexer"):
        r = requests.post(f"{API_URL}/documents", files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}, timeout=300)
        if r.ok:
            st.success(f"{r.json()['chunks']} extraits indexés.")
            st.rerun()
        else:
            st.error(r.json().get("detail", "Erreur d'indexation."))

    st.divider()
    st.markdown("**Exemples**")
    for example in [
        "Combien de jours de télétravail sont autorisés ?",
        "Quel est le plafond pour une nuit d'hôtel à Casablanca ?",
        "Que faire si je perds mon ordinateur portable ?",
    ]:
        if st.button(example, use_container_width=True):
            st.session_state.pending = example

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for src in message.get("sources", []):
            st.caption(src)

question = st.chat_input("Votre question…") or st.session_state.pop("pending", None)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche dans les documents…"):
            try:
                r = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=180)
                r.raise_for_status()
                data = r.json()
            except requests.RequestException as exc:
                data = {"answer": f"Erreur : {exc}", "sources": [], "latency_ms": 0}
        st.markdown(data["answer"])
        shown = [s for s in data["sources"] if s["cited"]] or data["sources"]
        captions = [f"📄 {s['source']}, page {s['page']} (pertinence {s['score']:.2f})" for s in shown]
        for c in captions:
            st.caption(c)
        if data.get("latency_ms"):
            st.caption(f"⏱ {data['latency_ms'] / 1000:.1f} s")
    st.session_state.messages.append({"role": "assistant", "content": data["answer"], "sources": captions})
