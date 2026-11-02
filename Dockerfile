FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Le modèle d'embeddings est téléchargé au build : le conteneur démarre ensuite sans Internet.
ARG EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

COPY . .

RUN useradd --create-home app && mkdir -p data/chroma && chown -R app /app
USER app

EXPOSE 8000 8501
CMD ["sh", "docker/start-api.sh"]
