FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf-cache

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir .

ARG MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${MODEL_NAME}')"

FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf-cache \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf-cache

RUN groupadd --system app && useradd --system --gid app --no-create-home --home /home/app app && \
    mkdir -p /home/app/data && chown -R app:app /home/app

WORKDIR /home/app

COPY --from=base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=base /usr/local/bin/vector-search-api /usr/local/bin/vector-search-api
COPY --from=base /opt/hf-cache /opt/hf-cache

USER app
EXPOSE 8000
ENV INDEX_PATH=/home/app/data/index.bin \
    STORE_PATH=/home/app/data/docs.jsonl
ENTRYPOINT ["vector-search-api"]
