# Build from the repo root: docker build -f server/Dockerfile -t mayaz-api .
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MALLOC_ARENA_MAX=2 PYTHONPATH=/srv
WORKDIR /srv
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && ! pip list --format=freeze | grep -Ei '^(torch|transformers|optimum|sentence-transformers|accelerate|scipy)='
# The mayaz library (numpy + tokenizers + onnxruntime only); the server imports it, never a copy of it.
COPY src/mayaz/__init__.py src/mayaz/text.py src/mayaz/runtime.py src/mayaz/inference.py mayaz/
COPY src/mayaz/model/__init__.py src/mayaz/model/render.py src/mayaz/model/tournament.py mayaz/model/
COPY src/mayaz/assets/tokenizer.json mayaz/assets/
COPY server/app app
COPY server/artifacts/temperatures-*.json artifacts/
COPY server/scripts/fetch_model.sh /tmp/
# Render passes service environment variables to the build as build args: MAYAZ_MODEL picks the tier
# (mayaz-17m | mayaz-32m | mayaz-68m) for both the download and the server. The model comes from the public
# v1.0.0 GitHub release and is checked against the sha256 in app/models.json.
ARG MAYAZ_MODEL=mayaz-32m
ENV MAYAZ_MODEL=${MAYAZ_MODEL}
RUN sh /tmp/fetch_model.sh artifacts app/models.json \
 && cp mayaz/assets/tokenizer.json artifacts/tokenizer.json
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --no-access-log"]
