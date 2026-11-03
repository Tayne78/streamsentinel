FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[kafka,ingest,ml,api]"

RUN useradd --create-home --uid 1000 app && mkdir -p /data /mlflow && chown app:app /data /mlflow
USER app

CMD ["uvicorn", "streamsentinel.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
