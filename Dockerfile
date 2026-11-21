FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY deploy ./deploy
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.13-slim AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    INDEX_PATH=/app/index.json
WORKDIR /app

RUN adduser --system --uid 10001 --no-create-home findex
COPY --from=builder --chown=findex:nogroup /app/.venv /app/.venv
COPY --chown=findex:nogroup deploy/index.json /app/index.json
COPY --chown=findex:nogroup deploy/corpus /app/deploy/corpus

USER findex
EXPOSE 8000
CMD ["sh", "-c", "uvicorn findex.web:app --host 0.0.0.0 --port ${PORT:-8000}"]
