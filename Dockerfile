FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY data ./data
COPY evals ./evals

RUN pip install --no-cache-dir ".[server]"

EXPOSE 8000
ENV CITEKIT_DOCS=/app/data/docs

CMD ["citekit", "serve", "--host", "0.0.0.0", "--port", "8000", "--docs", "/app/data/docs"]
