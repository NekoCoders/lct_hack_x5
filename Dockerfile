FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Europe/Moscow \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r app && useradd -r -g app app

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY server model /app/

RUN chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD curl -fsS http://127.0.0.1:8000/health >/dev/null || exit 1

CMD ["uvicorn", "server.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
