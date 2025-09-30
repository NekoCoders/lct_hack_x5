FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Europe/Moscow \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r app \
    && useradd -m -d /home/app -r -g app app

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY model /app/
COPY server /app/

RUN chown -R app:app /app /home/app
ENV HOME=/home/app \
    HF_HOME=/home/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/app/.cache/transformers \
    HUGGINGFACE_HUB_CACHE=/home/app/.cache/huggingface/hub
RUN mkdir -p $HF_HOME $TRANSFORMERS_CACHE $HUGGINGFACE_HUB_CACHE && \
    chown -R app:app /home/app/.cache

USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD curl -fsS http://127.0.0.1:8000/health >/dev/null || exit 1

CMD ["uvicorn", "server.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
