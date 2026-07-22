FROM python:3.12-slim

WORKDIR /app

# Layer-cache dependencies separately from source
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Non-root user; data dir for the sqlite checkpointer + registry
RUN useradd --create-home athena \
    && mkdir -p /data \
    && chown -R athena:athena /data /app
USER athena

ENV ATHENA_CHECKPOINTER=sqlite \
    ATHENA_DB_PATH=/data/athena.db

EXPOSE 8000 8501

# Default: the API. The UI service overrides the command (see docker-compose.yml).
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
