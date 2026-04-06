# ---------------------------------------------------------------------------
# Stage 1 — build: install Python dependencies
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS deps

WORKDIR /app

COPY sdk/ ./sdk/

RUN SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 pip install --no-cache-dir ./sdk flask

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=deps /usr/local/lib/python3.11/site-packages \
                 /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

COPY frontend/ ./

ENV GATEWAY_ADDRESS=gateway:50050

EXPOSE 8080

CMD ["python", "server.py"]
