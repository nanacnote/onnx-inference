# ---------------------------------------------------------------------------
# Stage 1 — dependency layer (cached separately from application code)
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS deps

WORKDIR /app

COPY requirements.txt .
# rapidocr pulls in opencv-python (GUI build) which links against X11/XCB.
# Force-replace it with the headless build — no X11 deps, same cv2 API.
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless

# ---------------------------------------------------------------------------
# Stage 2 — runtime image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# CELL must be supplied at build time, e.g. --build-arg CELL=bge-small-v15
ARG CELL
ARG PORT=50051

# Bake both values into the image so the CMD and healthcheck can reference them
# at runtime without requiring them to be passed again via -e.
ENV CELL=${CELL}
ENV PORT=${PORT}

WORKDIR /app

# Copy installed packages from the deps stage instead of re-downloading.
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy protos and compile all gRPC stubs so generated files are always
# consistent with the bundled proto definitions.
COPY protos/ ./protos/
RUN for proto in protos/*.proto; do \
      python -m grpc_tools.protoc \
        -I./protos \
        --python_out=./protos \
        --grpc_python_out=./protos \
        "$proto"; \
    done

# Copy only the target cell (model artifacts must be present before building).
COPY cells/${CELL}/ ./cells/${CELL}/

# Copy the in-container health probe used by the Docker HEALTHCHECK.
COPY healthcheck.py ./

EXPOSE ${PORT}

# exec replaces the shell so Python becomes PID 1 and receives signals directly.
CMD ["sh", "-c", "exec python cells/${CELL}/server.py"]
