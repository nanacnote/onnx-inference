# onnx-inference

A high-performance, zero-bloat AI inference server. gRPC + ONNX Runtime — no Transformers, no PyTorch.

## Architecture

Models live in self-contained **cells** under `cells/`. Each cell owns its `server.py` and `inference.py`, plus the model artifacts it needs. Adding a new model means adding a new directory; nothing else changes.

```
protos/                    # gRPC service definitions (one .proto per capability)
cells/
  <model-name>/
    server.py              # gRPC server
    inference.py           # ONNX wrapper
    weights/               # model artifacts — not committed; place them here before running
      .gitignore           # documents required files and keeps the directory tracked
client/
  <cell>_client.py         # gRPC client per cell type
nginx/
  Dockerfile               # builds the gateway image — bakes nginx.conf in at build time
  nginx.conf               # gRPC reverse proxy — routes by service path prefix to the correct cell
Dockerfile                 # parameterized — builds one cell per image via --build-arg CELL=<name>
docker-compose.yml         # model services + gateway, health checks and resource limits
healthcheck.py             # gRPC health probe used by Docker HEALTHCHECK
requirements.txt
```

## Cells

| Cell              | Service         | Port    | Proto             |
| ----------------- | --------------- | ------- | ----------------- |
| `bge-small-v15`   | Text embedding  | `50051` | `embedding.proto` |
| `kokoro-82m-v1.0` | Text-to-speech  | `50052` | `tts.proto`       |
| `rapid-pp-ocr-v4` | OCR             | `50053` | `ocr.proto`       |
| `smollm2-360m`    | Text generation | `50054` | `llm.proto`       |

## Gateway

An NGINX reverse proxy sits in front of all model services and exposes a single gRPC endpoint on **port 50050**. Callers connect there and target any service — NGINX routes the request transparently based on the gRPC path prefix (`/<package>.<Service>/`), which gRPC sets automatically. Clients do not need to know which port a model lives on.

| gRPC path prefix               | Routed to               | Timeout |
| ------------------------------ | ----------------------- | ------- |
| `/embedding.EmbeddingService/` | `bge-small-v15:50051`   | 30 s    |
| `/tts.TTSService/`             | `kokoro-82m-v1.0:50052` | 60 s    |
| `/ocr.OCRService/`             | `rapid-pp-ocr-v4:50053` | 30 s    |
| `/llm.LLMService/`             | `smollm2-360m:50054`    | 120 s   |

The gateway waits for all backend services to pass their Docker health checks before starting. Individual backend ports (50051–50054) remain mapped to the host for direct access during development.

## Quick Start

### 1. Create a virtual environment

```bash
python3.11 -m venv .venv && source .venv/bin/activate  # requires Python 3.11
pip install -r requirements.txt
```

### 2. Compile the proto stubs

Run this once from the repo root after a fresh clone, and again whenever any `.proto` file changes:

```bash
python sync_sdk_protos.py
```

### 3. Place model artifacts

Model files are not committed. Each cell has a `weights/` directory containing a `.gitignore` that lists exactly which files are required and where to get them. Place the files there before starting the server.

### 4. Start a cell

```bash
python cells/<model-name>/server.py
```

### 5. Query from the client

When running cells directly (without Docker), there is no gateway — pass `--host localhost:<port>` matching the port the cell started on.

```bash
python client/<cell>_client.py --host localhost:<port> <args>
```

When running via Docker Compose, all clients default to the gateway on `localhost:50050` and no `--host` flag is needed.

```bash
python client/<cell>_client.py <args>
```

See each client file for its specific arguments and flags.

## Docker

Each cell builds into its own image. The `Dockerfile` is parameterized via `--build-arg`.

### Compose (recommended)

```bash
# Build and start all cells + gateway
docker compose up --build

# Start a single cell (without the gateway)
docker compose up --build <cell-name>

# Start specific cells with the gateway
docker compose up --build bge-small-v15 kokoro-82m-v1.0 gateway
```

### Manual build and run

```bash
# Build the gateway
docker build -t onnx-inference/gateway:latest ./nginx

# Build a specific cell
docker build --build-arg CELL=<cell-name> --build-arg PORT=<port> \
  -t onnx-inference/<cell-name>:latest .

# Run
docker run --rm -p <port>:<port> onnx-inference/<cell-name>:latest
```
