---
description: "Full local development environment setup for onnx-inference. Use when setting up a fresh clone, troubleshooting a broken environment, or onboarding to the project."
agent: "agent"
tools: [run_in_terminal, file_search, get_errors]
---

> **Approval required**: Before command, show the exact command and ask the user to confirm before executing.

Walk through the complete local setup for onnx-inference and verify every step.

## Steps

### 1 — Python environment

Ensure Python 3.11 is active. If not:
```bash
python --version
```
Recommend `pyenv install 3.11` or a virtualenv if the version is wrong. Create and activate a venv using the `python3.11` binary:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Verify key packages loaded correctly:
```bash
python -c "import onnxruntime; import tokenizers; import grpc; import numpy; print('all deps OK')"
```

### 3 — Compile proto stubs

The generated stubs are gitignored and must be built after every fresh clone:
```bash
python sync_sdk_protos.py
```

This compiles all `.proto` files and syncs the generated stubs into the SDK package in one step.

Confirm that a `_pb2.py` and `_pb2_grpc.py` file now exist in `protos/` for every `.proto` file.

### 4 — Place model artifacts

Model files are **not** committed. Each cell has a `weights/` directory containing a `.gitignore` that lists exactly which files are required and where to source them. Place the files there before starting the server. If you need to export a model from HuggingFace, use the `/export-to-onnx` prompt.

### 5 — Start a cell server

```bash
python cells/<cell-name>/server.py
```

Expected log line: `onnx-inference server listening on 0.0.0.0:<port>`

### 6 — Smoke test with the client

Open a second terminal. When running cells directly (no gateway), pass `--host localhost:<port>` matching the port the cell started on:
```bash
# Embedding cell (port 50051)
python client/embedding_client.py "Hello from onnx-inference" --host localhost:50051
# TTS cell (port 50052)
python client/tts_client.py "Hello from onnx-inference" -o output.wav --host localhost:50052
```

Embedding: expected output includes `Dimensions: 384` and a non-zero `Vector[0:8]`.
TTS: expected output is `Saved N bytes → output.wav`.

## Troubleshooting

| Symptom                           | Fix                                                                     |
| --------------------------------- | ----------------------------------------------------------------------- |
| `ModuleNotFoundError: <cell>_pb2` | Re-run step 3                                                           |
| `FileNotFoundError: model.onnx`   | Complete step 4                                                         |
| `grpc._channel._InactiveRpcError` | Server is not running — complete step 5 first                           |
| Wrong Python version              | Use `pyenv local 3.11` or recreate venv with `python3.11 -m venv .venv` |
