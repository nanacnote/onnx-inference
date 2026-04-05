---
description: "Start a cell server for local development. Use when you want to run and test a specific cell during development."
argument-hint: "Cell name to start (e.g. bge-small-v15, kokoro-82m-v1.0, rapid-pp-ocr-v4)"
agent: "agent"
tools: [run_in_terminal, file_search]
---

> **Approval required**: Before each command, show the exact command and ask the user to confirm before executing.

Start the onnx-inference cell server for: **$ARGUMENTS**

## Pre-flight checks

### 1 — Resolve the cell name

List available cells so the correct directory name is used:
```bash
ls cells/
```

If `$ARGUMENTS` is ambiguous or empty, ask the user which cell to start before continuing.

### 2 — Verify the venv is active

```bash
python -c "import grpc; import onnxruntime; print('env OK')"
```

If it fails, activate the venv first:
```bash
source .venv/bin/activate
```

### 3 — Verify proto stubs exist

```bash
ls protos/*_pb2.py
```

If any stub is missing, compile them:
```bash
for proto in protos/*.proto; do
  python -m grpc_tools.protoc \
    -I./protos \
    --python_out=./protos \
    --grpc_python_out=./protos \
    "$proto"
done
```

## Start the server

```bash
python cells/<cell-name>/server.py
```

Expected log line: `onnx-inference server listening on 0.0.0.0:<port>`

The server runs in the foreground. Leave it running and open a second terminal to send requests.

## Smoke test

Look up the cell's port from the Cells table in `README.md`. When running locally (no gateway), pass `--host localhost:<port>`:
```bash
python client/<cell>_client.py --host localhost:<port> <args>
```

## Troubleshooting

| Symptom                     | Likely cause                                 | Fix                                                           |
| --------------------------- | -------------------------------------------- | ------------------------------------------------------------- |
| `ModuleNotFoundError: grpc` | venv not active                              | `source .venv/bin/activate`                                   |
| `No module named 'ocr_pb2'` | Proto stubs not compiled                     | Run the protoc loop above                                     |
| `StatusCode.UNAVAILABLE`    | Server not running or wrong port             | Check the server log and the port in README                   |
| Model load error on startup | Missing model artifact in weights/ directory | See `cells/<cell-name>/weights/.gitignore` for required files |
