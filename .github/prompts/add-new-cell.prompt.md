---
description: "Scaffold a new model cell under /cells. Use when adding a new ONNX model to the inference server."
argument-hint: "Cell name (e.g. minilm-l6-v2) and model type (e.g. embedding, classification)"
agent: "agent"
tools: [create_file, read_file, file_search]
---

> **Approval required**: Before command, show the exact command and ask the user to confirm before executing.

Scaffold a new onnx-inference cell for: **$ARGUMENTS**

## What to generate

Create the following structure under `cells/<cell-name>/`:

```
cells/<cell-name>/
  __init__.py       (empty)
  inference.py      (OnnxModel class)
  server.py         (gRPC server on 0.0.0.0:50051)
  weights/
    .gitignore      (lists required files, ignores everything else)
```

## Rules

1. **inference.py** must:
   - Resolve `weights_dir = Path(__file__).parent / "weights"` and load all model artifacts from there — no hardcoded paths, no runtime downloads.
   - Discover available ONNX input names at init time and only pass what the model declares (guard `token_type_ids` conditionally).
   - Implement the same pooling and normalisation strategy appropriate for the model type:
     - Embedding models → CLS pooling (`last_hidden_state[0, 0, :]`) + L2 normalise.
     - Classification models → softmax over the logits output.
   - Use strict type hints throughout. No `Any`. Return `list[float]`.

2. **server.py** must:
   - Import gRPC stubs from `../../protos` by injecting via `sys.path`.
   - Read the port from `int(os.environ.get("PORT", <default-port>))` — never hardcode the port.
   - Import and register the standard gRPC health service:
     ```python
     from grpc_health.v1 import health as grpc_health
     from grpc_health.v1 import health_pb2, health_pb2_grpc
     ```
     Create a `HealthServicer(experimental_non_blocking=True)`, add it to the server with `health_pb2_grpc.add_HealthServicer_to_server`, and call `health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)` after `server.start()`.
   - Validate that `request.text` is non-empty and return `INVALID_ARGUMENT` if not.
   - Use `logging.getLogger("onnx-inference.<cell-name>")`.
   - Register `signal.signal` handlers for both `SIGINT` and `SIGTERM` that:
     1. Set health status to `NOT_SERVING` before stopping.
     2. Call `server.stop(grace=5).wait()` and log a shutdown message.
   - Call `server.wait_for_termination()` in `serve()` after registering the signal handlers.

3. **Do not** add any `transformers` or `torch` imports anywhere.
4. **Do not** create `model.onnx` or `tokenizer.json` — those are placed by the user in the `weights/` directory.
5. Follow the existing [bge-small-v15 cell](../../cells/bge-small-v15/inference.py) as the canonical reference.

## After generating

Tell the user:
- Exactly which files were created.
- The proto compilation command if the proto has changed (run `sync_sdk_protos.py` immediately after to keep the SDK stubs in sync).
- A reminder to place model artifacts in `cells/<cell-name>/weights/` before running (the generated `.gitignore` there documents which files are needed).
- A reminder to add a new service entry to `docker-compose.yml` for the new cell, following the pattern of the existing services (unique port, appropriate memory limit, `start_period` tuned to the model's load time).
- A reminder to add a `location /<package>.<Service>/` block to `nginx/nginx.conf` pointing at the new cell's host and port, so the gateway routes to it.
