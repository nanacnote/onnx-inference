---
description: "Build and run onnx-inference with Docker. Use when packaging the server for deployment or testing the container image locally."
argument-hint: "Optional: cell name to target (e.g. bge-small-v15, kokoro-82m-v1.0, or 'all' for compose) or 'debug' for an interactive shell"
agent: "agent"
tools: [run_in_terminal, file_search]
---

> **Approval required**: Before command, show the exact command and ask the user to confirm before executing.

Build and run the onnx-inference Docker image for: **$ARGUMENTS**

## Pre-flight checks

1. Confirm model artifacts exist inside every cell's `weights/` directory before building — they are baked into the image. Check each cell that will be included:
   ```bash
   find cells/ -path '*/weights/*.onnx' -o -path '*/weights/tokenizer.json' -o -path '*/weights/*.npy' | sort
   ```
   If any expected artifact is missing, stop and ask the user to place it before continuing.

2. Confirm Docker daemon is running:
   ```bash
   docker info --format '{{.ServerVersion}}'
   ```

## Build

Each cell is its own image. Use `docker compose` to build all cells at once, or build a specific cell manually.

### Compose (build all cells + gateway)
```bash
docker compose build
```

### Single cell
Look up the cell's port from `docker-compose.yml`, then:
```bash
docker build --build-arg CELL=<cell-name> --build-arg PORT=<port> \
  -t onnx-inference/<cell-name>:latest .
```

Show the full build output. If it fails, report the exact error and suggest a fix.

## Run

### Compose (all cells + gateway)
```bash
docker compose up
```

The `gateway` service starts last — it waits for all model services to pass their health checks before accepting traffic.

### Single cell
```bash
docker run --rm -p <port>:<port> -e PORT=<port> onnx-inference/<cell-name>:latest
```

Expected log line inside the container: `onnx-inference server listening on 0.0.0.0:<port>`

## Smoke test (from host)

In a separate terminal, verify the container is reachable via the gateway on port 50050:
```bash
# Embedding cell
python client/embedding_client.py "Docker smoke test"
# TTS cell
python client/tts_client.py "Docker smoke test" -o output.wav
```

To test a backend directly (bypassing the gateway):
```bash
python client/embedding_client.py "Docker smoke test" --host localhost:50051
python client/tts_client.py "Docker smoke test" -o output.wav --host localhost:50052
```

## Debug mode

If `$ARGUMENTS` contains "debug", run an interactive shell against the target cell's image:
```bash
docker run --rm -it --entrypoint /bin/bash onnx-inference/<cell-name>:latest
```

## Useful one-liners

| Task                     | Command                                                                       |
| ------------------------ | ----------------------------------------------------------------------------- |
| List layers and sizes    | `docker history onnx-inference/<cell-name>:latest`                            |
| Inspect final image size | `docker image inspect onnx-inference/<cell-name>:latest --format '{{.Size}}'` |
| Tail container logs      | `docker logs -f <container-id>`                                               |
| Tail compose logs        | `docker compose logs -f`                                                      |
| Remove dangling images   | `docker image prune -f`                                                       |

## Notes

- The `Dockerfile` compiles proto stubs at build time — no need to pre-generate them.
- GPU support: swap `CPUExecutionProvider` in `inference.py` to `CUDAExecutionProvider` and use `nvidia/cuda` as the base image.
