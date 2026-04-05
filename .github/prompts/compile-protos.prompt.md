---
description: "Compile all .proto files in protos/ into Python gRPC stubs. Use when any proto file has changed or stubs are missing."
agent: "agent"
tools: [run_in_terminal, file_search, get_errors]
---

> **Approval required**: Before command, show the exact command and ask the user to confirm before executing.

Compile all onnx-inference proto definitions into Python stubs.

## Steps

1. Verify `grpcio-tools` is available. If `python -m grpc_tools.protoc --version` fails, run:
   ```bash
   pip install grpcio-tools
   ```

2. Run the compiler from the repo root:
   ```bash
   python sync_sdk_protos.py
   ```
   This compiles all `.proto` files in `protos/` and syncs the generated stubs into the SDK package in one step.

3. Confirm that a `_pb2.py` and `_pb2_grpc.py` file now exist in `protos/` for every `.proto` file.

4. Check each generated `_grpc.py` file's imports — if it contains `import <name>_pb2 as ...` without a relative `.` prefix, patch it to use a relative import so it resolves correctly when the protos directory is on `sys.path`:
   ```python
   # Before
   import embedding_pb2 as embedding__pb2
   # After
   from . import embedding_pb2 as embedding__pb2  # only if using package imports
   ```
   **Only apply this patch if the import would fail** given the project's sys.path injection pattern in `server.py` and the client scripts.

5. Report the output. If compilation produced any errors or warnings, show them in full.

## Notes

- The stubs are gitignored (`protos/*_pb2.py`, `protos/*_pb2_grpc.py`). Always recompile after cloning.
- Re-run this prompt any time any file under `protos/` is modified.
