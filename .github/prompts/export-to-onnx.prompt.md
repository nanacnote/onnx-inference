---
description: "Export a HuggingFace model to ONNX format for use in an onnx-inference cell. Use when preparing model.onnx and tokenizer.json for a new or existing cell."
argument-hint: "HuggingFace model ID (e.g. BAAI/bge-small-en-v1.5) and target cell name (e.g. bge-small-v15)"
agent: "agent"
tools: [run_in_terminal, file_search]
---

> **Approval required**: Before command, show the exact command and ask the user to confirm before executing.

Export a HuggingFace model to ONNX for onnx-inference.

Model / cell target: **$ARGUMENTS**

## Important constraint

This project does **not** include `transformers` or `torch` in `requirements.txt`. The export is a **one-time local operation** run outside the server environment. Use a separate venv or a throwaway environment so that the heavy dependencies never pollute the inference runtime.

## 1 — Create a temporary export environment

```bash
python3 -m venv /tmp/onnx-export-env
source /tmp/onnx-export-env/bin/activate
pip install --upgrade pip
pip install transformers torch optimum[exporters]
```

## 2 — Export with Optimum

Replace `<model-id>` and `<cell-dir>` with the values from `$ARGUMENTS`:

```bash
optimum-cli export onnx \
  --model <model-id> \
  --task feature-extraction \
  --opset 17 \
  <cell-dir>-export/
```

This writes `model.onnx`, `tokenizer.json`, and related vocab files into `<cell-dir>-export/`.

## 3 — Copy artifacts into the cell

Only the files the cell's `inference.py` reads are needed. Place them in the `weights/` subdirectory:

```bash
cp <cell-dir>-export/model.onnx     cells/<cell-name>/weights/model.onnx
cp <cell-dir>-export/tokenizer.json cells/<cell-name>/weights/tokenizer.json
```

The `weights/.gitignore` in each cell documents exactly which files are expected.

## 4 — Verify the model inputs

Print the model's declared input names so `inference.py` can be written correctly:

```python
import onnxruntime as ort
sess = ort.InferenceSession("cells/<cell-name>/weights/model.onnx", providers=["CPUExecutionProvider"])
for inp in sess.get_inputs():
    print(inp.name, inp.shape, inp.type)
```

Report the output to the user. The `inference.py` `BGEEmbedder` already guards `token_type_ids` conditionally, so no further changes are needed for standard encoder models.

## 5 — Cleanup (optional)

```bash
deactivate
rm -rf /tmp/onnx-export-env
```

## Notes

- If the model requires a custom export flag (e.g. `--no-post-process` or a specific `--framework`), add it to the `optimum-cli` command.
- For classification models, change `--task` to `text-classification`.
- The exported `tokenizer.json` is the Rust-tokenizers fast format. It is read directly by the `tokenizers` library — no `transformers` needed at inference time.
