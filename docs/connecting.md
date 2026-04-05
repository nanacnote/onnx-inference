# API Reference

All services are accessed over gRPC at a single address. Send your request there — the right model handles it.

## Endpoint

```
localhost:50050
```

---

## Python SDK

The fastest way to get started from Python. The SDK ships with bundled stubs — no protoc step required.

Get the wheel URL for the latest release from the [Releases page](https://github.com/nanacnote/onnx-inference/releases), then:

```bash
pip install https://github.com/nanacnote/onnx-inference/releases/download/<tag>/onnx_inference-<version>-py3-none-any.whl
```

```python
from onnx_inference import InferenceClient

with InferenceClient("localhost:50050") as client:
    vector  = client.embed("Hello world")
    audio   = client.synthesize("Hello world")
    regions = client.recognize(open("image.png", "rb").read())
    text    = client.generate("What is the capital of France?")
```

### `InferenceClient` reference

| Method       | Signature                                                                     | Returns                                                |
| ------------ | ----------------------------------------------------------------------------- | ------------------------------------------------------ |
| `embed`      | `embed(text: str)`                                                            | `list[float]` — 384-dimensional unit-normalised vector |
| `synthesize` | `synthesize(text: str, *, speed: float = 1.0)`                                | `bytes` — WAV audio (16-bit mono, 24 000 Hz)           |
| `recognize`  | `recognize(image: bytes)`                                                     | `list[OCRResult]` — one entry per detected text region |
| `generate`   | `generate(prompt: str, *, max_new_tokens: int = 0, temperature: float = 0.0)` | `str` — generated response                             |

`OCRResult` fields: `text: str`, `confidence: float`, `box: tuple[float, ...]` (flat quadrilateral `[x1,y1, x2,y2, x3,y3, x4,y4]`).

All methods raise `InferenceError` on failure. `InferenceError.code` is a `grpc.StatusCode`.

---

## Raw gRPC

For other languages, or when you need direct proto access. Obtain the `.proto` files from the `protos/` directory, then generate stubs for your language:

```bash
# Python example
for proto in protos/*.proto; do
  python -m grpc_tools.protoc \
    -I./protos \
    --python_out=./protos \
    --grpc_python_out=./protos \
    "$proto"
done
```

Add the directory containing the generated `*_pb2.py` files to your `sys.path` before importing them. The examples below assume `protos/` is on the path.

---

## Embedding

Encodes text into a 384-dimensional unit-normalised float vector, suitable for semantic search, similarity comparison, or clustering.

**Service:** `embedding.EmbeddingService`
**Method:** `GetEmbedding`

| Field  | Type   | Notes                         |
| ------ | ------ | ----------------------------- |
| `text` | string | The text to encode. Required. |

| Field        | Type           | Notes                      |
| ------------ | -------------- | -------------------------- |
| `vector`     | repeated float | 384-dimensional embedding. |
| `dimensions` | int            | Always `384`.              |

```python
import grpc, embedding_pb2, embedding_pb2_grpc

with grpc.insecure_channel("localhost:50050") as channel:
    stub = embedding_pb2_grpc.EmbeddingServiceStub(channel)
    resp = stub.GetEmbedding(embedding_pb2.EmbeddingRequest(text="Hello world"))

print(resp.dimensions)  # 384
print(list(resp.vector[:4]))
```

---

## Text-to-Speech

Converts text to spoken audio, returned as 16-bit mono PCM WAV at 24 000 Hz.

**Service:** `tts.TTSService`
**Method:** `Synthesize`

| Field   | Type   | Notes                                                  |
| ------- | ------ | ------------------------------------------------------ |
| `text`  | string | The text to speak. Required.                           |
| `speed` | float  | Speaking speed multiplier. Range 0.5–2.0, default 1.0. |

| Field         | Type  | Notes                                |
| ------------- | ----- | ------------------------------------ |
| `audio`       | bytes | WAV-encoded audio (PCM 16-bit mono). |
| `sample_rate` | int   | Always `24000`.                      |

```python
import grpc, tts_pb2, tts_pb2_grpc

with grpc.insecure_channel("localhost:50050") as channel:
    stub = tts_pb2_grpc.TTSServiceStub(channel)
    resp = stub.Synthesize(tts_pb2.SynthesizeRequest(text="Hello world", speed=1.0))

with open("output.wav", "wb") as f:
    f.write(resp.audio)
```

---

## OCR

Detects and reads text in an image. Accepts JPEG, PNG, BMP, TIFF, or WebP.

**Service:** `ocr.OCRService`
**Method:** `Recognize`

| Field   | Type  | Notes                              |
| ------- | ----- | ---------------------------------- |
| `image` | bytes | Raw encoded image bytes. Required. |

| Field     | Type                  | Notes                               |
| --------- | --------------------- | ----------------------------------- |
| `results` | repeated `TextResult` | One entry per detected text region. |

Each `TextResult`:

| Field        | Type           | Notes                                               |
| ------------ | -------------- | --------------------------------------------------- |
| `text`       | string         | Detected text.                                      |
| `confidence` | float          | Detection confidence, 0–1.                          |
| `box`        | repeated float | Flat quadrilateral: `[x1,y1, x2,y2, x3,y3, x4,y4]`. |

```python
import grpc, ocr_pb2, ocr_pb2_grpc

image_bytes = open("image.png", "rb").read()

with grpc.insecure_channel("localhost:50050") as channel:
    stub = ocr_pb2_grpc.OCRServiceStub(channel)
    resp = stub.Recognize(ocr_pb2.RecognizeRequest(image=image_bytes))

for r in resp.results:
    print(f"[{r.confidence:.2f}] {r.text}")
```

---

## Text Generation

Generates text from a prompt. Send the user message — instruction formatting is handled automatically.

**Service:** `llm.LLMService`
**Method:** `Generate`

| Field            | Type   | Notes                                                   |
| ---------------- | ------ | ------------------------------------------------------- |
| `prompt`         | string | The user message. Required.                             |
| `max_new_tokens` | int    | Maximum tokens to generate. `0` uses the default (256). |
| `temperature`    | float  | Sampling temperature. `0.0` = greedy (deterministic).   |

| Field  | Type   | Notes                    |
| ------ | ------ | ------------------------ |
| `text` | string | Generated response text. |

```python
import grpc, llm_pb2, llm_pb2_grpc

with grpc.insecure_channel("localhost:50050") as channel:
    stub = llm_pb2_grpc.LLMServiceStub(channel)
    resp = stub.Generate(llm_pb2.GenerateRequest(
        prompt="What is the capital of France?",
        max_new_tokens=128,
        temperature=0.0,
    ))

print(resp.text)
```
