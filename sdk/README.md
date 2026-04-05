# onnx-inference

Python SDK for the [onnx-inference](https://github.com/nanacnote/onnx-inference) gRPC gateway. Call embedding, text-to-speech, OCR, and text generation from a single client — no proto compilation step required.

## Installation

Download the wheel for the desired tag from the [Releases page](https://github.com/nanacnote/onnx-inference/releases) and install it directly:

```bash
# Replace <tag> and the wheel filename with the values from the release page
pip install https://github.com/nanacnote/onnx-inference/releases/download/<tag>/onnx_inference-<version>-py3-none-any.whl
```

Or install from a tag via Git (no download step):

```bash
pip install "onnx-inference @ git+https://github.com/nanacnote/onnx-inference.git@<tag>#subdirectory=sdk"
```

Requires Python 3.11+.

## Quick start

```python
from onnx_inference import InferenceClient

with InferenceClient("localhost:50050") as client:
    vector  = client.embed("Hello world")
    audio   = client.synthesize("Hello world")
    regions = client.recognize(open("image.png", "rb").read())
    text    = client.generate("What is the capital of France?")
```

## Services

### Embedding

Encodes text into a 384-dimensional unit-normalised float vector.

```python
vector: list[float] = client.embed("semantic search query")
```

### Text-to-Speech

Returns a complete WAV file as bytes — 16-bit mono PCM at 24 000 Hz.

```python
audio: bytes = client.synthesize("Hello world", speed=1.0)

with open("output.wav", "wb") as f:
    f.write(audio)
```

`speed` accepts 0.5–2.0, default `1.0`.

### OCR

Detects and reads text regions in an image. Accepts JPEG, PNG, BMP, TIFF, or WebP.

```python
from onnx_inference import OCRResult

regions: list[OCRResult] = client.recognize(open("image.png", "rb").read())

for r in regions:
    print(f"[{r.confidence:.2f}] {r.text}  box={r.box}")
```

`OCRResult` is a frozen dataclass with fields:

| Field        | Type                | Description                                         |
| ------------ | ------------------- | --------------------------------------------------- |
| `text`       | `str`               | Detected text.                                      |
| `confidence` | `float`             | Detection confidence, 0–1.                          |
| `box`        | `tuple[float, ...]` | Flat quadrilateral: `[x1,y1, x2,y2, x3,y3, x4,y4]`. |

### Text Generation

Generates text from a prompt. Instruction formatting is handled server-side — send only the user message.

```python
reply: str = client.generate(
    "Explain gRPC in one sentence.",
    max_new_tokens=128,
    temperature=0.0,
)
```

`max_new_tokens=0` uses the server default (256). `temperature=0.0` is greedy/deterministic.

## Error handling

All methods raise `InferenceError` on failure. The `code` attribute is a `grpc.StatusCode`.

```python
from onnx_inference import InferenceClient, InferenceError
import grpc

with InferenceClient("localhost:50050") as client:
    try:
        vector = client.embed("hello")
    except InferenceError as e:
        if e.code == grpc.StatusCode.UNAVAILABLE:
            print("gateway is down")
        else:
            raise
```

## Client lifecycle

Use as a context manager for automatic cleanup, or call `close()` manually for long-lived instances.

```python
# context manager (recommended)
with InferenceClient("localhost:50050") as client:
    ...

# long-lived
client = InferenceClient("localhost:50050")
try:
    ...
finally:
    client.close()
```

## License

MIT © Owusu K
