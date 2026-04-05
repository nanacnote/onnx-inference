"""Client for the OCR service (rapid-pp-ocr-v4), routed via the gateway.

Usage:
    python client/ocr_client.py image.png
    python client/ocr_client.py image.jpg --host localhost:50050
    python client/ocr_client.py image.png -o results.json
"""

from __future__ import annotations

import sys
import json
import argparse
import logging
from pathlib import Path

import grpc

_PROTOS_DIR = str(Path(__file__).parent.parent / "protos")
if _PROTOS_DIR not in sys.path:
    sys.path.insert(0, _PROTOS_DIR)

import ocr_pb2        # noqa: E402
import ocr_pb2_grpc   # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("onnx-inference.ocr-client")


def recognize(
    image_bytes: bytes,
    host: str = "localhost:50050",
) -> list[dict[str, object]]:
    """Send image bytes to the OCR server and return structured results."""
    with grpc.insecure_channel(host) as channel:
        stub = ocr_pb2_grpc.OCRServiceStub(channel)
        response: ocr_pb2.RecognizeResponse = stub.Recognize(
            ocr_pb2.RecognizeRequest(image=image_bytes)
        )
    return [
        {
            "text": r.text,
            "confidence": round(r.confidence, 4),
            "box": list(r.box),
        }
        for r in response.results
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="onnx-inference OCR client")
    parser.add_argument("image", metavar="IMAGE", help="path to the image file")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="write JSON results to this file instead of stdout",
    )
    parser.add_argument(
        "--host",
        default="localhost:50050",
        help="gateway host:port (default: localhost:50050)",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        logger.error("File not found: %s", image_path)
        sys.exit(1)

    image_bytes = image_path.read_bytes()
    logger.info("Connecting to %s …", args.host)
    results = recognize(image_bytes, host=args.host)

    output_json = json.dumps(results, ensure_ascii=False, indent=2)

    if args.output:
        Path(args.output).write_text(output_json, encoding="utf-8")
        logger.info("Saved %d result(s) → %s", len(results), args.output)
    else:
        for item in results:
            print(f"[{item['confidence']:.4f}] {item['text']}")


if __name__ == "__main__":
    main()
