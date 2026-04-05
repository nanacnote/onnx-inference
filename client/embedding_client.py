"""Client for the embedding service (BGE-small-v1.5), routed via the gateway.

Usage:
    python client/embedding_client.py "Your text here"
    python client/embedding_client.py "Your text here" -o vector.json
    python client/embedding_client.py "Your text here" --host localhost:50050
    echo "Your text here" | python client/embedding_client.py -
    python client/embedding_client.py notes.txt -o vector.json
"""

from __future__ import annotations

import json
import sys
import argparse
import logging
from pathlib import Path

import grpc

_PROTOS_DIR = str(Path(__file__).parent.parent / "protos")
if _PROTOS_DIR not in sys.path:
    sys.path.insert(0, _PROTOS_DIR)

import embedding_pb2  # noqa: E402
import embedding_pb2_grpc  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("onnx-inference.client")


def get_embedding(text: str, host: str = "localhost:50050") -> list[float]:
    """Request an embedding vector from the onnx-inference server.

    Args:
        text: The input string to embed.
        host: ``host:port`` of the running onnx-inference server.

    Returns:
        A unit-normalised list of floats representing the embedding.
    """
    with grpc.insecure_channel(host) as channel:
        stub = embedding_pb2_grpc.EmbeddingServiceStub(channel)
        response: embedding_pb2.EmbeddingResponse = stub.GetEmbedding(
            embedding_pb2.EmbeddingRequest(text=text)
        )
    return list(response.vector)


def main() -> None:
    parser = argparse.ArgumentParser(description="onnx-inference embedding client")
    parser.add_argument(
        "text",
        metavar="TEXT|-",
        help="text to embed, or '-' to read from stdin",
    )
    parser.add_argument("-o", "--output", default=None, help="write vector as JSON to this file (default: print to stdout)")
    parser.add_argument("--host", default="localhost:50050", help="gateway host:port (default: localhost:50050)")
    args = parser.parse_args()

    text = sys.stdin.read().strip() if args.text == "-" else (
        Path(args.text).read_text(encoding="utf-8").strip()
        if Path(args.text).is_file()
        else args.text
    )

    logger.info("Connecting to %s …", args.host)
    vector = get_embedding(text, args.host)

    if args.output:
        Path(args.output).write_text(json.dumps(vector), encoding="utf-8")
        logger.info("Saved %d dimensions → %s", len(vector), args.output)
    else:
        logger.info("Text      : %s", text)
        logger.info("Dimensions: %d", len(vector))
        logger.info("Vector[0:8]: %s", vector[:8])
        print(json.dumps(vector))


if __name__ == "__main__":
    main()
