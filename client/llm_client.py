"""Client for the LLM service (smollm2-360m), routed via the gateway.

Usage:
    python client/llm_client.py "What is the capital of France?"
    python client/llm_client.py "Write a haiku about the sea" --host localhost:50050
    python client/llm_client.py "Explain gravity" --max-tokens 512 --temperature 0.7
"""

from __future__ import annotations

import sys
import argparse
import logging
from pathlib import Path

import grpc

_PROTOS_DIR = str(Path(__file__).parent.parent / "protos")
if _PROTOS_DIR not in sys.path:
    sys.path.insert(0, _PROTOS_DIR)

import llm_pb2        # noqa: E402
import llm_pb2_grpc   # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("onnx-inference.llm-client")


def generate(
    prompt: str,
    host: str = "localhost:50050",
    max_new_tokens: int = 0,
    temperature: float = 0.0,
) -> str:
    """Send a prompt to the LLM server and return the generated text."""
    with grpc.insecure_channel(host) as channel:
        stub = llm_pb2_grpc.LLMServiceStub(channel)
        response: llm_pb2.GenerateResponse = stub.Generate(
            llm_pb2.GenerateRequest(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        )
    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="onnx-inference LLM client")
    parser.add_argument("prompt", metavar="PROMPT", help="prompt text to send to the model")
    parser.add_argument(
        "--host",
        default="localhost:50050",
        help="gateway host:port (default: localhost:50050)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        dest="max_tokens",
        help="maximum new tokens to generate (0 = server default of 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="sampling temperature (0.0 = greedy decoding)",
    )
    args = parser.parse_args()

    logger.info("Connecting to %s …", args.host)
    text = generate(
        args.prompt,
        host=args.host,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print(text)


if __name__ == "__main__":
    main()
