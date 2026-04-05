"""Client for the TTS service (kokoro-82m-v1.0), routed via the gateway.

Usage:
    python client/tts_client.py "Hello from onnx-inference"
    python client/tts_client.py "Hello" -o output.wav
    python client/tts_client.py "Hello" -o output.wav --host localhost:50050
    echo "Hello" | python client/tts_client.py -
    cat notes.txt | python client/tts_client.py -
    python client/tts_client.py notes.txt -o output.wav
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

import tts_pb2  # noqa: E402
import tts_pb2_grpc  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("onnx-inference.tts-client")


def synthesize(
    text: str,
    speed: float = 1.0,
    host: str = "localhost:50050",
) -> bytes:
    """Request WAV audio from the TTS server and return the raw bytes."""
    with grpc.insecure_channel(host) as channel:
        stub = tts_pb2_grpc.TTSServiceStub(channel)
        response: tts_pb2.SynthesizeResponse = stub.Synthesize(
            tts_pb2.SynthesizeRequest(text=text, speed=speed)
        )
    return response.audio


def main() -> None:
    parser = argparse.ArgumentParser(description="onnx-inference TTS client")
    parser.add_argument(
        "text",
        metavar="TEXT|-",
        help="text to synthesize, or '-' to read from stdin",
    )
    parser.add_argument("-o", "--output", default="output.wav", help="output WAV file (default: output.wav)")
    parser.add_argument("--host", default="localhost:50050", help="gateway host:port (default: localhost:50050)")
    args = parser.parse_args()

    text = sys.stdin.read().strip() if args.text == "-" else (
        Path(args.text).read_text(encoding="utf-8").strip()
        if Path(args.text).is_file()
        else args.text
    )

    logger.info("Connecting to %s …", args.host)
    audio = synthesize(text, host=args.host)

    Path(args.output).write_bytes(audio)
    logger.info("Saved %d bytes → %s", len(audio), args.output)


if __name__ == "__main__":
    main()
