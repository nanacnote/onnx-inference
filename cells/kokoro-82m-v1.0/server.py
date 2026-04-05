from __future__ import annotations

import os
import sys
import signal
import logging
from concurrent import futures
from pathlib import Path

import grpc
from grpc_health.v1 import health as grpc_health
from grpc_health.v1 import health_pb2, health_pb2_grpc

_PROTOS_DIR = str(Path(__file__).parent.parent.parent / "protos")
if _PROTOS_DIR not in sys.path:
    sys.path.insert(0, _PROTOS_DIR)

import tts_pb2  # noqa: E402  (generated stub)
import tts_pb2_grpc  # noqa: E402  (generated stub)

from inference import KokoroTTS  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("onnx-inference.kokoro-82m-v1.0")

_HOST = "0.0.0.0"
_PORT = int(os.environ.get("PORT", 50052))
_MAX_WORKERS = 2  # TTS is memory-heavy; keep concurrency conservative

# Voice: af_sky (American English, female)
# Model: kokoro-v1.0.fp16.onnx


class TTSServicer(tts_pb2_grpc.TTSServiceServicer):
    """gRPC servicer backed by the kokoro-82m-v1.0 model."""

    def __init__(self) -> None:
        logger.info("Loading kokoro-82m-v1.0 model…")
        self._tts = KokoroTTS()
        logger.info("Model ready.")

    def Synthesize(
        self,
        request: tts_pb2.SynthesizeRequest,
        context: grpc.ServicerContext,
    ) -> tts_pb2.SynthesizeResponse:
        if not request.text:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("'text' must not be empty.")
            return tts_pb2.SynthesizeResponse()

        speed = request.speed if request.speed > 0 else 1.0

        try:
            wav_bytes, sample_rate = self._tts.synthesize(
                request.text, speed=speed
            )
        except ValueError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return tts_pb2.SynthesizeResponse()

        return tts_pb2.SynthesizeResponse(audio=wav_bytes, sample_rate=sample_rate)


def serve() -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS))
    tts_pb2_grpc.add_TTSServiceServicer_to_server(TTSServicer(), server)

    health_servicer = grpc_health.HealthServicer(experimental_non_blocking=True)
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    listen_addr = f"{_HOST}:{_PORT}"
    server.add_insecure_port(listen_addr)
    server.start()
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    logger.info("onnx-inference server listening on %s", listen_addr)

    def _shutdown(signum, frame) -> None:
        logger.info("Shutting down gracefully…")
        health_servicer.set("", health_pb2.HealthCheckResponse.NOT_SERVING)
        server.stop(grace=10).wait()
        logger.info("Server stopped.")

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    server.wait_for_termination()


if __name__ == "__main__":
    serve()
