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

# Resolve the compiled proto stubs regardless of the working directory.
_PROTOS_DIR = str(Path(__file__).parent.parent.parent / "protos")
if _PROTOS_DIR not in sys.path:
    sys.path.insert(0, _PROTOS_DIR)

import llm_pb2        # noqa: E402  (generated stub)
import llm_pb2_grpc   # noqa: E402  (generated stub)

from inference import SmolLMInference  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("onnx-inference.smollm2-360m")

_HOST = "0.0.0.0"
_PORT = int(os.environ.get("PORT", 50054))
_MAX_WORKERS = 2  # generation is serial; workers > 1 only helps concurrent requests queue


class LLMServicer(llm_pb2_grpc.LLMServiceServicer):
    """gRPC servicer backed by SmolLM2-360M-Instruct."""

    def __init__(self) -> None:
        logger.info("Loading SmolLM2-360M-Instruct models…")
        self._model = SmolLMInference()
        logger.info("Models ready.")

    def Generate(
        self,
        request: llm_pb2.GenerateRequest,
        context: grpc.ServicerContext,
    ) -> llm_pb2.GenerateResponse:
        if not request.prompt:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("'prompt' must not be empty.")
            return llm_pb2.GenerateResponse()

        text = self._model.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )
        return llm_pb2.GenerateResponse(text=text)


def serve() -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS))
    llm_pb2_grpc.add_LLMServiceServicer_to_server(LLMServicer(), server)

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
        server.stop(grace=5).wait()
        logger.info("Server stopped.")

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    server.wait_for_termination()


if __name__ == "__main__":
    serve()
