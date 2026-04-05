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

import embedding_pb2  # noqa: E402  (generated stub)
import embedding_pb2_grpc  # noqa: E402  (generated stub)

from inference import BGEEmbedder  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("onnx-inference.bge-small-v15")

_HOST = "0.0.0.0"
_PORT = int(os.environ.get("PORT", 50051))
_MAX_WORKERS = 4


class EmbeddingServicer(embedding_pb2_grpc.EmbeddingServiceServicer):
    """gRPC servicer backed by the BGE-small-en-v1.5 ONNX model."""

    def __init__(self) -> None:
        logger.info("Loading BGE-small-en-v1.5 model…")
        self._embedder = BGEEmbedder()
        logger.info("Model ready.")

    def GetEmbedding(
        self,
        request: embedding_pb2.EmbeddingRequest,
        context: grpc.ServicerContext,
    ) -> embedding_pb2.EmbeddingResponse:
        if not request.text:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("'text' must not be empty.")
            return embedding_pb2.EmbeddingResponse()

        vector = self._embedder.embed(request.text)
        return embedding_pb2.EmbeddingResponse(
            vector=vector,
            dimensions=len(vector),
        )


def serve() -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS))
    embedding_pb2_grpc.add_EmbeddingServiceServicer_to_server(
        EmbeddingServicer(), server
    )

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
