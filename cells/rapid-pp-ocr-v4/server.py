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

import ocr_pb2        # noqa: E402
import ocr_pb2_grpc   # noqa: E402

from inference import RapidOCRInference  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("onnx-inference.rapid-pp-ocr-v4")

_HOST = "0.0.0.0"
_PORT = int(os.environ.get("PORT", 50053))
_MAX_WORKERS = 4


class OCRServicer(ocr_pb2_grpc.OCRServiceServicer):
    """gRPC servicer backed by RapidOCR ONNX models."""

    def __init__(self) -> None:
        logger.info("Loading RapidOCR models…")
        self._ocr = RapidOCRInference()
        logger.info("Models ready.")

    def Recognize(
        self,
        request: ocr_pb2.RecognizeRequest,
        context: grpc.ServicerContext,
    ) -> ocr_pb2.RecognizeResponse:
        if not request.image:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("'image' must not be empty.")
            return ocr_pb2.RecognizeResponse()

        try:
            results = self._ocr.recognize(request.image)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return ocr_pb2.RecognizeResponse()

        return ocr_pb2.RecognizeResponse(
            results=[
                ocr_pb2.TextResult(
                    text=r.text,
                    confidence=r.confidence,
                    box=r.box,
                )
                for r in results
            ]
        )


def serve() -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS))
    ocr_pb2_grpc.add_OCRServiceServicer_to_server(OCRServicer(), server)

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
