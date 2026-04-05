from __future__ import annotations

import grpc


class InferenceError(Exception):
    """Raised when a service call fails.

    Attributes:
        code: The gRPC status code returned by the server.
    """

    def __init__(self, message: str, code: grpc.StatusCode) -> None:
        super().__init__(message)
        self.code = code
