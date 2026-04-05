#!/usr/bin/env python
"""gRPC health probe — used as the Docker HEALTHCHECK command.

Usage:
    python healthcheck.py localhost:50051
"""
from __future__ import annotations

import sys

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc


def main() -> None:
    addr = sys.argv[1] if len(sys.argv) > 1 else "localhost:50051"
    try:
        channel = grpc.insecure_channel(addr)
        stub = health_pb2_grpc.HealthStub(channel)
        response = stub.Check(health_pb2.HealthCheckRequest(), timeout=5)
        sys.exit(0 if response.status == health_pb2.HealthCheckResponse.SERVING else 1)
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
