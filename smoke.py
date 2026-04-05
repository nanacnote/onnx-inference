#!/usr/bin/env python3
"""Smoke test — hits all four models through the gateway and prints results.

Usage:
    python smoke.py                  # defaults to localhost:50050
    python smoke.py <host:port>      # explicit gateway address

Requires the onnx-inference SDK to be installed:
    pip install https://github.com/nanacnote/onnx-inference/releases/download/<tag>/onnx_inference-<version>-py3-none-any.whl
"""

import os
import struct
import sys
import zlib

from onnx_inference import InferenceClient, InferenceError

HOST = sys.argv[1] if len(sys.argv) > 1 else "localhost:50050"
PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def check(name: str, fn) -> bool:
    try:
        result = fn()
        print(f"[{PASS}] {name}: {result}")
        return True
    except InferenceError as e:
        print(f"[{FAIL}] {name}: {e} (code={e.code})")
        return False
    except Exception as e:
        print(f"[{FAIL}] {name}: unexpected error — {e}")
        return False


def _make_stub_png() -> bytes:
    """Generate a valid 1x1 white RGB PNG using stdlib only."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))  # filter=None + RGB white
        + chunk(b"IEND", b"")
    )


def _ocr_image() -> bytes:
    if os.path.exists("test.png"):
        with open("test.png", "rb") as f:
            return f.read()
    return _make_stub_png()


print(f"Gateway: {HOST}\n")

results = []
with InferenceClient(HOST) as c:
    results.append(check(
        "embed",
        lambda: f"{len(c.embed('hello world'))} floats",
    ))
    results.append(check(
        "synthesize",
        lambda: f"{len(c.synthesize('hello world'))} bytes",
    ))
    results.append(check(
        "recognize",
        lambda: f"{len(c.recognize(_ocr_image()))} region(s)",
    ))
    results.append(check(
        "generate",
        lambda: repr(c.generate("Reply with one word: hello", max_new_tokens=16).strip()),
    ))

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
