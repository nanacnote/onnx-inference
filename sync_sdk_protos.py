#!/usr/bin/env python3
"""Compile proto stubs and sync them into the SDK package.

Run this from the repo root after any change to a .proto file:

    python sync_sdk_protos.py

What it does:
  1. Runs grpc_tools.protoc on every .proto file in protos/, writing the
     generated *_pb2.py and *_pb2_grpc.py stubs back into protos/.
  2. Copies each *_pb2.py into sdk/onnx_inference/_protos/ verbatim.
  3. Copies each *_pb2_grpc.py and rewrites the bare module import to a
     relative package import so it resolves correctly when bundled inside
     the onnx_inference._protos package:

         import foo_pb2 as foo__pb2
         →
         from . import foo_pb2 as foo__pb2
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent
PROTOS_DIR = REPO_ROOT / "protos"
SDK_PROTOS_DIR = REPO_ROOT / "sdk" / "onnx_inference" / "_protos"

_BARE_IMPORT_RE = re.compile(r"^import (\w+_pb2) as (\w+)$", re.MULTILINE)


def _compile() -> None:
    proto_files = sorted(PROTOS_DIR.glob("*.proto"))
    if not proto_files:
        print("warning: no .proto files found in protos/")
        return

    print("Compiling proto stubs…")
    for proto in proto_files:
        result = subprocess.run(
            [
                sys.executable, "-m", "grpc_tools.protoc",
                f"-I{PROTOS_DIR}",
                f"--python_out={PROTOS_DIR}",
                f"--grpc_python_out={PROTOS_DIR}",
                str(proto),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  error compiling {proto.name}:\n{result.stderr}", file=sys.stderr)
            sys.exit(result.returncode)
        print(f"  compiled  {proto.name}")


def _patch_grpc(source: str) -> str:
    return _BARE_IMPORT_RE.sub(r"from . import \1 as \2", source)


def _sync() -> None:
    copied, patched, skipped = 0, 0, 0

    print("\nSyncing stubs into SDK…")
    for src_file in sorted(PROTOS_DIR.glob("*_pb2*.py")):
        dst_file = SDK_PROTOS_DIR / src_file.name
        text = src_file.read_text(encoding="utf-8")

        if src_file.name.endswith("_pb2_grpc.py"):
            new_text = _patch_grpc(text)
            if dst_file.exists() and dst_file.read_text(encoding="utf-8") == new_text:
                skipped += 1
                continue
            dst_file.write_text(new_text, encoding="utf-8")
            patched += 1
        else:
            if dst_file.exists() and dst_file.read_text(encoding="utf-8") == text:
                skipped += 1
                continue
            shutil.copy2(src_file, dst_file)
            copied += 1

        print(f"  {'patched' if src_file.name.endswith('_grpc.py') else 'copied ':7s} {src_file.name}")

    print(f"\nDone — {copied} copied, {patched} patched, {skipped} unchanged.")


if __name__ == "__main__":
    for path, label in [(PROTOS_DIR, "protos/"), (SDK_PROTOS_DIR, "sdk/onnx_inference/_protos/")]:
        if not path.exists():
            print(f"error: {label} not found at {path}", file=sys.stderr)
            sys.exit(1)

    _compile()
    _sync()

