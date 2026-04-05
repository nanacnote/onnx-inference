from __future__ import annotations

import grpc

from ._protos import embedding_pb2, embedding_pb2_grpc
from ._protos import tts_pb2, tts_pb2_grpc
from ._protos import ocr_pb2, ocr_pb2_grpc
from ._protos import llm_pb2, llm_pb2_grpc
from ._exceptions import InferenceError
from ._models import OCRResult


class InferenceClient:
    """Client for all onnx-inference services.

    Manages a single shared channel to the gateway. Supports use as a
    context manager for automatic cleanup, or manual ``close()`` for
    long-lived instances.

    Args:
        address: ``host:port`` of the gateway. Defaults to ``localhost:50050``.

    Example — context manager::

        with InferenceClient("localhost:50050") as client:
            vector = client.embed("Hello world")
            audio  = client.synthesize("Hello world")

    Example — long-lived instance::

        client = InferenceClient("localhost:50050")
        try:
            print(client.generate("What is the capital of France?"))
        finally:
            client.close()
    """

    def __init__(self, address: str = "localhost:50050") -> None:
        self._channel = grpc.insecure_channel(address)
        self._embedding = embedding_pb2_grpc.EmbeddingServiceStub(self._channel)
        self._tts = tts_pb2_grpc.TTSServiceStub(self._channel)
        self._ocr = ocr_pb2_grpc.OCRServiceStub(self._channel)
        self._llm = llm_pb2_grpc.LLMServiceStub(self._channel)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying gRPC channel."""
        self._channel.close()

    def __enter__(self) -> InferenceClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Services
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """Encode *text* into a 384-dimensional unit-normalised float vector.

        Args:
            text: The text to encode. Must not be empty.

        Returns:
            A list of 384 floats representing the embedding.

        Raises:
            InferenceError: If the call fails or *text* is empty.
        """
        try:
            response = self._embedding.GetEmbedding(
                embedding_pb2.EmbeddingRequest(text=text)
            )
        except grpc.RpcError as exc:
            raise InferenceError(exc.details(), exc.code()) from exc
        return list(response.vector)

    def synthesize(self, text: str, *, speed: float = 1.0) -> bytes:
        """Convert *text* to speech and return WAV audio bytes.

        The returned bytes are a complete WAV file — 16-bit mono PCM at
        24 000 Hz — ready to write to disk or stream directly.

        Args:
            text: The text to speak. Must not be empty.
            speed: Speaking speed multiplier. Range 0.5–2.0, default 1.0.

        Returns:
            WAV-encoded audio as bytes.

        Raises:
            InferenceError: If the call fails or *text* is empty.
        """
        try:
            response = self._tts.Synthesize(
                tts_pb2.SynthesizeRequest(text=text, speed=speed)
            )
        except grpc.RpcError as exc:
            raise InferenceError(exc.details(), exc.code()) from exc
        return response.audio

    def recognize(self, image: bytes) -> list[OCRResult]:
        """Detect and read text regions in an image.

        Args:
            image: Raw encoded image bytes. Accepts JPEG, PNG, BMP, TIFF,
                or WebP.

        Returns:
            A list of :class:`OCRResult` objects, one per detected text
            region. Empty if no text was found.

        Raises:
            InferenceError: If the call fails or the image cannot be decoded.
        """
        try:
            response = self._ocr.Recognize(
                ocr_pb2.RecognizeRequest(image=image)
            )
        except grpc.RpcError as exc:
            raise InferenceError(exc.details(), exc.code()) from exc
        return [
            OCRResult(
                text=r.text,
                confidence=r.confidence,
                box=tuple(r.box),
            )
            for r in response.results
        ]

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 0,
        temperature: float = 0.0,
    ) -> str:
        """Generate text from *prompt*.

        The instruction template is applied server-side — send only the
        user message.

        Args:
            prompt: The user message. Must not be empty.
            max_new_tokens: Maximum tokens to generate. ``0`` uses the
                server default of 256.
            temperature: Sampling temperature. ``0.0`` (default) is greedy
                (deterministic). Higher values increase randomness.

        Returns:
            The generated response as a plain string.

        Raises:
            InferenceError: If the call fails or *prompt* is empty.
        """
        try:
            response = self._llm.Generate(
                llm_pb2.GenerateRequest(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            )
        except grpc.RpcError as exc:
            raise InferenceError(exc.details(), exc.code()) from exc
        return response.text
