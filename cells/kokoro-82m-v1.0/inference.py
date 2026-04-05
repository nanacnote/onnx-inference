# https://github.com/thewh1teagle/kokoro-onnx

from __future__ import annotations

import io
import wave

import numpy as np
from pathlib import Path
from kokoro_onnx import Kokoro

_DEFAULT_SPEED = 1.0


class KokoroTTS:
    """Wraps the kokoro-82m-v1.0 model for text-to-speech synthesis.

    Expects the following files to exist in the same directory:
        - kokoro-v1.0.fp16.onnx  (FP16 Kokoro model weights)
        - voice_af_sky.npy       (af_sky voice style embedding, raw float32 array)
    """

    def __init__(self) -> None:
        cell_dir = Path(__file__).parent
        weights_dir = cell_dir / "weights"
        self._kokoro = Kokoro(
            str(weights_dir / "kokoro-v1.0.fp16.onnx"),
            str(weights_dir / "voice_af_sky.npy"),
        )
        # Load the voice array directly so we can pass it as an ndarray,
        # bypassing kokoro-onnx's named-key lookup (which requires .npz format).
        self._voice: np.ndarray = np.load(str(weights_dir / "voice_af_sky.npy"))

    def synthesize(
        self,
        text: str,
        speed: float = _DEFAULT_SPEED,
    ) -> tuple[bytes, int]:
        """Synthesize *text* and return ``(wav_bytes, sample_rate)``.

        The WAV bytes are 16-bit mono PCM, ready to write to a file or stream.
        """
        samples, sample_rate = self._kokoro.create(
            text, voice=self._voice, speed=speed, lang="en-us"
        )
        return self._to_wav(samples, sample_rate), sample_rate

    @staticmethod
    def _to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(sample_rate)
            pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()
