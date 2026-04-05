# https://huggingface.co/BAAI/bge-small-en-v1.5

from __future__ import annotations

import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
import onnxruntime as ort


class BGEEmbedder:
    """Wraps the BGE-small-en-v1.5 ONNX model for text embedding.

    Expects the following files to exist in the same directory:
        - model.onnx
        - tokenizer.json
    """

    def __init__(self) -> None:
        cell_dir = Path(__file__).parent

        weights_dir = cell_dir / "weights"
        self._tokenizer: Tokenizer = Tokenizer.from_file(
            str(weights_dir / "tokenizer.json")
        )

        self._session = ort.InferenceSession(
            str(weights_dir / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )

        # Discover which inputs the exported model actually declares so we
        # don't blindly pass token_type_ids to models that omit it.
        self._input_names: set[str] = {
            inp.name for inp in self._session.get_inputs()
        }

    def embed(self, text: str) -> list[float]:
        """Return a unit-normalised embedding vector for *text*.

        Pipeline:
            1. Tokenize with the Rust tokenizers library.
            2. Run ONNX inference.
            3. CLS-pool the last_hidden_state (index 0 along the sequence axis).
            4. L2-normalise so the vector lies on the unit hypersphere.
        """
        encoding = self._tokenizer.encode(text)

        feeds: dict[str, np.ndarray] = {
            "input_ids": np.array([encoding.ids], dtype=np.int64),
            "attention_mask": np.array([encoding.attention_mask], dtype=np.int64),
        }

        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.array([encoding.type_ids], dtype=np.int64)

        # outputs[0] shape: [batch=1, seq_len, hidden_dim]
        last_hidden_state: np.ndarray = self._session.run(None, feeds)[0]

        # CLS token is always at position 0 in the sequence dimension.
        cls_vec: np.ndarray = last_hidden_state[0, 0, :]

        # L2 normalisation — small epsilon guards against near-zero vectors.
        norm: float = float(np.linalg.norm(cls_vec))
        normalised: np.ndarray = cls_vec / (norm + 1e-8)

        return normalised.tolist()
