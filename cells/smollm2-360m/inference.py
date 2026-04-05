# https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
import onnxruntime as ort


class SmolLMInference:
    """Wraps SmolLM2-360M-Instruct for autoregressive generation using pure onnxruntime.

    Expects the following files in the same directory as this module:
        - model_q4f16.onnx      (unified model — handles prefill and decode via KV cache)
        - tokenizer.json
        - tokenizer_config.json (to discover eos_token)

    The model uses float16 KV cache tensors with shape [batch, heads, past_seq_len, head_dim].
    On prefill the past caches are empty (past_seq_len=0). Each decode step feeds the
    previous present.* outputs back in as past_key_values.* inputs.

    Prompts are automatically wrapped in the ChatML instruct template expected by
    SmolLM2-360M-Instruct before tokenization.
    """

    _DEFAULT_MAX_NEW_TOKENS: int = 256

    def __init__(self) -> None:
        cell_dir = Path(__file__).parent
        providers = ["CPUExecutionProvider"]

        weights_dir = cell_dir / "weights"
        self._tokenizer = Tokenizer.from_file(str(weights_dir / "tokenizer.json"))
        self._tokenizer.no_padding()
        self._tokenizer.no_truncation()

        self._session = ort.InferenceSession(
            str(weights_dir / "model_q4f16.onnx"), providers=providers
        )

        all_inputs = self._session.get_inputs()
        self._kv_input_names: list[str] = [
            i.name for i in all_inputs if i.name.startswith("past_key_values.")
        ]
        # KV tensor shape: [batch, heads, past_seq_len, head_dim] — read from first KV input.
        kv_shape = next(i for i in all_inputs if i.name.startswith("past_key_values.")).shape
        self._kv_heads: int = kv_shape[1]
        self._kv_head_dim: int = kv_shape[3]

        # present.N.{key,value} outputs → past_key_values.N.{key,value} inputs for next step.
        self._present_to_past: dict[str, str] = {
            o.name: o.name.replace("present.", "past_key_values.", 1)
            for o in self._session.get_outputs()
            if o.name.startswith("present.")
        }

        # Resolve EOS token ID.
        self._eos_id: int | None = None
        cfg_path = weights_dir / "tokenizer_config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            raw = cfg.get("eos_token", "")
            eos_token: str = raw if isinstance(raw, str) else raw.get("content", "")
            if eos_token:
                self._eos_id = self._tokenizer.token_to_id(eos_token)
        if self._eos_id is None:
            for candidate in ("<|endoftext|>", "</s>", "<eos>"):
                tid = self._tokenizer.token_to_id(candidate)
                if tid is not None:
                    self._eos_id = tid
                    break

    def _empty_kv(self) -> dict[str, np.ndarray]:
        """Return empty (past_seq_len=0) KV cache for the first model call."""
        empty = np.zeros((1, self._kv_heads, 0, self._kv_head_dim), dtype=np.float16)
        return {name: empty for name in self._kv_input_names}

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 0,
        temperature: float = 0.0,
    ) -> str:
        """Return generated text for *prompt*.

        Args:
            prompt: User message text. The ChatML instruct template is applied automatically.
            max_new_tokens: Maximum tokens to generate. 0 uses the server default (256).
            temperature: Sampling temperature. 0.0 (default) → greedy decoding.
        """
        effective_max = max_new_tokens if max_new_tokens > 0 else self._DEFAULT_MAX_NEW_TOKENS

        formatted = (
            "<|im_start|>system\n"
            "You are a helpful AI assistant.\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{prompt}\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        encoding = self._tokenizer.encode(formatted)
        prompt_ids = np.array([encoding.ids], dtype=np.int64)
        prompt_len = prompt_ids.shape[1]

        # -- Prefill (empty KV cache, full prompt) --
        outs = self._session.run(None, {
            "input_ids": prompt_ids,
            "attention_mask": np.ones((1, prompt_len), dtype=np.int64),
            "position_ids": np.arange(prompt_len, dtype=np.int64).reshape(1, -1),
            **self._empty_kv(),
        })
        out_map = {o.name: v for o, v in zip(self._session.get_outputs(), outs)}

        next_token = self._pick_token(out_map["logits"][0, -1, :], temperature)
        past_kv: dict[str, np.ndarray] = {
            self._present_to_past[k]: v
            for k, v in out_map.items() if k in self._present_to_past
        }

        generated: list[int] = []

        # -- Decode loop (one token at a time, growing KV cache) --
        for step in range(effective_max):
            if next_token == self._eos_id:
                break
            generated.append(next_token)

            cur_pos = prompt_len + step
            outs = self._session.run(None, {
                "input_ids": np.array([[next_token]], dtype=np.int64),
                "attention_mask": np.ones((1, cur_pos + 1), dtype=np.int64),
                "position_ids": np.array([[cur_pos]], dtype=np.int64),
                **past_kv,
            })
            out_map = {o.name: v for o, v in zip(self._session.get_outputs(), outs)}

            next_token = self._pick_token(out_map["logits"][0, -1, :], temperature)
            past_kv = {
                self._present_to_past[k]: v
                for k, v in out_map.items() if k in self._present_to_past
            }

        return self._tokenizer.decode(generated)

    @staticmethod
    def _pick_token(logits: np.ndarray, temperature: float) -> int:
        if temperature <= 0.0:
            return int(np.argmax(logits))
        shifted = logits / temperature
        shifted -= shifted.max()
        probs = np.exp(shifted)
        probs /= probs.sum()
        return int(np.random.choice(len(probs), p=probs))
