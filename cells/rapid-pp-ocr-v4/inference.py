# https://github.com/RapidAI/RapidOCR
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image
from rapidocr import RapidOCR


@dataclass
class OCRResult:
    text: str
    confidence: float
    box: list[float]  # flat [x1,y1, x2,y2, x3,y3, x4,y4]


class RapidOCRInference:
    """Wraps RapidOCR using its bundled PP-OCRv4 models.

    No external model files are required — models ship inside the
    rapidocr package and are resolved automatically at runtime.
    """

    def __init__(self) -> None:
        self._engine = RapidOCR()

    def recognize(self, image_bytes: bytes) -> list[OCRResult]:
        """Run OCR on raw image bytes and return detected text regions."""
        try:
            img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Could not decode image: {exc}") from exc

        img_np = np.array(img_pil, dtype=np.uint8)
        ocr_output = self._engine(img_np)

        if ocr_output.boxes is None or ocr_output.txts is None:
            return []

        output: list[OCRResult] = []
        for box_raw, text, confidence in zip(ocr_output.boxes, ocr_output.txts, ocr_output.scores):
            flat_box = [float(coord) for point in box_raw for coord in point]
            output.append(OCRResult(
                text=str(text),
                confidence=float(confidence),
                box=flat_box,
            ))
        return output
