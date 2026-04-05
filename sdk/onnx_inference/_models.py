from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OCRResult:
    """A single detected text region returned by the OCR service.

    Attributes:
        text: The detected text string.
        confidence: Detection confidence in the range [0, 1].
        box: Flat quadrilateral bounding box as eight floats:
            ``[x1, y1, x2, y2, x3, y3, x4, y4]``.
    """

    text: str
    confidence: float
    box: tuple[float, ...]
