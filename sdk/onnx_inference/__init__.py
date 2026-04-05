from ._client import InferenceClient
from ._exceptions import InferenceError
from ._models import OCRResult

try:
    from ._version import version as __version__
except ImportError:  # package not installed from a built distribution
    __version__ = "0.0.0.dev0"

__all__ = ["InferenceClient", "InferenceError", "OCRResult", "__version__"]
