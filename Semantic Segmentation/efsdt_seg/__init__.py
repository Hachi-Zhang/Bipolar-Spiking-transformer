"""EFSDT semantic-segmentation plugin for MMSegmentation."""
from importlib.util import find_spec
from .models import SegmentationAttention, SignedSSAQuantizer
__version__ = "0.1.0"

def register_all_modules() -> None:
    """Import all registry-backed components."""
    if find_spec("mmengine") is None or find_spec("mmseg") is None:
        raise ImportError("Install MMEngine and MMSegmentation to register the plugin")
    from .models import qfpn, qfpn_head, sdtv3  # noqa: F401

if find_spec("mmengine") is not None and find_spec("mmseg") is not None:
    register_all_modules()

__all__ = ["SegmentationAttention", "SignedSSAQuantizer", "register_all_modules"]
