"""Optional deep-video baseline entry points."""

from __future__ import annotations


def deep_video_available() -> bool:
    """Return whether torch/torchvision are importable."""
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True
