"""Optional adapter boundary for MuJoCo RGB rollout rendering."""

from __future__ import annotations


def render_rollouts() -> None:
    """Explain why rendering is not implemented inside the CV package."""
    raise NotImplementedError(
        "Headless MuJoCo rendering should be implemented as an optional walker "
        "adapter so this CV repository remains decoupled from RL internals."
    )
