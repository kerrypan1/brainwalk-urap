from __future__ import annotations

from .internvl import InternVL


class InternVLSmall(InternVL):
    """
    File Summary
    ------------
    Thin subclass of `InternVL` that switches to a smaller checkpoint to reduce
    VRAM usage while keeping the same runtime behavior and API
    """

    name = "intern_s"

    def __init__(self, model_name=None, num_frames=16, max_new_tokens=256):
        # Only override the default model id while inheriting the rest
        super().__init__(
            model_name=model_name or "OpenGVLab/InternVL2-1B",
            num_frames=num_frames,
            max_new_tokens=max_new_tokens,
        )
