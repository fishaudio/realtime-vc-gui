from dataclasses import dataclass
from typing import ClassVar

from rtvc.plugins.base import WithSpeaker, dropdown, slider


@dataclass
class DiffusionPlugin(WithSpeaker):
    id: ClassVar[str] = "diffusion"

    sample_method: dropdown(
        [
            ("None", "none"),
            ("PLMS", "plms"),
        ]
    ) = "none"
    sample_interval: slider(1, 100, 5) = 20
    skip_steps: slider(0, 1000, 10) = 0
