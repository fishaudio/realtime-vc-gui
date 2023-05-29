from dataclasses import dataclass
from typing import ClassVar

from rtvc.plugins.base import WithSpeaker, dropdown, slider


@dataclass
class RVCPlugin(WithSpeaker):
    id: ClassVar[str] = "rvc"

    index_ratio: slider(0, 100, 1) = 20
