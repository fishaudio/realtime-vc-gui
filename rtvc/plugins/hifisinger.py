from dataclasses import dataclass
from typing import ClassVar

from rtvc.plugins.base import WithSpeaker, slider


@dataclass
class HiFiSingerPlugin(WithSpeaker):
    id: ClassVar[str] = "hifisinger"
