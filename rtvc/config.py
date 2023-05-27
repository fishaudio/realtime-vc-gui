import locale
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import sys
import yaml


if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app 
    # path into variable _MEIPASS'.
    application_path = Path(sys._MEIPASS)
else:
    application_path = Path(__file__).parent

@dataclass
class Config:
    theme: Literal["auto", "light", "dark"] = "auto"
    locale: str = locale.getdefaultlocale()[0]
    backend: str = "http://localhost:6844/voiceChangeModel"

    input_device: str | None = None
    output_device: str | None = None

    db_threshold: int = -30
    pitch_shift: int = 0
    sample_duration: int = 1000
    fade_duration: int = 80
    extra_duration: int = 50
    input_denoise: bool = False
    output_denoise: bool = False
    sample_rate: int = 44100
    sola_search_duration: int = 12
    buffer_num: int = 4

    @property
    def sample_frames(self):
        return self.sample_duration * self.sample_rate // 1000

    @property
    def fade_frames(self):
        return self.fade_duration * self.sample_rate // 1000

    @property
    def extra_frames(self):
        return self.extra_duration * self.sample_rate // 1000

    @property
    def sola_search_frames(self):
        return self.sola_search_duration * self.sample_rate // 1000


config_path = Path.home() / ".rtvc" / "config.yaml"
config = Config()


def load_config():
    global config

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = Config(**yaml.safe_load(f.read()))

    return config


def save_config():
    if not config_path.parent.exists():
        config_path.parent.mkdir(parents=True)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.__dict__, f)


# Auto load config
load_config()
save_config()
