import locale
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class Config:
    theme: Literal["auto", "light", "dark"] = "auto"
    locale: str = locale.getdefaultlocale()[0]
    backend: str = "http://localhost:8080/rtvc"

    input_device: str | None = None
    output_device: str | None = None

    db_threshold: int = -30
    pitch_shift: int = 0
    sample_duration: int = 1000
    fade_duration: int = 80
    extra_duration: int = 50
    input_denoise: bool = False
    output_denoise: bool = False


config_path = Path.home() / ".rtvc" / "config.yaml"
config = Config()


def load_config():
    global config

    if config_path.exists():
        with open(config_path, "r") as f:
            config = Config(**yaml.safe_load(f.read()))

    return config


def save_config():
    if not config_path.parent.exists():
        config_path.parent.mkdir(parents=True)

    with open(config_path, "w") as f:
        yaml.safe_dump(config.__dict__, f)


# Auto load config
load_config()
save_config()
