import locale
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    locale: str = locale.getdefaultlocale()[0]


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
