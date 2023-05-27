from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSlider,
    QVBoxLayout,
)

from rtvc.config import config
from rtvc.i18n import _t


def slider(minimum: int, maximum: int, step: int = 1) -> int:
    namespace = dict(min=minimum, max=maximum, step=step)
    return type("Slider", (int,), namespace)


def input() -> str:
    return type("Input", (str,), {})


def checkbox() -> bool:
    return type("Checkbox", (bool,), {})


def dropdown(options: list[tuple[str, str]]) -> str:
    return type("Dropdown", (str,), dict(options=options))


@dataclass
class WithSpeaker:
    speaker: input() = "0"


def render_plugin(plugin_cls: dataclass) -> QGroupBox:
    box = QGroupBox()
    layout = QVBoxLayout()

    # Inspect all the fields of the plugin class
    fields = plugin_cls.__dataclass_fields__
    class_id = plugin_cls.id
    _t_key = f"plugins.{class_id}"
    box.setTitle(_t(f"{_t_key}.title"))
    plugin_config = plugin_cls(**config.plugins.get(class_id, {}))

    get_value_funcs = {}

    for key, value in fields.items():
        type = value.type.__name__
        if type not in ["Slider", "Input", "Checkbox", "Dropdown"]:
            continue

        row = QHBoxLayout()
        row.setAlignment(Qt.AlignmentFlag.AlignLeft)

        if type == "Slider":
            slider = QSlider()
            slider.setOrientation(Qt.Orientation.Horizontal)
            slider.setMinimum(value.type.min)
            slider.setMaximum(value.type.max)
            slider.setSingleStep(value.type.step)
            slider.setTickInterval(value.type.step)
            value_label = QLabel(f"{slider.value()}")
            slider.setValue(getattr(plugin_config, key))
            slider.valueChanged.connect(lambda value: value_label.setText(str(value)))
            get_value_funcs[key] = lambda: slider.value()
            row.addWidget(QLabel(_t(f"{_t_key}.{key}.label")))
            row.addWidget(slider)
            row.addWidget(value_label)

        elif type == "Input":
            line_edit = QLineEdit()
            line_edit.setText(getattr(plugin_config, key))
            get_value_funcs[key] = lambda: line_edit.text()
            row.addWidget(QLabel(_t(f"{_t_key}.{key}.label")))
            row.addWidget(line_edit)

        elif type == "Checkbox":
            checkbox = QCheckBox()
            checkbox.setChecked(getattr(plugin_config, key))
            get_value_funcs[key] = lambda: checkbox.isChecked()
            row.addWidget(QLabel(_t(f"{_t_key}.{key}.label")))
            row.addWidget(checkbox)

        elif type == "Dropdown":
            dropdown = QComboBox()
            dropdown.setMinimumWidth(200)
            dropdown.addItems([item[0] for item in value.type.options])
            for i, item in enumerate(value.type.options):
                if item[1] == getattr(plugin_config, key):
                    dropdown.setCurrentIndex(i)
            get_value_funcs[key] = lambda: dropdown.currentText()
            row.addWidget(QLabel(_t(f"{_t_key}.{key}.label")))
            row.addWidget(dropdown)

        layout.addLayout(row)
    box.setLayout(layout)

    return box
