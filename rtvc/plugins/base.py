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
    QMessageBox,
    QSlider,
    QVBoxLayout,
)

from rtvc.config import config
from rtvc.i18n import _t


def slider(
    minimum: int, maximum: int, step: int = 1, map_key: str | None = None
) -> int:
    namespace = dict(min=minimum, max=maximum, step=step, map_key=map_key)
    return type("Slider", (int,), namespace)


def input(map_key: str | None = None) -> str:
    namespace = dict(map_key=map_key)
    return type("Input", (str,), namespace)


def checkbox(map_key: str | None = None) -> bool:
    namespace = dict(map_key=map_key)
    return type("Checkbox", (bool,), namespace)


def dropdown(options: list[tuple[str, str]], map_key: str | None = None) -> str:
    namespace = dict(options=options, map_key=map_key)
    return type("Dropdown", (str,), namespace)


@dataclass
class WithSpeaker:
    # Backward compatibility
    speaker: input(map_key="sSpeakId") = "0"


def render_plugin(plugin_cls: dataclass) -> QGroupBox:
    layout = QVBoxLayout()

    # Inspect all the fields of the plugin class
    fields = plugin_cls.__dataclass_fields__
    class_id = plugin_cls.id
    _t_key = f"plugins.{class_id}"

    try:
        plugin_config = plugin_cls(**config.plugins.get(class_id, {}))
    except TypeError as e:
        # Popup a message
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setText(_t(f"config.error"))
        msg.setInformativeText(str(e))
        msg.exec()

        plugin_config = plugin_cls()

    get_value_funcs = {}
    key_mappping = {}

    for key, value in fields.items():
        type = value.type.__name__
        if type not in ["Slider", "Input", "Checkbox", "Dropdown"]:
            continue

        if hasattr(value.type, "map_key") and value.type.map_key is not None:
            key_mappping[key] = value.type.map_key

        row = QHBoxLayout()
        row.setAlignment(Qt.AlignmentFlag.AlignLeft)

        if type == "Slider":
            slider = QSlider()
            slider.setOrientation(Qt.Orientation.Horizontal)
            slider.setMinimum(value.type.min)
            slider.setMaximum(value.type.max)
            slider.setSingleStep(value.type.step)
            slider.setTickInterval(value.type.step)
            slider.setValue(getattr(plugin_config, key))
            value_label = QLabel(f"{slider.value()}")
            slider.valueChanged.connect(
                lambda value, value_label=value_label: value_label.setText(str(value))
            )
            get_value_funcs[key] = lambda slider=slider: slider.value()
            row.addWidget(QLabel(_t(f"{_t_key}.{key}.label")))
            row.addWidget(slider)
            slider.setToolTip(_t(f"{_t_key}.{key}.tooltip"))
            row.addWidget(value_label)

        elif type == "Input":
            line_edit = QLineEdit()
            line_edit.setText(getattr(plugin_config, key))
            get_value_funcs[key] = lambda line_edit=line_edit: line_edit.text()
            row.addWidget(QLabel(_t(f"{_t_key}.{key}.label")))
            line_edit.setToolTip(_t(f"{_t_key}.{key}.tooltip"))
            row.addWidget(line_edit)

        elif type == "Checkbox":
            checkbox = QCheckBox()
            checkbox.setChecked(getattr(plugin_config, key))
            get_value_funcs[key] = lambda checkbox=checkbox: checkbox.isChecked()
            row.addWidget(QLabel(_t(f"{_t_key}.{key}.label")))
            checkbox.setToolTip(_t(f"{_t_key}.{key}.tooltip"))
            row.addWidget(checkbox)

        elif type == "Dropdown":
            dropdown = QComboBox()
            dropdown.setMinimumWidth(200)
            dropdown.addItems([item[0] for item in value.type.options])
            for i, item in enumerate(value.type.options):
                if item[1] == getattr(plugin_config, key):
                    dropdown.setCurrentIndex(i)
            get_value_funcs[key] = lambda dropdown=dropdown: dropdown.currentText()
            row.addWidget(QLabel(_t(f"{_t_key}.{key}.label")))
            dropdown.setToolTip(_t(f"{_t_key}.{key}.tooltip"))
            row.addWidget(dropdown)

        layout.addLayout(row)

    return (
        layout,
        lambda: {key: func() for key, func in get_value_funcs.items()},
        key_mappping,
    )
