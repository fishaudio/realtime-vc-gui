import os
import queue
import sys
import threading
import time
from io import BytesIO

import librosa
import noisereduce as nr
import numpy as np
import pkg_resources
import qdarktheme
import requests
import sounddevice as sd
import soundfile as sf
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from scipy.signal import convolve

from rtvc.audio import get_devices
from rtvc.config import application_path, config, load_config, save_config
from rtvc.i18n import _t, language_map
from rtvc.plugins import ALL_PLUGINS
from rtvc.plugins.base import render_plugin


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowIcon(QIcon(str(application_path / "assets" / "icon.png")))

        version = pkg_resources.get_distribution("rtvc").version
        # remove +editable if it exists
        version = version.split("+")[0]
        self.setWindowTitle(_t("title").format(version=version))

        self.main_layout = QVBoxLayout()
        # Stick to the top
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.setup_ui_settings()
        self.setup_backend_settings()
        self.setup_device_settings()
        self.setup_audio_settings()
        self.plugin_layout = QGroupBox()
        self.main_layout.addWidget(self.plugin_layout)
        self.setup_plugin_settings()
        self.setup_action_buttons()
        self.setLayout(self.main_layout)

        # Use size hint to set a reasonable size
        self.setMinimumWidth(900)

        # Voice Conversion Thread
        self.thread = None
        self.vc_status = threading.Event()

    def setup_ui_settings(self):
        # we have language and backend settings in the first row
        row = QHBoxLayout()
        row.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # set up a theme combo box
        row.addWidget(QLabel(_t("theme.name")))
        self.theme_combo = QComboBox()
        self.theme_combo.addItem(_t("theme.auto"), "auto")
        self.theme_combo.addItem(_t("theme.light"), "light")
        self.theme_combo.addItem(_t("theme.dark"), "dark")
        self.theme_combo.setCurrentText(_t(f"theme.{config.theme}"))
        self.theme_combo.currentIndexChanged.connect(self.change_theme)
        self.theme_combo.setMinimumWidth(100)
        row.addWidget(self.theme_combo)

        # set up language combo box
        row.addWidget(QLabel(_t("i18n.language")))
        self.language_combo = QComboBox()

        for k, v in language_map.items():
            self.language_combo.addItem(v, k)

        self.language_combo.setCurrentText(language_map.get(config.locale, 'en_US'))
        self.language_combo.currentIndexChanged.connect(self.change_language)
        self.language_combo.setMinimumWidth(150)
        row.addWidget(self.language_combo)

        # setup plugin combo box
        row.addWidget(QLabel(_t("plugins.name")))
        self.plugin_combo = QComboBox()
        self.plugin_combo.addItem(_t("plugins.none.name"), "none")
        for plugin in ALL_PLUGINS:
            self.plugin_combo.addItem(_t(f"plugins.{plugin.id}.name"), plugin.id)

        if config.current_plugin is not None:
            self.plugin_combo.setCurrentText(
                _t(f"plugins.{config.current_plugin}.name")
            )
        else:
            self.plugin_combo.setCurrentText(_t("plugins.none.name"))

        self.plugin_combo.currentIndexChanged.connect(self.change_plugin)
        self.plugin_combo.setMinimumWidth(150)
        row.addWidget(self.plugin_combo)

        # save button
        self.save_button = QPushButton(_t("config.save"))
        self.save_button.clicked.connect(self.save_config)
        row.addWidget(self.save_button)

        # load button
        self.load_button = QPushButton(_t("config.load"))
        self.load_button.clicked.connect(self.load_config)
        row.addWidget(self.load_button)

        self.main_layout.addLayout(row)

    def setup_device_settings(self):
        # second row: a group box for audio device settings
        row = QGroupBox(_t("audio_device.name"))
        row_layout = QGridLayout()
        row_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # fetch devices
        input_devices, output_devices = get_devices()

        # input device
        row_layout.addWidget(QLabel(_t("audio_device.input")), 0, 0)
        self.input_device_combo = QComboBox()
        for device in input_devices:
            self.input_device_combo.addItem(device["name"], device["id"])

        # find the current device from config
        if config.input_device is not None:
            for i in range(self.input_device_combo.count()):
                if self.input_device_combo.itemData(i) == config.input_device:
                    self.input_device_combo.setCurrentIndex(i)
                    break
            else:
                # not found, use default
                self.input_device_combo.setCurrentIndex(0)
                config.input_device = self.input_device_combo.itemData(0)

        self.input_device_combo.setFixedWidth(300)
        row_layout.addWidget(self.input_device_combo, 0, 1)

        # output device
        row_layout.addWidget(QLabel(_t("audio_device.output")), 1, 0)
        self.output_device_combo = QComboBox()
        for device in output_devices:
            self.output_device_combo.addItem(device["name"], device["id"])

        # find the current device from config
        if config.output_device is not None:
            for i in range(self.output_device_combo.count()):
                if self.output_device_combo.itemData(i) == config.output_device:
                    self.output_device_combo.setCurrentIndex(i)
                    break
            else:
                # not found, use default
                self.output_device_combo.setCurrentIndex(0)
                config.output_device = self.output_device_combo.itemData(0)

        self.input_device_combo.setFixedWidth(300)
        row_layout.addWidget(self.output_device_combo, 1, 1)

        row.setLayout(row_layout)

        self.main_layout.addWidget(row)

    def setup_audio_settings(self):
        # third row: a group box for audio settings
        row = QGroupBox(_t("audio.name"))
        row_layout = QGridLayout()

        # db_threshold, pitch_shift
        row_layout.addWidget(QLabel(_t("audio.db_threshold")), 0, 0)
        self.db_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.db_threshold_slider.setMinimum(-60)
        self.db_threshold_slider.setMaximum(0)
        self.db_threshold_slider.setSingleStep(1)
        self.db_threshold_slider.setTickInterval(1)
        self.db_threshold_slider.setValue(config.db_threshold)
        row_layout.addWidget(self.db_threshold_slider, 0, 1)
        self.db_threshold_label = QLabel(f"{config.db_threshold} dB")
        self.db_threshold_label.setFixedWidth(50)
        row_layout.addWidget(self.db_threshold_label, 0, 2)
        self.db_threshold_slider.valueChanged.connect(
            lambda v: self.db_threshold_label.setText(f"{v} dB")
        )

        row_layout.addWidget(QLabel(_t("audio.pitch_shift")), 0, 3)
        self.pitch_shift_slider = QSlider(Qt.Orientation.Horizontal)
        self.pitch_shift_slider.setMinimum(-24)
        self.pitch_shift_slider.setMaximum(24)
        self.pitch_shift_slider.setSingleStep(1)
        self.pitch_shift_slider.setTickInterval(1)
        self.pitch_shift_slider.setValue(config.pitch_shift)
        row_layout.addWidget(self.pitch_shift_slider, 0, 4)
        self.pitch_shift_label = QLabel(f"{config.pitch_shift}")
        self.pitch_shift_label.setFixedWidth(50)
        row_layout.addWidget(self.pitch_shift_label, 0, 5)
        self.pitch_shift_slider.valueChanged.connect(
            lambda v: self.pitch_shift_label.setText(f"{v}")
        )

        # performance related
        # sample_duration, fade_duration
        row_layout.addWidget(QLabel(_t("audio.sample_duration")), 1, 0)
        self.sample_duration_slider = QSlider(Qt.Orientation.Horizontal)
        self.sample_duration_slider.setMinimum(100)
        self.sample_duration_slider.setMaximum(3000)
        self.sample_duration_slider.setSingleStep(100)
        self.sample_duration_slider.setTickInterval(100)
        self.sample_duration_slider.setValue(config.sample_duration)
        row_layout.addWidget(self.sample_duration_slider, 1, 1)
        self.sample_duration_label = QLabel(f"{config.sample_duration / 1000:.1f} s")
        self.sample_duration_label.setFixedWidth(50)
        row_layout.addWidget(self.sample_duration_label, 1, 2)
        self.sample_duration_slider.valueChanged.connect(
            lambda v: self.sample_duration_label.setText(f"{v / 1000:.1f} s")
        )

        row_layout.addWidget(QLabel(_t("audio.fade_duration")), 1, 3)
        self.fade_duration_slider = QSlider(Qt.Orientation.Horizontal)
        self.fade_duration_slider.setMinimum(10)
        self.fade_duration_slider.setMaximum(150)
        self.fade_duration_slider.setSingleStep(10)
        self.fade_duration_slider.setTickInterval(10)
        self.fade_duration_slider.setValue(config.fade_duration)
        row_layout.addWidget(self.fade_duration_slider, 1, 4)
        self.fade_duration_label = QLabel(f"{config.fade_duration / 1000:.2f} s")
        self.fade_duration_label.setFixedWidth(50)
        row_layout.addWidget(self.fade_duration_label, 1, 5)
        self.fade_duration_slider.valueChanged.connect(
            lambda v: self.fade_duration_label.setText(f"{v / 1000:.2f} s")
        )

        # Extra duration, input denoise, output denoise in next row
        row_layout.addWidget(QLabel(_t("audio.extra_duration")), 2, 0)
        self.extra_duration_slider = QSlider(Qt.Orientation.Horizontal)
        self.extra_duration_slider.setMinimum(50)
        self.extra_duration_slider.setMaximum(1000)
        self.extra_duration_slider.setSingleStep(10)
        self.extra_duration_slider.setTickInterval(10)
        self.extra_duration_slider.setValue(config.extra_duration)
        row_layout.addWidget(self.extra_duration_slider, 2, 1)
        self.extra_duration_label = QLabel(f"{config.extra_duration / 1000:.2f} s")
        self.extra_duration_label.setFixedWidth(50)
        row_layout.addWidget(self.extra_duration_label, 2, 2)
        self.extra_duration_slider.valueChanged.connect(
            lambda v: self.extra_duration_label.setText(f"{v / 1000:.2f} s")
        )

        self.input_denoise_checkbox = QCheckBox()
        self.input_denoise_checkbox.setText(_t("audio.input_denoise"))
        self.input_denoise_checkbox.setChecked(config.input_denoise)
        row_layout.addWidget(self.input_denoise_checkbox, 2, 3)

        self.output_denoise_checkbox = QCheckBox()
        self.output_denoise_checkbox.setText(_t("audio.output_denoise"))
        self.output_denoise_checkbox.setChecked(config.output_denoise)
        row_layout.addWidget(self.output_denoise_checkbox, 2, 4)

        row.setLayout(row_layout)
        self.main_layout.addWidget(row)

    def setup_backend_settings(self):
        widget = QGroupBox()
        widget.setTitle(_t("backend.title"))
        row = QHBoxLayout()

        # protocol
        row.addWidget(QLabel(_t("backend.protocol_label")))
        self.backend_protocol = QComboBox()
        self.backend_protocol.setMinimumWidth(75)
        self.backend_protocol.addItems(["v1"])
        self.backend_protocol.setCurrentText("v1")
        row.addWidget(self.backend_protocol)

        # set up backend (url) input, and a test button
        row.addWidget(QLabel(_t("backend.name")))
        self.backend_input = QLineEdit()
        self.backend_input.setText(config.backend)
        row.addWidget(self.backend_input)

        self.test_button = QPushButton(_t("backend.test"))
        self.test_button.clicked.connect(self.test_backend)
        row.addWidget(self.test_button)

        widget.setLayout(row)
        self.main_layout.addWidget(widget)

    def setup_plugin_settings(self):
        plugin_id = config.current_plugin

        if plugin_id is None:
            self.get_plugin_config = lambda: dict()
            self.plugin_key_mapping = dict()
            self.plugin_layout.hide()
            return

        self.plugin_layout.show()
        self.plugin_layout.setTitle(_t(f"plugins.{plugin_id}.name"))

        if self.plugin_layout.layout():
            # remove the old layout
            QWidget().setLayout(self.plugin_layout.layout())

        # Find the plugin class from the config
        for plugin_cls in ALL_PLUGINS:
            if plugin_cls.id != plugin_id:
                continue

            layout, get_value_func, key_mappping = render_plugin(plugin_cls)
            self.get_plugin_config = get_value_func
            self.plugin_key_mapping = key_mappping
            self.plugin_layout.setLayout(layout)

        # resize the window to fit the new layout
        self.resize(self.sizeHint())

    def setup_action_buttons(self):
        row = QWidget()
        row_layout = QHBoxLayout()
        row_layout.addStretch(1)

        self.start_button = QPushButton(_t("action.start"))
        self.start_button.clicked.connect(self.start_conversion)
        row_layout.addWidget(self.start_button)

        self.stop_button = QPushButton(_t("action.stop"))
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_conversion)
        row_layout.addWidget(self.stop_button)

        self.latency_label = QLabel(_t("action.latency").format(latency=0))
        row_layout.addWidget(self.latency_label)

        row.setLayout(row_layout)
        self.main_layout.addWidget(row)

    def change_theme(self, index):
        config.theme = self.theme_combo.itemData(index)

        save_config()
        qdarktheme.setup_theme(config.theme)

    def change_language(self, index):
        config.locale = self.language_combo.itemData(index)
        save_config()

        # pop up a message box to tell user app will restart
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Warning)
        msg_box.setText(_t("i18n.restart_msg"))
        msg_box.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        msg_box.setDefaultButton(QMessageBox.StandardButton.No)
        ret = msg_box.exec()

        if ret == QMessageBox.StandardButton.Yes:
            os.execv(sys.argv[0], sys.argv)

    def change_plugin(self, index):
        config.current_plugin = self.plugin_combo.itemData(index)
        if config.current_plugin == "none":
            config.current_plugin = None

        self.setup_plugin_settings()

    def test_backend(self):
        backend = self.backend_input.text()

        try:
            response = requests.options(backend, timeout=5)
        except:
            response = None

        message_box = QMessageBox()

        if response is not None and response.status_code == 200:
            message_box.setIcon(QMessageBox.Icon.Information)
            message_box.setText(_t("backend.test_succeed"))
            config.backend = backend
            save_config()
        else:
            message_box.setIcon(QMessageBox.Icon.Question)
            message_box.setText(_t("backend.test_failed"))

        message_box.exec()

    def save_config(self, save_to_file=True):
        config.backend = self.backend_input.text()
        config.input_device = self.input_device_combo.currentData()
        config.output_device = self.output_device_combo.currentData()
        config.db_threshold = self.db_threshold_slider.value()
        config.pitch_shift = self.pitch_shift_slider.value()
        config.sample_duration = self.sample_duration_slider.value()
        config.fade_duration = self.fade_duration_slider.value()
        config.extra_duration = self.extra_duration_slider.value()
        config.input_denoise = self.input_denoise_checkbox.isChecked()
        config.output_denoise = self.output_denoise_checkbox.isChecked()
        config.plugins[config.current_plugin] = self.get_plugin_config()

        save_config()

        # pop up a message box to tell user if they want to save the config to a file
        if not save_to_file:
            return

        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Question)
        msg_box.setText(_t("config.save_msg"))
        msg_box.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        msg_box.setDefaultButton(QMessageBox.StandardButton.No)

        ret = msg_box.exec()
        if ret == QMessageBox.StandardButton.No:
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self, _t("config.save_title"), "", "YAML (*.yaml)"
        )

        if not file_name:
            return

        save_config(file_name)

    def load_config(self):
        # pop up a message box to select a config file
        file_name, _ = QFileDialog.getOpenFileName(
            self, _t("config.load_title"), "", "YAML (*.yaml)"
        )

        if not file_name:
            return

        load_config(file_name)
        save_config()

        # pop up a message box to tell user app will restart
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.setText(_t("config.load_msg"))
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()

        os.execv(sys.argv[0], sys.argv)

    def start_conversion(self):
        self.save_config(save_to_file=False)

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        # Create windows and buffers
        self.input_wav = np.zeros(
            (
                config.sample_frames
                + config.fade_frames
                + config.sola_search_frames
                + 2 * config.extra_frames,
            ),
            dtype=np.float32,
        )
        self.sola_buffer = np.zeros(config.fade_frames)
        self.fade_in_window = (
            np.sin(np.pi * np.linspace(0, 0.5, config.fade_frames)) ** 2
        )
        self.fade_out_window = (
            np.sin(np.pi * np.linspace(0.5, 1, config.fade_frames)) ** 2
        )

        self.vc_status.set()
        self.in_queue = queue.Queue()
        self.out_queue = queue.Queue()
        self.vc_thread = threading.Thread(target=self.vc_worker)
        self.bg_thread = threading.Thread(target=self.bg_worker)
        self.vc_thread.start()
        self.bg_thread.start()

    def stop_conversion(self):
        self.vc_status.clear()
        self.vc_thread.join()
        self.bg_thread.join()

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def vc_worker(self):
        with sd.Stream(
            callback=self.audio_callback,
            blocksize=config.sample_frames,
            samplerate=config.sample_rate,
            dtype="float32",
            device=(config.input_device, config.output_device),
        ):
            while self.vc_status.is_set():
                sd.sleep(config.sample_duration)

    def audio_callback(self, indata, outdata, frames, times, status):
        # push to queue
        self.in_queue.put((indata.copy(), outdata.shape[1], time.time()))

        try:
            outdata[:] = self.out_queue.get_nowait()
        except queue.Empty:
            outdata[:] = 0

    def bg_worker(self):
        while self.vc_status.is_set():
            indata, channels, in_time = self.in_queue.get()

            try:
                outdata = self.worker_step(indata)
                self.latency_label.setText(
                    _t("action.latency").format(latency=(time.time() - in_time) * 1000)
                )
            except:
                import traceback

                traceback.print_exc()

                self.vc_status.clear()
                self.latency_label.setText(_t("action.error"))
                outdata = np.zeros((config.sample_frames,), dtype=np.float32)

            self.out_queue.put(outdata.repeat(channels).reshape((-1, channels)))

    def worker_step(self, indata):
        indata = librosa.to_mono(indata.T)

        if config.input_denoise:
            indata = nr.reduce_noise(y=indata, sr=config.sample_rate)

        # db threshold
        if config.db_threshold != -60:
            frame_length = 2048
            hop_length = 1024

            rms = librosa.feature.rms(
                y=indata, frame_length=frame_length, hop_length=hop_length
            )
            rms_db = librosa.amplitude_to_db(rms, ref=1.0)[0] < config.db_threshold

            for i in range(len(rms_db)):
                if rms_db[i]:
                    indata[i * hop_length : (i + 1) * hop_length] = 0

        # Rolling buffer
        self.input_wav[:] = np.concatenate(
            [
                self.input_wav[config.sample_frames :],
                indata,
            ]
        )

        buffer = BytesIO()
        sf.write(buffer, self.input_wav, config.sample_rate, format="wav")
        buffer.seek(0)

        safe_pad_length = (
            config.extra_frames - config.fade_frames
        ) / config.sample_rate - 0.03
        safe_pad_length = max(0, safe_pad_length)

        data = {
            "fSafePrefixPadLength": str(safe_pad_length),
            "fPitchChange": str(config.pitch_shift),
            "sampleRate": str(config.sample_rate),
        }

        # Override plugin settings, and apply key mapping
        if (
            config.current_plugin is not None
            and config.current_plugin in config.plugins
        ):
            for k, v in config.plugins[config.current_plugin].items():
                if k in self.plugin_key_mapping:
                    k = self.plugin_key_mapping[k]

                data[k] = str(v)

        response = requests.post(
            config.backend,
            files={
                "sample": ("audio.wav", buffer, "audio/wav"),
            },
            data=data,
        )

        assert response.status_code == 200, f"Failed to request"

        buffer.close()

        with BytesIO(response.content) as buffer:
            buffer.seek(0)
            infer_wav, _ = librosa.load(buffer, sr=config.sample_rate, mono=True)

        infer_wav = infer_wav[
            -config.sample_frames
            - config.fade_frames
            - config.sola_search_frames
            - config.extra_frames : -config.extra_frames
        ]

        # Sola alignment
        sola_target = infer_wav[None, : config.sola_search_frames + config.fade_frames]
        sola_kernel = np.flip(self.sola_buffer[None])

        cor_nom = convolve(
            sola_target,
            sola_kernel,
            mode="valid",
        )
        cor_den = np.sqrt(
            convolve(
                np.square(sola_target),
                np.ones((1, config.fade_frames)),
                mode="valid",
            )
            + 1e-8
        )
        sola_offset = np.argmax(cor_nom[0] / cor_den[0])

        output_wav = infer_wav[sola_offset : sola_offset + config.sample_frames]
        output_wav[: config.fade_frames] *= self.fade_in_window
        output_wav[: config.fade_frames] += self.sola_buffer * self.fade_out_window

        if sola_offset < config.sola_search_frames:
            self.sola_buffer = infer_wav[
                sola_offset
                + config.sample_frames : sola_offset
                + config.sample_frames
                + config.fade_frames
            ]
        else:
            self.sola_buffer = infer_wav[-config.fade_frames :]

        # Denoise
        if config.output_denoise:
            output_wav = nr.reduce_noise(y=output_wav, sr=config.sample_rate)

        return output_wav
