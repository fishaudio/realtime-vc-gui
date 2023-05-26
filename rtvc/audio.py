import sys
import time

import sounddevice as sd

from rtvc.config import config


def get_devices(update: bool = True):
    if update:
        sd._terminate()
        sd._initialize()

    devices = sd.query_devices()
    hostapis = sd.query_hostapis()

    for hostapi in hostapis:
        for device_idx in hostapi["devices"]:
            devices[device_idx]["hostapi_name"] = hostapi["name"]

    input_devices = [
        {"id": idx, "name": f"{d['name']} ({d['hostapi_name']})"}
        for idx, d in enumerate(devices)
        if d["max_input_channels"] > 0
    ]

    output_devices = [
        {"id": idx, "name": f"{d['name']} ({d['hostapi_name']})"}
        for idx, d in enumerate(devices)
        if d["max_output_channels"] > 0
    ]

    return input_devices, output_devices
