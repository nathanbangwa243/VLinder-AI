"""
 OpenVINO Profiler
 Function for getting devices hardware information using IE python API

 Copyright (c) 2018 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

# pylint: disable=import-error,no-name-in-module
from openvino.inference_engine import IECore


def load_available_hardware_info() -> list:
    devices = []
    ie_core = IECore()
    for device in ie_core.available_devices:
        if device not in ('CPU', 'MYRIAD', 'GPU'):
            continue
        info = load_device_info(ie_core, device)
        if device == 'CPU':
            info['OPTIMIZATION_CAPABILITIES'].append('FP16')
        devices.append(info)

    if not devices:
        devices = [
            {
                'DEVICE': 'CPU',
                'DEFAULT_CONFIGURATION': [],
                'FULL_DEVICE_NAME': 'CPU',
                'RANGE_FOR_ASYNC_INFER_REQUESTS': {
                    'MIN': 1,
                    'MAX': 6,
                    'STEP': 1
                },
                'RANGE_FOR_STREAMS': {
                    'MIN': 1,
                    'MAX': 6,
                    'STEP': 1
                },
                'OPTIMIZATION_CAPABILITIES': [
                    'WINOGRAD',
                    'INT8',
                    'FP32',
                    'FP16',
                ]
            }
        ]
    return devices


def load_device_info(ie_core, device) -> dict:
    return {
        'DEVICE': device,
        **load_supported_metrics(ie_core, device),
        'DEFAULT_CONFIGURATION': load_default_configuration(ie_core, device) if device != 'MYRIAD' else []
    }


def load_supported_metrics(ie_core, device) -> dict:
    needed_keys = {
        'AVAILABLE_DEVICES',
        'FULL_DEVICE_NAME',
        'OPTIMIZATION_CAPABILITIES',
        'RANGE_FOR_ASYNC_INFER_REQUESTS',
        'RANGE_FOR_STREAMS'
    }
    supported_metrics = {}
    device_supported_metrics = set(ie_core.get_metric(device, 'SUPPORTED_METRICS')).intersection(needed_keys)
    for metric in device_supported_metrics:
        try:
            metric_val = ie_core.get_metric(device, metric)
            supported_metrics[metric] = perform_param(metric_val, 'RANGE' in metric)
        except AttributeError:
            continue
    return supported_metrics


def load_default_configuration(ie_core, device) -> dict:
    default_configuration = {}
    for cfg in ie_core.get_metric(device, 'SUPPORTED_CONFIG_KEYS'):
        try:
            cfg_val = ie_core.get_config(device, cfg)
            default_configuration[cfg] = perform_param(cfg_val)
        except AttributeError:
            continue
    return default_configuration


def perform_param(metric, is_range=True):
    if isinstance(metric, (list, tuple)):
        res = []
        if is_range:
            metric_range = {
                'MIN': int(metric[0]) or 1,
                'MAX': int(metric[1]) if len(metric) >= 2 else 1,
                'STEP': int(metric[2]) if len(metric) >= 3 else 1
            }
            return metric_range
        for source_val in metric:
            try:
                val = int(source_val)
            except ValueError:
                val = str(source_val)
            res.append(val)
        return res
    if isinstance(metric, dict):
        str_param_repr = ''
        for key, value in metric.items():
            str_param_repr += '{}: {}\n'.format(key, value)
        return str_param_repr
    try:
        return int(metric)
    except ValueError:
        return str(metric)
