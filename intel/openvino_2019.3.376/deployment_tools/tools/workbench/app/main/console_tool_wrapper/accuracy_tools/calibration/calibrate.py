"""
 OpenVINO Profiler
 Calibration cli tool

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
# !/usr/bin/env python
import argparse
from pathlib import Path

from app.main.console_tool_wrapper.calibration_tool.stages import PythonStages
from app.main.jobs.utils.yml_abstractions import Model
from app.main.console_tool_wrapper.accuracy_tools.calibration.calibration_progress_tracker \
    import CalibrationProgressTracker

# pylint: disable=import-error,no-name-in-module
from openvino.inference_engine import IENetwork

try:
    from openvino.tools.calibration.calibrator import Calibrator
    from openvino.tools.calibration.calibration_configuration import CalibrationConfiguration
except ImportError:
    from tools.calibration.calibrator import Calibrator
    from tools.calibration.calibration_configuration import CalibrationConfiguration


# pylint: disable=invalid-name, protected-access
parser = argparse.ArgumentParser()
parser.add_argument('-y',
                    '--yml_file',
                    help='Path to yaml configuration',
                    type=str,
                    required=True)
parser.add_argument('-th',
                    '--threshold',
                    help='Threshold for calibration',
                    type=float,
                    required=True)
parser.add_argument('-tp',
                    '--tuned-model-path',
                    help='Folder where to save tuned model',
                    type=str,
                    required=True)


def calculate_activation_steps(config):
    activation_upper_boundary = 100
    activation_lower_boundary = config.threshold_boundary
    activation_step = config.threshold_step

    # number of infers in the main loop of the calibration process
    return (activation_upper_boundary - activation_lower_boundary) / activation_step


def main():
    args = parser.parse_args()
    yml_config_path = args.yml_file
    threshold = args.threshold
    tuned_model_path = args.tuned_model_path

    CalibrationProgressTracker.current_stage = PythonStages.calibration_initialization.value
    config = Model.from_yml(yml_config_path)

    annotation_conversion = config.dataset.annotation.annotation_conversion

    # calibration tool requires Path instances
    for key in annotation_conversion:
        if key.endswith(('dir', 'file')):
            annotation_conversion[key] = Path(annotation_conversion[key])
    config.dataset.data_source = Path(config.dataset.data_source)
    config.launcher.model = Path(config.launcher.model)
    config.launcher.weights = Path(config.launcher.weights)
    config.launcher.cpu_extensions = Path(config.launcher.cpu_extensions)

    net = IENetwork(model=str(config.launcher.model),
                    weights=str(config.launcher.weights))

    calibration_config = CalibrationConfiguration(
        config=config.to_dict(),
        precision='INT8',
        model=str(config.launcher.model),
        weights=str(config.launcher.weights),
        output_model='{}_i8.xml'.format(tuned_model_path),
        output_weights='{}_i8.bin'.format(tuned_model_path),
        device=config.launcher.device,
        batch_size=net.batch_size,
        threshold=threshold,
        benchmark_iterations_count=1,
        ignore_layer_names=None,
        ignore_layer_names_path=None,
        ignore_layer_types=['FullyConnected'],
        ignore_layer_types_path=None,
        tmp_directory='',
        progress='wb_calibration_reporter',
        cpu_extension=str(config.launcher.cpu_extensions),
        gpu_extension=None,
        threshold_boundary=95.0,
        threshold_step=0.5)

    calibrator = Calibrator(calibration_config)

    CalibrationProgressTracker.counter = 0
    CalibrationProgressTracker.progress = 0
    CalibrationProgressTracker.n_main_infers = calculate_activation_steps(calibrator._configuration)
    CalibrationProgressTracker.n_layer_infers = len(calibrator._calibrator.get_quantization_layers())
    CalibrationProgressTracker.batch_size = config.launcher.batch

    network = calibrator.run()

    if network is not None:
        print('[ INT8 python ] Progress: {} done'.format(100))
        print('[ INT8 python ]: SUCCESSFULLY CALIBRATED MODEL')
    else:
        print('[ INT8 python ]: FAILED TO CALIBRATE MODEL WITH GIVEN THRESHOLD')


if __name__ == '__main__':
    main()
