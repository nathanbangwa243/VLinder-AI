"""
 OpenVINO Profiler
 Class for storing calibration stages

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

from enum import Enum


class PythonStages(Enum):
    calibration_initialization = 'Calibration initialization'
    model_loading = 'Loading Model'
    collecting_fp32_statistics = 'Collecting fp32 initial statistics'
    int8_conversion = 'Checking accuracy with all layers converted to int8'
    return_back_to_fp32 = 'Returning layers back to fp32'


# percents for stage
CALIBRATION_STAGE_PROGRESS = {
    PythonStages.calibration_initialization: 0.,
    PythonStages.model_loading: 0.,
    PythonStages.collecting_fp32_statistics: 15.,
    PythonStages.int8_conversion: 60.,
    PythonStages.return_back_to_fp32: 25.,
}


class CalibrationStages:
    stages = {
        'int8_tuning': 1,
        'Parsing input': 0.01,
        'Loading  plugin': 0.01,
        'Loading network files': 0.01,
        'Preparing input blobs': 0.01,
        'Collecting accuracy metric in FP32 mode': 0.2,
        'Verification of network accuracy': 0.06,
        'Validate int8 accuracy': 0.01,
        'Accuracy of all layers conversion does not correspond': 0.1,
        'Write calibrated network to IR file': 0.1,
    }

    @classmethod
    def get_stages(cls) -> tuple:
        return tuple(cls.stages.keys())
