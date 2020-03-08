"""
 OpenVINO Profiler
 Class for calibration tool's progress tracking

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
import math

from accuracy_checker.progress_reporters import ProgressReporter
from accuracy_checker.logging import print_info

from app.main.console_tool_wrapper.calibration_tool.stages import CALIBRATION_STAGE_PROGRESS, PythonStages


class CalibrationProgressTracker(ProgressReporter):
    __provider__ = 'wb_calibration_reporter'

    n_main_infers = None
    n_layer_infers = None
    progress = 0
    counter = 0
    batch_size = 1
    current_stage = None

    def __init__(self, dataset_size=None):
        super(CalibrationProgressTracker, self).__init__(dataset_size)
        self.main_infer_step = 0
        self.minor_infer_step = 0
        self.first_infer_step = 0
        self.batches_total = 0

    # pylint: disable=unused-argument
    def update(self, _batch_id=None, batch_size=None):
        current_batch = _batch_id + 1
        progress = 0
        if current_batch % 5 == 0:
            update_step = self.get_update_step()
            self.update_progress(update_step * 5)
        elif current_batch == self.batches_total:
            n_last_batches = current_batch % 5
            update_step = self.get_update_step()
            self.update_progress(update_step * n_last_batches)

        CalibrationProgressTracker.progress += progress

    def get_update_step(self):
        update_step = self.main_infer_step
        if CalibrationProgressTracker.counter == 0:
            update_step = self.first_infer_step
        elif CalibrationProgressTracker.counter > CalibrationProgressTracker.n_main_infers:
            update_step = self.minor_infer_step
        return update_step

    @staticmethod
    def update_progress(progress: float):
        CalibrationProgressTracker.progress += progress
        print_info('[ INT8 python ] Progress: {:.2f}% done'.format(CalibrationProgressTracker.progress))

    def reset(self, dataset_size=None):
        super().reset(dataset_size)
        self.batches_total = int(math.ceil(dataset_size * 1. / CalibrationProgressTracker.batch_size))
        print_info('[ INT8 python ] {} objects total'.format(dataset_size))
        print_info('[ INT8 python ] {} batches total'.format(self.batches_total))
        current_stage = CalibrationProgressTracker.current_stage
        if CalibrationProgressTracker.counter == 0:
            CalibrationProgressTracker.current_stage = PythonStages.collecting_fp32_statistics.value
        elif CalibrationProgressTracker.counter <= CalibrationProgressTracker.n_main_infers \
                and current_stage != PythonStages.int8_conversion:
            CalibrationProgressTracker.current_stage = PythonStages.int8_conversion.value
        elif CalibrationProgressTracker.counter > CalibrationProgressTracker.n_main_infers \
                and current_stage != PythonStages.return_back_to_fp32.value:
            CalibrationProgressTracker.current_stage = PythonStages.return_back_to_fp32.value

        self.main_infer_step = CALIBRATION_STAGE_PROGRESS[PythonStages.int8_conversion]\
                                     / CalibrationProgressTracker.n_main_infers / self.batches_total
        self.minor_infer_step = CALIBRATION_STAGE_PROGRESS[PythonStages.return_back_to_fp32]\
                                      / CalibrationProgressTracker.n_layer_infers / self.batches_total
        self.first_infer_step = CALIBRATION_STAGE_PROGRESS[PythonStages.collecting_fp32_statistics]\
                                      / self.batches_total

    # pylint: disable=unused-argument
    @staticmethod
    def finish(objects_processed=True):
        CalibrationProgressTracker.counter += 1
        print_info('[ INT8 python ] {} inference done'.format(CalibrationProgressTracker.counter))
