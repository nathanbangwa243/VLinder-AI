"""
 OpenVINO Profiler
 Progress tracker for accuracy checker tool

 Copyright (c) 2019 Intel Corporation

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

try:
    from openvino.tools.accuracy_checker.accuracy_checker.progress_reporters import ProgressReporter
except ImportError:
    from accuracy_checker.progress_reporters import ProgressReporter


class AccuracyProgressTracker(ProgressReporter):
    __provider__ = 'wb_accuracy_reporter'

    update_callback = None
    progress = 0
    batch_size = 1

    def __init__(self, dataset_size=None):
        super(AccuracyProgressTracker, self).__init__(dataset_size)
        self.update_step = 0
        self.batches_total = 0

    # pylint: disable=unused-argument
    def update(self, _batch_id=None, batch_size=None):
        current_batch = _batch_id + 1
        if current_batch % 5 == 0:
            self.update_progress(self.update_step * 5)
        elif current_batch == self.batches_total:
            n_last_batches = current_batch % 5
            self.update_progress(self.update_step * n_last_batches)

    @staticmethod
    def update_progress(progress: float):
        AccuracyProgressTracker.progress += progress
        print('[ Accuracy checker ] Progress: {:.2f}% done'.format(AccuracyProgressTracker.progress))

    def reset(self, dataset_size=None):
        super().reset(dataset_size)
        self.batches_total = int(math.ceil(dataset_size * 1. / AccuracyProgressTracker.batch_size))
        print('[ Accuracy checker ] {} objects total'.format(dataset_size))
        print('[ Accuracy checker ] {} batches total'.format(self.batches_total))

        self.update_step = 100 / self.batches_total

    # pylint: disable=no-self-use
    def finish(self, objects_processed=True):
        print('[ Accuracy checker ] measurement done')
