"""
 OpenVINO Profiler
 Accuracy cli tool

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
import json
import numpy as np
from app.main.jobs.utils.yml_abstractions import Model
from app.main.console_tool_wrapper.accuracy_tools.accuracy.accuracy_progress_tracker import AccuracyProgressTracker

try:
    from openvino.tools.accuracy_checker.accuracy_checker.evaluators import ModelEvaluator
    from openvino.tools.accuracy_checker.accuracy_checker.progress_reporters import ProgressReporter
except ImportError:
    from accuracy_checker.evaluators import ModelEvaluator
    from accuracy_checker.progress_reporters import ProgressReporter

PARSER = argparse.ArgumentParser()
PARSER.add_argument('-y',
                    '--yml_config',
                    help='Yaml configuration as a string',
                    type=str,
                    required=False)


class MetricsStatefulCallback:
    def __init__(self):
        self._values = list()
        self._latencies = list()

    def __call__(self, value, latency=None):
        self._values.append(value)
        self._latencies.append(latency)

    @property
    def values(self):
        return self._values

    @property
    def latencies(self):
        return self._latencies


def fetch_accuracy_from_presenter(presenter_values):
    value, _, _, _, _, meta = presenter_values
    accuracy = 0
    if np.isscalar(value):
        accuracy = value
    # In case of vectorized metric with one number 'value'l could be array with zero shape
    elif meta.get('calculate_mean', True) or (isinstance(value, np.ndarray) and not value.shape):
        accuracy = np.mean(value)
    elif value.shape:
        accuracy = value[0]
    return accuracy


def main():
    args = PARSER.parse_args()
    config = Model.from_dict(json.loads(args.yml_config))

    annotation_conversion = config.dataset.annotation.annotation_conversion

    # calibration tool requires Path instances
    for key in annotation_conversion:
        if key.endswith(('dir', 'file')):
            annotation_conversion[key] = Path(annotation_conversion[key])
    config.dataset.data_source = Path(config.dataset.data_source)
    config.launcher.model = Path(config.launcher.model)
    config.launcher.weights = Path(config.launcher.weights)
    config.launcher.cpu_extensions = Path(config.launcher.cpu_extensions)

    model_evaluator = ModelEvaluator.from_configs(config.launcher.to_dict(), config.dataset.to_dict())
    AccuracyProgressTracker.batch_size = config.launcher.batch
    progress_reporter = ProgressReporter.provide((
        'wb_accuracy_reporter'
    ))
    progress_reporter.reset(model_evaluator.dataset.size)
    model_evaluator.process_dataset(None, progress_reporter=progress_reporter)
    #
    model_evaluator_callback = MetricsStatefulCallback()

    model_evaluator.compute_metrics(ignore_results_formatting=False,
                                    output_callback=model_evaluator_callback)

    presenter_values = model_evaluator_callback.values[0]
    accuracy = fetch_accuracy_from_presenter(presenter_values)

    print('[ Accuracy checker] Accuracy: {:.2f}%'.format(accuracy * 100))


if __name__ == '__main__':
    main()
