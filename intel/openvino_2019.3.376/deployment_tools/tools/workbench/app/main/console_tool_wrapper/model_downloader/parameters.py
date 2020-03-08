"""
 OpenVINO Profiler
 Class for storing int8 calibration cli params

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

import os
import sys

from app.error.job_error import ModelDownloaderError
from app.main.jobs.model_downloader.model_downloader_config import ModelDownloaderConfig
from app.main.jobs.tools_runner.console_parameters import ConsoleToolParameters
from config.constants import MODEL_DOWNLOADER_PATH


class ModelDownloaderParameters(ConsoleToolParameters):
    def __init__(self, config: ModelDownloaderConfig = None, path=MODEL_DOWNLOADER_PATH):
        super(ModelDownloaderParameters, self).__init__()
        self.path = sys.executable
        self.exe = os.path.join(path, 'downloader.py')
        self.set_parameter('name', config.name)
        self.set_parameter('output_dir', config.output)
        self.set_parameter('progress_format', 'json')

    def __str__(self, parameter_prefix='--'):
        if not self.exe:
            raise ModelDownloaderError('Name of application to launch is not set', None)
        exe_path = '{} {}'.format(self.path, self.exe)
        params = ' '.join(
            ['{p}{k} {v}'.format(p=parameter_prefix, k=key, v=value) for key, value in self.params.items()])
        return exe_path + ' ' + params
