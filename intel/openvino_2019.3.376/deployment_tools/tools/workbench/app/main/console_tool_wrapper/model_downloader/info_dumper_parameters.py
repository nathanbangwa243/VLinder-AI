"""
 OpenVINO Profiler
 Class for cli output of calibration tool

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

import sys

import os

from app.main.jobs.tools_runner.console_parameters import ConsoleToolParameters
from config.constants import MODEL_DOWNLOADER_PATH


class InfoDumperParameters(ConsoleToolParameters):
    def __init__(self, path=MODEL_DOWNLOADER_PATH):
        super(InfoDumperParameters, self).__init__()
        self.path = sys.executable
        self.exe = os.path.join(path, 'info_dumper.py')
        self.set_parameter('all', '')

    def __str__(self, parameter_prefix='--'):
        exe_path = '{} {}'.format(self.path, self.exe)
        params = ' '.join(
            ['{p}{k} {v}'.format(p=parameter_prefix, k=key, v=value) for key, value in self.params.items()])
        return exe_path + ' ' + params
