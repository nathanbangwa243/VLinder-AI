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

from app.error.job_error import ModelOptimizerError
from app.main.jobs.tools_runner.console_parameters import ConsoleToolParameters
from config.constants import MODEL_OPTIMIZER_PATH


class ModelOptimizerParameters(ConsoleToolParameters):
    def __init__(self, mo_args: dict, environment: dict = None):
        super().__init__(environment)
        self.params = {key: value for key, value in mo_args.items() if value not in [True, False, None]}
        self.flags = [key for key, value in mo_args.items() if value is True]
        self.path = sys.executable
        self.exe = os.path.join(MODEL_OPTIMIZER_PATH, 'mo.py')

    def __str__(self, parameter_prefix='--'):
        if not self.exe:
            raise ModelOptimizerError('Name of application to launch is not set', None)
        exe_path = '{} {}'.format(self.path, self.exe)
        args = [
            "{p}{k} '{v}'".format(p=parameter_prefix, k=key, v=str(value).replace("'", r"'\''"))
            for key, value in self.params.items()
        ]
        args += [
            '{p}{f}'.format(p=parameter_prefix, f=flag)
            for flag in self.flags
        ]
        return exe_path + ' ' + ' '.join(args)
