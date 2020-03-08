"""
 OpenVINO Profiler
 Interface class for manage console tool's parameters

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

from app.error.general_error import GeneralError


class ConsoleToolParameters:
    def __init__(self, environment: dict = None):
        self.params = {}
        self.path = ''
        self.exe = ''
        self.environment = environment

    def set_parameter(self, name: str, value: str):
        self.params[name] = value

    def get_parameter(self, name: str):
        return self.params.get(name, None)

    def __str__(self, parameter_prefix='-'):
        if not self.exe:
            raise GeneralError('Name of application to launch is not set')
        exe_path = os.path.join(self.path, self.exe)
        params = ' '.join([
            "{p}{k} '{v}'".format(p=parameter_prefix, k=key, v=str(value).replace("'", r"'\''"))
            for key, value in self.params.items()
        ])
        return exe_path + ' ' + params
