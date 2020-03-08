"""
 OpenVINO Profiler
 Data class for accuracy checker's parameters

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
from app.main.jobs.tools_runner.console_parameters import ConsoleToolParameters


class AccuracyCheckerCliParameters(ConsoleToolParameters):

    def __init__(self):
        super(AccuracyCheckerCliParameters, self).__init__()
        self.path = sys.executable

    def __str__(self, parameter_prefix='-'):
        if not self.exe:
            raise AssertionError('Name of application did not set')
        exe_path = self.path + ' ' + self.exe
        params = ' '.join(
            ['{p}{k} {v}'.format(p=parameter_prefix, k=key, v=value) for key, value in self.params.items()])
        return exe_path + ' ' + params
