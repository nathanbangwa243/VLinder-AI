"""
 OpenVINO Profiler
 Class for handling runtime representation reports from benchmark application

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


class RuntimeRepresentationReport:
    def __init__(self, path: str = ''):
        self.path_file = path
        try:
            with open(self.path_file) as file:
                self.content = file.read().replace('\n', '')
        except OSError as error:
            raise error
