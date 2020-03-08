"""
 OpenVINO Profiler
 Utilities function

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
from config.constants import ROOT_FOLDER


def get_version():
    version_txt_path = os.path.join(ROOT_FOLDER, 'version.txt')
    if not os.path.isfile(version_txt_path):
        return 'develop'
    with open(version_txt_path, 'r') as file_descr:
        lines = file_descr.readlines()
    return lines[0]
