"""
 OpenVINO Profiler
 Class for dataset creation configuration

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

from app.main.jobs.interfaces.iconfig import IConfig


class MoveModelFromDownloaderConfig(IConfig):
    def __init__(self, session_id, data):
        super().__init__(session_id)
        self.destination_path = data['destinationPath']
        self.source_path = data['sourcePath']
