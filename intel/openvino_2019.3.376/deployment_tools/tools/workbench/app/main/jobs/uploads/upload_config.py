"""
 OpenVINO Profiler
 Class for configuration of model creation

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


class UploadConfig(IConfig):
    def __init__(self, session_id: str, data: dict):
        previous_job_id = data['previous_job_id'] if 'previous_job_id' in data else None
        super().__init__(session_id, previous_job_id)
        self.file_type = None

    def json(self) -> dict:
        return {
            'sessionId': self.session_id,
            'fileType': self.file_type,
        }
