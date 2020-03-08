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

from app.main.jobs.uploads.upload_config import UploadConfig


class DatasetUploadConfig(UploadConfig):
    def __init__(self, session_id: str, data: dict):
        super().__init__(session_id, data)
        self.file_type = 'dataset'
        self.name = data['name']
        self.format = data['type']

    def json(self) -> dict:
        json_message = super().json()
        json_message.update({
            'name': self.name,
            'format': self.format,
        })
        return json_message
