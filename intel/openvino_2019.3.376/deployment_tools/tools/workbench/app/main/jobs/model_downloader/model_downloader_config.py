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
import os

from app.main.jobs.uploads.models.model_upload_config import ModelUploadConfig
from config.constants import MODEL_DOWNLOADS_FOLDER


class ModelDownloaderConfig(ModelUploadConfig):
    def __init__(self, session_id, data: dict):
        super().__init__(session_id, data)
        self.result_model_id = data['resultModelId']
        self.output = os.path.join(MODEL_DOWNLOADS_FOLDER, str(self.result_model_id))
        self.name = data['name']

    def json(self) -> dict:
        return {
            'name': self.name,
            'output': self.output,
            'sessionId': self.session_id,
        }
