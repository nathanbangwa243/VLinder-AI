"""
 OpenVINO Profiler
 Class for configuration of model optimizer

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

from app.main.jobs.interfaces.iconfig import IConfig
from app.main.models.model_downloader_conversion_jobs_model import ModelDownloaderConversionJobsModel
from config.constants import MODEL_DOWNLOADS_FOLDER


class TopologyConvertConfig(IConfig):
    def __init__(self, session_id, result_model_id: int, data: ModelDownloaderConversionJobsModel):
        super(TopologyConvertConfig, self).__init__(session_id)
        self.name = data['name']
        self.result_model_id = result_model_id
        self.dir = os.path.join(MODEL_DOWNLOADS_FOLDER, str(result_model_id))
        self.path = data['path']
        self.args = data['args']

    def json(self) -> dict:
        return {
            'name': self.name,
            'dir': self.dir,
            'path': self.path,
            'resultModelId': self.result_model_id,
            'sessionId': self.session_id,
            'args': self.args,
        }
