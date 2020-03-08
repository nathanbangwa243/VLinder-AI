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
from app.main.jobs.uploads.models.model_upload_config import ModelUploadConfig
from app.main.models.model_optimizer_job_model import ModelOptimizerJobModel


class ModelOptimizerConfig(ModelUploadConfig):
    def __init__(self, session_id, data: ModelOptimizerJobModel):
        super().__init__(session_id, data)
        self.original_topology_id = data['original_topology_id']
        self.result_model_id = data['result_model_id']
        self.mo_args = data['mo_args']

    def json(self) -> dict:
        return {
            'original_topology_id': self.original_topology_id,
            'result_model_id': self.result_model_id,
            'mo_args': self.mo_args,
            'sessionId': self.session_id,
        }
