"""
 OpenVINO Profiler
 Class for dataset generation emit message configuration

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

from app.main.job_factory.config import CeleryDBAdapter
from app.main.jobs.datasets.dataset_generator_config import DatasetGeneratorConfig
from app.main.jobs.datasets.dataset_upload_emit_msg import DatasetUploadEmitMessage
from app.main.models.datasets_model import DatasetsModel


class DatasetGeneratorEmitMessage(DatasetUploadEmitMessage):

    def __init__(self, job, job_id: int, config: DatasetGeneratorConfig, weight: float):
        super().__init__(job, job_id, config, weight)

    def full_json(self):
        session = CeleryDBAdapter.session()
        dataset = session.query(DatasetsModel).get(self.job_id)
        json_message = dataset.json()
        json_message['status']['progress'] = self.total_progress
        session.close()
        return json_message
