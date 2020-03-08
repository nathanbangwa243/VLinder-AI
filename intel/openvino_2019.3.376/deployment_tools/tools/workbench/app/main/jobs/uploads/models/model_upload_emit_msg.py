"""
 OpenVINO Profiler
 Class for uploading file emit message

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

from app.main.jobs.job_types_enum import JobTypesEnum
from app.main.jobs.uploads.models.model_upload_config import ModelUploadConfig
from app.main.jobs.uploads.upload_emit_msg import UploadEmitMessage
from app.main.models.topologies_model import TopologiesModel


class ModelUploadEmitMessage(UploadEmitMessage):
    event = 'model'

    def __init__(self, job, artifact_id: int, config: ModelUploadConfig, weight: float, previous_progress: float = 0):
        super().__init__(job, artifact_id, config, weight, previous_progress)

    def full_json(self):
        json_message = super().full_json()
        topology = TopologiesModel.query.get(self.job_id)
        if topology.converted_from:
            topology = topology.converted_from_record
        json_message['stages'] = [{
            'progress': topology.uploaded_progress,
            'name': topology.status.value,
            'stage': JobTypesEnum.iuploader_type.value
        }]
        return json_message
