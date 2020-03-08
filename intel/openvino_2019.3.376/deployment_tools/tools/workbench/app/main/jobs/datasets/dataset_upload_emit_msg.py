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
from app.main.job_factory.config import CeleryDBAdapter
from app.main.jobs.datasets.dataset_upload_config import DatasetUploadConfig
from app.main.jobs.job_types_enum import JobTypesEnum
from app.main.jobs.uploads.upload_emit_msg import UploadEmitMessage
from app.main.models.datasets_model import DatasetsModel


class DatasetUploadEmitMessage(UploadEmitMessage):
    event = 'dataset'

    def __init__(self, job, artifact_id: int, config: DatasetUploadConfig, weight: float, previous_progress: float = 0):
        super().__init__(job, artifact_id, config, weight, previous_progress)

    def full_json(self):
        json_message = super().full_json()
        json_message['fileType'] = 'dataset'
        if self.from_celery:
            session = CeleryDBAdapter.session()
            dataset = session.query(DatasetsModel).get(self.artifact_id)
        else:
            dataset = DatasetsModel.query.get(self.artifact_id)

        json_message['stages'] = [{
            'progress': dataset.uploaded_progress,
            'name': dataset.status.value,
            'stage': JobTypesEnum.iuploader_type.value
        }]
        if self.from_celery:
            session.close()
        return json_message

    def update_extract_progress(self, new_progress):
        self.find_job_by_name('extracting').progress = new_progress
        self.emit_message()

    def update_validate_progress(self, new_progress):
        self.find_job_by_name('validating').progress = new_progress
        self.emit_message()
