"""
 OpenVINO Profiler
 Class for creation job for dataset creation

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

from app.main.job_factory.config import CeleryDBAdapter
from app.main.jobs.datasets.dataset_upload_config import DatasetUploadConfig
from app.main.jobs.datasets.dataset_upload_emit_msg import DatasetUploadEmitMessage
from app.main.jobs.interfaces.iemit_message import IEmitMessageStage
from app.main.jobs.interfaces.ijob import IJob
from app.main.jobs.job_types_enum import JobTypesEnum
from app.main.jobs.utils.archive_extractor import Extractor
from app.main.jobs.utils.database_functions import set_status_in_db
from app.main.models.enumerates import StatusEnum
from app.main.models.datasets_model import DatasetsModel
from app.main.models.factory import write_record
from app.main.utils.utils import remove_dir, get_size_of_files
from config.constants import UPLOAD_FOLDER_DATASETS, UPLOADS_FOLDER


class DatasetExtractorJob(IJob):
    db_table = DatasetsModel

    def __init__(self, job_id: int, config: DatasetUploadConfig, weight: float):
        super().__init__(JobTypesEnum.iuploader_type, DatasetUploadEmitMessage(self, job_id, config, weight))

    def run(self):
        self.emit_message.add_stage(IEmitMessageStage('extracting', progress=0), silent=True)
        session = CeleryDBAdapter.session()
        dataset = session.query(DatasetsModel).get(self.emit_message.artifact_id)
        file = dataset.files[0]
        if dataset.status == StatusEnum.cancelled:
            return
        uploaded_archive_path = file.path
        session.close()
        self.unpack(file.name, dataset.id, uploaded_archive_path, UPLOAD_FOLDER_DATASETS)
        session = CeleryDBAdapter.session()
        dataset = session.query(DatasetsModel).get(self.emit_message.job_id)
        dataset.path = os.path.join(UPLOAD_FOLDER_DATASETS, str(dataset.id))
        dataset.size = get_size_of_files(dataset.path)
        dataset.progress = self.emit_message.total_progress
        write_record(dataset, session)
        session.close()
        remove_dir(os.path.join(UPLOADS_FOLDER, str(self.emit_message.job_id)))

    def unpack(self, file_id, name, path, upload_folder):
        os.makedirs(upload_folder, exist_ok=True)
        extractor = Extractor(file_id, name, path, self, upload_folder)
        extractor.extract_archive()

    def on_failure(self):
        session = CeleryDBAdapter.session()
        set_status_in_db(DatasetsModel, self.emit_message.job_id, StatusEnum.error, session)
        session.close()
        remove_dir(os.path.join(UPLOAD_FOLDER_DATASETS, str(self.emit_message.job_id)))
