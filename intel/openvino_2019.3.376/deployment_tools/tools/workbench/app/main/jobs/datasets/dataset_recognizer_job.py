"""
 OpenVINO Profiler
 Dataset recognizer.

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
from app.error.inconsistent_upload_error import UnknownDatasetError
from app.main.job_factory.config import CeleryDBAdapter
from app.main.jobs.datasets.dataset_upload_config import DatasetUploadConfig
from app.main.jobs.datasets.dataset_upload_emit_msg import DatasetUploadEmitMessage
from app.main.jobs.interfaces.iemit_message import IEmitMessageStage
from app.main.jobs.interfaces.ijob import IJob
from app.main.jobs.job_types_enum import JobTypesEnum
from app.main.jobs.utils.yml_templates import ConfigRegistry
from app.main.models.datasets_model import DatasetsModel
from app.main.models.enumerates import StatusEnum
from app.main.models.factory import write_record
from app.main.utils.utils import remove_dir


class DatasetRecognizerJob(IJob):
    db_table = DatasetsModel
    event = 'dataset'

    def __init__(self, job_id: int, config: DatasetUploadConfig, weight: float):
        super().__init__(JobTypesEnum.iuploader_type, DatasetUploadEmitMessage(self, job_id, config, weight))

    def run(self):
        self.emit_message.add_stage(IEmitMessageStage('Recognizing', progress=0), silent=True)
        session = CeleryDBAdapter.session()
        dataset = session.query(DatasetsModel).get(self.emit_message.job_id)
        if dataset.status == StatusEnum.cancelled:
            session.close()
            return
        dataset_type = self.recognize(dataset.path)
        if not dataset_type:
            error_message = 'Unknown dataset type'
            dataset.status = StatusEnum.error
            dataset.error_message = error_message
            write_record(dataset, session)
            self.emit_message.add_error(error_message)
            remove_dir(dataset.path)
            session.close()
            raise UnknownDatasetError(error_message)
        dataset.dataset_type = dataset_type
        write_record(dataset, session)
        session.close()

    @staticmethod
    def recognize(path):
        for dataset_type, dataset_adapter in ConfigRegistry.dataset_adapter_registry.items():
            if dataset_adapter.recognize(path):
                return dataset_type
        return None
