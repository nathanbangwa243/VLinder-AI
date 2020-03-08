"""
 OpenVINO Profiler
 Class for created dataset validation

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

from accuracy_checker.annotation_converters.format_converter import BaseFormatConverter

from app.error.inconsistent_upload_error import InconsistentDatasetError
from app.main.job_factory.config import CeleryDBAdapter
from app.main.jobs.datasets.dataset_upload_config import DatasetUploadConfig
from app.main.jobs.datasets.dataset_upload_emit_msg import DatasetUploadEmitMessage
from app.main.jobs.interfaces.iemit_message import IEmitMessageStage
from app.main.jobs.interfaces.ijob import IJob
from app.main.jobs.job_types_enum import JobTypesEnum

from app.main.models.enumerates import StatusEnum
from app.main.models.datasets_model import DatasetsModel
from app.main.models.factory import write_record
from app.main.jobs.utils.yml_templates.registry import ConfigRegistry
from app.main.utils.utils import remove_dir


class DatasetValidatorJob(IJob):
    db_table = DatasetsModel

    def __init__(self, job_id: int, config: DatasetUploadConfig, weight: float):
        super().__init__(JobTypesEnum.dataset_validator_type, DatasetUploadEmitMessage(self, job_id, config, weight))

    def run(self):
        """
        Validate dataset.

        Validation has 2 steps:
        1. Checking that the dataset structure matches the expected dataset type.
           Performed during `BaseDatasetAdapter` subclass instantiation.
           If instantiation fails — the dataset is not correct.
        2. Checking that the images match the annotations.
           Performed by Accuracy Checker's `BaseFormatConverter.convert()`.
           Resulting annotation is ignored, only errors are checked.
        """

        self.emit_message.add_stage(IEmitMessageStage('validating', progress=0), silent=True)
        session = CeleryDBAdapter.session()
        dataset = session.query(DatasetsModel).get(self.emit_message.job_id)
        session.close()

        if dataset.status == StatusEnum.cancelled:
            return

        try:
            conversion_params = ConfigRegistry.dataset_adapter_registry[dataset.dataset_type](dataset.path).params
        except InconsistentDatasetError as exception:
            self.emit_message.add_error(exception.message)
            raise

        annotation_converter = BaseFormatConverter.provide(conversion_params['converter'], conversion_params)
        try:
            conversion_result = annotation_converter.convert(
                check_content=True,
                progress_callback=self.emit_message.update_validate_progress,
                progress_interval=100  # The callback is called after every 100 checked images.
            )
        except Exception as exception:
            self.emit_message.add_error(str(exception))
            raise

        if conversion_result.content_check_errors:
            self.emit_message.add_error(conversion_result.content_check_errors)
            raise InconsistentDatasetError(conversion_result.content_check_errors)

        session = CeleryDBAdapter.session()
        dataset.status = StatusEnum.ready
        dataset.progress = 100
        write_record(dataset, session)
        session.close()
        self.emit_message.update_validate_progress(100)

    def on_failure(self):
        session = CeleryDBAdapter.session()
        dataset = session.query(DatasetsModel).get(self.emit_message.job_id)
        dataset.status = StatusEnum.error
        write_record(dataset, session)
        remove_dir(dataset.path)
        session.close()
