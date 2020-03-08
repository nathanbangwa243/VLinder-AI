"""
 OpenVINO Profiler
 Class for creation job for dataset generation

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

import logging as log
import os
import numpy as np

import cv2

from app.main.job_factory.config import CeleryDBAdapter
from app.main.jobs.interfaces.iemit_message import IEmitMessageStage
from app.main.jobs.interfaces.ijob import IJob
from app.main.jobs.job_types_enum import JobTypesEnum
from app.main.jobs.datasets.dataset_generator_config import DatasetGeneratorConfig
from app.main.jobs.datasets.dataset_generator_emit_msg import DatasetGeneratorEmitMessage
from app.main.jobs.utils.database_functions import set_status_in_db
from app.main.models.enumerates import StatusEnum
from app.main.models.datasets_model import DatasetsModel
from app.main.models.factory import write_record
from app.main.utils.utils import create_empty_dir, remove_dir, get_size_of_files
from config.constants import UPLOAD_FOLDER_DATASETS


class DistributionLaw:
    def __init__(self, law, params):
        self.law = law
        if law not in ['Gaussian', 'power']:
            raise AssertionError('Unsupported distribution lower: {}'.format(law))
        self.params = params

    def random_generator(self, size):
        if self.law == 'Gaussian':
            return np.random.normal(self.params['loc'], self.params['scale'], size)
        if self.law == 'power':
            return np.random.power(self.params['a'], size)
        return None


class DatasetGeneratorJob(IJob):
    db_table = DatasetsModel
    event = 'dataset'

    def __init__(self, job_id: int, config: DatasetGeneratorConfig, weight: float):
        super().__init__(JobTypesEnum.add_generated_dataset_type,
                         DatasetGeneratorEmitMessage(self, job_id, config, weight))

    def run(self):
        parameters = self.emit_message.config
        dataset_id = self.emit_message.job_id
        current_job = self.emit_message.add_stage(IEmitMessageStage('Setup dataset parameters', weight=0.1))
        session = CeleryDBAdapter.session()
        dataset = session.query(DatasetsModel).get(self.emit_message.job_id)
        dataset.status = StatusEnum.running
        dataset_path = dataset.path
        write_record(dataset, session)
        session.close()
        image_size = parameters.width * parameters.height * parameters.channels
        create_empty_dir(dataset_path)
        try:
            random_generator = DistributionLaw(parameters.dist_law, parameters.params_dist).random_generator
        except AssertionError as exception:
            self.emit_message.add_error(str(exception))
            raise
        self.emit_message.update_progress(current_job, 100)
        current_job = self.emit_message.add_stage(IEmitMessageStage('Generate dataset', weight=0.9))
        log.debug('Starting of generating dataset %s', dataset_id)
        index = 0
        while index < self.emit_message.config.size:
            file_name = os.path.join(dataset_path, '{}.jpg'.format(index))
            cv2.imwrite(file_name, random_generator(image_size).reshape(parameters.height,
                                                                        parameters.width,
                                                                        parameters.channels).astype(np.uint8))
            percent = (index / (parameters.size + 2)) * 100

            if index % np.ceil(parameters.size / 10) == 0:
                self.emit_message.update_progress(current_job, percent)
            with open(os.path.join(dataset_path, parameters.name + '.txt'), 'a') as desc_file:
                desc_file.write('{}.jpg 0\n'.format(index))
            index += 1

        session = CeleryDBAdapter.session()
        dataset = session.query(DatasetsModel).get(self.emit_message.job_id)
        dataset.progress = 100
        dataset.status = StatusEnum.ready
        dataset.size = get_size_of_files(dataset_path)
        write_record(dataset, session)
        session.close()

        self.emit_message.update_progress(current_job, 100)
        log.debug('Finish of generating dataset %s', dataset_id)

    def on_failure(self):
        session = CeleryDBAdapter.session()
        set_status_in_db(DatasetsModel, self.emit_message.job_id, StatusEnum.error, session)
        session.close()
        remove_dir(os.path.join(UPLOAD_FOLDER_DATASETS, self.emit_message.job_id))
