"""
 OpenVINO Profiler
 Class for job for copy files

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
import shutil

from app.main.job_factory.config import CeleryDBAdapter
from app.main.jobs.move_model_from_downloader.move_model_from_downloader_config import MoveModelFromDownloaderConfig
from app.main.jobs.interfaces.iemit_message import IEmitMessage
from app.main.jobs.interfaces.ijob import IJob
from app.main.jobs.job_types_enum import JobTypesEnum
from app.main.jobs.utils.database_functions import set_status_in_db
from app.main.models.artifacts_model import ArtifactsModel
from app.main.models.enumerates import StatusEnum
from app.main.models.topologies_model import TopologiesModel
from app.main.utils.utils import create_empty_dir
from app.main.utils.utils import remove_dir


class MoveModelFromDownloaderJob(IJob):
    db_table = ArtifactsModel

    def __init__(self, job_id: int, config: MoveModelFromDownloaderConfig, weight: float):
        super().__init__(JobTypesEnum.iuploader_type, IEmitMessage(self, job_id, config, weight))

    def run(self):
        session = CeleryDBAdapter.session()
        precision_str = session.query(TopologiesModel).get(self.emit_message.job_id).precision.value
        session.close()
        create_empty_dir(self.emit_message.config.destination_path)
        src_dir = os.path.join(self.emit_message.config.source_path, precision_str)
        for file_name in os.listdir(src_dir):
            full_file_name = os.path.join(src_dir, file_name)
            if os.path.isfile(full_file_name):
                shutil.move(full_file_name, self.emit_message.config.destination_path)
        remove_dir(src_dir)

    def on_failure(self):
        session = CeleryDBAdapter.session()
        self.emit_message.add_error('')
        set_status_in_db(self.db_table, self.emit_message.job_id, StatusEnum.error, session)
        session.close()
        remove_dir(os.path.join(self.emit_message.config.destination_path, self.emit_message.config.path))
