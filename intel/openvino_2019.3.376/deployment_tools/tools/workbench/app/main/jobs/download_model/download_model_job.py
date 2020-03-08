"""
 OpenVINO Profiler
 Class for creation job for downloading tuned model

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
import tarfile

from app.main.job_factory.config import CeleryDBAdapter
from app.main.jobs.interfaces.iconfig import IConfig
from app.main.jobs.interfaces.ijob import IJob
from app.main.jobs.job_types_enum import JobTypesEnum
from app.main.jobs.download_model.download_model_emit_msg import DownloadModelEmitMessage
from app.main.models.download_configs_model import DownloadConfigsModel
from app.main.models.enumerates import StatusEnum
from app.main.models.factory import write_record
from app.main.models.int8_autotune_model import Int8AutotuneJobsModel
from app.main.models.jobs_model import JobsModel
from app.main.models.projects_model import ProjectsModel
from app.main.models.topologies_model import TopologiesModel


def pack_model(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        for file in os.listdir(source_dir):
            _, file_extension = os.path.splitext(file)
            if file_extension in ['.bin', '.xml']:
                tar.add(os.path.join(source_dir, file), arcname=os.path.basename(file))


class DownloadModelJob(IJob):
    db_table = TopologiesModel
    ext = '.tar.gz'

    def __init__(self, job_id: int, content: IConfig, weight: float):
        super(DownloadModelJob, self).__init__(JobTypesEnum.download_model_type,
                                               DownloadModelEmitMessage(self, job_id, content, weight))
        self.model_id = self.find_model_id(self.emit_message.config.job_id)

    def run(self):
        session = CeleryDBAdapter.session()
        config = session.query(DownloadConfigsModel).get(self.emit_message.job_id)
        config.status = StatusEnum.running
        write_record(config, session)
        session.close()
        self.emit_message.update_progress(self.emit_message.stage, 0)
        source_dir = self.find_source_dir(self.model_id)
        if not self.archive_exists(self.model_id)[0]:
            pack_model(DownloadModelJob.archive_path(source_dir, self.model_id), source_dir)
        session = CeleryDBAdapter.session()
        config = session.query(DownloadConfigsModel).get(self.emit_message.job_id)
        config.status = StatusEnum.ready
        config.progress = 100
        write_record(config, session)
        session.close()
        self.emit_message.update_progress(self.emit_message.stage, 100)

    @staticmethod
    def archive_path(source_folder, model_id):
        return os.path.join(source_folder, str(model_id) + DownloadModelJob.ext)

    @staticmethod
    def archive_exists(model_id):
        source_folder = DownloadModelJob.find_source_dir(model_id)
        file_path = DownloadModelJob.archive_path(source_folder, model_id)
        return os.path.isfile(file_path), file_path

    def on_failure(self):
        try:
            source_folder = self.find_source_dir(self.model_id)
            file_path = self.archive_path(source_folder, self.model_id)
            os.remove(file_path)
        except OSError:
            pass

    @staticmethod
    def find_source_dir(topology_id):
        session = CeleryDBAdapter.session()
        topology = session.query(TopologiesModel).get(topology_id)
        path = topology.path
        session.close()
        return path

    @staticmethod
    def find_model_id(job_id):
        session = CeleryDBAdapter.session()
        int8_config = session.query(Int8AutotuneJobsModel).get(job_id)
        if int8_config:
            topology_id = int8_config.result_model_id
        else:
            job = session.query(JobsModel).get(job_id)
            project = session.query(ProjectsModel).get(job.project_id)
            topology = session.query(TopologiesModel).get(project.model_id)
            topology_id = topology.id
        session.close()
        return topology_id
