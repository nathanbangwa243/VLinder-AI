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
import json

from sqlalchemy import desc

from app.main.job_factory.config import CeleryDBAdapter
from app.main.jobs.model_downloader.model_downloader_config import ModelDownloaderConfig
from app.main.jobs.uploads.models.model_upload_emit_msg import ModelUploadEmitMessage
from app.main.jobs.utils.database_functions import set_status_in_db
from app.main.jobs.utils.utils import get_stages_status
from app.main.models.enumerates import StatusEnum
from app.main.models.factory import write_record
from app.main.models.model_downloader_conversion_jobs_model import ModelDownloaderConversionJobsModel
from app.main.models.model_downloader_model import ModelDownloaderModel
from app.main.models.topologies_model import TopologiesModel


class ModelDownloaderEmitMessage(ModelUploadEmitMessage):
    def __init__(self, job, job_id, config: ModelDownloaderConfig, weight):
        super(ModelDownloaderEmitMessage, self).__init__(job, job_id, config, weight)
        self.last_emitted_percent = 0
        self.previous_accumulated_progress = 0
        self.weight = weight
        self.path = None

    def full_json(self):
        session = CeleryDBAdapter.session()
        record = session.query(ModelDownloaderModel).filter_by(job_id=self.job_id).first()
        topology = session.query(TopologiesModel).filter_by(id=record.result_model_id).first()
        conversion_record = (
            session.query(ModelDownloaderConversionJobsModel)
            .filter_by(result_model_id=record.result_model_id)
            .order_by(desc(ModelDownloaderConversionJobsModel.creation_timestamp))
            .first()
        )
        json_message = topology.short_json()
        json_message['stages'] = get_stages_status(record.job_id, session)
        session.close()
        if conversion_record:
            json_message['mo'] = {}
            if conversion_record.conversion_args:
                json_message['mo']['params'] = {}
                json_message['mo']['params']['dataType'] = json.loads(conversion_record.conversion_args)['precision']
        return json_message

    def set_previous_accumulated_progress(self, val):
        self.previous_accumulated_progress = val

    def update_progress(self, name: str, new_value: float):
        self.find_job_by_name(name).progress = new_value
        self.update_progress_in_database()
        self.emit_progress()

    def update_progress_in_database(self):
        job_progress = self.local_progress
        session = CeleryDBAdapter.session()
        record = session.query(ModelDownloaderModel).filter_by(job_id=self.job_id).first()
        record.progress = job_progress
        write_record(record, session)
        session.close()

    def set_error_to_database(self, message):
        session = CeleryDBAdapter.session()
        job_record = session.query(ModelDownloaderModel).get(self.job_id)
        set_status_in_db(ModelDownloaderModel, self.job_id, StatusEnum.error, session, message)
        set_status_in_db(TopologiesModel, job_record.result_model_id, StatusEnum.error, session, message)
        session.close()

    @property
    def total_progress(self):
        session = CeleryDBAdapter.session()
        artifact = session.query(TopologiesModel).get(self.config.result_model_id)
        current_job_record = session.query(ModelDownloaderModel).get(self.job_id)
        total_progress = current_job_record.progress * self.weight
        progress = self.previous_accumulated_progress + total_progress
        artifact.progress = progress if progress < 99 else 99
        progress = artifact.progress
        write_record(artifact, session)
        session.close()
        return progress

    @property
    def local_progress(self):
        return sum([job.progress * job.weight for job in self.jobs])
