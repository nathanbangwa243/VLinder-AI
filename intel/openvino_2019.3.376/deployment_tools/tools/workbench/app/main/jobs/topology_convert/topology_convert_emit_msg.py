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

from app.main.console_tool_wrapper.model_downloader.converter_stages import TopologyConvertStages
from app.main.jobs.utils.utils import get_stages_status
from app.main.models.model_downloader_conversion_jobs_model import ModelDownloaderConversionJobsModel
from app.main.models.factory import write_record
from app.main.models.topologies_model import TopologiesModel
from app.main.job_factory.config import CeleryDBAdapter
from app.main.jobs.topology_convert.topology_convert_config import TopologyConvertConfig
from app.main.jobs.interfaces.iemit_message import IEmitMessage


class TopologyConvertEmitMessage(IEmitMessage):
    namespace = '/upload'
    event = 'model'
    stages = TopologyConvertStages

    def __init__(self, job, job_id, config: TopologyConvertConfig, weight):
        super(TopologyConvertEmitMessage, self).__init__(job, job_id, config, weight)
        self.last_emitted_percent = 0
        self.previous_accumulated_progress = 0
        self.weight = weight

    def short_json(self):
        return self.full_json()

    def full_json(self):
        session = CeleryDBAdapter.session()

        model = session.query(TopologiesModel).get(self.config.result_model_id)

        record = (
            session.query(ModelDownloaderConversionJobsModel)
                .filter_by(result_model_id=self.config.result_model_id)
                .order_by(desc(ModelDownloaderConversionJobsModel.creation_timestamp))
                .first()
        )

        session.close()

        json_message = model.short_json()

        json_message['stages'] = get_stages_status(record.job_id, session)
        session.close()
        if record.conversion_args:
            json_message = {
                'mo': {
                    'params': {
                        'dataType': json.loads(record.conversion_args)['precision']
                    }
                }
            }

        return json_message

    def update_percent(self, percent):
        current_stage = self.get_current_job()
        current_stage.progress = percent
        self.update_progress_for_optimize()
        self.emit_progress()

    def set_previous_accumulated_progress(self, val):
        self.previous_accumulated_progress = val

    def update_progress_for_optimize(self):
        job_progress = self.local_progress
        session = CeleryDBAdapter.session()
        record = session.query(ModelDownloaderConversionJobsModel).filter_by(job_id=self.job_id).first()
        record.progress = job_progress
        write_record(record, session)

    @property
    def total_progress(self):
        session = CeleryDBAdapter.session()
        artifact = session.query(TopologiesModel).get(self.config.result_model_id)
        current_job_record = session.query(ModelDownloaderConversionJobsModel).get(self.job_id)
        total_progress = current_job_record.progress * self.weight
        artifact.progress = self.previous_accumulated_progress + total_progress
        progress = artifact.progress
        write_record(artifact, session)
        session.close()
        return progress
