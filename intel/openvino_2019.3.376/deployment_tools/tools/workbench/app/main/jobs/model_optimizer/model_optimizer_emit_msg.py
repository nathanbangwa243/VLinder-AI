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

from app.main.forms.model_optimizer import MOForm
from app.main.job_factory.config import CeleryDBAdapter
from app.main.jobs.model_optimizer.model_optimizer_config import ModelOptimizerConfig
from app.main.jobs.uploads.models.model_upload_emit_msg import ModelUploadEmitMessage
from app.main.jobs.utils.utils import get_stages_status
from app.main.models.enumerates import StatusEnum
from app.main.models.factory import write_record
from app.main.models.model_optimizer_job_model import ModelOptimizerJobModel
from app.main.models.model_optimizer_scan_model import ModelOptimizerScanJobsModel
from app.main.models.topologies_model import TopologiesModel
from app.main.console_tool_wrapper.model_optimizer.stages import ModelOptimizerStages


class ModelOptimizerEmitMessage(ModelUploadEmitMessage):
    namespace = '/upload'
    stages = ModelOptimizerStages

    def __init__(self, job, artifact_id: int, config: ModelOptimizerConfig, weight: float):
        super().__init__(job, artifact_id, config, weight)
        self.last_emitted_percent = 0
        self.previous_progress = 0
        self.weight = weight

    def short_json(self):
        return self.full_json()

    def full_json(self):
        session = CeleryDBAdapter.session()
        record = (
            session.query(ModelOptimizerJobModel)
                .filter_by(result_model_id=self.config.result_model_id)
                .order_by(desc(ModelOptimizerJobModel.creation_timestamp))
                .first()
        )
        json_message = record.result_model.short_json()
        json_message['stages'] = get_stages_status(record.job_id, session)
        if record.mo_args:
            json_message['mo'] = {
                'params': MOForm.to_params(json.loads(record.mo_args))
            }
        if record.status == StatusEnum.error:
            json_message['status']['errorMessage'] = record.error_message
            json_message['mo']['errorMessage'] = record.detailed_error_message
        mo_analyzed_job = (
            session.query(ModelOptimizerScanJobsModel)
                .filter_by(topology_id=self.config.original_topology_id)
                .first()
        )
        if mo_analyzed_job and mo_analyzed_job.information:
            json_message['mo']['analyzedParams'] = json.loads(mo_analyzed_job.information)
        session.close()
        return json_message

    def update_percent(self, percent):
        self.get_current_job().progress = percent

        session = CeleryDBAdapter.session()

        mo_job_record = (
            session.query(ModelOptimizerJobModel)
                .filter_by(result_model_id=self.config.result_model_id)
                .order_by(desc(ModelOptimizerJobModel.creation_timestamp))
                .first()
        )
        mo_job_record.progress = self.local_progress
        write_record(mo_job_record, session)

        artifact = session.query(TopologiesModel).get(self.config.result_model_id)
        artifact.progress = self.total_progress
        write_record(artifact, session)

        session.close()

        self.emit_progress()

    @property
    def total_progress(self):
        return self.previous_progress + (self.local_progress * self.weight)
