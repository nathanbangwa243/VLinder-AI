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

from app.main.job_factory.config import CeleryDBAdapter
from app.main.jobs.model_optimizer_scan.model_optimizer_scan_config import ModelOptimizerScanConfig
from app.main.jobs.uploads.models.model_upload_emit_msg import ModelUploadEmitMessage
from app.main.jobs.utils.utils import get_stages_status
from app.main.models.enumerates import StatusEnum
from app.main.models.factory import write_record
from app.main.models.model_optimizer_scan_model import ModelOptimizerScanJobsModel
from app.main.models.topologies_model import TopologiesModel


class ModelOptimizerScanEmitMessage(ModelUploadEmitMessage):
    stages = {'Initialisation': 1}

    def __init__(self, job, artifact_id: int, config: ModelOptimizerScanConfig, weight: float):
        super().__init__(job, artifact_id, config, weight)

    def short_json(self):
        json_data = self.full_json()
        return json_data

    def full_json(self):
        session = CeleryDBAdapter.session()
        record = session.query(ModelOptimizerScanJobsModel).filter_by(topology_id=self.config.topology_id).first()
        topology = session.query(TopologiesModel).get(self.artifact_id)
        json_message = topology.short_json()
        if record.information:
            params = json.loads(record.information)
            json_message['mo'] = {
                'analyzedParams': params
            }
        if record.status == StatusEnum.error:
            json_message['status']['errorMessage'] = record.error_message
            if 'mo' not in json_message:
                json_message['mo'] = {}
            json_message['mo']['errorMessage'] = record.error_message
        json_message['stages'] = get_stages_status(record.job_id, session)
        session.close()
        return json_message

    def update_percent(self, percent):
        self.get_current_job().progress = percent

        session = CeleryDBAdapter.session()

        mo_job_record = (
            session.query(ModelOptimizerScanJobsModel)
                .filter_by(topology_id=self.config.topology_id)
                .first()
        )
        mo_job_record.progress = self.local_progress
        write_record(mo_job_record, session)

        artifact = session.query(TopologiesModel).get(self.config.topology_id)
        artifact.progress = self.total_progress
        write_record(artifact, session)

        session.close()

        self.emit_progress()

    def update_model_optimizer_scan_result(self, results: str):
        session = CeleryDBAdapter.session()
        mo_job_record = (
            session.query(ModelOptimizerScanJobsModel)
                .filter_by(topology_id=self.config.topology_id)
                .first()
        )
        mo_job_record.information = self.cleanup_results(results)
        mo_job_record.progress = 100
        mo_job_record.status = StatusEnum.ready
        write_record(mo_job_record, session)
        session.close()

    @staticmethod
    def cleanup_results(mo_scan_results: str) -> str:
        res = json.loads(mo_scan_results)
        res['intermediate'] = tuple(res['intermediate'].keys())
        return json.dumps(res)

    @property
    def local_progress(self):
        return sum([job.progress * self.stages[job.job_type] for job in self.jobs])
