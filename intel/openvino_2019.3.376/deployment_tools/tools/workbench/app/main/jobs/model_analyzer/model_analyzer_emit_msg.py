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
from app.main.job_factory.config import CeleryDBAdapter
from app.main.jobs.uploads.models.model_upload_emit_msg import ModelUploadEmitMessage
from app.main.jobs.utils.utils import get_stages_status
from app.main.models.factory import write_record
from app.main.models.topologies_model import TopologiesModel
from app.main.models.topology_analysis_jobs_model import TopologyAnalysisJobsModel


class ModelAnalyzerEmitMessage(ModelUploadEmitMessage):
    def __init__(self, job, artifact_id, config, weight: float, previous_progress: float = 0):
        super().__init__(job, artifact_id, config, weight)
        self.previous_progress = previous_progress

    def full_json(self):
        session = CeleryDBAdapter.session()
        model = session.query(TopologiesModel).get(self.job_id)
        model_analysis = session.query(TopologyAnalysisJobsModel).filter_by(model_id=self.job_id).first()
        json_message = model.short_json()
        json_message['stages'] = get_stages_status(model_analysis.job_id, session)
        session.close()
        return json_message

    def update_analyze_progress(self, new_progress):
        self.find_job_by_name('analyzing').progress = new_progress
        self.emit_message()

    @property
    def total_progress(self):
        session = CeleryDBAdapter.session()
        artifact = session.query(TopologiesModel).get(self.job_id)
        progress = self.local_progress * self.weight + self.previous_progress
        artifact.progress = progress
        write_record(artifact, session)
        session.close()
        return progress

    @property
    def local_progress(self):
        return sum([job.progress for job in self.jobs])