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
from app.extensions_factories.database import get_db
from app.main.job_factory.config import CeleryDBAdapter
from app.main.jobs.interfaces.iemit_message import IEmitMessage
from app.main.jobs.uploads.upload_config import UploadConfig
from app.main.models.artifacts_model import ArtifactsModel
from app.main.models.factory import write_record


class UploadEmitMessage(IEmitMessage):
    namespace = '/upload'

    def __init__(self, job, artifact_id: int, config: UploadConfig, weight: float, previous_progress: float = 0):
        super().__init__(job, artifact_id, config, weight)
        self.artifact_id = artifact_id
        self.from_celery = False
        self.previous_progress = previous_progress

    def full_json(self):
        if self.from_celery:
            session = CeleryDBAdapter.session()
            artifact = session.query(self.job.db_table).get(self.artifact_id)
        else:
            session = get_db().session
            artifact = ArtifactsModel.query.get(self.artifact_id)
        json_message = artifact.json()
        if self.from_celery:
            session.close()
        json_message.update({
            'creationTimestamp': self.date,
        })
        return json_message

    def update_upload_progress(self, new_progress: float):
        self.find_job_by_name('uploading').progress = new_progress
        self.emit_message()

    @property
    def total_progress(self):
        if self.from_celery:
            session = CeleryDBAdapter.session()
            artifact = session.query(ArtifactsModel).get(self.artifact_id)
        else:
            session = get_db().session
            artifact = ArtifactsModel.query.get(self.artifact_id)
        progress = self.local_progress * self.weight + self.previous_progress
        artifact.progress = progress
        write_record(artifact, session)
        if self.from_celery:
            session.close()
        return progress

    @property
    def local_progress(self):
        return sum([job.progress * job.weight for job in self.jobs])
