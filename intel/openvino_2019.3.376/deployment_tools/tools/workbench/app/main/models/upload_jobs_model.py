"""
 OpenVINO Profiler
 Class for ORM model described Job

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
from sqlalchemy import Column, Integer, ForeignKey
from sqlalchemy.orm import relationship, backref

from app.main.models.jobs_model import JobsModel


class UploadJobsModel(JobsModel):
    __tablename__ = 'upload_jobs'

    __mapper_args__ = {
        'polymorphic_identity': __tablename__
    }

    job_id = Column(Integer, ForeignKey('jobs.job_id'), primary_key=True)
    artifact_id = Column(Integer, ForeignKey('artifacts.id'), nullable=False)

    artifact = relationship('ArtifactsModel',
                            cascade='delete,all',
                            backref=backref('upload_job', lazy='subquery', cascade='delete,all', ),
                            foreign_keys=[artifact_id])

    def __init__(self, data: dict):
        super().__init__(data)
        self.artifact_id = data['artifactId']

    def json(self) -> dict:
        return {
            'artifactId': self.artifact_id,
            'previousJobId': self.parent_job,
            'session_id': self.session_id,
            'date': self.creation_timestamp,
            'status': self.status_to_json()
        }

    def status_to_json(self):
        status = {
            'name': self.status.value,
            'progress': self.progress
        }
        if self.error_message:
            status['errorMessage'] = self.error_message
        return status
