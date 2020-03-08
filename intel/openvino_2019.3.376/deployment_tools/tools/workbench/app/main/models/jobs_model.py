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
from sqlalchemy import Column, String, Integer, ForeignKey, Float
from sqlalchemy.orm import relationship, backref

from app.main.models.base_model import BaseModel
from app.main.models.enumerates import STATUS_ENUM_SCHEMA, StatusEnum


class JobsModel(BaseModel):
    __tablename__ = 'jobs'

    job_type = Column(String(30))

    __mapper_args__ = {
        'polymorphic_identity': 'job',
        'polymorphic_on': job_type
    }

    job_id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String, nullable=True)

    project_id = Column(Integer, ForeignKey('projects.id'), nullable=True)
    parent_job = Column(Integer, nullable=True)
    session_id = Column(String, nullable=True)

    progress = Column(Float, nullable=False)

    status = Column(STATUS_ENUM_SCHEMA, nullable=False, default=StatusEnum.queued)
    error_message = Column(String, nullable=True)

    project = relationship('ProjectsModel', backref=backref('jobs', lazy='subquery', cascade='delete,all'),
                           foreign_keys=[project_id])

    def __init__(self, data: dict):
        self.project_id = data['projectId'] if 'projectId' in data else None
        self.parent_job = data['previousJobId'] if 'previousJobId' in data else None
        self.progress = 0
        self.session_id = data['session_id']

    def json(self) -> dict:
        if self.project:
            project_json = self.project.json()
            project_json['projectId'] = project_json['id']
            del project_json['id']
        else:
            project_json = {}
        return {
            **project_json,
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
