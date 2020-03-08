"""
 OpenVINO Profiler
 Class for ORM model described an Artifacts Job

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

from sqlalchemy import Column, Integer, Float, String
from sqlalchemy.orm import relationship, backref

from app.main.models.base_model import BaseModel
from app.main.models.enumerates import StatusEnum, STATUS_ENUM_SCHEMA

# pylint: disable=unused-import
from app.main.models.files_model import FilesModel


class ArtifactsModel(BaseModel):
    __tablename__ = 'artifacts'

    id = Column(Integer, primary_key=True, autoincrement=True)

    name = Column(String, nullable=False, default='artifact')
    path = Column(String, nullable=True)
    size = Column(Float, nullable=True, default=0.0)
    session_id = Column(String, nullable=True)

    progress = Column(Float, nullable=False, default=0)
    status = Column(STATUS_ENUM_SCHEMA, nullable=False, default=StatusEnum.queued)
    error_message = Column(String, nullable=True)

    task_id = Column(String, nullable=True)

    files = relationship('FilesModel', backref=backref('artifact', lazy='subquery'), cascade='delete,all')

    def __init__(self, name, session_id):
        self.name = name
        self.session_id = session_id

    def json(self):
        return {
            'sessionId': self.session_id,
            'id': self.id,
            'name': self.name,
            'size': self.size,
            'path': self.path,
            'date': self.creation_timestamp.timestamp() * 1000,  # Milliseconds.
            'status': self.status_to_json(),
        }

    def status_to_json(self):
        status = {
            'name': self.status.value,
            'progress': self.progress
        }
        if self.error_message:
            status['errorMessage'] = self.error_message
        return status

    @property
    def uploaded_progress(self):
        file_sizes = sum([f.size for f in self.files])
        file_weights = {f.name: f.size / file_sizes for f in self.files}
        return sum([f.progress * file_weights[f.name] for f in self.files])
