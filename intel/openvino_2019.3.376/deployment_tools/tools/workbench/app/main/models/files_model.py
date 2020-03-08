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

import os

from sqlalchemy import Column, Integer, Float, String, Enum, ForeignKey

from app.extensions_factories.database import get_db
from app.main.models.base_model import BaseModel
from app.main.models.enumerates import StatusEnum
from app.main.models.factory import write_record


class FilesModel(BaseModel):
    __tablename__ = 'files'

    id = Column(Integer, primary_key=True, autoincrement=True)

    artifact_id = Column(Integer, ForeignKey('artifacts.id'), nullable=False)

    name = Column(String, nullable=False)
    path = Column(String, nullable=True)

    size = Column(Float, nullable=True)
    uploaded_blob_size = Column(Float, nullable=True, default=0)

    session_id = Column(String, nullable=True)

    progress = Column(Float, nullable=False, default=0)
    status = Column(Enum(StatusEnum), nullable=False, default=StatusEnum.queued)
    error_message = Column(String, nullable=True)

    def __init__(self, name, artifact_id, size, session_id):
        self.name = name
        self.artifact_id = artifact_id
        self.size = size
        self.session_id = session_id

    def json(self):
        return {
            'id': self.id,
            'name': self.name,
            'size': self.size,
            'date': self.creation_timestamp.timestamp(),
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

    @classmethod
    def create_files(cls, files: dict, artifact_id: int, session_id: str) -> dict:
        result = {}
        for file_name, file_data in files.items():
            file_record = cls(file_data['name'], artifact_id, file_data['size'], session_id)
            write_record(file_record, get_db().session)
            file_record.path = os.path.join(file_record.artifact.path, str(file_record.name))
            write_record(file_record, get_db().session)
            result[file_name] = file_record.id
        return result
