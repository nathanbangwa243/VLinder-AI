"""
 OpenVINO Profiler
 Class for ORM model described an Datasets

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

from sqlalchemy import String, Integer, Column, ForeignKey
from sqlalchemy.orm import relationship

from app.main.models.enumerates import MODEL_PRECISION_ENUM_SCHEMA, ModelPrecisionEnum
from app.main.models.jobs_model import JobsModel


class ModelDownloaderModel(JobsModel):
    __tablename__ = 'model_download'

    __mapper_args__ = {
        'polymorphic_identity': __tablename__
    }

    job_id = Column(Integer, ForeignKey('jobs.job_id'), primary_key=True)
    name = Column(String, nullable=False)
    precision = Column(MODEL_PRECISION_ENUM_SCHEMA, nullable=False)
    result_model_id = Column(Integer, ForeignKey('topologies.id'), nullable=True)
    path = Column(String, nullable=True)

    result_model = relationship('TopologiesModel', foreign_keys=[result_model_id])

    def __init__(self, data):
        super().__init__(data)
        self.name = data['name']
        self.precision = ModelPrecisionEnum(data['precision'])
        self.path = data['path']

    def json(self):
        return {
            'name': self.name,
            'precision': self.precision,
            'path': self.path,
            'resultModelId': self.result_model_id,
        }
