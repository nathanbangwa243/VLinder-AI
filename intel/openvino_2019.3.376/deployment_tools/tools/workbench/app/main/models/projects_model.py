"""
 OpenVINO Profiler
 Base Class for ORM models

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

from app.main.models.base_model import BaseModel
from app.main.models.enumerates import DevicesEnum, DEVICE_ENUM_SCHEMA, OPTIMIZATION_TYPE_ENUM_SCHEMA, \
    OptimizationTypesEnum


class ProjectsModel(BaseModel):
    __tablename__ = 'projects'

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey('topologies.id'), nullable=False)
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False)
    target = Column(DEVICE_ENUM_SCHEMA, nullable=False)
    optimization_type = Column(OPTIMIZATION_TYPE_ENUM_SCHEMA, nullable=False)

    topology = relationship('TopologiesModel', backref=backref('projects', lazy='subquery', cascade='delete,all'),
                            foreign_keys=[model_id])
    dataset = relationship('DatasetsModel', backref=backref('projects', lazy='subquery', cascade='delete,all'),
                           foreign_keys=[dataset_id])

    def __init__(self, model_id: int, dataset_it: int, target: DevicesEnum,
                 optimization_type: OptimizationTypesEnum):
        self.model_id = model_id
        self.dataset_id = dataset_it
        self.target = target
        self.optimization_type = optimization_type

    def json(self):
        return {
            'id': self.id,
            'modelId': self.model_id,
            'datasetId': self.dataset_id,
            'device': self.target.value,
            'optimizationType': self.optimization_type.value,
        }
