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
import json

from sqlalchemy import Integer, Column, ForeignKey, Text
from sqlalchemy.orm import relationship

from app.main.models.jobs_model import JobsModel


class ModelOptimizerJobModel(JobsModel):
    __tablename__ = 'model_optimizer'
    __mapper_args__ = {
        'polymorphic_identity': __tablename__
    }

    job_id = Column(Integer, ForeignKey('jobs.job_id'), primary_key=True)
    original_topology_id = Column(Integer, ForeignKey('topologies.id'), nullable=False)
    result_model_id = Column(Integer, ForeignKey('topologies.id'), nullable=False)
    mo_args = Column(Text, nullable=True)
    detailed_error_message = Column(Text, nullable=True)

    original_topology = relationship('TopologiesModel', foreign_keys=[original_topology_id])
    result_model = relationship('TopologiesModel', foreign_keys=[result_model_id])

    def __init__(self, data):
        super().__init__(data)
        self.original_topology_id = data['original_topology_id']
        self.result_model_id = data['result_model_id']
        if 'mo_args' in data:
            self.mo_args = json.dumps(data['mo_args'])

    def json(self):
        return {
            'original_topology_id': self.original_topology_id,
            'result_model_id': self.result_model_id,
            'mo_args': json.loads(self.mo_args) if self.mo_args else None,
        }
