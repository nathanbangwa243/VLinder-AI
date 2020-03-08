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


class ModelOptimizerScanJobsModel(JobsModel):
    __tablename__ = 'model_optimizer_analysis'

    __mapper_args__ = {
        'polymorphic_identity': __tablename__
    }

    job_id = Column(Integer, ForeignKey('jobs.job_id'), primary_key=True)
    topology_id = Column(Integer, ForeignKey('topologies.id'), nullable=False)
    information = Column(Text, nullable=True)

    topology = relationship('TopologiesModel', foreign_keys=[topology_id])

    def __init__(self, data):
        super().__init__(data)
        self.topology_id = data['topology_id']

    def json(self):
        return {
            'topology_id': self.topology_id,
            'information': json.loads(self.information) if self.information else None,
        }

    def short_json(self):
        json_message = self.json()
        del json_message['topology_id']
        del json_message['information']['intermediate']
        return json_message
