"""
 OpenVINO Profiler
 Class for ORM model described a Model

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
from sqlalchemy import Column, ForeignKey, Integer, Float
from sqlalchemy.orm import relationship

from app.main.models.jobs_model import JobsModel


class TopologyAnalysisJobsModel(JobsModel):
    __tablename__ = 'topology_analysis_jobs'

    __mapper_args__ = {
        'polymorphic_identity': __tablename__
    }

    job_id = Column(Integer, ForeignKey('jobs.job_id'), primary_key=True)
    model_id = Column(Integer, ForeignKey('topologies.id'), nullable=False)
    batch = Column(Integer, default=1, nullable=False)
    g_flops = Column(Float, nullable=True)
    g_iops = Column(Float, nullable=True)
    maximum_memory = Column(Float, nullable=True)
    minimum_memory = Column(Float, nullable=True)
    m_params = Column(Float, nullable=True)
    sparsity = Column(Float, nullable=True)

    topology = relationship('TopologiesModel', back_populates='analysis')

    def __init__(self, data: dict):
        super().__init__(data)
        self.model_id = data['model_id']

    def set_analysis_data(self, analyze_data: dict, batch: int = 1):
        self.batch = batch
        self.g_flops = analyze_data['g_flops']
        self.g_iops = analyze_data['g_iops']
        self.maximum_memory = analyze_data['max_mem']
        self.minimum_memory = analyze_data['min_mem']
        self.m_params = analyze_data['m_params']
        self.sparsity = analyze_data['sparsity']

    def json(self):
        return {
            'm_params': self.m_params,
            'maximum_memory': self.maximum_memory,
            'minimum_memory': self.minimum_memory,
            'g_flops': self.g_flops,
            'sparsity': self.sparsity,
            'g_iops': self.g_iops,
        }
