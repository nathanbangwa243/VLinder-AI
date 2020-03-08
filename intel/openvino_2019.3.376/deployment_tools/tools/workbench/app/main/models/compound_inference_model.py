"""
 OpenVINO Profiler
 Class for ORM model described an Infer Job

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
import math
from sqlalchemy import Column, Integer, ForeignKey, Float
from sqlalchemy.orm import relationship

from app.main.models.jobs_model import JobsModel


class CompoundInferenceJobsModel(JobsModel):
    __tablename__ = 'compound_inferences_jobs'

    __mapper_args__ = {
        'polymorphic_identity': __tablename__
    }

    job_id = Column(Integer, ForeignKey('jobs.job_id'), primary_key=True)

    min_batch = Column(Integer, nullable=False)
    max_batch = Column(Integer, nullable=False)
    step_batch = Column(Integer, nullable=False)
    min_nireq = Column(Integer, nullable=False)
    max_nireq = Column(Integer, nullable=False)
    step_nireq = Column(Integer, nullable=False)
    inference_time = Column(Float, nullable=False)

    inference_results = relationship('InferenceResultsModel', backref='compound_inference_job')

    def __init__(self, data):
        super().__init__(data)
        self.min_batch = data['minBatch']
        self.max_batch = data['maxBatch']
        self.step_batch = data['stepBatch']
        self.min_nireq = data['minNireq']
        self.max_nireq = data['maxNireq']
        self.step_nireq = data['stepNireq']
        self.inference_time = data['inferenceTime']
        self.progress = 0

    def json(self) -> dict:
        return {
            **super().json(),
            'minBatch': self.min_batch,
            'maxBatch': self.max_batch,
            'stepBatch': self.step_batch,
            'minNireq': self.min_batch,
            'maxNireq': self.max_batch,
            'stepNireq': self.step_batch,
            'inferenceTime': self.inference_time
        }

    @property
    def num_single_inferences(self):
        return math.ceil((self.max_batch - self.min_batch + 1) / self.step_batch) * \
               math.ceil((self.max_nireq - self.min_nireq + 1) / self.step_nireq)
