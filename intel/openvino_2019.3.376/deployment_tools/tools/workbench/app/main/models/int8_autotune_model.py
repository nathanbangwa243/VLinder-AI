"""
 OpenVINO Profiler
 Class for ORM model described an Int8Calibration Job

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
from sqlalchemy import Column, Integer, Float, ForeignKey, Text
from app.main.models.jobs_model import JobsModel


class Int8AutotuneJobsModel(JobsModel):
    __tablename__ = 'int8_autotune_jobs'

    __mapper_args__ = {
        'polymorphic_identity': __tablename__
    }

    job_id = Column(Integer, ForeignKey('jobs.job_id'), primary_key=True)

    threshold = Column(Float, nullable=False)
    subset_size = Column(Integer, nullable=False, default=100)
    result_model_id = Column(Integer, ForeignKey('topologies.id'), nullable=True)

    batch = Column(Integer, nullable=False)
    calibration_config = Column(Text, nullable=True)

    def __init__(self, data):
        super(Int8AutotuneJobsModel, self).__init__(data)
        self.batch = data['batch']
        self.threshold = data['threshold']
        self.subset_size = data['subsetSize']
        self.calibration_config = data['calibrationConfig']

    def json(self) -> dict:
        return {
            **super(Int8AutotuneJobsModel, self).json(),
            'batch': self.batch,
            'threshold': self.threshold,
            'calibration_config': self.calibration_config
        }
