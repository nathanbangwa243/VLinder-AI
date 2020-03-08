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
from sqlalchemy import Column, Integer, ForeignKey, Float, Text
from app.main.models.jobs_model import JobsModel


class AccuracyJobsModel(JobsModel):
    __tablename__ = 'accuracy_jobs'

    __mapper_args__ = {
        'polymorphic_identity': __tablename__
    }

    job_id = Column(Integer, ForeignKey('jobs.job_id'), primary_key=True)

    accuracy = Column(Float, nullable=True)
    accuracy_config = Column(Text, nullable=True)

    def __init__(self, data):
        super().__init__(data)
        self.project_id = data['projectId']
        self.accuracy_config = data['accuracyConfig']

    def json(self) -> dict:
        return {
            **super().json(),
            'accuracy': self.accuracy,
            'accuracy_config': self.accuracy_config
        }
