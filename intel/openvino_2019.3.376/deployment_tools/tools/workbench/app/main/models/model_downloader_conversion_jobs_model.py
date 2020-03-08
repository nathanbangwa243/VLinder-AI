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

from sqlalchemy import String, Integer, Column, ForeignKey, Text
from sqlalchemy.orm import relationship

from app.main.models.jobs_model import JobsModel


class ModelDownloaderConversionJobsModel(JobsModel):
    __tablename__ = 'model_downloader_convert_jobs'
    __mapper_args__ = {
        'polymorphic_identity': __tablename__
    }

    job_id = Column(Integer, ForeignKey('jobs.job_id'), primary_key=True)
    result_model_id = Column(Integer, ForeignKey('topologies.id'), nullable=True)
    conversion_args = Column(Text, nullable=True)
    path = Column(String, nullable=False)

    result_model = relationship('TopologiesModel', foreign_keys=[result_model_id])

    def __init__(self, data):
        super().__init__(data)
        self.path = data['path']

    def json(self):
        return {
            'path': self.path,
            'args': self.conversion_args
        }
