"""
 OpenVINO Profiler
 Class for ORM model described a dataset generation config

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

from sqlalchemy import Column, Integer, ForeignKey, String
from app.main.models.jobs_model import JobsModel


class DownloadConfigsModel(JobsModel):
    __tablename__ = 'downloads_configs'

    __mapper_args__ = {
        'polymorphic_identity': __tablename__
    }
    job_id = Column(Integer, ForeignKey('jobs.job_id'), primary_key=True)

    path = Column(String, nullable=True)
    name = Column(String, nullable=True)

    def __init__(self, data):
        super(DownloadConfigsModel, self).__init__(data)
        self.path = data['path']
        self.name = data['name']

    def json(self):
        return {
            'jobId': self.job_id
        }
