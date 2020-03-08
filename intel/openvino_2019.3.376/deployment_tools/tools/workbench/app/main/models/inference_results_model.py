"""
 OpenVINO Profiler
 Class for ORM model described an Inference Job

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

from sqlalchemy import Column, Float, Text, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import JSON

from app.main.models.base_model import BaseModel
from app.main.models.enumerates import StatusEnum, STATUS_ENUM_SCHEMA


class InferenceResultsModel(BaseModel):
    __tablename__ = 'inference_results'

    id = Column(Integer, primary_key=True, autoincrement=True)

    job_id = Column(Integer, ForeignKey('compound_inferences_jobs.job_id'), nullable=False)

    latency = Column(Float, nullable=True)
    throughput = Column(Float, nullable=True)
    total_execution_time = Column(Float, nullable=True)

    perf_counters = Column(JSON, nullable=True)
    exec_graph = Column(Text, nullable=True)

    batch = Column(Integer, nullable=False, default=1)
    nireq = Column(Integer, nullable=False, default=1)

    status = Column(STATUS_ENUM_SCHEMA, nullable=False, default=StatusEnum.queued)
    progress = Column(Float, nullable=False, default=0)

    def __init__(self, data):
        self.job_id = data['jobId']
        self.update(data)

    def update(self, results):
        exec_info = None

        if results and 'execInfo' in results:
            exec_info = results['execInfo']

        if exec_info:
            self.latency = exec_info['latency'] if 'latency' in exec_info else None
            self.throughput = exec_info['throughput'] if 'throughput' in exec_info else None
            self.total_execution_time = exec_info['totalExecTime'] if 'totalExecTime' in exec_info else None
            if 'batch' in exec_info:
                self.batch = exec_info['batch']
            if 'nireq' in exec_info:
                self.nireq = exec_info['nireq']

        if 'pc' in results:
            self.perf_counters = results['pc']
        if 'execGraph' in results:
            self.exec_graph = results['execGraph']

    def json(self):
        return {
            'execInfo': {
                'nireq': self.nireq,
                'batch': self.batch,
                'latency': self.latency,
                'throughput': self.throughput,
                'totalExecutionTime': self.total_execution_time,
            },
            'pc': self.perf_counters,
            'execGraph': self.exec_graph
        }
