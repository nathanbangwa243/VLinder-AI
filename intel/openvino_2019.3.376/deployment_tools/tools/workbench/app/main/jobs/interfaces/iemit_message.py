"""
 OpenVINO Profiler
 Interface for emit message classes

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
import time

from app import get_socket_io
from app.main.job_factory.config import CeleryDBAdapter
from app.main.jobs.interfaces.iconfig import IConfig
from app.main.jobs.utils.database_functions import set_status_in_db
from app.main.models.enumerates import StatusEnum


class IEmitMessageStage:
    def __init__(self,
                 job_type: str,
                 name: str = None,
                 progress: float = 0.0,
                 error_message: str = None,
                 weight: float = 1):
        self.job_type = job_type
        self.name = name if name else self.job_type
        self.progress = progress
        self.error_message = error_message
        self.weight = weight


class IEmitMessage:
    stages = None

    def __init__(self, job, job_id: int, config: IConfig, weight: float, socket_io=None):
        self.job = job
        self.jobs = []
        self.job_id = job_id
        self.config = config
        self.date = time.time()
        self.socket_io = socket_io
        self.weight = weight
        self.last_emitted_percent = 0

    def update_progress(self, name: str, new_value: float):
        self.find_job_by_name(name).progress = new_value
        self.emit_progress()

    @property
    def local_progress(self):
        return sum([job.progress * self.stages.stages[job.job_type] for job in self.jobs])

    @property
    def total_progress(self):
        return self.local_progress * self.weight

    def add_stage(self, stage: IEmitMessageStage, silent: bool = False) -> str:
        for job in self.jobs:
            job.progress = 100  # if add new job, all previously are finished
        count = len([j for j in self.jobs if j.job_type == stage.job_type])
        stage.name = self.generate_name(stage.job_type, count)
        self.jobs.append(stage)
        if not silent:
            self.emit_message()
        return stage.name

    def add_error(self, message, name=None):
        if not self.jobs:
            self.add_stage(IEmitMessageStage('Job failed'))
        job = self.find_job_by_name(name)
        if not job:
            job = self.get_current_job()
        job.error_message = message
        self.set_error_to_database(message)
        self.emit_message()

    def set_error_to_database(self, message):
        session = CeleryDBAdapter.session()
        set_status_in_db(self.job.db_table, self.job_id, StatusEnum.error, session, message)
        session.close()

    def find_job_by_name(self, name: str) -> [IEmitMessageStage, None]:
        jobs = list(filter(lambda job: job.name == name, self.jobs))
        if jobs:
            return jobs[0]
        return None

    def get_current_job(self) -> (IEmitMessageStage, None):
        return self.jobs[-1] if self.jobs else None

    def short_json(self) -> dict:
        return self.full_json()

    def full_json(self):
        return {
            'creationTimestamp': self.date,
            'jobId': self.job_id,
            'type': self.job.job_type.value,
            'config': self.config.json()
        }

    def emit_progress(self):
        if not self.last_emitted_percent or (self.total_progress - self.last_emitted_percent) > 1:
            self.last_emitted_percent = self.total_progress
            self.emit_message()

    def emit_message(self):
        socket_io = self.socket_io or get_socket_io()
        socket_io.emit(self.event, self.short_json(), namespace=self.namespace)
        socket_io.sleep(0)


    @staticmethod
    def generate_name(name, count):
        return '{} ({})'.format(name, count) if count else name
