"""
 OpenVINO Profiler
 Interface for a common backend job

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
import logging as log

from app.main.job_factory.config import CeleryDBAdapter
from app.main.jobs.interfaces.iemit_message import IEmitMessage
from app.main.models.factory import write_record


class IJob:
    db_table = None
    subprocess = []

    def __init__(self, job_type, emit_message: IEmitMessage):
        self.job_type = job_type
        self.emit_message = emit_message
        self.celery_task = None

    def run(self):
        raise NotImplementedError

    def set_task_id(self, task_id: str):
        session = CeleryDBAdapter.session()
        record = session.query(self.db_table).get(self.emit_message.job_id)
        if record:
            record.task_id = task_id
            write_record(record, session)
        session.close()

    def on_failure(self):
        log.debug('[ TEARDOWN JOB ] %s %s', self.job_type, self.emit_message.job_id)
