"""
 OpenVINO Profiler
 Class for accuracy web socket message

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
from app.main.jobs.accuracy.accuracy_config import AccuracyConfig
from app.main.jobs.interfaces.iemit_message import IEmitMessage

from app.main.models.accuracy_model import AccuracyJobsModel


class AccuracyEmitMessage(IEmitMessage):
    event = 'accuracy'
    namespace = '/accuracy'

    def __init__(self, job, job_id: int, config: AccuracyConfig, weight: float):
        super(AccuracyEmitMessage, self).__init__(job, job_id, config, weight)
        self.last_emitted_percent = 0

    def full_json(self):
        session = CeleryDBAdapter.session()
        accuracy_job = session.query(AccuracyJobsModel).get(self.job_id)
        accuracy = accuracy_job.accuracy
        status = accuracy_job.status.value
        error_message = accuracy_job.error_message
        session.close()
        message = {
            **super().full_json(),
            'accuracy': accuracy,
            'status': {
                'progress': self.get_current_job().progress,
                'name': status
            }
        }
        if error_message:
            message['status']['errorMessage'] = error_message
        return message

    def update_percent(self, percent: float = 0.0):
        current_stage = self.get_current_job()
        current_stage.progress = percent
        log.debug('[ ACCURACY ]: Update progress of stage %s percent: %s', current_stage.name, percent)
        self.update_progress_in_database(self.total_progress)
        self.emit_message()

    def update_progress_in_database(self, progress):
        session = CeleryDBAdapter.session()
        accuracy_record = session.query(AccuracyJobsModel).filter_by(job_id=self.job_id).first()
        accuracy_record.progress = progress
        session.add(accuracy_record)
        session.commit()
        session.close()

    @property
    def local_progress(self):
        return self.get_current_job().progress
