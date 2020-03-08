"""
 OpenVINO Profiler
 Class for running Jobs as celery tasks

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

import gc
import os
import sys
import signal
from flask_socketio import SocketIO
from sqlalchemy.exc import SQLAlchemyError

from app import get_celery, get_config
from app.error.general_error import GeneralError
from app.main.job_factory.config import CeleryDBAdapter
from app.main.job_factory.job_factory import JobFromDBFactory
from app.main.jobs.feed.feed_emit_msg import FeedEmitMessage
from app.main.jobs.registries.code_registry import CodeRegistry
from app.main.utils.safe_runner import safe_run
from config.constants import SERVER_MODE

CELERY = get_celery()


class Task(CELERY.Task):
    config = get_config()[SERVER_MODE]
    socket_io = None
    job = None

    # pylint: disable=unused-argument
    def run(self, previous_task_return_value, job_type: str, job_id: int, **kwargs):
        # The previous_task_return_value argument is required by celery chain method
        # as it uses partial argument rule
        signal.signal(signal.SIGTERM, lambda signum, frame: self.terminate())
        self.socket_io = SocketIO(message_queue=self.config.broker_url)
        data = kwargs['data'] if 'data' in kwargs else {}
        progress_weight = kwargs['progress_weight'] if 'progress_weight' in kwargs else 1
        self.job = JobFromDBFactory.create_job(job_type, job_id, self.socket_io, data, progress_weight, self)
        self.job.set_task_id(self.request.id)
        self.job.run()

    # pylint: disable=unused-argument
    @staticmethod
    def after_return(*args):
        CeleryDBAdapter.session.remove()

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        log.debug('[ TASK %s FAILED ] ', task_id)
        log.debug('[ TASK ARGS ] : %s', args)
        log.debug('[ TASK KWARGS ] : %s', kwargs)
        if isinstance(exc, GeneralError):
            message = str(exc)
            error_code = exc.get_error_code()
        elif isinstance(exc, SQLAlchemyError):
            message = 'Unable to update information in database'
            error_code = CodeRegistry.get_database_error_code()
        else:
            message = str(exc)
            error_code = 500
        log.error('Server Exception', exc_info=einfo)
        FeedEmitMessage.socket_io = self.socket_io
        if self.job:
            safe_run(self.job.on_failure)()
        FeedEmitMessage.emit(error_code, message)
        gc.collect()

    # pylint: disable=unused-argument
    @staticmethod
    def on_success(*args):
        gc.collect()

    def terminate(self):
        self.request.callbacks = None
        self.request.chain = None
        for pid in self.job.subprocess:
            try:
                os.kill(pid, 0)
            except OSError:
                pass
            else:
                os.kill(pid, signal.SIGTERM)
        get_celery().control.revoke(self.request.id, terminate=True, signal='SIGKILL')
        CeleryDBAdapter.session.remove()
        sys.exit(1)


TASK = CELERY.register_task(Task())
