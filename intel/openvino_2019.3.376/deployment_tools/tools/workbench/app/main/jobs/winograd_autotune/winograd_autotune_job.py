"""
 OpenVINO Profiler
 Class for Winograd job

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

from app.error.job_error import WinogradAutotuneError
from app.main.console_tool_wrapper.winograd_tool.winograd_tool_console_output_parser import WinogradConsoleOutputParser
from app.main.console_tool_wrapper.winograd_tool.winograd_tool_error_message_processor import \
    WinogradErrorMessageProcessor
from app.main.console_tool_wrapper.winograd_tool.winograd_tool_parameters import WinogradParameters
from app.main.console_tool_wrapper.winograd_tool.winograd_tool_stages import WinogradToolStages
from app.main.job_factory.config import CeleryDBAdapter
from app.main.jobs.interfaces.ijob import IJob
from app.main.jobs.job_types_enum import JobTypesEnum
from app.main.jobs.tools_runner.runner import run_console_tool
from app.main.jobs.utils.database_functions import set_status_in_db
from app.main.jobs.winograd_autotune.winograd_autotune_config import WinogradAutotuneConfig
from app.main.jobs.winograd_autotune.winograd_autotune_emit_msg import WinogradAutotuneEmitMessage
from app.main.models.enumerates import StatusEnum
from app.main.models.factory import write_record
from app.main.models.topologies_model import TopologiesModel
from app.main.models.winograd_autotune_model import WinogradAutotuneJobsModel


class WinogradAutotuneJob(IJob):
    db_table = WinogradAutotuneJobsModel

    def __init__(self, job_id: int, config: WinogradAutotuneConfig, weight):
        super(WinogradAutotuneJob, self).__init__(JobTypesEnum.winograd_autotune_type,
                                                  WinogradAutotuneEmitMessage(self, job_id, config, weight))

    def run(self):
        session = CeleryDBAdapter.session()
        winograd_autotune_job = session.query(WinogradAutotuneJobsModel).get(self.emit_message.job_id)
        new_model = session.query(TopologiesModel).get(winograd_autotune_job.result_model_id)
        tuned_path = new_model.path
        set_status_in_db(WinogradAutotuneJobsModel, self.emit_message.job_id, StatusEnum.running, session)
        session.close()
        self.emit_message.emit_progress()
        parameters = WinogradParameters(self.emit_message.config)
        parameters.set_parameter('o', tuned_path)
        parser = WinogradConsoleOutputParser(self.emit_message, WinogradToolStages.get_stages())

        return_code, message = run_console_tool(parameters, parser)
        if return_code:
            message = parser.get_error() if not message else message
            error_message = WinogradErrorMessageProcessor.recognize_error(message, 'winograd autotuner')
            self.emit_message.add_error(error_message)
            raise WinogradAutotuneError(error_message, self.emit_message.job_id)
        session = CeleryDBAdapter.session()
        set_status_in_db(WinogradAutotuneJobsModel, self.emit_message.job_id, StatusEnum.ready, session)
        winograd_autotune_job = session.query(WinogradAutotuneJobsModel).get(self.emit_message.job_id)
        winograd_autotune_job.progress = 100
        write_record(winograd_autotune_job, session)
        set_status_in_db(WinogradAutotuneJobsModel, self.emit_message.job_id, StatusEnum.ready, session, force=True)
        session.close()
        self.emit_message.emit_message()

    def on_failure(self):
        session = CeleryDBAdapter.session()
        set_status_in_db(WinogradAutotuneJobsModel, self.emit_message.job_id, StatusEnum.error, session)
        winograd_job = session.query(WinogradAutotuneJobsModel).get(self.emit_message.job_id)
        result_model_id = winograd_job.result_model_id
        session.close()
        set_status_in_db(TopologiesModel, result_model_id, StatusEnum.error, session)
