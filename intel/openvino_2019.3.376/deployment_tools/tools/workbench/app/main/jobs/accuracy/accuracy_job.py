"""
 OpenVINO Profiler
 Class for accuracy job

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
import json
import os

from app.error.job_error import AccuracyError

from app.main.console_tool_wrapper.accuracy_tools.accuracy.accuracy_parser import AccuracyParser
from app.main.console_tool_wrapper.accuracy_tools.accuracy_cli_parameters import AccuracyCheckerCliParameters
from app.main.jobs.interfaces.iemit_message import IEmitMessageStage
from app.main.jobs.interfaces.ijob import IJob
from app.main.jobs.job_types_enum import JobTypesEnum
from app.main.job_factory.config import CeleryDBAdapter
from app.main.jobs.accuracy.accuracy_config import AccuracyConfig
from app.main.jobs.accuracy.accuracy_emit_msg import AccuracyEmitMessage
from app.main.jobs.tools_runner.runner import run_console_tool
from app.main.models.topologies_model import TopologiesModel
from app.main.models.datasets_model import DatasetsModel
from app.main.models.accuracy_model import AccuracyJobsModel
from app.main.models.enumerates import StatusEnum
from app.main.models.factory import write_record
from app.main.jobs.utils.database_functions import set_status_in_db
from app.main.models.projects_model import ProjectsModel
from app.main.jobs.utils.accuracy_utils import construct_accuracy_tool_config
from config.constants import PYTHON_CLI_FOLDER


class AccuracyJob(IJob):
    db_table = AccuracyJobsModel

    def __init__(self, job_id: int, config: AccuracyConfig, weight: float):
        super(AccuracyJob, self).__init__(JobTypesEnum.accuracy_type,
                                          AccuracyEmitMessage(self, job_id, config, weight))

    def run(self):
        session = CeleryDBAdapter.session()
        accuracy_job = session.query(AccuracyJobsModel).get(self.emit_message.job_id)
        project = session.query(ProjectsModel).get(accuracy_job.project_id)
        original_model = session.query(TopologiesModel).get(project.model_id)
        dataset_model = session.query(DatasetsModel).get(project.dataset_id)
        accuracy_job.status = StatusEnum.running

        config = construct_accuracy_tool_config(original_model, dataset_model, project.target)

        accuracy_config = json.dumps(config.to_dict())
        accuracy_job.accuracy_config = accuracy_config

        write_record(accuracy_job, session)

        session.close()
        self.emit_message.add_stage(IEmitMessageStage(job_type='accuracy'))

        cli_params = AccuracyCheckerCliParameters()
        cli_params.exe = os.path.join(PYTHON_CLI_FOLDER, os.path.join('accuracy', 'check_accuracy.py'))
        log.debug(accuracy_config)
        cli_params.set_parameter('y', "\'{}\'".format(accuracy_config))

        cli_parser = AccuracyParser(self.emit_message, None)
        code, error = run_console_tool(cli_params, cli_parser, self)
        if code:
            self.emit_message.add_error('Accuracy tool failed')
            raise AccuracyError(error, self.emit_message.job_id)

        session = CeleryDBAdapter.session()
        accuracy_job = session.query(AccuracyJobsModel).get(self.emit_message.job_id)
        accuracy_job.accuracy = round(cli_parser.accuracy, 3)
        accuracy_job.status = StatusEnum.ready
        write_record(accuracy_job, session)
        session.close()
        self.emit_message.update_percent(100)

    def on_failure(self):
        session = CeleryDBAdapter.session()
        set_status_in_db(AccuracyJobsModel, self.emit_message.job_id, StatusEnum.error, session)
        session.close()
