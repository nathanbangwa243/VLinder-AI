"""
 OpenVINO Profiler
 Class for int8 calibration job

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
import os
import json
import logging as log

from app.error.job_error import Int8AutotuneError
from app.main.console_tool_wrapper.accuracy_tools.accuracy_cli_parameters import AccuracyCheckerCliParameters
from app.main.console_tool_wrapper.accuracy_tools.parser import ProgressParser
from app.main.job_factory.config import CeleryDBAdapter
from app.main.jobs.interfaces.ijob import IJob
from app.main.jobs.job_types_enum import JobTypesEnum
from app.main.jobs.int8_autotune.int8_autotune_config import Int8AutoTuneConfig
from app.main.jobs.int8_autotune.int8_autotune_emit_msg import Int8AutoTuneEmitMessage
from app.main.jobs.tools_runner.runner import run_console_tool
from app.main.jobs.utils.database_functions import set_status_in_db
from app.main.models.compound_inference_model import CompoundInferenceJobsModel
from app.main.models.enumerates import StatusEnum, DevicesEnum
from app.main.models.factory import write_record
from app.main.models.int8_autotune_model import Int8AutotuneJobsModel
from app.main.models.projects_model import ProjectsModel
from app.main.models.datasets_model import DatasetsModel
from app.main.models.topologies_model import TopologiesModel
from app.main.utils.utils import remove_dir, create_empty_dir
from app.main.jobs.interfaces.iemit_message import IEmitMessageStage
from app.main.jobs.utils.accuracy_utils import construct_accuracy_tool_config
from config.constants import PYTHON_CLI_FOLDER


class Int8AutotuneJob(IJob):
    db_table = Int8AutotuneJobsModel

    def __init__(self, job_id: int, config: Int8AutoTuneConfig, weight):
        super(Int8AutotuneJob, self).__init__(JobTypesEnum.int8autotune_type,
                                              Int8AutoTuneEmitMessage(self, job_id, config, weight))

    def _run_python_calibration(self,
                                tuned_path: str,
                                int8_model_id: int):
        try:
            self.emit_message.emit_progress()
            session = CeleryDBAdapter.session()

            new_int8_model = session.query(TopologiesModel).get(int8_model_id)
            int8_job = session.query(Int8AutotuneJobsModel).get(self.emit_message.job_id)
            original_model = session.query(TopologiesModel).get(new_int8_model.optimized_from)

            project_model = session.query(ProjectsModel).get(int8_job.project_id)
            dataset_model = session.query(DatasetsModel).get(project_model.dataset_id)

            config = construct_accuracy_tool_config(original_model, dataset_model, DevicesEnum.cpu)
            config.dataset.subsample_size = '{}%'.format(int8_job.subset_size)

            int8_job.status = StatusEnum.running
            int8_job.calibration_config = json.dumps(config.to_dict())

            write_record(int8_job, session)
            session.close()

            tuned_model_path = os.path.join(tuned_path, str(self.emit_message.job_id))
            yml_file = '{}.yml'.format(tuned_model_path)

            config.dump_to_yml(yml_file)

            cli_params = AccuracyCheckerCliParameters()
            cli_params.exe = os.path.join(PYTHON_CLI_FOLDER, os.path.join('calibration', 'calibrate.py'))
            cli_params.set_parameter('y', yml_file)
            cli_params.set_parameter('th', self.emit_message.config.threshold)
            cli_params.set_parameter('tp', tuned_model_path)

            self.emit_message.add_stage(IEmitMessageStage(job_type='int8_tuning'))
            cli_parser = ProgressParser(self.emit_message, None)
            code, error = run_console_tool(cli_params, cli_parser, self)
            if code:
                self.emit_message.add_error('Calibration tool failed')
                raise Int8AutotuneError(error, self.emit_message.job_id)
            self._update_db_on_success()
            self.emit_message.emit_message()

        except Exception as exc:
            log.debug('[ INT8 python ] ERROR: calibration job failed')
            log.debug(exc)
            remove_dir(tuned_path)
            self.emit_message.add_error('Calibration tool failed')
            raise Int8AutotuneError(str(exc), self.emit_message.job_id)

    def _update_db_on_success(self):
        session = CeleryDBAdapter.session()
        set_status_in_db(Int8AutotuneJobsModel, self.emit_message.job_id, StatusEnum.ready, session, force=True)
        session.close()

    def on_failure(self):
        session = CeleryDBAdapter.session()
        set_status_in_db(Int8AutotuneJobsModel, self.emit_message.job_id, StatusEnum.error, session)
        int8_job = session.query(Int8AutotuneJobsModel).get(self.emit_message.job_id)
        result_model_id = int8_job.result_model_id
        set_status_in_db(TopologiesModel, result_model_id, StatusEnum.error, session)
        project = session.query(ProjectsModel).filter_by(model_id=result_model_id).first()
        compound_job = session.query(CompoundInferenceJobsModel).filter_by(project_id=project.id).first()
        compound_job_id = compound_job.job_id
        set_status_in_db(CompoundInferenceJobsModel, compound_job_id, StatusEnum.error, session)
        set_status_in_db(TopologiesModel, result_model_id, StatusEnum.error, session)
        session.close()

    def run(self):
        session = CeleryDBAdapter.session()
        int8_job = session.query(Int8AutotuneJobsModel).get(self.emit_message.job_id)
        new_int8_model = session.query(TopologiesModel).get(int8_job.result_model_id)
        int8_model_id = new_int8_model.id
        tuned_path = new_int8_model.path
        session.close()
        create_empty_dir(tuned_path)
        self._run_python_calibration(tuned_path, int8_model_id)

    @staticmethod
    def calculate_activation_steps(config):
        activation_upper_boundary = 100
        activation_lower_boundary = config.threshold_boundary
        activation_step = config.threshold_step

        # number of infers in the main loop of the calibration process
        return (activation_upper_boundary - activation_lower_boundary) / activation_step
