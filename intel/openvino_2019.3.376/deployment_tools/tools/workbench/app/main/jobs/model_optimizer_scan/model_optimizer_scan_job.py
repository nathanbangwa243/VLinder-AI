"""
 OpenVINO Profiler
 Class for job of model creation

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

import re
from app.error.job_error import ModelOptimizerError
from app.main.console_tool_wrapper.model_optimizer.model_optimizer_scan_console_output_parser import \
    ModelOptimizerScanParser
from app.main.job_factory.config import CeleryDBAdapter
from app.main.jobs.interfaces.ijob import IJob
from app.main.jobs.job_types_enum import JobTypesEnum
from app.main.jobs.model_optimizer_scan.model_optimizer_scan_config import ModelOptimizerScanConfig
from app.main.jobs.model_optimizer_scan.model_optimizer_scan_emit_msg import ModelOptimizerScanEmitMessage
from app.main.jobs.tools_runner.runner import run_console_tool
from app.main.jobs.utils.database_functions import set_status_in_db
from app.main.jobs.utils.utils import resolve_file_args
from app.main.models.enumerates import StatusEnum
from app.main.models.factory import write_record
from app.main.models.model_optimizer_scan_model import ModelOptimizerScanJobsModel
from app.main.models.topologies_model import TopologiesModel
from app.main.console_tool_wrapper.model_optimizer.parameters import ModelOptimizerParameters
from app.main.console_tool_wrapper.model_optimizer.stages import ModelOptimizerStages


class ModelOptimizerScanJob(IJob):
    event = 'model'
    db_table = TopologiesModel

    def __init__(self, topology_id: int, config: ModelOptimizerScanConfig, weight: float):
        super().__init__(
            JobTypesEnum.model_optimizer_type,
            ModelOptimizerScanEmitMessage(self, topology_id, config, weight)
        )
        self.emit_message.event = self.event

    def run(self):
        emit_msg = self.emit_message
        config = emit_msg.config

        session = CeleryDBAdapter.session()

        topology = session.query(TopologiesModel).get(config.topology_id)
        log.debug('[ MODEL OPTIMIZER ] Analyzing model %s', topology.name)

        mo_job_record = session.query(ModelOptimizerScanJobsModel).filter_by(topology_id=config.topology_id).first()
        mo_job_id = mo_job_record.job_id
        mo_job_record.status = StatusEnum.running
        write_record(mo_job_record, session)

        resolve_file_args(emit_msg.job_id, config, topology)
        session.close()

        parameters = ModelOptimizerParameters(config.mo_args, {'MO_ENABLED_TRANSFORMS': 'ANALYSIS_JSON_PRINT'})

        parser = ModelOptimizerScanParser(self.emit_message, ModelOptimizerStages.get_stages())

        return_code, message = run_console_tool(parameters, parser, self)

        if return_code:
            match = re.search(r': (.+)\.\s+For more information please refer to Model Optimizer FAQ', message)
            if match:
                short_error_message = match.group(1)
            elif 'FRAMEWORK ERROR' in message:
                short_error_message = 'Invalid topology'
            else:
                short_error_message = 'Model Optimizer Scan failed'

            log.error('[ MODEL OPTIMIZER ] [ ERROR ]: %s', short_error_message)

            session = CeleryDBAdapter.session()
            set_status_in_db(ModelOptimizerScanJobsModel, mo_job_id, StatusEnum.error, session, short_error_message)
            mo_job_record = session.query(ModelOptimizerScanJobsModel).filter_by(topology_id=config.topology_id).first()
            mo_job_record.error_message = short_error_message
            mo_job_record.detailed_error_message = re.sub(r'^.*ERROR \]\s*', '', re.sub(r'(\n\s*)+', '\n', message))
            write_record(mo_job_record, session)

            set_status_in_db(TopologiesModel, emit_msg.job_id, StatusEnum.error, session, short_error_message)
            session.close()

            self.emit_message.emit_message()

            raise ModelOptimizerError(short_error_message, self.emit_message.job_id)

        session = CeleryDBAdapter.session()
        mo_job_record = session.query(ModelOptimizerScanJobsModel).get(mo_job_id)
        mo_job_record.progress = 100
        mo_job_record.status = StatusEnum.ready
        write_record(mo_job_record, session)
        session.close()

        self.emit_message.emit_message()

    def on_failure(self):
        super().on_failure()
        session = CeleryDBAdapter.session()
        mo_job_record = session.query(ModelOptimizerScanJobsModel).filter_by(
            topology_id=self.emit_message.config.topology_id).first()
        mo_job_record.status = StatusEnum.error
        write_record(mo_job_record, session)
        set_status_in_db(TopologiesModel, self.emit_message.config.topology_id, StatusEnum.error, session)
        session.close()

    def set_task_id(self, task_id):
        session = CeleryDBAdapter.session()
        resulting_topology = session.query(TopologiesModel).get(self.emit_message.config.topology_id)
        resulting_topology.task_id = task_id
        write_record(resulting_topology, session)
        session.close()
