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

import os
import json
import re

from sqlalchemy import desc

from app.error.job_error import ModelOptimizerError
from app.main.job_factory.config import CeleryDBAdapter
from app.main.jobs.interfaces.ijob import IJob
from app.main.jobs.job_types_enum import JobTypesEnum
from app.main.jobs.model_optimizer.model_optimizer_config import ModelOptimizerConfig
from app.main.jobs.model_optimizer.model_optimizer_emit_msg import ModelOptimizerEmitMessage
from app.main.jobs.tools_runner.runner import run_console_tool
from app.main.jobs.utils.database_functions import set_status_in_db
from app.main.jobs.utils.utils import resolve_file_args
from app.main.models.enumerates import StatusEnum
from app.main.models.factory import write_record
from app.main.models.model_optimizer_job_model import ModelOptimizerJobModel
from app.main.models.topologies_model import TopologiesModel
from app.main.console_tool_wrapper.model_optimizer.console_output_parser import ModelOptimizerParser
from app.main.console_tool_wrapper.model_optimizer.parameters import ModelOptimizerParameters
from app.main.console_tool_wrapper.model_optimizer.stages import ModelOptimizerStages
from app.main.utils.utils import create_empty_dir, remove_dir, get_size_of_files
from config.constants import ORIGINAL_FOLDER, UPLOAD_FOLDER_MODELS


class ModelOptimizerJob(IJob):
    event = 'model'
    db_table = TopologiesModel

    def __init__(self, artifact_id: int, config: ModelOptimizerConfig, weight: float):
        super().__init__(
            JobTypesEnum.model_optimizer_type,
            ModelOptimizerEmitMessage(self, artifact_id, config, weight)
        )
        self.emit_message.event = self.event

    def run(self):
        emit_msg = self.emit_message
        config = emit_msg.config

        session = CeleryDBAdapter.session()

        original_topology = session.query(TopologiesModel).get(config.original_topology_id)
        log.debug('[ MODEL OPTIMIZER ] Optimizing model %s', original_topology.name)

        mo_job_record = (
            session.query(ModelOptimizerJobModel)
                .filter_by(result_model_id=config.result_model_id)
                .order_by(desc(ModelOptimizerJobModel.creation_timestamp))
                .first()
        )
        mo_job_id = mo_job_record.job_id
        mo_job_record.status = StatusEnum.running

        resulting_topology = session.query(TopologiesModel).get(config.result_model_id)
        resulting_topology.converted_from = config.original_topology_id
        resulting_topology.status = StatusEnum.running
        resulting_topology.path = os.path.join(UPLOAD_FOLDER_MODELS, str(config.result_model_id), ORIGINAL_FOLDER)
        write_record(resulting_topology, session)
        create_empty_dir(resulting_topology.path)

        resolve_file_args(emit_msg.job_id, config, original_topology)
        mo_job_record.mo_args = json.dumps(config.mo_args)
        write_record(mo_job_record, session)

        config.mo_args.update({
            'model_name': original_topology.name,
            'framework': original_topology.framework.value,
            'output_dir': resulting_topology.path,
            'steps': True,
        })

        session.close()

        parameters = ModelOptimizerParameters(config.mo_args)
        parser = ModelOptimizerParser(self.emit_message, ModelOptimizerStages.get_stages())
        return_code, message = run_console_tool(parameters, parser, self)

        if return_code:
            match = re.search(r': (.+)\.\s+For more information please refer to Model Optimizer FAQ', message)
            short_error_message = match.group(1) if match else 'Model Optimizer failed'

            log.error('[ MODEL OPTIMIZER ] [ ERROR ]: %s', short_error_message)

            session = CeleryDBAdapter.session()

            mo_job_record = session.query(ModelOptimizerJobModel).get(mo_job_id)
            mo_job_record.status = StatusEnum.error
            mo_job_record.error_message = short_error_message
            mo_job_record.detailed_error_message = re.sub(r'\[ ERROR \]\s*', '', re.sub(r'(\n\s*)+', '\n', message))
            write_record(mo_job_record, session)

            resulting_topology = session.query(TopologiesModel).get(config.result_model_id)
            resulting_topology.status = StatusEnum.error
            resulting_topology.error_message = short_error_message
            write_record(resulting_topology, session)

            session.close()

            self.emit_message.emit_message()

            raise ModelOptimizerError(short_error_message, self.emit_message.job_id)

        session = CeleryDBAdapter.session()

        mo_job_record = session.query(ModelOptimizerJobModel).get(mo_job_id)
        mo_job_record.progress = 100
        mo_job_record.status = StatusEnum.ready
        write_record(mo_job_record, session)

        resulting_topology = session.query(TopologiesModel).get(config.result_model_id)
        resulting_topology.size = get_size_of_files(resulting_topology.path)
        write_record(resulting_topology, session)

        session.close()

        self.emit_message.emit_message()

    def on_failure(self):
        super().on_failure()
        session = CeleryDBAdapter.session()
        set_status_in_db(TopologiesModel, self.emit_message.config.result_model_id, StatusEnum.error, session)
        remove_dir(session.query(TopologiesModel).get(self.emit_message.config.result_model_id).path)
        session.close()

    def set_task_id(self, task_id):
        session = CeleryDBAdapter.session()
        resulting_topology = session.query(TopologiesModel).get(self.emit_message.config.result_model_id)
        resulting_topology.task_id = task_id
        write_record(resulting_topology, session)
        session.close()
