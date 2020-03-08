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

from app.main.console_tool_wrapper.model_downloader.converter_error_message_processor import (
    TopologyConvertErrorMessageProcessor)
from app.main.console_tool_wrapper.model_downloader.converter_parameters import TopologyConvertParameters
from app.main.console_tool_wrapper.model_downloader.converter_parser import TopologyConvertParser
from app.main.console_tool_wrapper.model_downloader.converter_stages import TopologyConvertStages
from app.main.jobs.topology_convert.topology_convert_config import TopologyConvertConfig
from app.main.jobs.topology_convert.topology_convert_emit_msg import TopologyConvertEmitMessage
from app.main.models.model_downloader_conversion_jobs_model import ModelDownloaderConversionJobsModel
from app.error.job_error import ModelOptimizerError
from app.main.job_factory.config import CeleryDBAdapter
from app.main.jobs.interfaces.ijob import IJob
from app.main.jobs.job_types_enum import JobTypesEnum
from app.main.jobs.tools_runner.runner import run_console_tool
from app.main.jobs.utils.database_functions import set_status_in_db
from app.main.models.enumerates import StatusEnum
from app.main.models.factory import write_record
from app.main.models.model_downloader_model import ModelDownloaderModel
from app.main.models.topologies_model import TopologiesModel


class TopologyConvertJob(IJob):
    db_table = TopologiesModel

    def __init__(self, job_id: int, config: TopologyConvertConfig, weight: float):
        super().__init__(JobTypesEnum.model_convert_type,
                         TopologyConvertEmitMessage(self, job_id, config, weight))
        self.previous_task_id = None

    def run(self):
        emit_msg = self.emit_message
        config = emit_msg.config
        log.debug('[ MODEL DOWNLOADER CONVERT ] Converting model %s', config.name)

        session = CeleryDBAdapter.session()
        convert_model = session.query(ModelDownloaderConversionJobsModel).get(self.emit_message.job_id)
        download_model = session.query(ModelDownloaderModel).filter_by(
            result_model_id=convert_model.result_model_id
        ).first()
        if convert_model.conversion_args is None or (download_model and download_model.status != StatusEnum.ready):
            log.debug('[ MODEL DOWNLOADER CONVERT ] Model Converter args or %s files are not in place yet, skipping.',
                      convert_model.result_model.name)
            # Once the downloader started, its id is in the Topology instance.
            # Each next convert request is skipped if downloader is still running.
            # However, in the Topology instance, we need to switch task id back to the original Downloader job.
            self.set_task_id(self.previous_task_id)
            session.close()
            self.celery_task.request.chain = None
            return
        convert_model.status = StatusEnum.running
        artifact = session.query(TopologiesModel).get(config.result_model_id)
        artifact.status = StatusEnum.running
        emit_msg.set_previous_accumulated_progress(artifact.progress)
        write_record(convert_model, session)
        write_record(artifact, session)
        session.close()

        parameters = TopologyConvertParameters(config)
        parser = TopologyConvertParser(self.emit_message, TopologyConvertStages.get_stages())
        return_code, message = run_console_tool(parameters, parser, self)

        if return_code:
            job_name = self.emit_message.get_current_job().name if self.emit_message.get_current_job() else None
            error = TopologyConvertErrorMessageProcessor.recognize_error(message, job_name)
            log.error('[ MODEL DOWNLOADER CONVERT ] [ ERROR ]: %s', error)
            session = CeleryDBAdapter.session()
            set_status_in_db(ModelDownloaderConversionJobsModel, self.emit_message.job_id, StatusEnum.error, session,
                             error)
            set_status_in_db(TopologiesModel, config.result_model_id, StatusEnum.error, session, error)
            session.close()
            self.emit_message.add_error('Model optimizer failed: {}'.format(error), return_code)
            raise ModelOptimizerError(error, self.emit_message.job_id)
        session = CeleryDBAdapter.session()
        convert_model = session.query(ModelDownloaderConversionJobsModel).get(self.emit_message.job_id)
        convert_model.progress = 100
        convert_model.status = StatusEnum.ready
        write_record(convert_model, session)
        model = session.query(TopologiesModel).get(config.result_model_id)
        model.path = config.output
        write_record(model, session)
        session.close()
        self.emit_message.emit_message()

    def on_failure(self):
        super().on_failure()
        session = CeleryDBAdapter.session()
        set_status_in_db(TopologiesModel, self.emit_message.config.result_model_id, StatusEnum.error, session)
        set_status_in_db(ModelDownloaderConversionJobsModel, self.emit_message.job_id, StatusEnum.error, session)
        session.close()

    def set_task_id(self, task_id):
        session = CeleryDBAdapter.session()
        topology_convert_job = session.query(ModelDownloaderConversionJobsModel).filter_by(
            job_id=self.emit_message.job_id).first()
        topology = topology_convert_job.result_model
        self.previous_task_id = topology.task_id
        topology.task_id = task_id
        write_record(topology, session)
        topology_convert_job.task_id = task_id
        write_record(topology_convert_job, session)
        session.close()
