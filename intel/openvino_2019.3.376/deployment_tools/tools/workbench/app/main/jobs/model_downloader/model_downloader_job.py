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

from app.error.job_error import ModelDownloaderError
from app.main.console_tool_wrapper.model_downloader.console_output_parser import ModelDownloaderParser
from app.main.console_tool_wrapper.model_downloader.error_message_processor import ModelDownloaderErrorMessageProcessor
from app.main.console_tool_wrapper.model_downloader.parameters import ModelDownloaderParameters
from app.main.job_factory.config import CeleryDBAdapter
from app.main.jobs.interfaces.ijob import IJob
from app.main.jobs.job_types_enum import JobTypesEnum
from app.main.jobs.model_downloader.model_downloader_config import ModelDownloaderConfig
from app.main.jobs.model_downloader.model_downloader_emit_msg import ModelDownloaderEmitMessage
from app.main.jobs.tools_runner.runner import run_console_tool
from app.main.jobs.utils.database_functions import set_status_in_db
from app.main.models.enumerates import StatusEnum
from app.main.models.factory import write_record
from app.main.models.model_downloader_model import ModelDownloaderModel
from app.main.models.topologies_model import TopologiesModel
from app.main.utils.utils import remove_dir


class ModelDownloaderJob(IJob):
    db_table = TopologiesModel

    def __init__(self, job_id: int, config: ModelDownloaderConfig, weight: float):
        super().__init__(JobTypesEnum.model_downloader_type,
                         ModelDownloaderEmitMessage(self, job_id, config, weight))

    def run(self):
        emit_msg = self.emit_message
        config = emit_msg.config
        log.debug('[ MODEL DOWNLOADER ] Downloading model %s', config.name)
        self.emit_message.emit_message()
        parser = ModelDownloaderParser(self.emit_message, self.emit_message.stages.get_stages())
        parameters = self.setup_parameters(config)
        session = CeleryDBAdapter.session()
        artifact = session.query(TopologiesModel).get(config.result_model_id)
        artifact.status = StatusEnum.running
        download_model = session.query(ModelDownloaderModel).get(self.emit_message.job_id)
        download_model.status = StatusEnum.running
        emit_msg.set_previous_accumulated_progress(artifact.progress)
        session.add(artifact)
        write_record(download_model, session)
        session.close()
        return_code, message = run_console_tool(parameters, parser, self)
        if return_code or 'Error' in message:
            job_name = self.emit_message.get_current_job().name if self.emit_message.get_current_job() else None
            error = ModelDownloaderErrorMessageProcessor.recognize_error(message, job_name)
            session = CeleryDBAdapter.session()
            set_status_in_db(ModelDownloaderModel, self.emit_message.job_id, StatusEnum.error, session, error)
            set_status_in_db(TopologiesModel, config.result_model_id, StatusEnum.error, session, error)
            session.close()
            log.error('[ MODEL_DOWNLOADER ] [ ERROR ]: %s', error)
            self.emit_message.add_error('Model downloader failed: {}'.format(error))
            raise ModelDownloaderError(error, self.emit_message.job_id)
        for job in self.emit_message.jobs:
            job.progress = 100
        download_model = session.query(ModelDownloaderModel).get(self.emit_message.job_id)
        download_model.progress = 100
        download_model.status = StatusEnum.ready
        write_record(download_model, session)
        session.close()
        self.emit_message.emit_message()

    def on_failure(self):
        super().on_failure()
        remove_dir(self.emit_message.config.output)

    def set_task_id(self, task_id):
        session = CeleryDBAdapter.session()
        downloader_job = session.query(ModelDownloaderModel).filter_by(job_id=self.emit_message.job_id).first()
        topology = session.query(TopologiesModel).get(downloader_job.result_model_id)
        topology.task_id = task_id
        write_record(topology, session)
        downloader_job.task_id = task_id
        write_record(downloader_job, session)
        session.close()

    def setup_parameters(self, config):
        try:
            parameters = ModelDownloaderParameters(config=config)
        except ValueError as error:
            raise ModelDownloaderError(str(error), self.emit_message.job_id)
        return parameters
