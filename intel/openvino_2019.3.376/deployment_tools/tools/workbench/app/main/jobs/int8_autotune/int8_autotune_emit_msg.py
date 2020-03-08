"""
 OpenVINO Profiler
 Class for int8 autotune emit message

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

from app.main.console_tool_wrapper.calibration_tool.stages import CalibrationStages
from app.main.job_factory.config import CeleryDBAdapter
from app.main.jobs.interfaces.iemit_message import IEmitMessage, IEmitMessageStage
from app.main.jobs.int8_autotune.int8_autotune_config import Int8AutoTuneConfig
from app.main.jobs.utils.traversal import get_top_level_model_id
from app.main.models.enumerates import OptimizationTypesEnum, DevicesEnum
from app.main.models.factory import write_record
from app.main.models.int8_autotune_model import Int8AutotuneJobsModel
from app.main.models.projects_model import ProjectsModel


class Int8AutoTuneEmitMessage(IEmitMessage):
    event = 'tune_int8'
    namespace = '/{}'.format(event)
    stages = CalibrationStages

    def __init__(self, job, job_id: int, config: Int8AutoTuneConfig, weight):
        super(Int8AutoTuneEmitMessage, self).__init__(job, job_id, config, weight)
        self.last_emitted_percent = 0

    def add_stage(self, stage: IEmitMessageStage, silent: bool = False) -> str:
        super(Int8AutoTuneEmitMessage, self).add_stage(stage, silent)
        if self.get_current_job().job_type != CalibrationStages.get_stages()[-1]:
            self.update_progress_in_database(self.total_progress)
        else:
            self.update_progress_in_database(100)
            self.emit_message()
        return self.get_current_job().name

    def full_json(self):
        session = CeleryDBAdapter.session()
        int8_record = session.query(Int8AutotuneJobsModel).filter_by(job_id=self.job_id).first()
        progress = int8_record.progress
        status = int8_record.status.value
        error_message = int8_record.error_message
        project_record = session.query(ProjectsModel).filter_by(
            model_id=int8_record.result_model_id,
            dataset_id=self.config.dataset_id,
            target=DevicesEnum(self.config.device),
            optimization_type=OptimizationTypesEnum(self.job.job_type.value)
        ).first()
        session.close()
        original_model_id = get_top_level_model_id(project_record.id)
        message = {
            'creationTimestamp': self.date,
            'jobId': self.job_id,
            'type': self.job.job_type.value,
            'config': self.config.json(),
            'projectId': project_record.id,
            'originalModelId': original_model_id,
            'status': {
                'progress': progress,
                'name': status
            }
        }
        if error_message:
            message['status']['errorMessage'] = error_message
        return message

    def update_percent(self, percent: float = 0.0):
        current_stage = self.get_current_job()
        current_stage.progress = percent
        log.debug('[ INT8 ]: Update progress of stage %s percent: %s', current_stage.name, percent)
        self.update_progress_in_database(self.total_progress)
        self.emit_progress()

    def update_progress_in_database(self, progress):
        session = CeleryDBAdapter.session()
        int8_record = session.query(Int8AutotuneJobsModel).filter_by(job_id=self.job_id).first()
        int8_record.progress = progress
        write_record(int8_record, session)
        session.close()
