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


from app.main.console_tool_wrapper.winograd_tool.winograd_tool_stages import WinogradToolStages
from app.main.job_factory.config import CeleryDBAdapter
from app.main.jobs.interfaces.iemit_message import IEmitMessage, IEmitMessageStage
from app.main.jobs.utils.traversal import get_top_level_model_id
from app.main.jobs.winograd_autotune.winograd_autotune_config import WinogradAutotuneConfig
from app.main.models.enumerates import DevicesEnum, OptimizationTypesEnum
from app.main.models.factory import write_record
from app.main.models.projects_model import ProjectsModel
from app.main.models.winograd_autotune_model import WinogradAutotuneJobsModel


class WinogradAutotuneEmitMessage(IEmitMessage):
    event = 'winograd_autotune'
    namespace = '/{}'.format(event)
    stages = WinogradToolStages

    def __init__(self, job, job_id: int, config: WinogradAutotuneConfig, weight):
        super(WinogradAutotuneEmitMessage, self).__init__(job, job_id, config, weight)
        self.last_emitted_percent = 0

    def full_json(self):
        session = CeleryDBAdapter.session()
        job_record = session.query(WinogradAutotuneJobsModel).filter_by(job_id=self.job_id).first()
        progress = job_record.progress
        status = job_record.status.value
        error_message = job_record.error_message
        project_record = session.query(ProjectsModel).filter_by(
            model_id=job_record.result_model_id,
            dataset_id=self.config.dataset_id,
            target=DevicesEnum(self.config.device),
            optimization_type=OptimizationTypesEnum(self.job.job_type.value)
        ).first()
        session.close()
        message = {
            'creationTimestamp': self.date,
            'jobId': self.job_id,
            'type': self.job.job_type.value,
            'config': self.config.json(),
            'projectId': project_record.id,
            'originalModelId': get_top_level_model_id(project_record.id),
            'status': {
                'name': status,
                'progress': progress
            }
        }
        if error_message:
            message['status']['errorMessage'] = error_message
        return message

    def add_stage(self, stage: IEmitMessageStage, silent: bool = False) -> str:
        super(WinogradAutotuneEmitMessage, self).add_stage(stage, silent=True)
        self.update_progress_in_database(self.local_progress)
        self.emit_message()
        return stage.name

    def update_progress_in_database(self, progress):
        session = CeleryDBAdapter.session()
        record = session.query(WinogradAutotuneJobsModel).filter_by(job_id=self.job_id).first()
        record.progress = progress
        write_record(record, session)
        session.close()
