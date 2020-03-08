"""
 OpenVINO Profiler
 Class for emit message of tuned model downloading

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


from app.main.job_factory.config import CeleryDBAdapter
from app.main.jobs.interfaces.iconfig import IConfig
from app.main.jobs.interfaces.iemit_message import IEmitMessage, IEmitMessageStage
from app.main.models.download_configs_model import DownloadConfigsModel
from app.main.models.enumerates import StatusEnum


class DownloadModelEmitMessage(IEmitMessage):
    event = 'download'
    namespace = '/download'

    def __init__(self, job, job_id: int, content: IConfig, weight: float):
        super(DownloadModelEmitMessage, self).__init__(job, job_id, content, weight=weight)
        self.status = None
        self.stage = self.add_stage(IEmitMessageStage('download', 'download'), silent=True)

    def full_json(self):
        session = CeleryDBAdapter.session()
        config = session.query(DownloadConfigsModel).get(self.job_id)
        status = config.status.value
        error_message = config.error_message
        project_id = config.project_id
        path = config.path
        name = config.name
        session.close()
        message = {
            **super(DownloadModelEmitMessage, self).full_json(),
            'projectId': project_id,
            'path': path,
            'name': name,
            'status': {
                'progress': self.total_progress,
                'name': status
            }
        }
        if error_message:
            message['status']['errorMessage'] = error_message
        return message

    def update_progress(self, name, new_value):
        if new_value == 100:
            self.status = StatusEnum.ready.value
        super(DownloadModelEmitMessage, self).update_progress(name, new_value)

    @property
    def local_progress(self):
        return self.get_current_job().progress
