"""
 OpenVINO Profiler
 Class for storing winograd parameters

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
from app.error.entry_point_error import InconsistentConfigError
from app.main.jobs.interfaces.iconfig import IConfig
from app.main.models.enumerates import DevicesEnum


class WinogradAutotuneConfig(IConfig):
    def __init__(self, session_id: str, data: dict):
        super(WinogradAutotuneConfig, self).__init__(session_id)
        try:
            self.model_id = data['modelId']
            self.dataset_id = data['datasetId']
            self.device = DevicesEnum.cpu.value
        except KeyError as error:
            raise InconsistentConfigError(
                'Data of the request  does not contain field {} needed for running sample'.format(error))

    def json(self) -> dict:
        return {
            'modelId': self.model_id,
            'datasetId': self.dataset_id,
        }
