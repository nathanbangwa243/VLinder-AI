"""
 OpenVINO Profiler
 Class for storing int8 calibration parameters

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


class Int8AutoTuneConfig(IConfig):
    def __init__(self, session_id: str, data: dict):
        super(Int8AutoTuneConfig, self).__init__(session_id)
        try:
            self.sample_type = data['taskType']
            self.model_id = data['modelId']
            self.dataset_id = data['datasetId']
            self.threshold = data['threshold']
            self.device = 'CPU'
            self.batch = data['batch']
        except KeyError as error:
            raise InconsistentConfigError(
                'Data of the request  does not contain field {} needed for running sample'.format(error))

    def json(self) -> dict:
        return {
            'modelType': self.sample_type,
            'modelId': self.model_id,
            'datasetId': self.dataset_id,
            'device': self.device,
            'threshold': self.threshold,
            'batch': self.batch,
            'jobId': self.previous_job_id,
            'sessionId': self.session_id
        }
