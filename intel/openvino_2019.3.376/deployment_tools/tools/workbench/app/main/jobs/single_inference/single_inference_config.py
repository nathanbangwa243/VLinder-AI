"""
 OpenVINO Profiler
 Class for inference configuration data

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


class SingleInferenceConfig(IConfig):
    def __init__(self, session_id, data):
        super(SingleInferenceConfig, self).__init__(session_id)
        try:
            self.model_id = data['modelId']
            self.previous_job_id = data['previousJobId'] if 'previousJobId' in data else None
            self.dataset_id = data['datasetId']
            self.device = data['device']
            self.batch = data['batch']
            self.nireq = data['nireq']
            self.inference_time = data['inferenceTime']
        except KeyError as error:
            raise InconsistentConfigError(
                'Data of the request  does not contain field {} needed for running sample'.format(error))

    def json(self) -> dict:
        return {
            'modelId': self.model_id,
            'datasetId': self.dataset_id,
            'device': self.device,
            'batch': self.batch,
            'nireq': self.nireq,
            'inferenceTime': self.inference_time,
            'previousJobId': self.previous_job_id,
            'sessionId': self.session_id
        }
