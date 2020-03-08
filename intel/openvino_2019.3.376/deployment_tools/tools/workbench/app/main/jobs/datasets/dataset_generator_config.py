"""
 OpenVINO Profiler
 Class for dataset generation configuration

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
from app.error.inconsistent_upload_error import InconsistentDatasetError
from app.main.jobs.datasets.dataset_upload_config import DatasetUploadConfig


class DatasetGeneratorConfig(DatasetUploadConfig):
    def __init__(self, session_id: str, data: dict):
        super().__init__(session_id, data)
        try:
            self.size = data['numberOfImages']
            self.channels = data['channels']
            self.width = data['width']
            self.height = data['height']
            self.dist_law = data['distLaw']
            self.params_dist = data['distLawParams']
        except KeyError as error:
            raise InconsistentDatasetError('Config of the dataset does not contain {}'.format(str(error)))

    def json(self) -> dict:
        return {
            'numberOfImages': self.size,
            'name': self.name,
            'channels': self.channels,
            'width': self.width,
            'height': self.height,
            'distLaw': self.dist_law,
            'sessionId': self.session_id,
        }
