"""
 OpenVINO Profiler
 Class for ORM model described a dataset generation config

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
from sqlalchemy import Column, Integer, ForeignKey, String, JSON

from app.main.models.base_model import BaseModel
from app.main.models.enumerates import StatusEnum, STATUS_ENUM_SCHEMA


class DatasetGenerationConfigsModel(BaseModel):
    __tablename__ = 'dataset_generation_configs'

    result_dataset_id = Column(Integer, ForeignKey('datasets.id'), primary_key=True)
    number_images = Column(Integer, nullable=False)
    channels = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    dist_law = Column(String, nullable=False)
    dist_law_params = Column(JSON, nullable=False)

    status = Column(STATUS_ENUM_SCHEMA, nullable=False, default=StatusEnum.queued)
    error_message = Column(String, nullable=True)

    # pylint: disable=too-many-arguments
    def __init__(self, dataset_id, number_images, channels, width, height, dist_law, params_dist):
        self.result_dataset_id = dataset_id
        self.number_images = number_images
        self.channels = channels
        self.width = width
        self.height = height
        self.dist_law = dist_law
        self.dist_law_params = params_dist

    def json(self) -> dict:
        return {
            'numberOfImages': self.number_images,
            'channels': self.channels,
            'width': self.width,
            'height': self.height,
            'distLaw': self.dist_law,
            'distLawParams': self.dist_law_params
        }
