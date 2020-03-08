"""
 OpenVINO Profiler
 Class for ORM model described an Datasets

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
from sqlalchemy import Integer, Column, ForeignKey

from app.main.models.artifacts_model import ArtifactsModel
from app.main.models.enumerates import DATASET_TYPES_ENUM_SCHEMA


class DatasetsModel(ArtifactsModel):
    __tablename__ = 'datasets'

    id = Column(Integer, ForeignKey('artifacts.id'), primary_key=True)
    dataset_type = Column(DATASET_TYPES_ENUM_SCHEMA, nullable=True)

    def short_json(self):
        return self.json()

    def json(self):
        return {
            **super().json(),
            'type': self.dataset_type.value if self.dataset_type else None,
            'readiness': self.status.value
        }
