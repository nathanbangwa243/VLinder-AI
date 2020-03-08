"""
 OpenVINO Profiler
 An ORM entity that stores topology metadata.

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
from sqlalchemy import Integer, Column, Text
from sqlalchemy.orm import relationship

from app.main.models.enumerates import TASK_ENUM_SCHEMA, TASK_METHOD_ENUM_SCHEMA
from app.main.models.base_model import BaseModel


class TopologiesMetaDataModel(BaseModel):
    __tablename__ = 'topologies_metadata'

    id = Column(Integer, primary_key=True, autoincrement=True)

    topology_type = Column(TASK_METHOD_ENUM_SCHEMA, nullable=True)
    task_type = Column(TASK_ENUM_SCHEMA, nullable=True)
    advanced_configuration = Column(Text, nullable=True)

    topologies = relationship('TopologiesModel', back_populates='meta', lazy='joined')
