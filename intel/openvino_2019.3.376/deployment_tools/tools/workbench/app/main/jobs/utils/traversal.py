"""
 OpenVINO Profiler
 Utils to traverse through models

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
from app.main.models.projects_model import ProjectsModel
from app.main.models.topologies_model import TopologiesModel


def get_top_level_model_id(project_id: int) -> int:
    session = CeleryDBAdapter.session()
    project = session.query(ProjectsModel).get(project_id)
    model_id = project.model_id
    while True:
        parent_model = session.query(TopologiesModel).get(model_id)
        if not parent_model.optimized_from:
            session.close()
            return parent_model.id
        model_id = parent_model.optimized_from
