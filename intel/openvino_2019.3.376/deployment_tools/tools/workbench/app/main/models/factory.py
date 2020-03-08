"""
 OpenVINO Profiler
 Classes and functions for adding records to datasets

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
from app.main.models.accuracy_model import AccuracyJobsModel
from app.main.models.compound_inference_model import CompoundInferenceJobsModel
from app.main.models.download_configs_model import DownloadConfigsModel
from app.main.models.enumerates import OptimizationTypesEnum, DevicesEnum
from app.main.models.int8_autotune_model import Int8AutotuneJobsModel
from app.main.models.jobs_model import JobsModel
from app.main.models.projects_model import ProjectsModel
from app.main.models.winograd_autotune_model import WinogradAutotuneJobsModel


def create_project(job_type: OptimizationTypesEnum, model_id: int, dataset_id: int, target: DevicesEnum, session):
    project = ProjectsModel.query.filter_by(model_id=model_id, dataset_id=dataset_id, target=target).first()
    if not project:
        project = ProjectsModel(model_id, dataset_id, target, job_type)
        write_record(project, session)
    return project.id


def get_job_by_id(job_id: int):
    common_job = JobsModel.query.get(job_id)
    tables = {
        CompoundInferenceJobsModel.__tablename__: CompoundInferenceJobsModel,
        Int8AutotuneJobsModel.__tablename__: Int8AutotuneJobsModel,
        WinogradAutotuneJobsModel.__tablename__: WinogradAutotuneJobsModel,
        AccuracyJobsModel.__tablename__: AccuracyJobsModel,
        DownloadConfigsModel.__tablename__: DownloadConfigsModel,
    }
    job = tables[common_job.job_type].query.get(job_id)
    return job or None


def write_record(record, session):
    session.add(record)
    session.commit()
