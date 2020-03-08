"""
 OpenVINO Profiler
 Endpoints to work with inference utilities

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
import os
from celery import chain
from flask import jsonify, request
from app.extensions_factories.database import get_db
from app.main.api_endpoints import adds_session_id, check_expired_jobs, INT8AUTOTUNE_API
from app.main.api_endpoints.inference import generate_queue_of_single_inference_tasks
from app.main.api_endpoints.utils import delete_model_from_db
from app.main.jobs.job_types_enum import JobTypesEnum
from app.main.models.compound_inference_model import CompoundInferenceJobsModel
from app.main.models.enumerates import OptimizationTypesEnum, ModelPrecisionEnum, SupportedFrameworksEnum, StatusEnum, \
    DevicesEnum, ModelSourceEnum
from app.main.models.factory import create_project, write_record
from app.main.models.int8_autotune_model import Int8AutotuneJobsModel
from app.main.models.topologies_model import TopologiesModel
from app.main.models.topology_analysis_jobs_model import TopologyAnalysisJobsModel
from app.main.tasks.task import TASK
from app.main.utils.safe_runner import safe_run
from app.utils.jobs_weights import JobsWeight
from config.constants import ORIGINAL_FOLDER


@INT8AUTOTUNE_API.route('/calibrate', methods=['POST'])
@safe_run
@adds_session_id
@check_expired_jobs
def run_int8autotune(session_id: str):
    data = request.get_json()
    int8_data = data['int8AutotuneConfig']
    compound_inference_data = data['compoundInferenceConfig']

    model_id = int8_data['modelId']
    dataset_id = int8_data['datasetId']
    device = DevicesEnum(int8_data['device'])

    project_id = create_project(OptimizationTypesEnum.int8autotune, model_id, dataset_id, device, get_db().session)

    original_model = TopologiesModel.query.get(model_id)

    int8_data['session_id'] = session_id
    int8_data['projectId'] = project_id

    compound_inference_data['session_id'] = session_id

    int8_data['taskType'] = original_model.meta.task_type
    int8_data['taskMethod'] = original_model.meta.topology_type
    int8_data['calibrationConfig'] = ''
    int8_job = Int8AutotuneJobsModel(int8_data)
    write_record(int8_job, get_db().session)

    model_path = original_model.path
    if ORIGINAL_FOLDER in original_model.path:
        model_path = os.path.dirname(original_model.path)
    tuned_path = os.path.join(model_path, str(int8_job.job_id))

    new_int8_model = TopologiesModel(
        name='{}_{}'.format(original_model.name, int8_job.job_id),
        framework=SupportedFrameworksEnum.openvino,
        metadata_id=original_model.metadata_id,
        session_id=session_id
    )

    new_int8_model.path = tuned_path
    new_int8_model.optimized_from = original_model.id
    new_int8_model.precision = ModelPrecisionEnum.mixed
    new_int8_model.status = StatusEnum.running
    new_int8_model.source = ModelSourceEnum.ir
    write_record(new_int8_model, get_db().session)

    # check existing projects
    model_id = new_int8_model.id
    dataset_id = compound_inference_data['datasetId']
    device = DevicesEnum(compound_inference_data['device'])
    inference_project_id = create_project(OptimizationTypesEnum.int8autotune, model_id, dataset_id, device,
                                          get_db().session)

    int8_job = Int8AutotuneJobsModel.query.get(int8_job.job_id)
    int8_job.result_model_id = model_id
    write_record(int8_job, get_db().session)
    analysis_data = TopologyAnalysisJobsModel({
        'model_id': new_int8_model.id,
        'session_id': session_id,
        'previousJobId': int8_job.job_id,
    })
    write_record(analysis_data, get_db().session())
    infer_data = {
        **compound_inference_data,
        'previousJobId': int8_job.job_id,
        'projectId': inference_project_id
    }
    infer_job = CompoundInferenceJobsModel(infer_data)
    write_record(infer_job, get_db().session)

    weights = JobsWeight.int8_model()
    tasks = list()
    tasks.append(TASK.subtask(args=(None, JobTypesEnum.int8autotune_type.value, int8_job.job_id),
                              kwargs={
                                  'progress_weight': weights[JobTypesEnum.int8autotune_type]
                              }))
    tasks.append(TASK.subtask(args=(JobTypesEnum.model_analyzer_type.value, model_id),
                              kwargs={
                                  'progress_weight': weights[JobTypesEnum.model_analyzer_type],
                              }
                              ))
    tasks[0].on_failure = lambda: delete_model_from_db(int8_job.job_id)
    tasks_queue = generate_queue_of_single_inference_tasks(infer_data, infer_job.job_id,
                                                           start_tasks=tasks)

    chain(tasks_queue).apply_async()

    return jsonify({'jobId': int8_job.job_id})
