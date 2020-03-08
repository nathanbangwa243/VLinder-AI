"""
 OpenVINO Profiler
 Endpoints to work with inference

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
from celery import chain
from flask import jsonify, request

from app.extensions_factories.database import get_db
from app.main.api_endpoints import adds_session_id, check_expired_jobs, INFERENCE_API
from app.main.api_endpoints.utils import generate_queue_of_single_inference_tasks, get_unique_job_configs, \
    get_jobs_for_config
from app.main.jobs.utils.traversal import get_top_level_model_id
from app.main.models.compound_inference_model import CompoundInferenceJobsModel
from app.main.models.enumerates import OptimizationTypesEnum, DevicesEnum
from app.main.models.factory import create_project, write_record
from app.main.utils.safe_runner import safe_run


@INFERENCE_API.route('/inference', methods=['POST'])
@safe_run
@adds_session_id
@check_expired_jobs
def run_compound_inference(session_id: str):
    data = request.get_json()
    data['session_id'] = session_id
    model_id = data['modelId']
    dataset_id = data['datasetId']
    device = DevicesEnum(data['device'])
    project_id = create_project(OptimizationTypesEnum.inference, model_id, dataset_id, device, get_db().session)
    data['projectId'] = project_id
    job_record = CompoundInferenceJobsModel(data)
    write_record(job_record, get_db().session)
    tasks_queue = generate_queue_of_single_inference_tasks(data, job_record.job_id)
    chain(tasks_queue).apply_async()
    original_model_id = get_top_level_model_id(project_id)
    return jsonify({
        'jobId': job_record.job_id,
        'projectId': project_id,
        'originalModelId': original_model_id
    })


@INFERENCE_API.route('/inference-history/<project_id>', methods=['GET'])
@safe_run
@adds_session_id
def get_jobs_by_project_id(session_id: str, project_id):
    result = []
    configs = get_unique_job_configs(project_id)
    for config in configs:
        result.append(get_jobs_for_config(config, project_id, session_id))
    return jsonify(result)
