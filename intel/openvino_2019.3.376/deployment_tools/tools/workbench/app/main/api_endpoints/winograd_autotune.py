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

from app.error.entry_point_error import InconsistentConfigError
from app.extensions_factories.database import get_db
from app.main.api_endpoints import adds_session_id, check_expired_jobs, WINOGRAD_AUTOTUNE_API
from app.main.api_endpoints.inference import generate_queue_of_single_inference_tasks
from app.main.jobs.job_types_enum import JobTypesEnum
from app.main.models.compound_inference_model import CompoundInferenceJobsModel
from app.main.models.enumerates import OptimizationTypesEnum, DevicesEnum, SupportedFrameworksEnum, StatusEnum
from app.main.models.factory import create_project, write_record
from app.main.models.topologies_model import TopologiesModel
from app.main.models.topology_analysis_jobs_model import TopologyAnalysisJobsModel
from app.main.models.winograd_autotune_model import WinogradAutotuneJobsModel
from app.main.tasks.task import TASK
from app.main.utils.safe_runner import safe_run
from config.constants import ORIGINAL_FOLDER


@WINOGRAD_AUTOTUNE_API.route('/winograd', methods=['POST'])
@safe_run
@adds_session_id
@check_expired_jobs
def run_winograd_autotune(session_id: str):
    data = request.get_json()

    winograd_data = data['winogradAutotuneConfig']
    compound_inference_data = data['compoundInferenceConfig']

    model_id = winograd_data['modelId']
    dataset_id = winograd_data['datasetId']
    device = DevicesEnum.cpu

    project_id = create_project(OptimizationTypesEnum.winograd_autotune,
                                model_id,
                                dataset_id,
                                device,
                                get_db().session)

    original_model = TopologiesModel.query.get(model_id)

    winograd_data['session_id'] = session_id
    winograd_data['projectId'] = project_id

    compound_inference_data['session_id'] = session_id

    winograd_job = WinogradAutotuneJobsModel(winograd_data)

    write_record(winograd_job, get_db().session)

    model_path = original_model.path
    if ORIGINAL_FOLDER in original_model.path:
        model_path = os.path.dirname(original_model.path)
    tuned_model_path = os.path.join(model_path, str(winograd_job.job_id))

    new_winograd_model = TopologiesModel(
        name='{}_{}'.format(original_model.name, winograd_job.job_id),
        framework=SupportedFrameworksEnum.openvino,
        metadata_id=original_model.metadata_id,
        session_id=session_id
    )
    new_winograd_model.path = tuned_model_path
    new_winograd_model.optimized_from = original_model.id
    new_winograd_model.precision = original_model.precision
    new_winograd_model.status = StatusEnum.running
    write_record(new_winograd_model, get_db().session)

    winograd_model_id = new_winograd_model.id
    dataset_id = compound_inference_data['datasetId']
    device = DevicesEnum(compound_inference_data['device'])
    if device != DevicesEnum.cpu:
        raise InconsistentConfigError(message='Device {} does not support Winograd optimization'.format(device.value))

    inference_project_id = create_project(OptimizationTypesEnum.winograd_autotune,
                                          winograd_model_id,
                                          dataset_id,
                                          device,
                                          get_db().session)

    winograd_job = WinogradAutotuneJobsModel.query.get(winograd_job.job_id)
    winograd_job.result_model_id = winograd_model_id
    write_record(winograd_job, get_db().session)

    infer_data = {
        **compound_inference_data,
        'previousJobId': winograd_job.job_id,
        'projectId': inference_project_id
    }

    inference_job = CompoundInferenceJobsModel(infer_data)
    write_record(inference_job, get_db().session)
    analysis_data = TopologyAnalysisJobsModel({
        'model_id': winograd_model_id,
        'session_id': session_id,
        'previousJobId': winograd_job.job_id,
    })
    write_record(analysis_data, get_db().session())
    tasks = list()
    tasks.append(TASK.subtask(args=(None,
                                    JobTypesEnum.winograd_autotune_type.value,
                                    winograd_job.job_id)))
    tasks.append(TASK.subtask(args=(JobTypesEnum.model_analyzer_type.value,
                                    winograd_model_id)))

    tasks_queue = generate_queue_of_single_inference_tasks(infer_data, inference_job.job_id,
                                                           start_tasks=tasks)

    chain(tasks_queue).apply_async()

    return jsonify({'jobId': winograd_job.job_id})
