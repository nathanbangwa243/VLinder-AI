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
from flask import jsonify, request

from app.extensions_factories.database import get_db
from app.main.api_endpoints import adds_session_id, check_expired_jobs, ACCURACY_API
from app.main.jobs.job_types_enum import JobTypesEnum
from app.main.jobs.utils.yml_templates.config_converter import AccuracyConfigConverter
from app.main.jobs.utils.yml_templates.utils import get_default_config_for_topologies
from app.main.models.factory import write_record
from app.main.models.accuracy_model import AccuracyJobsModel
from app.main.utils.safe_runner import safe_run
from app.main.tasks.task import TASK


@ACCURACY_API.route('/accuracy', methods=['POST'])
@safe_run
@adds_session_id
@check_expired_jobs
def run_accuracy_check(session_id: str):
    data = request.get_json()
    data['session_id'] = session_id
    data['projectId'] = data['projectId']
    data['accuracyConfig'] = ''

    accuracy_job = AccuracyJobsModel(data)

    write_record(accuracy_job, get_db().session)
    TASK.apply_async(args=(None, JobTypesEnum.accuracy_type.value, accuracy_job.job_id),
                     task_id=str(accuracy_job.job_id))
    return jsonify({'jobId': accuracy_job.job_id})


@ACCURACY_API.route('/default-accuracy-configs', methods=['GET'])
@safe_run
def get_defaults_accuracy():
    result = []
    for item in get_default_config_for_topologies():
        result.append(AccuracyConfigConverter.from_accuracy_representation(item))
    return jsonify(result)
