"""
 OpenVINO Profiler
 Model Optimizer related endpoints

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

import json

from celery import chain
from flask import jsonify, request

from app.extensions_factories.database import get_db
from app.main.api_endpoints import MODEL_OPTIMIZER_API, check_expired_jobs, adds_session_id
from app.main.api_endpoints.model_downloader import convert_downloaded_model
from app.main.api_endpoints.utils import save_pipeline_config
from app.main.forms.model_optimizer import MOForm
from app.main.jobs.job_types_enum import JobTypesEnum
from app.main.jobs.utils.database_functions import set_status_in_db
from app.main.models.enumerates import StatusEnum
from app.main.models.factory import write_record
from app.main.models.model_optimizer_job_model import ModelOptimizerJobModel
from app.main.models.topologies_model import TopologiesModel
from app.main.tasks.task import TASK
from app.main.utils.safe_runner import safe_run
from app.utils.jobs_weights import JobsWeight


def convert(mo_job_record: ModelOptimizerJobModel, data: dict, chain_progress_weight: dict):
    """Validate MO params, prepare them, update MO job record and launch MO chain."""

    pipeline_config = data.get('pipelineConfigFile', None)
    if pipeline_config:
        del data['pipelineConfigFile']
        save_pipeline_config(pipeline_config, mo_job_record.original_topology_id)
    mo_form = MOForm(data, mo_job_record.original_topology.framework.value)
    if mo_form.is_invalid:
        set_status_in_db(ModelOptimizerJobModel, mo_job_record.job_id, StatusEnum.error, get_db().session)
        set_status_in_db(TopologiesModel, mo_job_record.result_model_id, StatusEnum.error, get_db().session)
        return jsonify({'errors': mo_form.errors}), 400

    mo_job_record.mo_args = json.dumps(mo_form.get_args())
    write_record(mo_job_record, get_db().session)

    chain([
        TASK.subtask(
            args=(None, JobTypesEnum.model_optimizer_type.value, mo_job_record.job_id),
            kwargs={'progress_weight': chain_progress_weight[JobTypesEnum.model_optimizer_type]}
        ),
        TASK.subtask(
            args=(JobTypesEnum.model_analyzer_type.value, mo_job_record.result_model_id),
            kwargs={
                'progress_weight': chain_progress_weight[JobTypesEnum.model_analyzer_type],
            }
        )
    ]).apply_async()

    return jsonify({
        'irId': mo_job_record.result_model_id,
        'modelOptimizerJobId': mo_job_record.job_id,
    })


@MODEL_OPTIMIZER_API.route('/convert', methods=['POST'])
@safe_run
@check_expired_jobs
def convert_update():
    """
    Update MO args record and launch MO job.

    Launches conversion with specified params
    for existing IR and MO job record.

    Used as part of original model uploading flow.
    """

    data = request.get_json()

    # pylint: disable=fixme
    # TODO: Extract this condition to separate endpoint
    if data.get('topologyId', None):
        convert_downloaded_model(data)
        return jsonify({})

    mo_job_id = data.pop('modelOptimizerJobId')
    mo_job_record = ModelOptimizerJobModel.query.get(mo_job_id)
    if not mo_job_record:
        return 'Model optimisation record with id {} was not found in the database'.format(mo_job_id), 404
    return convert(mo_job_record, data, JobsWeight.upload_and_convert_openvino_model())


@MODEL_OPTIMIZER_API.route('/convert-edit', methods=['POST'])
@safe_run
@adds_session_id
@check_expired_jobs
def convert_edit(session_id):
    """Rerun IR conversion with changed MO params."""

    data = request.get_json()
    topology_id = data.pop('irId')

    topology = TopologiesModel.query.get(topology_id)
    if not topology:
        return 'Model with id {} was not found in the database'.format(topology_id), 404

    mo_job_record = ModelOptimizerJobModel({
        'original_topology_id': topology.converted_from,
        'result_model_id': topology_id,
        'session_id': session_id,
    })
    write_record(mo_job_record, get_db().session)

    topology.progress = 0
    topology.status = StatusEnum.queued
    topology.error_message = None

    return convert(mo_job_record, data, JobsWeight.model_optimizer())
