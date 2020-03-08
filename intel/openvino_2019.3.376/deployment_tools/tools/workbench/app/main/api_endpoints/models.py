"""
 OpenVINO Profiler
 Endpoints to work with models

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
import json

from flask import jsonify, request
from sqlalchemy import and_, desc

from app.main.models.files_model import FilesModel
from app.extensions_factories.database import get_db
from app.main.api_endpoints import MODELS_API, check_expired_jobs, adds_session_id
from app.main.api_endpoints.utils import try_load_configuration, on_new_chunk_received, delete_model_from_db, \
    prepare_data_for_mo_pipeline
from app.main.forms.model_optimizer import MOForm
from app.main.jobs.job_types_enum import JobTypesEnum
from app.main.models.accuracy_model import AccuracyJobsModel
from app.main.models.enumerates import SupportedFrameworksEnum, DevicesEnum, ModelSourceEnum, StatusEnum
from app.main.models.factory import write_record
from app.main.models.model_downloader_conversion_jobs_model import ModelDownloaderConversionJobsModel
from app.main.models.model_optimizer_job_model import ModelOptimizerJobModel
from app.main.models.model_optimizer_scan_model import ModelOptimizerScanJobsModel
from app.main.models.projects_model import ProjectsModel
from app.main.models.topologies_metadata_model import TopologiesMetaDataModel
from app.main.models.topologies_model import TopologiesModel
from app.main.models.topology_analysis_jobs_model import TopologyAnalysisJobsModel
from app.main.models.upload_jobs_model import UploadJobsModel
from app.main.tasks.task import TASK
from app.main.utils.safe_runner import safe_run
from app.main.utils.utils import create_empty_dir
from config.constants import UPLOAD_FOLDER_MODELS, ORIGINAL_FOLDER


@MODELS_API.route('/models/all', methods=['GET'])
@safe_run
@check_expired_jobs
def get_all_models():
    """
    Return original IRs.

    Returns a list of objects, representing original IR models.
    A model is returned, if its status is "running" or "ready",
    or if it failed in Model Optimizer.

    Used by the frontend to fill the models table on the wizard page.
    """

    models = (
        TopologiesModel.query
            .filter(and_(
            TopologiesModel.status.notin_([StatusEnum.queued, StatusEnum.cancelled]),
            TopologiesModel.framework == SupportedFrameworksEnum.openvino,
        ))
            .filter(TopologiesModel.optimized_from.is_(None))
    )
    result = []
    for model in models:
        model_dict = model.short_json()
        last_mo_job_record = (
            ModelOptimizerJobModel.query
                .filter_by(result_model_id=model.id)
                .order_by(desc(ModelOptimizerJobModel.creation_timestamp))
                .first()
        )
        last_conversion_job = (
            ModelDownloaderConversionJobsModel.query
                .filter_by(result_model_id=model.id)
                .order_by(desc(ModelDownloaderConversionJobsModel.creation_timestamp))
                .first()
        )
        mo_analyzed_job = (
            ModelOptimizerScanJobsModel.query
                .filter_by(topology_id=model.converted_from)
                .first()
        )
        # Erroneous models should be shown only if they failed in Model Optimizer.
        if (model.status == StatusEnum.error
                and (not last_mo_job_record or last_mo_job_record.status != StatusEnum.error)):
            continue
        if last_mo_job_record:
            model_dict['mo'] = {}
            if last_mo_job_record.mo_args:
                model_dict['mo']['params'] = MOForm.to_params(json.loads(last_mo_job_record.mo_args))
            model_dict['mo']['errorMessage'] = last_mo_job_record.detailed_error_message
        if last_conversion_job:
            model_dict['mo'] = {}
            if last_conversion_job.conversion_args:
                model_dict['mo']['params'] = {}
                model_dict['mo']['params']['dataType'] = json.loads(last_conversion_job.conversion_args)['precision']
        if mo_analyzed_job and mo_analyzed_job.information:
            model_dict['mo']['analyzedParams'] = mo_analyzed_job.short_json()['information']
        result.append(model_dict)
    return jsonify(result)


@MODELS_API.route('/model/<int:model_id>', methods=['GET'])
@safe_run
@check_expired_jobs
def model_info(model_id: int):
    model = TopologiesModel.query.get(model_id)
    if not model:
        return 'Model with id {} was not found on database'.format(model_id), 404
    return jsonify(model.json())


@MODELS_API.route('/model/<int:model_id>', methods=['POST'])
@adds_session_id
@safe_run
@check_expired_jobs
def update_model_advanced_configuration(session_id: str, model_id: int):
    config = request.get_json()

    try_load_configuration(config)

    dataset_id = config['datasetId']
    target = DevicesEnum(config['device'])

    model = TopologiesModel.query.get(model_id)
    if not model:
        return 'Model with id {} was not found in the database'.format(model_id), 404

    model.meta.advanced_configuration = json.dumps(config)
    write_record(model, get_db().session)

    affected_topologies_ids = [t.id for t in model.meta.topologies]
    projects = (
        ProjectsModel.query
            .filter(ProjectsModel.model_id.in_(affected_topologies_ids))
            .filter_by(dataset_id=dataset_id, target=target)
    )
    affected_projects_ids = [p.id for p in projects]

    for project_id in affected_projects_ids:
        data = {
            'session_id': session_id,
            'projectId': project_id,
            'accuracyConfig': ''
        }
        accuracy_job = AccuracyJobsModel(data)
        write_record(accuracy_job, get_db().session)
        TASK.apply_async(args=(None, JobTypesEnum.accuracy_type.value, accuracy_job.job_id))

    return jsonify({'modelIds': affected_topologies_ids, 'projectIds': affected_projects_ids})


@MODELS_API.route('/model/<int:model_id>', methods=['PUT'])
@safe_run
@check_expired_jobs
def set_model_advanced_configuration(model_id: int):
    config = request.get_json()

    try_load_configuration(config)

    model = TopologiesModel.query.get(model_id)
    if not model:
        return 'Model with id {} was not found in the database'.format(model_id), 404

    model.meta.task_type = config['taskType']
    model.meta.topology_type = config['taskMethod']
    model.meta.advanced_configuration = json.dumps(config)
    write_record(model, get_db().session)

    return jsonify(model.short_json())


@MODELS_API.route('/model-upload', methods=['POST'])
@safe_run
@adds_session_id
@check_expired_jobs
def upload_model(session_id: str):
    data = request.get_json()

    model_name = data['modelName']
    framework = SupportedFrameworksEnum(data['framework'])
    files = data['files']

    metadata = TopologiesMetaDataModel()
    write_record(metadata, get_db().session)

    topology = TopologiesModel(model_name, framework, metadata.id, session_id)
    topology.source = ModelSourceEnum.ir if framework == SupportedFrameworksEnum.openvino else ModelSourceEnum.original
    write_record(topology, get_db().session)
    topology.path = os.path.join(UPLOAD_FOLDER_MODELS, str(topology.id), ORIGINAL_FOLDER)
    write_record(topology, get_db().session)
    create_empty_dir(topology.path)

    upload_job = UploadJobsModel({'session_id': session_id, 'artifactId': topology.id})
    write_record(upload_job, get_db().session)

    files_ids = FilesModel.create_files(files, topology.id, session_id)
    topology.size = round(sum(f.size for f in topology.files) / 2 ** (10 * 2))  # bytes / 2**10 = mb
    write_record(topology, get_db().session)
    result = {'modelItem': topology.short_json(), 'files': files_ids, }

    if framework != SupportedFrameworksEnum.openvino:
        converted_topology, model_optimizer_job = prepare_data_for_mo_pipeline(topology, upload_job.job_id, session_id)
        result['modelItem'] = converted_topology.short_json()
        result['modelItem']['modelOptimizerJobId'] = model_optimizer_job.job_id
    else:
        analysis_data = TopologyAnalysisJobsModel(
            {'session_id': session_id, 'model_id': topology.id, 'previousJobId': upload_job.job_id, })
        write_record(analysis_data, get_db().session)
    result['modelItem']['originalModelFramework'] = framework.value
    return jsonify(result)


@MODELS_API.route('/model-upload/<int:file_id>', methods=['POST'])
@safe_run
@check_expired_jobs
def write_model(file_id: int):
    file_record = FilesModel.query.get(file_id)
    if not file_record:
        return 'File record with id {} was not found on the database'.format(file_id), 404
    res = on_new_chunk_received(request, file_id)
    return jsonify(res)


@MODELS_API.route('/model/<int:model_id>', methods=['DELETE'])
@safe_run
@check_expired_jobs
def delete_model(model_id: int):
    delete_model_from_db(model_id)
    return jsonify({'id': model_id})
