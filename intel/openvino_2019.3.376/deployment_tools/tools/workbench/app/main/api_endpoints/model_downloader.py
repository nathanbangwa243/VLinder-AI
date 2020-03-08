"""
 OpenVINO Profiler
 Endpoints to work with model downloader

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
import os

from celery import chain
from flask import request, jsonify

from app.extensions_factories.database import get_db
from app.main.api_endpoints import adds_session_id, MODELS_API
from app.main.console_tool_wrapper.model_downloader.utils import fetch_downloadable_models
from app.main.jobs.job_types_enum import JobTypesEnum
from app.main.models.enumerates import TaskEnum, ModelPrecisionEnum, SupportedFrameworksEnum, ModelSourceEnum
from app.main.models.factory import write_record
from app.main.models.model_downloader_conversion_jobs_model import ModelDownloaderConversionJobsModel
from app.main.models.model_downloader_model import ModelDownloaderModel
from app.main.models.omz_topology_model import OMZTopologyModel
from app.main.models.topologies_metadata_model import TopologiesMetaDataModel
from app.main.models.topologies_model import TopologiesModel
from app.main.models.topology_analysis_jobs_model import TopologyAnalysisJobsModel
from app.main.tasks.task import TASK
from app.main.utils.safe_runner import safe_run
from app.utils.jobs_weights import JobsWeight
from config.constants import ORIGINAL_FOLDER, UPLOAD_FOLDER_MODELS, MODEL_DOWNLOADS_FOLDER


@MODELS_API.route('/downloader-models', methods=['GET'])
@safe_run
def list_download_models():
    if not OMZTopologyModel.query.all():
        fetch_downloadable_models()
    supported_topologies = OMZTopologyModel. \
        query. \
        filter(OMZTopologyModel.task_type.in_([TaskEnum.object_detection, TaskEnum.classification])). \
        filter(OMZTopologyModel.precision.notin_([ModelPrecisionEnum.i1, ])). \
        all()
    return jsonify([model.json() for model in supported_topologies])


@MODELS_API.route('/downloader-models', methods=['POST'])
@adds_session_id
@safe_run
def download_model(session_id: str):
    data = request.get_json()
    precision = ModelPrecisionEnum(data['precision'])
    model_name = data['modelName']

    topology = OMZTopologyModel.query.filter_by(name=model_name, precision=precision).first()

    metadata = TopologiesMetaDataModel()
    write_record(metadata, get_db().session)
    new_model = TopologiesModel(model_name, SupportedFrameworksEnum.openvino, metadata.id, session_id)
    new_model.source = ModelSourceEnum.omz
    new_model.precision = precision
    new_model.downloaded_from = topology.id
    write_record(new_model, get_db().session)

    new_model.path = os.path.join(UPLOAD_FOLDER_MODELS, str(new_model.id), ORIGINAL_FOLDER)

    new_model.meta.task_type = topology.task_type
    new_model.meta.topology_type = topology.topology_type
    new_model.meta.advanced_configuration = topology.advanced_configuration
    write_record(new_model, get_db().session)

    new_model_json = new_model.short_json()
    new_model_json['session_id'] = session_id

    tasks = []

    weights = JobsWeight.download_model()

    download_job_record = ModelDownloaderModel(new_model_json)
    download_job_record.result_model_id = new_model.id

    write_record(download_job_record, get_db().session)
    tasks.append(TASK.subtask(args=(None, JobTypesEnum.model_downloader_type.value, download_job_record.job_id),
                              kwargs={'progress_weight': weights[JobTypesEnum.model_downloader_type]}))
    analysis_data = TopologyAnalysisJobsModel({
        'session_id': session_id,
        'model_id': new_model.id,
    })
    write_record(analysis_data, get_db().session)

    if topology.framework != SupportedFrameworksEnum.openvino:
        weights = JobsWeight.download_source_model()

        convert_job_record = ModelDownloaderConversionJobsModel(new_model_json)
        convert_job_record.result_model_id = new_model.id
        convert_job_record.parent_job = download_job_record.job_id
        write_record(convert_job_record, get_db().session)

        converter_args = [JobTypesEnum.model_convert_type.value, convert_job_record.job_id]
        tasks.append(TASK.subtask(args=converter_args,
                                  kwargs={'progress_weight': weights[JobTypesEnum.model_convert_type]}))
        analysis_data.parent_job = convert_job_record.job_id
    else:
        weights = JobsWeight.download_openvino_model()
        analysis_data.parent_job = download_job_record.job_id
    write_record(analysis_data, get_db().session)
    source_path = os.path.join(MODEL_DOWNLOADS_FOLDER, str(new_model.id), topology.path)
    destination_path = new_model.path

    ir_postprocessing(tasks, source_path, destination_path, new_model.id, weights)

    chain(tasks).apply_async()

    result = new_model.short_json()
    result['originalModelFramework'] = topology.framework.value
    return jsonify(result)


def convert_downloaded_model(data: dict):
    topology_id = data['topologyId']
    topology = TopologiesModel.query.get(topology_id)
    topology.precision = ModelPrecisionEnum(data['dataType'])
    omz_topology = OMZTopologyModel.query.filter_by(name=topology.name).first()
    convert_job_record = ModelDownloaderConversionJobsModel.query.filter_by(result_model_id=topology_id).first()
    convert_job_record.conversion_args = json.dumps(({
        'precision': data['dataType'],
    }))
    write_record(convert_job_record, get_db().session)
    weights = JobsWeight.download_source_model()
    tasks = [TASK.subtask(args=[None, JobTypesEnum.model_convert_type.value, convert_job_record.job_id],
                          kwargs={'progress_weight': weights[JobTypesEnum.model_convert_type]}), ]
    source_path = os.path.join(MODEL_DOWNLOADS_FOLDER, str(topology_id), omz_topology.path)
    destination_path = topology.path
    ir_postprocessing(tasks, source_path, destination_path, topology.id, weights)

    chain(tasks).apply_async()
    return jsonify({})


def ir_postprocessing(tasks: list, source_path: str, destination_path: str, job_id: int, weight: dict):
    move_file_args = [JobTypesEnum.move_model_from_downloader_type.value, job_id]
    if not tasks:
        move_file_args.insert(0, None)
    tasks.append(TASK.subtask(args=move_file_args,
                              kwargs={'data': {
                                  'sourcePath': source_path,
                                  'destinationPath': destination_path
                              },
                                  'progress_weight': weight[JobTypesEnum.move_model_from_downloader_type]
                              }))
    tasks.append(TASK.subtask(args=(JobTypesEnum.model_analyzer_type.value, job_id),
                              kwargs={'progress_weight': weight[JobTypesEnum.model_analyzer_type], }))
