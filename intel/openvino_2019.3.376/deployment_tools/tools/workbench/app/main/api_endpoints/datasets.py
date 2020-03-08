"""
 OpenVINO Profiler
 Endpoints to work with datasets

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

from flask import jsonify, request

from app.extensions_factories.database import get_db
from app.main.api_endpoints import check_expired_jobs, adds_session_id, DATASETS_API
from app.main.api_endpoints.utils import delete_dataset_from_db, on_new_chunk_received
from app.main.jobs.job_types_enum import JobTypesEnum
from app.main.models.enumerates import StatusEnum, DatasetTypesEnum
from app.main.models.dataset_generation_configs_model import DatasetGenerationConfigsModel
from app.main.models.datasets_model import DatasetsModel
from app.main.models.factory import write_record
from app.main.models.files_model import FilesModel
from app.main.tasks.task import TASK
from app.main.utils.safe_runner import safe_run
from app.main.utils.utils import get_dataset_folder, create_empty_dir
from config.constants import UPLOADS_FOLDER


@DATASETS_API.route("/datasets", methods=['GET'])
@safe_run
@check_expired_jobs
def datasets():
    list_datasets = DatasetsModel.query.filter(DatasetsModel.status.notin_([StatusEnum.queued,
                                                                            StatusEnum.cancelled,
                                                                            StatusEnum.error])).all()
    return jsonify([dataset.json() for dataset in list_datasets])


@DATASETS_API.route("/dataset/<dataset_id>", methods=['GET'])
@safe_run
@check_expired_jobs
def dataset_info(dataset_id):
    return jsonify(DatasetsModel.query.get(dataset_id).json())


@DATASETS_API.route("/dataset-generate", methods=['POST'])
@safe_run
@adds_session_id
@check_expired_jobs
def generate_dataset(session_id: str):
    number_images = request.get_json()['numberOfImages']
    name = request.get_json()['datasetName']
    channels = request.get_json()['channels']
    width = request.get_json()['width']
    height = request.get_json()['height']
    dist_law = request.get_json()['distLaw']
    params_dist = request.get_json()['distLawParams']
    dataset = DatasetsModel(name, session_id)
    dataset.dataset_type = DatasetTypesEnum.imagenet.value
    write_record(dataset, get_db().session)
    dataset.path = get_dataset_folder(str(dataset.id))
    write_record(dataset, get_db().session)
    config = DatasetGenerationConfigsModel(dataset.id, number_images, channels, width, height, dist_law, params_dist)
    write_record(config, get_db().session)
    TASK.apply_async((None, JobTypesEnum.add_generated_dataset_type.value, dataset.id), task_id=str(dataset.id))
    return jsonify(dataset.json())


@DATASETS_API.route('/dataset-upload', methods=['POST'])
@safe_run
@adds_session_id
@check_expired_jobs
def create_dataset(session_id: str):
    data = request.get_json()
    name = data['datasetName']
    files = data['files']
    dataset = DatasetsModel(name=name, session_id=session_id)
    write_record(dataset, get_db().session)
    dataset.path = os.path.join(UPLOADS_FOLDER, str(dataset.id))
    write_record(dataset, get_db().session)
    files_ids = FilesModel.create_files(files, dataset.id, session_id)
    dataset.size = round(sum([f.size for f in dataset.files]) / (1024 ** 2))
    write_record(dataset, get_db().session)
    create_empty_dir(dataset.path)
    return jsonify({'datasetItem': dataset.short_json(), 'files': files_ids, })


@DATASETS_API.route("/dataset-upload/<int:file_id>", methods=['POST'])
@safe_run
@check_expired_jobs
def write_dataset(file_id: int):
    file_record = FilesModel.query.get(file_id)
    if not file_record:
        return 'Dataset with id {} was not found on the database'.format(file_id), 404
    res = on_new_chunk_received(request, file_id)
    return jsonify(res)


@DATASETS_API.route('/dataset/<int:dataset_id>', methods=['DELETE'])
@safe_run
@check_expired_jobs
def delete_dataset(dataset_id: int):
    delete_dataset_from_db(dataset_id)
    return jsonify({'id': dataset_id})
