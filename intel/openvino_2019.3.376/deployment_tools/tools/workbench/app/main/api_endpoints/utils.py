"""
 OpenVINO Profiler
 Endpoints utility functions

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
from datetime import datetime, timedelta
from pathlib import Path

import math

from celery import chain
from sqlalchemy import and_, orm, or_

from app import get_celery
from app.error.general_error import GeneralError
from app.error.inconsistent_upload_error import InconsistentModelConfigurationError
from app.main.jobs.datasets.dataset_upload_config import DatasetUploadConfig
from app.main.jobs.datasets.dataset_upload_emit_msg import DatasetUploadEmitMessage
from app.main.jobs.interfaces.iemit_message import IEmitMessageStage
from app.main.jobs.uploads.models.model_upload_config import ModelUploadConfig
from app.main.jobs.uploads.models.model_upload_emit_msg import ModelUploadEmitMessage

from app.main.jobs.uploads.upload_emit_msg import UploadEmitMessage
from app.main.jobs.utils.traversal import get_top_level_model_id
from app.main.models import UploadJobsModel
from app.main.models.accuracy_model import AccuracyJobsModel
from app.main.jobs.utils.database_functions import set_status_in_db
from app.main.models.files_model import FilesModel
from app.main.models.model_downloader_model import ModelDownloaderModel
from app.main.models.model_optimizer_job_model import ModelOptimizerJobModel
from app.main.models.dataset_generation_configs_model import DatasetGenerationConfigsModel
from app.main.models.model_optimizer_scan_model import ModelOptimizerScanJobsModel
from app.main.models.projects_model import ProjectsModel
from app.main.models.topology_analysis_jobs_model import TopologyAnalysisJobsModel

from app.extensions_factories.database import get_db
from app.main.jobs.interfaces.exec_info import ExecInfo
from app.main.models.compound_inference_model import CompoundInferenceJobsModel
from app.main.models.inference_results_model import InferenceResultsModel
from app.main.models.jobs_model import JobsModel
from app.main.models.model_downloader_conversion_jobs_model import ModelDownloaderConversionJobsModel
from app.main.models.winograd_autotune_model import WinogradAutotuneJobsModel
from app.main.utils.utils import remove_dir, get_size_of_files
from app.main.jobs.job_types_enum import JobTypesEnum
from app.main.models.artifacts_model import ArtifactsModel
from app.main.models.datasets_model import DatasetsModel
from app.main.models.enumerates import STATUS_PRIORITY, DevicesEnum, SupportedFrameworksEnum, OptimizationTypesEnum, \
    ModelSourceEnum
from app.main.models.factory import get_job_by_id
from app.main.models.enumerates import StatusEnum
from app.main.models.factory import write_record
from app.main.models.int8_autotune_model import Int8AutotuneJobsModel
from app.main.models.topologies_model import TopologiesModel
from app.main.jobs.utils.yml_abstractions import Postprocessing, Preprocessing, Metric
from app.main.tasks.task import TASK
from app.main.models.enumerates import TaskEnum, TaskMethodEnum
from app.utils.jobs_weights import JobsWeight


def restore_pipeline_from_last_job_by(job_id: int) -> list:
    last_job_id = job_id

    pipeline = list()
    while last_job_id is not None:
        pipeline.append(last_job_id)
        last_job = JobsModel.query.get(last_job_id)
        parent_job_id = JobsModel.query.get(last_job_id).parent_job if last_job else None
        if not parent_job_id:
            break
        last_job_id = parent_job_id
    return pipeline[::-1]


def get_upload_task_ids(job_id: int) -> list:
    ids = get_artifact_task_ids(job_id)
    return ids or []


def get_task_id_for_job(job_id: int) -> str or None:
    job = JobsModel.query.get(job_id)
    return job.task_id if job else None


def get_artifact_task_ids(job_id: int) -> list:
    artifact_job = ArtifactsModel.query.get(job_id)
    return [artifact_job.task_id] if artifact_job and artifact_job.id else []


def cancel_job_in_db(job_id: int):
    job = JobsModel.query.get(job_id)
    if job:
        job.status = StatusEnum.cancelled
        write_record(job, get_db().session)
        compound_inference_job = CompoundInferenceJobsModel.query.get(job_id)
        if not compound_inference_job:
            return
        inference_results = compound_inference_job.inference_results
        for inference_result in inference_results:
            if inference_result.status in (StatusEnum.running, StatusEnum.queued):
                inference_result.status = StatusEnum.cancelled
                write_record(inference_result, get_db().session)


def cancel_upload_in_db(artifact_id: int) -> bool:
    artifact = DatasetsModel.query.get(artifact_id)
    if not artifact:
        artifact = TopologiesModel.query.get(artifact_id)
        if artifact:
            model_optimize = ModelOptimizerJobModel.query.filter_by(result_model_id=artifact_id).first()
            if model_optimize:
                set_status_in_db(ModelOptimizerJobModel, model_optimize.job_id, StatusEnum.cancelled, get_db().session)
            model_downloader = ModelDownloaderModel.query.filter_by(result_model_id=artifact_id).first()
            if model_downloader:
                set_status_in_db(ModelDownloaderModel, model_downloader.job_id, StatusEnum.cancelled, get_db().session)
    if artifact:
        set_status_in_db(ArtifactsModel, artifact_id, StatusEnum.cancelled, get_db().session)
        return True
    return False


def delete_job_from_db(job_id: int):
    job = get_job_by_id(job_id)
    if not job:
        return

    children = JobsModel.query.filter_by(parent_job=job_id).all()

    for child_job in children:
        delete_job_from_db(child_job.job_id)

    dependent_inference_results = InferenceResultsModel.query.filter_by(job_id=job_id).all()

    delete_rows(dependent_inference_results, get_db().session)
    delete_rows([job], get_db().session)


def load_compound_inference_job(job_id, inference_result_ids: tuple = None):
    job_record = CompoundInferenceJobsModel.query.filter_by(job_id=job_id).first()
    if not job_record:
        raise ValueError
    config = job_record.json()
    results, status = collect_compound_inference_result(job_id, inference_result_ids)
    job_data = {
        'jobType': JobTypesEnum.compound_inference_type.value,
        'jobId': job_id,
        'status': status.value,
        'creationTimestamp': job_record.creation_timestamp.timestamp(),
        'config': config,
        'result': results,
    }
    return job_data


def collect_compound_inference_result(job_id, inference_result_ids: tuple = None):
    if inference_result_ids is not None:
        result_records = InferenceResultsModel.query.filter(InferenceResultsModel.id.in_(inference_result_ids)).all()
    else:
        result_records = InferenceResultsModel.query.filter_by(job_id=job_id).all()

    results = []
    current_status = StatusEnum.ready

    for result in result_records:
        results.append(result.json())
        if STATUS_PRIORITY[result.status] > STATUS_PRIORITY[current_status]:
            current_status = result.status
    return results, current_status


def load_int8_job_from_database(job_id):
    job_record = Int8AutotuneJobsModel.query.filter_by(job_id=job_id).first()
    if not job_record:
        raise ValueError
    data = {
        'jobType': JobTypesEnum.int8autotune_type.value,
        'jobId': job_id,
        'status': StatusEnum(job_record.status).value,
        'creationTimestamp': job_record.creation_timestamp.timestamp(),
        'config': job_record.json()
    }
    return data


def load_winograd_job_from_database(job_id):
    job_record = WinogradAutotuneJobsModel.query.filter_by(job_id=job_id).first()
    if not job_record:
        raise ValueError
    data = {
        'jobType': JobTypesEnum.winograd_autotune_type.value,
        'jobId': job_id,
        'status': job_record.status.value,
        'creationTimestamp': job_record.creation_timestamp.timestamp(),
        'config': job_record.json()
    }
    return data


def write_chunk(upload_id, request):
    file_record = FilesModel.query.get(upload_id)
    artifact = file_record.artifact
    chunk = request.files['file'].stream.read()
    file_name = os.path.join(artifact.path, file_record.name)

    with open(file_name, "ab") as file:
        file.write(chunk)

    if file_record.uploaded_blob_size:
        file_record.uploaded_blob_size += len(chunk)
    else:
        file_record.uploaded_blob_size = len(chunk)
    file_record.progress = file_record.uploaded_blob_size / file_record.size * 100
    write_record(file_record, get_db().session)


def on_new_chunk_received(request, file_id: int):
    file_record = FilesModel.query.get(file_id)
    artifact = file_record.artifact

    if not artifact or artifact.status == StatusEnum.cancelled or file_record.status == StatusEnum.cancelled:
        return {}
    try:
        write_chunk(file_id, request)
    except OSError:
        return 'Internal server error', 500

    if TopologiesModel.query.get(file_record.artifact_id):
        emit_message = create_upload_emit_message_for_topology(file_record)
    elif DatasetsModel.query.get(file_record.artifact_id):
        emit_message = create_upload_emit_message_for_dataset(file_record)
    else:
        return 'Cannot find artifact for this file {}'.format(file_id), 404

    uploaded_progress = update_artifact_upload_progress(file_id, emit_message)

    if uploaded_progress >= 100 or all(f.uploaded_blob_size == f.size for f in artifact.files):
        celery_tasks_chain = []
        if TopologiesModel.query.get(artifact.id):
            upload_job = UploadJobsModel.query.filter_by(artifact_id=artifact.id).first()
            upload_job.status = StatusEnum.ready
            upload_job.progress = 100
            write_record(upload_job, get_db().session)
            celery_tasks_chain = create_tasks_chain_for_upload_model(artifact.id)
        elif DatasetsModel.query.get(artifact.id):
            celery_tasks_chain = create_tasks_chain_for_upload_dataset(artifact.id)
        artifact.size = get_size_of_files(artifact.path)
        write_record(artifact, get_db().session)
        set_status_in_db(ArtifactsModel, artifact.id, StatusEnum.running, get_db().session)
        try:
            write_record(artifact, get_db().session)
        except orm.exc.StaleDataError:
            pass

        # pylint: disable=fixme
        # TODO: Remove as soon as Model Optimizer fixes filenames handling.
        rename_mxnet_files(artifact.id)
        if celery_tasks_chain:
            chain(celery_tasks_chain).apply_async()
    return {}


# pylint: disable=fixme
# TODO: Remove as soon as Model Optimizer fixes filenames handling.
def rename_mxnet_files(artifact_id: int):
    model = TopologiesModel.query.get(artifact_id)
    if model and model.framework == SupportedFrameworksEnum.mxnet:
        files = model.files
        for file in files:
            old_path = Path(file.path)
            new_name = model.name + {'.params': '-00001.params', '.json': '-symbol.json'}[old_path.suffix]
            new_path = old_path.parent / new_name
            os.rename(str(old_path), str(new_path))
            file.path = str(new_path)
            file.name = new_name
            write_record(file, get_db().session)


def update_artifact_upload_progress(file_id: int, emit_message: UploadEmitMessage) -> float:
    file_record = FilesModel.query.get(file_id)
    artifact = file_record.artifact

    if file_record.uploaded_blob_size == file_record.size:
        file_status = StatusEnum.ready
    else:
        file_status = StatusEnum.running

    uploaded_progress = min(artifact.uploaded_progress, 100)

    artifact.progress = uploaded_progress * emit_message.weight
    topology = TopologiesModel.query.get(artifact.id)

    total_progress = uploaded_progress

    if topology and topology.framework != SupportedFrameworksEnum.openvino:
        mo_job = ModelOptimizerJobModel.query.filter_by(original_topology_id=artifact.id).first()
        result_topology = mo_job.result_model
        weights = JobsWeight.upload_and_convert_openvino_model()
        result_topology.status = StatusEnum.running
        result_topology.progress = uploaded_progress * weights[JobTypesEnum.iuploader_type]
        write_record(result_topology, get_db().session)
        total_progress = result_topology.progress

    write_record(artifact, get_db().session)

    set_status_in_db(FilesModel, file_id, file_status, get_db().session)

    if artifact.progress == 100:
        set_status_in_db(ArtifactsModel, artifact.id, StatusEnum.ready, get_db().session)
    else:
        set_status_in_db(ArtifactsModel, artifact.id, StatusEnum.running, get_db().session)

    emit_message.add_stage(IEmitMessageStage('uploading', progress=total_progress))
    return uploaded_progress


def create_tasks_chain_for_upload_model(model_id: int) -> list:
    topology = TopologiesModel.query.get(model_id)
    if topology.framework != SupportedFrameworksEnum.openvino:
        topology = TopologiesModel.query.filter_by(converted_from=model_id).first()
        model_id = topology.id
        weights = JobsWeight.upload_and_convert_openvino_model()
        return [
            TASK.subtask(args=tuple([None, JobTypesEnum.model_optimizer_scan_type.value, model_id]),
                         kwargs={'progress_weight': weights[JobTypesEnum.model_optimizer_scan_type]})]
    weights = JobsWeight.upload_openvino_model()
    return [
        TASK.subtask(args=tuple([None, JobTypesEnum.model_analyzer_type.value, model_id]),
                     kwargs={'progress_weight': weights[JobTypesEnum.model_analyzer_type]})]


def create_tasks_chain_for_upload_dataset(dataset_id: int) -> list:
    weights = JobsWeight.upload_dataset()
    return [
        TASK.subtask(args=tuple([None, JobTypesEnum.dataset_extractor_type.value, dataset_id]),
                     kwargs={'progress_weight': weights[JobTypesEnum.dataset_extractor_type]}),
        TASK.subtask(args=tuple([JobTypesEnum.dataset_recognizer_type.value, dataset_id]),
                     kwargs={'progress_weight': weights[JobTypesEnum.dataset_recognizer_type]}),
        TASK.subtask(args=tuple([JobTypesEnum.dataset_validator_type.value, dataset_id]),
                     kwargs={'progress_weight': weights[JobTypesEnum.dataset_validator_type]})]


def is_descendant_of(target_model_id: int, model_id: int) -> bool:
    model = TopologiesModel.query.get(model_id)
    if not model.optimized_from:
        return False
    if model.optimized_from == target_model_id:
        return True
    return is_descendant_of(target_model_id, model.optimized_from)


def get_job_json(job_id):
    loaders = (load_compound_inference_job, load_int8_job_from_database, load_winograd_job_from_database)
    for loader in loaders:
        try:
            return loader(job_id)
        except ValueError:
            pass
    raise GeneralError


def remove_hanging_uploads():
    delta = datetime.utcnow() - timedelta(seconds=30)

    downloading_topologies = ModelDownloaderModel.query.all()
    downloading_topologies_result_ids = [m.result_model_id for m in downloading_topologies]
    optimizing_topologies = ModelOptimizerJobModel.query.all()
    optimizing_topologies_result_ids = [m.result_model_id for m in optimizing_topologies]
    omz_topologies_ids = {*downloading_topologies_result_ids, *optimizing_topologies_result_ids}
    non_omz_topologies = TopologiesModel.query \
        .filter(TopologiesModel.last_modified < delta) \
        .filter(TopologiesModel.status != StatusEnum.ready) \
        .filter(TopologiesModel.id.notin_(list(omz_topologies_ids))) \
        .filter_by(optimized_from=None).all()

    datasets = DatasetsModel.query.filter(
        and_(DatasetsModel.last_modified < delta,
             TopologiesModel.status != StatusEnum.ready)).all()

    map(lambda topology: delete_model_from_db(topology.id), non_omz_topologies)
    map(lambda dataset: delete_dataset_from_db(dataset.id), datasets)

    bad_inference_results = InferenceResultsModel.query.filter(
        InferenceResultsModel.status.in_([StatusEnum.cancelled, StatusEnum.error]))

    map(lambda job: delete_job_from_db(job.job_id), bad_inference_results)

    bad_int8_jobs = Int8AutotuneJobsModel.query.filter(
        Int8AutotuneJobsModel.status.in_([StatusEnum.cancelled, StatusEnum.error]))

    map(lambda job: delete_job_from_db(job.job_id), bad_int8_jobs)


def delete_model_from_db(model_id: int):
    all_models = TopologiesModel.query.all()
    derived_models = tuple(filter(lambda m: is_descendant_of(model_id, m.id), all_models))
    derived_models_ids = tuple(map(lambda m: m.id, derived_models))

    for derived_model_id in derived_models_ids:
        delete_model_from_db(derived_model_id)

    derived_scope = model_related_information(derived_models_ids)

    for rows in derived_scope:
        delete_rows(rows, get_db().session)

    parent_int8 = Int8AutotuneJobsModel.query.filter_by(result_model_id=model_id).all()
    parent_winograd = WinogradAutotuneJobsModel.query.filter_by(result_model_id=model_id).all()
    parent_mo = ModelOptimizerJobModel.query.filter(or_(ModelOptimizerJobModel.original_topology_id == model_id,
                                                        ModelOptimizerJobModel.result_model_id == model_id)).all()

    delete_rows([*parent_int8, *parent_winograd, *parent_mo], get_db().session)
    project_ids = tuple(map(lambda p: p.id, ProjectsModel.query.filter(ProjectsModel.model_id == model_id).all()))

    all_accuracy_results = AccuracyJobsModel.query \
        .filter(AccuracyJobsModel.project_id.in_(project_ids)) \
        .all()
    delete_rows(all_accuracy_results, get_db().session)

    original_scope = model_related_information((model_id,))
    for rows in original_scope:
        delete_rows(rows, get_db().session)

    model = TopologiesModel.query.get(model_id)

    if model:
        model_path = model.path
        delete_rows([model], get_db().session)
        remove_dir(model_path)


def delete_dataset_from_db(dataset_id: int):
    derived_scope = dataset_related_information(dataset_id)

    tuple(map(lambda el: delete_rows(el, get_db().session), derived_scope))

    dataset = DatasetsModel.query.get(dataset_id)

    if dataset:
        dataset_path = dataset.path
        delete_rows((dataset,), get_db().session)
        remove_dir(dataset_path)


def model_related_information(model_ids: tuple) -> tuple:
    projects = ProjectsModel.query.filter(ProjectsModel.model_id.in_(model_ids)).all()

    all_project_ids = tuple(map(lambda p: p.id, projects))

    all_accuracy_results = AccuracyJobsModel.query \
        .filter(AccuracyJobsModel.project_id.in_(all_project_ids)) \
        .all()

    run_results, compound_configs = projects_related_information(all_project_ids)

    analysis = TopologyAnalysisJobsModel.query.filter(TopologyAnalysisJobsModel.model_id.in_(model_ids)).all()

    optimizes = ModelOptimizerJobModel.query.filter(ModelOptimizerJobModel.result_model_id.in_(model_ids)).all()
    converts = ModelDownloaderConversionJobsModel.query.filter(
        ModelDownloaderConversionJobsModel.result_model_id.in_(model_ids)).all()
    downloads = ModelDownloaderModel.query.filter(ModelDownloaderModel.result_model_id.in_(model_ids)).all()
    return analysis, optimizes, converts, downloads, run_results, all_accuracy_results, compound_configs, projects


def dataset_related_information(dataset_id: int):
    projects = ProjectsModel.query.filter_by(dataset_id=dataset_id).all()
    all_project_ids = list(map(lambda p: p.id, projects))

    run_results, compound_configs = projects_related_information(all_project_ids)
    dataset_generator_config = DatasetGenerationConfigsModel.query.filter(
        DatasetGenerationConfigsModel.result_dataset_id == dataset_id).all()

    all_int8_results = Int8AutotuneJobsModel.query \
        .filter(Int8AutotuneJobsModel.project_id.in_(all_project_ids)) \
        .all()
    all_accuracy_results = AccuracyJobsModel.query \
        .filter(AccuracyJobsModel.project_id.in_(all_project_ids)) \
        .all()

    all_winograd_results = WinogradAutotuneJobsModel.query \
        .filter(WinogradAutotuneJobsModel.project_id.in_(all_project_ids)) \
        .all()

    return run_results, compound_configs, all_int8_results, all_winograd_results, \
           all_accuracy_results, projects, dataset_generator_config


def projects_related_information(project_ids: iter) -> tuple:
    compound_configs = CompoundInferenceJobsModel.query \
        .filter(CompoundInferenceJobsModel.project_id.in_(project_ids)).all()

    all_infer_config_ids = map(lambda i: i.job_id, compound_configs)

    inference_results = InferenceResultsModel.query \
        .filter(InferenceResultsModel.job_id.in_(all_infer_config_ids)).all()

    return inference_results, compound_configs


def delete_rows(rows, session):
    for i in rows:
        session.delete(i)
    session.commit()


def generate_queue_of_single_inference_tasks(data: dict, job_id: int, start_tasks: list = None, previous_weight=0):
    min_nireq = data['minNireq']
    max_nireq = data['maxNireq']
    step_nireq = data['stepNireq']

    min_batch = data['minBatch']
    max_batch = data['maxBatch']
    step_batch = data['stepBatch']

    queue = []
    if start_tasks:
        for task in start_tasks:
            queue.append(task)
    num_runs = math.ceil((max_batch - min_batch + 1) / step_batch) * math.ceil((max_nireq - min_nireq + 1) / step_nireq)
    weight_single_run = (1 - previous_weight) / num_runs
    for batch in range(min_batch, max_batch + 1, step_batch):
        for nireq in range(min_nireq, max_nireq + 1, step_nireq):
            queue.append(TASK.subtask(args=(JobTypesEnum.single_inference_type.value, job_id),
                                      kwargs={'data': ExecInfo(batch, nireq).json(),
                                              'progress_weight': weight_single_run}))
            inference_result = InferenceResultsModel({'jobId': job_id,
                                                      'execInfo': {
                                                          'batch': batch,
                                                          'nireq': nireq}})
            write_record(inference_result, get_db().session)

    if not start_tasks:
        queue.pop(0)
        queue.insert(0, TASK.subtask(args=tuple([None, JobTypesEnum.single_inference_type.value, job_id]),
                                     kwargs={'data': ExecInfo(min_batch, min_nireq).json(),
                                             'progress_weight': weight_single_run}))
        get_db().session().commit()
    return queue


def find_projects(model_id: int, all_levels: bool) -> tuple:
    if model_id:
        all_models = TopologiesModel.query.all()
        derived_models_id = []
        if all_levels:
            derived_models = filter(lambda m: is_descendant_of(target_model_id=model_id, model_id=m.id), all_models)
            derived_models_id = tuple(map(lambda model: model.id, derived_models))
        projects = ProjectsModel.query. \
            filter(ProjectsModel.model_id.in_([*derived_models_id, model_id])).all()
        return filter_projects(projects)
    if all_levels:
        all_models = TopologiesModel.query.all()
        all_models_ids = map(lambda model: model.id, all_models)
        projects = ProjectsModel.query.filter(ProjectsModel.model_id.in_(all_models_ids)).all()
    else:
        all_original_models = TopologiesModel.query.filter(TopologiesModel.optimized_from.is_(None)).all()
        all_original_models_ids = map(lambda model: model.id, all_original_models)
        projects = ProjectsModel.query.filter(ProjectsModel.model_id.in_(all_original_models_ids)).all()

    return filter_projects(projects)


def filter_projects(projects):
    projects_id = tuple(map(lambda project: project.id, projects))
    int_8_failed = Int8AutotuneJobsModel.query \
        .filter(Int8AutotuneJobsModel.project_id.in_(projects_id)) \
        .filter(Int8AutotuneJobsModel.status.in_([StatusEnum.cancelled, StatusEnum.error])) \
        .all()
    int8_models_ids = tuple(map(lambda el: el.result_model_id, int_8_failed))
    int8_result_projects = tuple(map(lambda el: find_project_by(el).id, int8_models_ids))

    winograd_failed = WinogradAutotuneJobsModel.query \
        .filter(WinogradAutotuneJobsModel.project_id.in_(projects_id)) \
        .filter(WinogradAutotuneJobsModel.status.in_([StatusEnum.cancelled, StatusEnum.error])) \
        .all()
    winograd_projects_ids = tuple(map(lambda el: el.result_model_id, winograd_failed))
    ignore_projects = tuple(filter(lambda el: el, map(find_project_by, winograd_projects_ids)))
    winograd_result_projects = tuple(map(lambda el: el.id, ignore_projects))

    res_projects = filter(lambda project: project.id not in [
        *int8_result_projects,
        *winograd_result_projects
    ], projects)
    return tuple(res_projects)


def find_project_by(model_id):
    resulting_model = TopologiesModel.query.get(model_id)
    return ProjectsModel.query.filter_by(model_id=resulting_model.id).first()


def project_json(project: ProjectsModel):
    jobs = CompoundInferenceJobsModel.query.filter_by(project_id=project.id)
    good_jobs = jobs.filter(CompoundInferenceJobsModel.status.notin_([StatusEnum.cancelled, StatusEnum.error]))
    jobs = good_jobs if good_jobs.count() > 0 else jobs
    first_job = jobs.order_by(CompoundInferenceJobsModel.creation_timestamp.asc()).first()
    last_job = jobs.order_by(CompoundInferenceJobsModel.creation_timestamp.desc()).first()
    job_status = CompoundInferenceJobsModel.query.get(last_job.job_id) if last_job else None
    status = job_status.status.value if job_status else StatusEnum.ready.value
    error_message = job_status.error_message if job_status else None
    progress = job_status.progress if job_status else 100
    generated_dataset_id = DatasetGenerationConfigsModel.query.get(project.dataset_id)
    model = TopologiesModel.query.get(project.model_id)
    is_accuracy_available = not generated_dataset_id
    optimization_params = {}
    if project.optimization_type == OptimizationTypesEnum.int8autotune:
        int8_job = Int8AutotuneJobsModel.query.filter_by(result_model_id=project.model_id).first()
        optimization_params['threshold'] = int8_job.threshold
        optimization_params['subsetSize'] = int8_job.subset_size
    project_info = project.json()
    del project_info['optimizationType']
    res = {
        **project_info,
        'creationTimestamp': first_job.creation_timestamp if first_job else None,
        'precision': model.precision.value,
        'configParameters': {
            'optimizationType': project.optimization_type.value,
            **optimization_params
        },
        'analysisData': None,
        'execInfo': None,
        'parentId': None,
        'isAccuracyCheckAvailable': is_accuracy_available,
        'status': {
            'name': status,
            'progress': progress
        }
    }
    if error_message:
        res['status']['errorMessage'] = error_message
    return res


def fill_with_exec_info(result):
    for info in result:
        jobs = CompoundInferenceJobsModel.query.filter_by(project_id=info['id'])
        job_ids = map(lambda job: job.job_id, jobs)
        best_infer_results = InferenceResultsModel.query \
            .filter(InferenceResultsModel.job_id.in_(job_ids), InferenceResultsModel.throughput.isnot(None)) \
            .order_by(InferenceResultsModel.throughput.desc()) \
            .first()
        if best_infer_results:
            info['execInfo'] = best_infer_results.json()['execInfo']
        else:
            info['execInfo'] = {
                'throughput': None,
                'latency': None,
                'batch': None,
                'nireq': None,
            }


def fill_with_analysis_data(result):
    for info in result:
        model_id = info['modelId']
        data = TopologyAnalysisJobsModel.query.filter_by(model_id=model_id).first()
        if data:
            info['analysisData'] = data.json()


def fill_projects_with_model_and_dataset_names(projects: list):
    for project in projects:
        original_model_id = get_top_level_model_id(project['id'])

        model = TopologiesModel.query.get(original_model_id)
        if model:
            project['modelName'] = model.name
        dataset_id = project['datasetId']
        dataset = DatasetsModel.query.get(dataset_id)
        if dataset:
            project['datasetName'] = dataset.name


def connect_with_parents(result):
    for res in result:
        model = TopologiesModel.query.get(res['modelId'])
        if model.optimized_from:
            parent_project = ProjectsModel.query. \
                filter_by(model_id=model.optimized_from,
                          dataset_id=res['datasetId'],
                          target=DevicesEnum(res['device'])).first()
            res['parentId'] = parent_project.id


def get_derived_projects(project: ProjectsModel) -> list:
    original_model_id = project.model_id
    generated_models = TopologiesModel.query.filter_by(optimized_from=original_model_id).all()
    generated_models_ids = [model.id for model in generated_models]
    derived_projects = ProjectsModel.query \
        .filter(ProjectsModel.model_id.in_(generated_models_ids)) \
        .filter_by(dataset_id=project.dataset_id) \
        .filter_by(target=project.target) \
        .all()
    res = [*derived_projects]
    for derived_project in derived_projects:
        res.extend(get_derived_projects(derived_project))
    return derived_projects


def get_unique_job_configs(project_id):
    configs = []
    project = ProjectsModel.query.get(project_id)
    if not project:
        return configs
    all_project_jobs = CompoundInferenceJobsModel.query.filter_by(project_id=project_id).all()
    target = ProjectsModel.query.get(project_id).target.value
    for job in all_project_jobs:
        current_config_job = list(
            filter(lambda r, j=job:
                   r['minBatch'] == j.min_batch and r['maxBatch'] == j.max_batch and
                   r['minNireq'] == j.min_nireq and r['maxNireq'] == j.max_nireq and
                   r['stepBatch'] == j.step_batch and r['stepNireq'] == j.step_nireq,
                   configs))
        if not current_config_job:
            configs.append({
                'minBatch': job.min_batch,
                'maxBatch': job.max_batch,
                'stepBatch': job.step_batch,
                'minNireq': job.min_nireq,
                'maxNireq': job.max_nireq,
                'stepNireq': job.step_nireq,
                'device': target,
                'inferenceTime': job.inference_time
            })
    return configs


def get_jobs_for_config(config, project_id, session_id):
    inference_jobs = CompoundInferenceJobsModel.query.filter_by(project_id=project_id,
                                                                min_batch=config['minBatch'],
                                                                max_batch=config['maxBatch'],
                                                                step_batch=config['stepBatch'],
                                                                min_nireq=config['minNireq'],
                                                                max_nireq=config['maxNireq'],
                                                                step_nireq=config['stepNireq'])
    last_job = inference_jobs.order_by(CompoundInferenceJobsModel.creation_timestamp.desc()).first()
    status = last_job.status if last_job else StatusEnum.ready
    progress = last_job.progress if last_job else 0
    error_message = last_job.error_message
    result = {
        'compoundJobId': last_job.job_id,
        'projectId': project_id,
        'sessionId': session_id,
        'config': config,
        'status': {
            'name': status.value,
            'progress': progress,
            'errorMessage': error_message
        },
        'currentTimestamp': last_job.creation_timestamp,
        'execInfo': get_exec_info_for_config(config, project_id)
    }
    return result


def get_exec_info_for_config(config, project_id):
    exec_info = []
    project_inference_jobs = CompoundInferenceJobsModel.query.filter_by(project_id=project_id)
    project_inference_job_ids = list(map(lambda job: job.job_id, project_inference_jobs))

    min_batch = config['minBatch']
    max_batch = config['maxBatch']
    step_batch = config['stepBatch']

    min_nireq = config['minNireq']
    max_nireq = config['maxNireq']
    step_nireq = config['stepNireq']

    for batch in range(min_batch, max_batch + step_batch, step_batch):
        for nireq in range(min_nireq, max_nireq + step_nireq, step_nireq):
            specified_jobs_for_batch_nireq_project = InferenceResultsModel.query \
                .filter(InferenceResultsModel.job_id.in_(project_inference_job_ids)) \
                .filter_by(batch=batch, nireq=nireq) \
                .filter(InferenceResultsModel.throughput.isnot(None)) \
                .filter(InferenceResultsModel.latency.isnot(None)) \
                .filter(InferenceResultsModel.total_execution_time.isnot(None))
            best_job = specified_jobs_for_batch_nireq_project.order_by(InferenceResultsModel.throughput.desc()).first()
            if not best_job:
                specified_jobs_for_batch_nireq_project = InferenceResultsModel.query \
                    .filter(InferenceResultsModel.job_id.in_(project_inference_job_ids)) \
                    .filter_by(batch=batch, nireq=nireq)
                best_job = specified_jobs_for_batch_nireq_project.order_by(
                    InferenceResultsModel.throughput.desc()).first()
            if best_job:
                exec_info.append({
                    'throughput': best_job.throughput,
                    'latency': best_job.latency,
                    'batch': batch,
                    'nireq': nireq,
                    'totalExecTime': best_job.total_execution_time,
                    'inferenceResultId': best_job.id
                })
    return exec_info


def try_load_configuration(config: dict):
    try:
        if not TaskEnum.has_value(config['taskType']):
            raise InconsistentModelConfigurationError('Incorrect task type value: {}'.format(config['taskType']))
        if not TaskMethodEnum.has_value(config['taskMethod'].lower()):
            raise InconsistentModelConfigurationError('Incorrect task method value: {}'.format(config['taskMethod']))
        Postprocessing.from_list(config['postprocessing'])
        Preprocessing.from_list(config['preprocessing'])
        Metric.from_list(config['metric'])
    except KeyError as err:
        raise InconsistentModelConfigurationError('Configuration does not contain required field "{}"'.format(err))
    except Exception as exc:
        raise InconsistentModelConfigurationError('Incorrect configuration data', str(exc))


def create_upload_emit_message_for_topology(file: FilesModel) -> ModelUploadEmitMessage:
    topology = TopologiesModel.query.get(file.artifact_id)
    if not topology:
        raise FileNotFoundError('Cannot find Topology for this file')
    if topology.framework != SupportedFrameworksEnum.openvino:
        weight = JobsWeight.upload_source_model()[JobTypesEnum.iuploader_type]
        mo_job = ModelOptimizerJobModel.query.filter_by(original_topology_id=file.artifact_id).first()
        artifact_id = mo_job.result_model_id
    else:
        weight = JobsWeight.upload_openvino_model()[JobTypesEnum.iuploader_type]
        artifact_id = file.artifact_id
    topology = TopologiesModel.query.get(file.artifact_id)
    config = ModelUploadConfig(topology.session_id, topology.short_json())
    emit_message = ModelUploadEmitMessage(None, artifact_id, config, weight)
    emit_message.date = topology.creation_timestamp.timestamp()
    return emit_message


def create_upload_emit_message_for_dataset(file: FilesModel) -> UploadEmitMessage:
    dataset_id = file.artifact_id
    dataset = DatasetsModel.query.get(dataset_id)
    config = DatasetUploadConfig(dataset.session_id, dataset.short_json())
    emit_message = DatasetUploadEmitMessage(None, dataset_id, config,
                                            JobsWeight.upload_dataset()[JobTypesEnum.iuploader_type])
    emit_message.date = dataset.creation_timestamp.timestamp()
    return emit_message


def cancel_tasks(all_jobs: list):
    for job in all_jobs:
        if job and job.status == StatusEnum.running:
            cancel_task_by_task_id(job.task_id)
        cancel_job_in_db(job.job_id)


def cancel_task_by_task_id(task_id: str):
    celery = get_celery()
    celery.control.revoke(task_id, terminate=True, wait=True, signal='SIGTERM')


def prepare_data_for_mo_pipeline(topology: TopologiesModel, upload_job_id: int, session_id: str):
    converted_model = TopologiesModel(topology.name, SupportedFrameworksEnum.openvino, topology.metadata_id, session_id)
    converted_model.source = ModelSourceEnum.original
    converted_model.converted_from = topology.id
    write_record(converted_model, get_db().session)
    model_optimizer_scan_job = ModelOptimizerScanJobsModel({
        'topology_id': topology.id,
        'previousJobId': upload_job_id,
        'session_id': session_id,
    })
    write_record(model_optimizer_scan_job, get_db().session)
    model_optimizer_job = ModelOptimizerJobModel({
        'session_id': session_id,
        'name': topology.name,
        'original_topology_id': topology.id,
        'result_model_id': converted_model.id,
        'previousJobId': model_optimizer_scan_job.job_id,
    })
    write_record(model_optimizer_job, get_db().session)
    analysis_data = TopologyAnalysisJobsModel({'session_id': session_id, 'model_id': converted_model.id})
    analysis_data.parent_job = model_optimizer_job.job_id
    write_record(analysis_data, get_db().session)
    return converted_model, model_optimizer_job


def save_pipeline_config(content: str, topology_id: int):
    topology = TopologiesModel.query.get(topology_id)
    pipeline_config_path = os.path.join(topology.path, 'pipeline.config')
    with open(pipeline_config_path, 'w+') as pipeline_config_file:
        pipeline_config_file.writelines(content)
    size = get_size_of_files(pipeline_config_path)
    config_file_record = FilesModel('pipeline.config', topology_id, size, topology.session_id)
    config_file_record.progress = 100
    config_file_record.status = StatusEnum.ready
    config_file_record.uploaded_blob_size = size
    config_file_record.path = pipeline_config_path
    write_record(config_file_record, get_db().session)
