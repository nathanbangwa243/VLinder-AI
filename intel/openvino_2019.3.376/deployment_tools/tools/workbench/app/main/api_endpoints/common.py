"""
 OpenVINO Profiler
 Common Endpoints

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
from app.main.api_endpoints import check_expired_jobs, COMMON_API
from app.main.api_endpoints.utils import get_job_json, cancel_job_in_db, delete_job_from_db, \
    restore_pipeline_from_last_job_by, find_projects, project_json, fill_with_exec_info, connect_with_parents, \
    get_derived_projects, delete_rows, load_compound_inference_job, fill_with_analysis_data, \
    fill_projects_with_model_and_dataset_names, get_upload_task_ids, cancel_upload_in_db, cancel_tasks, \
    get_task_id_for_job, cancel_task_by_task_id
from app.main.models.accuracy_model import AccuracyJobsModel
from app.main.models.compound_inference_model import CompoundInferenceJobsModel
from app.main.models.enumerates import StatusEnum
from app.main.models.factory import get_job_by_id
from app.main.models.inference_results_model import InferenceResultsModel
from app.main.models.int8_autotune_model import Int8AutotuneJobsModel
from app.main.models.jobs_model import JobsModel
from app.main.models.projects_model import ProjectsModel
from app.main.models.winograd_autotune_model import WinogradAutotuneJobsModel
from app.main.utils.safe_runner import safe_run


@COMMON_API.route('/job-info/<int:job_id>', methods=['GET'])
@safe_run
@check_expired_jobs
def get_job_info(job_id):
    inference_result_id = request.args.get('inferenceResultId', default=None)
    if inference_result_id:
        job_data = load_compound_inference_job(job_id, (inference_result_id,))
        job_data['result'] = job_data['result'][0]
    else:
        job_data = get_job_json(job_id)
    return jsonify(job_data) if job_data else ('No such job: {}'.format(job_id), 404)


@COMMON_API.route('/cancel-uploading/<int:artifact_id>', methods=['PUT'])
@safe_run
@check_expired_jobs
def cancel_uploading(artifact_id: int):
    for task_id in get_upload_task_ids(artifact_id):
        if task_id:
            cancel_task_by_task_id(task_id)
    cancel_upload_in_db(artifact_id)
    return jsonify({'id': artifact_id})


@COMMON_API.route('/cancel-job/<int:job_id>', methods=['PUT'])
@safe_run
@check_expired_jobs
def cancel_job(job_id: int):
    task_id = get_task_id_for_job(job_id)
    if task_id:
        cancel_task_by_task_id(task_id)
    cancel_job_in_db(job_id)
    return jsonify({'jobId': job_id})


@COMMON_API.route('/delete/<job_id>', methods=['DELETE'])
@safe_run
@check_expired_jobs
def delete(job_id):
    delete_job_from_db(job_id)
    return jsonify({})


@COMMON_API.route('/exec-jobs-history', methods=['GET'])
@safe_run
@check_expired_jobs
def job_history():
    all_results = InferenceResultsModel.query \
        .filter(InferenceResultsModel.status == StatusEnum.ready) \
        .all()
    unique_job_ids = set(result.job_id for result in all_results)
    result = []
    for inference_job_id in unique_job_ids:
        pipeline_job_ids = restore_pipeline_from_last_job_by(inference_job_id)
        single_pipeline = []
        for job_id in pipeline_job_ids:
            job_json = get_job_json(job_id)
            job = {
                'jobId': job_json['jobId'],
                'config': job_json['config'],
                'type': job_json['jobType'],
                'finished': job_json['status'] in (StatusEnum.ready.value,
                                                   StatusEnum.error.value,
                                                   StatusEnum.cancelled.value),
                'creationTimestamp': job_json['creationTimestamp']
            }
            single_pipeline.append(job)
        job_model = JobsModel.query.get(pipeline_job_ids[0])
        project = ProjectsModel.query.get(job_model.project_id)
        model_id = project.model_id
        dataset_id = project.dataset_id
        model_dataset_pairs = list(
            filter(lambda r, m_id=model_id, d_id=dataset_id: r['modelId'] == m_id and r['datasetId'] == d_id, result))
        if len(model_dataset_pairs) > 1:
            return 'Internal server error', 500
        if not model_dataset_pairs:
            result.append({
                'modelId': model_id,
                'datasetId': dataset_id,
                'pipelines': []
            })
            index = len(result) - 1
        else:
            index = result.index(model_dataset_pairs[0])
        result[index]['pipelines'].append(single_pipeline)
    return jsonify(result)


@COMMON_API.route('/projects-info/', methods=['GET'])
@safe_run
def projects_info():
    include_exec_info = bool(request.args.get('includeExecInfo'))
    defined_model_id = request.args.get('modelId')
    if defined_model_id:
        defined_model_id = int(defined_model_id)
    all_levels = bool(request.args.get('allLevels'))
    projects = find_projects(defined_model_id, all_levels)
    result = [project_json(project) for project in projects]
    if include_exec_info:
        fill_with_exec_info(result)
        for res in result:
            latest_accuracy_job = AccuracyJobsModel.query \
                .filter_by(project_id=res['id']).order_by(AccuracyJobsModel.creation_timestamp.desc()).first()
            res['execInfo']['accuracy'] = latest_accuracy_job.accuracy if latest_accuracy_job else None
    fill_with_analysis_data(result)
    connect_with_parents(result)
    fill_projects_with_model_and_dataset_names(result)
    return jsonify(result)


@COMMON_API.route('/delete-project/<int:project_id>', methods=['DELETE'])
@safe_run
def delete_project(project_id: int):
    project = ProjectsModel.query.get(project_id)
    if not project:
        return 'Project with id {} was not found'.format(project), 404
    derived_projects = get_derived_projects(project)
    derived_projects_ids = [i.id for i in derived_projects]

    jobs = JobsModel.query.filter(JobsModel.project_id.in_([*derived_projects_ids, project.id])).all()
    jobs_ids = tuple(map(lambda job: job.job_id, jobs))

    all_jobs = []
    inference_results = []

    int8_job = Int8AutotuneJobsModel.query.filter_by(result_model_id=project.model_id).first()
    winograd_job = WinogradAutotuneJobsModel.query.filter_by(result_model_id=project.model_id).first()
    if int8_job:
        all_jobs.append(int8_job)

    if winograd_job:
        all_jobs.append(winograd_job)

    table_rows = JobsModel.query.filter(JobsModel.job_id.in_(jobs_ids)).all()

    for table_row in table_rows:
        all_jobs.append(get_job_by_id(table_row.job_id))
        if CompoundInferenceJobsModel.query.get(table_row.job_id):
            for res in InferenceResultsModel.query.filter_by(job_id=table_row.job_id).all():
                inference_results.append(res)

    cancel_tasks(all_jobs)

    delete_rows(inference_results, get_db().session)
    delete_rows(all_jobs, get_db().session)
    delete_rows(all_jobs, get_db().session)
    delete_rows(derived_projects, get_db().session)
    delete_rows([project], get_db().session)

    return jsonify({'id': project.id})
