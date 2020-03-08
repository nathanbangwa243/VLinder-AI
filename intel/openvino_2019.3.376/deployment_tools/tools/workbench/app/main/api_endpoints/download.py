"""
 OpenVINO Profiler
 Endpoints to work with downloading of files

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
from app.main.api_endpoints import check_expired_jobs, DOWNLOAD_API, adds_session_id
from app.main.jobs.download_model.download_model_job import DownloadModelJob
from app.main.jobs.job_types_enum import JobTypesEnum
from app.main.models.artifacts_model import ArtifactsModel
from app.main.models.download_configs_model import DownloadConfigsModel
from app.main.models.factory import write_record
from app.main.models.projects_model import ProjectsModel
from app.main.tasks.task import TASK
from app.main.utils.safe_runner import safe_run


@DOWNLOAD_API.route('/archive/<project_id>')
@safe_run
@adds_session_id
@check_expired_jobs
def archive_model(session_id, project_id):
    project = ProjectsModel.query.get(project_id)
    artifact = ArtifactsModel.query.get(project.model_id)

    exists, path = DownloadModelJob.archive_exists(artifact.id)
    if exists:
        return jsonify({
            'jobId': None,
            'message': 'archive already exists',
            'path': path
        })

    name = request.args.get('name')
    download_job = DownloadConfigsModel(dict(session_id=session_id, projectId=project_id, path=path, name=name))
    write_record(download_job, get_db().session)

    TASK.apply_async(args=(None, JobTypesEnum.download_model_type.value, download_job.job_id))

    return jsonify({'jobId': download_job.job_id})
