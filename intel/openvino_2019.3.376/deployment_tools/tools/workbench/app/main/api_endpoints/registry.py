"""
 OpenVINO Profiler
 Endpoints to work with states and registry

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
import time

from flask import jsonify, send_file

from app.main.api_endpoints import check_expired_jobs, REGISTRY_API
from app.main.jobs.registries.code_registry import CodeRegistry
from app.main.models.accuracy_model import AccuracyJobsModel
from app.main.models.compound_inference_model import CompoundInferenceJobsModel
from app.main.models.enumerates import StatusEnum
from app.main.models.int8_autotune_model import Int8AutotuneJobsModel
from app.main.models.jobs_model import JobsModel
from app.main.models.winograd_autotune_model import WinogradAutotuneJobsModel
from app.main.utils.device_info import load_available_hardware_info
from app.main.utils.safe_runner import safe_run
from app.utils.logger import InitLogger, FileHandler
from app.utils.utils import get_version
from config.constants import TF_CUSTOM_OPERATIONS_CONFIGS


@REGISTRY_API.route("/sync", methods=['GET'])
@safe_run
@check_expired_jobs
def sync():
    jobs_tables = (table.__tablename__ for table in (AccuracyJobsModel, CompoundInferenceJobsModel,
                                                     Int8AutotuneJobsModel, WinogradAutotuneJobsModel))
    running_jobs_number = JobsModel.query.filter(
        JobsModel.job_type.in_(jobs_tables),
        JobsModel.status.in_((StatusEnum.queued, StatusEnum.running))
    ).count()
    return jsonify({
        'time': time.time(),
        'version': get_version(),
        'codes': CodeRegistry.CODES,
        'devices': load_available_hardware_info(),
        'taskIsRunning': running_jobs_number > 0,
        'availableTfConfigs': list(sorted(TF_CUSTOM_OPERATIONS_CONFIGS.keys())),
    })


@REGISTRY_API.route("/get-log", methods=['POST'])
@safe_run
@check_expired_jobs
def get_log():
    FileHandler.tail_file(InitLogger.log_file)
    return send_file(InitLogger.log_file, as_attachment=True)
