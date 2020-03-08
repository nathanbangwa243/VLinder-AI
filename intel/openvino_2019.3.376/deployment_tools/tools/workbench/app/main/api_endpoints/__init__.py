"""
 OpenVINO Profiler
 Creating Blueprints of existing endpoints

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
from functools import wraps
from flask import request, Blueprint

from app.main.api_endpoints.utils import remove_hanging_uploads

COMMON_API = Blueprint('common_api', __name__)
DATASETS_API = Blueprint('datasets_api', __name__)
INFERENCE_API = Blueprint('inference_api', __name__)
INT8AUTOTUNE_API = Blueprint('int8autotune_api', __name__)
WINOGRAD_AUTOTUNE_API = Blueprint('winograd_autotune_api', __name__)
ACCURACY_API = Blueprint('accuracy_api', __name__)
MODELS_API = Blueprint('models_api', __name__)
MODEL_OPTIMIZER_API = Blueprint('model_optimizer_api', __name__)
MODEL_DOWNLOADER_API = Blueprint('model_downloader_api', __name__)
REGISTRY_API = Blueprint('registry_api', __name__)
DOWNLOAD_API = Blueprint('download_api', __name__)


def adds_session_id(func):
    @wraps(func)
    def decorated_function(*args, **kwargs):
        args = tuple([request.headers.get('SESSION-ID'), *args])
        return func(*args, **kwargs)

    return decorated_function


def check_expired_jobs(func):
    @wraps(func)
    def decorated_function(*args, **kwargs):
        remove_hanging_uploads()
        res = func(*args, **kwargs)
        return res

    return decorated_function


# pylint: disable=wrong-import-position
from . import (
    datasets, models, int8autotune, accuracy, registry,
    download, inference, common, winograd_autotune, model_optimizer,
    model_downloader
)
