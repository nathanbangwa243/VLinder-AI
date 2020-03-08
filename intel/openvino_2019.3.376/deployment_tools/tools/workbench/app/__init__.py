"""
 OpenVINO Profiler
 Entry point for defining the Flask instance

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
from flask import Flask
from flask_restful import Api

from app.config.application import get_config
from app.extensions_factories.database import init_db_app
from app.extensions_factories.socket_io import get_socket_io
from app.extensions_factories.celery import get_celery, init_celery_app


def create_app() -> Flask:
    """Create an application."""
    app = Flask(__name__)

    Api(app)

    from app.main.api_endpoints import COMMON_API
    from app.main.api_endpoints import REGISTRY_API
    from app.main.api_endpoints import DOWNLOAD_API
    from app.main.api_endpoints import WINOGRAD_AUTOTUNE_API
    from app.main.api_endpoints import DATASETS_API, MODELS_API, MODEL_DOWNLOADER_API, MODEL_OPTIMIZER_API
    from app.main.api_endpoints import INFERENCE_API, INT8AUTOTUNE_API, ACCURACY_API

    app.register_blueprint(COMMON_API, url_prefix='/api')
    app.register_blueprint(DATASETS_API, url_prefix='/api')
    app.register_blueprint(INFERENCE_API, url_prefix='/api')
    app.register_blueprint(INT8AUTOTUNE_API, url_prefix='/api')
    app.register_blueprint(WINOGRAD_AUTOTUNE_API, url_prefix='/api')
    app.register_blueprint(MODELS_API, url_prefix='/api')
    app.register_blueprint(MODEL_OPTIMIZER_API, url_prefix='/api')
    app.register_blueprint(MODEL_DOWNLOADER_API, url_prefix='/api')
    app.register_blueprint(REGISTRY_API, url_prefix='/api')
    app.register_blueprint(DOWNLOAD_API, url_prefix='/api')
    app.register_blueprint(ACCURACY_API, url_prefix='/api')

    return app


def configure_app(app: Flask, config):
    app.config.from_object(config)

    socket_io = get_socket_io()
    socket_io.init_app(app, async_mode='eventlet', message_queue=config.broker_url)

    init_celery_app(app)
    init_db_app(app)
