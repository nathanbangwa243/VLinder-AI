"""
 OpenVINO Profiler
 Classes and functions for configure Flask application

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

from config.constants import DATA_FOLDER


class Config:
    # app
    app_host = '127.0.0.1'
    app_port = int(os.getenv('API_PORT', '5676'))

    # celery config
    rabbitmq_password = os.getenv('RABBITMQ_PASSWORD', 'openvino')
    broker_host = 'localhost'
    broker_url = 'amqp://openvino:{}@{}/openvino_vhost'.format(rabbitmq_password, broker_host)
    celery_backend_url = 'rpc://'
    worker_prefetch_multiplier = 1
    task_acks_late = True
    imports = ['app.main.tasks.task']


class ProductionConfig(Config):
    # database config
    database_password = os.getenv('DB_PASSWORD', 'openvino')
    db_url = 'localhost:5432'
    SQLALCHEMY_DATABASE_URI = 'postgresql://openvino:{}@{}/workbench'.format(database_password, db_url)
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class TestingConfig(Config):
    TESTING = True
    WTF_CSRF_ENABLED = False
    SQLALCHEMY_DATABASE_URI = 'sqlite:///{}/test.db'.format(DATA_FOLDER)


def get_config():
    return {
        'testing': TestingConfig,
        'development': ProductionConfig,
        'production': ProductionConfig
    }
