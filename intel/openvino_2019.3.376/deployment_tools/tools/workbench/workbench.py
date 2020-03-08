"""
 OpenVINO Profiler
 Entry point for launching profiler BE

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

from app import create_app, get_socket_io, get_config, configure_app
from app.utils.logger import InitLogger
from config.constants import SERVER_MODE

InitLogger.init_logger()

APP = create_app()

CONFIG = get_config()[SERVER_MODE]
configure_app(APP, CONFIG)

if __name__ == '__main__':
    SOCKET_IO = get_socket_io()
    SOCKET_IO.run(APP, host=CONFIG.app_host, port=CONFIG.app_port)
