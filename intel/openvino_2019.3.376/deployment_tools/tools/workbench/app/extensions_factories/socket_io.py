"""
 OpenVINO Profiler
 Classes and functions creating and working with instance of Flask SocketIO

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
from flask_socketio import SocketIO
import eventlet

from config.constants import SERVER_MODE


eventlet.monkey_patch(socket=True)


def get_socket_io():
    return get_socket_io.SOCKET_IO


if SERVER_MODE == 'development':
    get_socket_io.SOCKET_IO = SocketIO(cors_allowed_origins='*')
else:
    get_socket_io.SOCKET_IO = SocketIO()
