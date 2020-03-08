"""
 OpenVINO Profiler
 Class for inference emit message

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
from app import get_socket_io


class FeedEmitMessage:
    namespace = '/feed'
    event = 'events'
    socket_io = get_socket_io()

    @staticmethod
    def emit(code, message, details=None):
        message = {
            'code': code,
            'message': message,
            'details': details or ''
        }
        FeedEmitMessage.socket_io.emit(FeedEmitMessage.event, message, namespace=FeedEmitMessage.namespace)
        FeedEmitMessage.socket_io.sleep(0)
