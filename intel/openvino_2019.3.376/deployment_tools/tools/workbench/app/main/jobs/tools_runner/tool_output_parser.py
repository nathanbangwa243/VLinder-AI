"""
 OpenVINO Profiler
 Interface class for parsing outputs of console tools

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

from app.main.jobs.interfaces.iemit_message import IEmitMessage


class ConsoleToolOutputParser:
    def __init__(self, emit_message: IEmitMessage, stage_types: tuple = ()):
        self.stage_types = stage_types
        self.emit_message = emit_message
        self.stdout = ''

    def parse(self, string: str):
        raise NotImplementedError
