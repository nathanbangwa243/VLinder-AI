"""
 OpenVINO Profiler
 Class for parsing progress logs

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
import re
from app.main.jobs.interfaces.iemit_message import IEmitMessage
from app.main.jobs.tools_runner.tool_output_parser import ConsoleToolOutputParser


class ProgressParser(ConsoleToolOutputParser):

    def __init__(self, emit_message: IEmitMessage, stage_types: tuple):
        ConsoleToolOutputParser.__init__(self, emit_message, stage_types)

    def parse(self, string: str):
        if re.search(r'Progress:\s\d+.\d+%\sdone$', string):
            pattern = r'\d+.\d+'
            pattern = re.compile(pattern)
            matches = re.findall(pattern, string)
            if len(matches) == 1:
                val = float(matches[0])
                self.emit_message.update_percent(val)
