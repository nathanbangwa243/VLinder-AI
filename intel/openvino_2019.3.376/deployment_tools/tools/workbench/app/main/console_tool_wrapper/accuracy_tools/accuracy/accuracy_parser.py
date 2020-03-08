"""
 OpenVINO Profiler
 Accuracy checker parser

 Copyright (c) 2019 Intel Corporation

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

from app.main.console_tool_wrapper.accuracy_tools.parser import ProgressParser
from app.main.jobs.interfaces.iemit_message import IEmitMessage


class AccuracyParser(ProgressParser):
    def __init__(self, emit_message: IEmitMessage, stage_types: tuple):
        ProgressParser.__init__(self, emit_message, stage_types)
        self.accuracy = 0

    def parse(self, string: str):
        super(AccuracyParser, self).parse(string)
        if re.search(r'Accuracy:\s\d+.\d+%$', string):
            pattern = r'\d+.\d+'
            pattern = re.compile(pattern)
            matches = re.findall(pattern, string)
            if len(matches) == 1:
                val = float(matches[0])
                self.accuracy = val
