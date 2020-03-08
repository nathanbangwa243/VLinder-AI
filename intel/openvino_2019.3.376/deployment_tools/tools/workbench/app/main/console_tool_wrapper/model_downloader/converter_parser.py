"""
 OpenVINO Profiler
 Class for parsing output of convert.py of Model Downloader

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

from app.error.job_error import ModelOptimizerError
from app.main.jobs.interfaces.iemit_message import IEmitMessageStage
from app.main.jobs.topology_convert.topology_convert_emit_msg import TopologyConvertEmitMessage
from app.main.jobs.tools_runner.tool_output_parser import ConsoleToolOutputParser


class TopologyConvertParser(ConsoleToolOutputParser):
    def __init__(self, emit_message: TopologyConvertEmitMessage, stage_types: tuple):
        super().__init__(emit_message, stage_types)
        self.counter = 0
        self.emit_message.add_stage(IEmitMessageStage(self.stage_types[0], self.stage_types[0]), silent=True)

    def parse(self, string: str):
        if '[ ERROR ]' in string:
            self.emit_message.add_error(string)
            raise ModelOptimizerError(string, 1)
        percent = min(0.025 * self.counter, 99)  # Until Model Optimizer provides progress bar.
        self.emit_message.update_percent(percent)
        self.counter += 1
