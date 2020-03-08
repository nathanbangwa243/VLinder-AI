"""
 OpenVINO Profiler
 Class for cli output of winograd tool

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

from app.error.job_error import WinogradAutotuneError
from app.main.console_tool_wrapper.benchmark_app.stages import BenchmarkAppStages
from app.main.console_tool_wrapper.winograd_tool.winograd_tool_stages import WinogradToolStages
from app.main.jobs.interfaces.iemit_message import IEmitMessageStage
from app.main.jobs.single_inference.single_inference_emit_msg import SingleInferenceEmitMessage
from app.main.jobs.tools_runner.tool_output_parser import ConsoleToolOutputParser


class WinogradConsoleOutputParser(ConsoleToolOutputParser):

    def __init__(self, emit_message: SingleInferenceEmitMessage, stage_types: tuple):
        ConsoleToolOutputParser.__init__(self, emit_message, stage_types)
        self.is_error = False

    def parse(self, string: str):
        if not string:
            return
        self.stdout += ('\n' + string)
        for stage in self.stage_types:
            stage_pattern = r'.*'.join(stage.split(' '))
            if re.search('.*{s}.*'.format(s=stage_pattern), string):
                self.emit_message.add_stage(
                    IEmitMessageStage(stage, weight=BenchmarkAppStages.get_weight_for_stage(stage)))
                return
        for error in WinogradToolStages.errors:
            error_pattern = r'.*'.join(error.split(' '))
            if re.search('.*{s}.*'.format(s=error_pattern), string):
                self.emit_message.add_error(string)
                raise WinogradAutotuneError(string, 1)

    def get_error(self):
        stdout = self.stdout.split('\n')
        error = []
        for line in stdout:
            if '[ ERROR ]' in line or error:
                error.append(line)
        return '\n'.join(error)
