"""
 OpenVINO Profiler
 Class for cli output of benchmark app

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

from app.error.job_error import CompoundInferenceError
from app.main.console_tool_wrapper.benchmark_app.stages import BenchmarkAppStages
from app.main.jobs.interfaces.iemit_message import IEmitMessageStage
from app.main.jobs.single_inference.single_inference_emit_msg import SingleInferenceEmitMessage
from app.main.jobs.tools_runner.tool_output_parser import ConsoleToolOutputParser


class BenchmarkConsoleOutputParser(ConsoleToolOutputParser):

    def __init__(self, emit_message: SingleInferenceEmitMessage, stages: tuple):
        ConsoleToolOutputParser.__init__(self, emit_message, stages)

    def parse(self, string: str):
        self.stdout += ('\n' + string)
        for stage in self.stage_types:
            stage_pattern = r'.*'.join(stage.split(' '))
            # Find starting of stages
            if re.search('.*{s}.*'.format(s=stage_pattern), string):
                stage = IEmitMessageStage(stage, weight=BenchmarkAppStages.get_weight_for_stage(stage))
                self.emit_message.add_stage(stage)
                return
            if re.search(r'\[\s*ERROR\s*\]', string):
                self.emit_message.add_error(string)
                raise CompoundInferenceError(string, self.emit_message.job_id)
