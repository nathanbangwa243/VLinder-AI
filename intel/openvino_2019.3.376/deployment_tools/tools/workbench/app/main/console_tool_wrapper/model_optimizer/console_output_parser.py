"""
 OpenVINO Profiler
 Class for cli output of calibration tool

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

from app.error.job_error import ModelOptimizerError
from app.main.console_tool_wrapper.model_optimizer.stages import ModelOptimizerStages
from app.main.jobs.interfaces.iemit_message import IEmitMessageStage
from app.main.jobs.model_optimizer.model_optimizer_emit_msg import ModelOptimizerEmitMessage
from app.main.jobs.tools_runner.tool_output_parser import ConsoleToolOutputParser


class ModelOptimizerParser(ConsoleToolOutputParser):
    # 0.015 - empirical value. MO progress increases approximately by this value between parse() calls.
    speed_by_stage = {stage: 0.015 / weight for stage, weight in ModelOptimizerStages.stages.items()}

    def __init__(self, emit_message: ModelOptimizerEmitMessage, stage_types: tuple):
        super().__init__(emit_message, stage_types)
        initial_stage = 'Initialisation'
        self.emit_message.add_stage(
            IEmitMessageStage(initial_stage, weight=ModelOptimizerStages.get_weight_for_stage(initial_stage)),
            silent=True
        )
        self.counter = 0

    def parse(self, string: str):
        if '[ ERROR ]' in string:
            self.emit_message.add_error(string)
            raise ModelOptimizerError(string, 1)

        step = re.match(r'\[ INFO \] (.*) step', string)
        if step:
            step = step.groups()[0]
            self.emit_message.add_stage(IEmitMessageStage(step, weight=ModelOptimizerStages.get_weight_for_stage(step)))
            self.counter = 0

        current_stage = self.emit_message.get_current_job().name
        speed = self.speed_by_stage[current_stage]
        percent = min(speed * self.counter, 99)
        self.emit_message.update_percent(percent)
        self.counter += 1
