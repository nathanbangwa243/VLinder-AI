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
import json

from app.error.job_error import ModelOptimizerError
from app.main.console_tool_wrapper.model_optimizer.stages import ModelOptimizerStages
from app.main.jobs.interfaces.iemit_message import IEmitMessageStage
from app.main.jobs.model_optimizer.model_optimizer_emit_msg import ModelOptimizerEmitMessage
from app.main.jobs.tools_runner.tool_output_parser import ConsoleToolOutputParser


class ModelOptimizerScanParser(ConsoleToolOutputParser):

    def __init__(self, emit_message: ModelOptimizerEmitMessage, stage_types: tuple):
        super().__init__(emit_message, stage_types)
        initial_stage = 'Initialisation'
        self.emit_message.add_stage(
            IEmitMessageStage(initial_stage, weight=ModelOptimizerStages.get_weight_for_stage(initial_stage)),
            silent=True
        )
        self.counter = 0
        self.buffer = ''

    def parse(self, string: str):
        if '[ ERROR ]' in string:
            self.emit_message.add_error(string)
            raise ModelOptimizerError(string, 1)

        speed = 0.75
        percent = min(speed * self.counter, 99)
        self.counter += 1
        self.emit_message.update_percent(percent)

        try:
            json.loads(string)
            console_output = string
        except json.decoder.JSONDecodeError:
            return
        self.emit_message.update_model_optimizer_scan_result(console_output)
