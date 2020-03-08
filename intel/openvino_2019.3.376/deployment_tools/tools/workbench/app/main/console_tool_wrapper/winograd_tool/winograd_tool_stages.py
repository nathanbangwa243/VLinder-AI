"""
 OpenVINO Profiler
 Class for storing winograd tool stages

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
from app.main.console_tool_wrapper.inference_engine_tool.stages import Stages


class WinogradToolStages(Stages):
    stages = {
        'Start preparation step': 0.1,
        'Collect performance statistics from original model': 0.2,
        'Enable winograd for all convolution layers': 0.2,
        'Choose winograd primitive for layers that benefit from this this optimization': 0.2,
        'Write optimized network to IR file': 0.3
    }
    errors = [
        'Winograd primitive does not speed up this model',
        'Winograd disabled for all layers'
    ]
