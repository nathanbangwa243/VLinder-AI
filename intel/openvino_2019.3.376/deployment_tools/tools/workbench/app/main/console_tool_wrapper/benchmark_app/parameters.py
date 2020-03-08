"""
 OpenVINO Profiler
 Class for storing int8 calibration cli params

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
from app.main.jobs.single_inference.single_inference_config import SingleInferenceConfig
from app.main.console_tool_wrapper.inference_engine_tool.parameters import Parameters
from config.constants import IE_BIN_PATH


class BenchmarkAppParameters(Parameters):
    def __init__(self, config: SingleInferenceConfig, path: str = IE_BIN_PATH):
        super(BenchmarkAppParameters, self).__init__(path, 'benchmark_app', config)
        self.params['b'] = config.batch
        self.params['nireq'] = config.nireq
        if self.params['d'] != 'MYRIAD':
            self.params['nstreams'] = config.nireq
        self.params['t'] = int(config.inference_time)
