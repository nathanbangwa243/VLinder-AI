"""
 OpenVINO Profiler
 Class for processing errors from Benchmark application

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
from app.main.console_tool_wrapper.inference_engine_tool.error_message_processor import ErrorMessageProcessor


class BenchmarkErrorMessageProcessor(ErrorMessageProcessor):
    match_error = {
        'no images found':
            'Unable to load dataset',
        'no inputs info is provided':
            'Inference Engine is unable to find information about model inputs',
        'only networks with one input are supported':
            'Multi-input models are not supported by benchmark app',
        'unsupported model for batch size changing in automatic mode':
            'Batch can not be changed for the model',
        'the plugin does not support FP':
            'This device does not support this precision',
        'Unsupported precision FP32':
            'This device does not support this precision'
    }

    @staticmethod
    def general_error(stage):
        return 'Benchmarking application failed in stage {}'.format(stage) \
            if stage else 'Failed to execute the benchmarking application'
