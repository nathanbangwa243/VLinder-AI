"""
 OpenVINO Profiler
 Class for storing parameters of running the benchmark application

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


# pylint: disable=too-many-instance-attributes
class BenchmarkAppParameters:
    def __init__(self, data):
        self.path_to_model = data.path_to_model
        self.path_to_input = data.path_to_input
        self.path_to_extension = data.path_to_extension
        self.time = data.inference_time
        self.target_device = 'CPU'
        self.api_type = 'sync'
        self.number_infer_requests = 1
        self.number_iterations = 1
        self.number_threads = 1
        self.perf_counts = False
        self.number_streams = '1'
        self.batch_size = 1
        self.exec_graph_path = None
        self.infer_threads_pinning = 'YES'
        self.stream_output = False
        self.progress = False
        self.path_to_cldnn_config = False

    def set_exec_graph_path(self, path: str):
        self.exec_graph_path = path
