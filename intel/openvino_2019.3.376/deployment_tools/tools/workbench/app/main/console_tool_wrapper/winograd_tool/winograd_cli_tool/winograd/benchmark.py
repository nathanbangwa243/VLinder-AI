"""
 OpenVINO Profiler
 Functions for run the benchmark application

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

import os
# pylint: disable=import-error,no-name-in-module
from openvino.tools.benchmark.benchmark import Benchmark
from openvino.tools.benchmark.utils.infer_request_wrap import InferRequestsQueue
from openvino.tools.benchmark.utils.inputs_filling import get_inputs
from openvino.tools.benchmark.utils.logging import logger
from openvino.tools.benchmark.utils.progress_bar import ProgressBar
from openvino.tools.benchmark.utils.utils import next_step, read_network, config_network_inputs, dump_exec_graph, \
    get_number_iterations, process_help_inference_string


def benchmark_app(args):
    try:
        # ------------------------------ 2. Loading Inference Engine ---------------------------------------------------
        next_step()

        benchmark = Benchmark(args.target_device, args.number_infer_requests,
                              args.number_iterations, args.time, args.api_type)

        benchmark.add_extension(args.path_to_extension, args.path_to_cldnn_config)

        version = benchmark.get_version_info()

        logger.info(version)

        # --------------------- 3. Read the Intermediate Representation of the network ---------------------------------
        next_step()

        ie_network = read_network(args.path_to_model)

        # --------------------- 4. Resizing network to match image sizes and given batch -------------------------------

        next_step()
        if args.batch_size and args.batch_size != ie_network.batch_size:
            benchmark.reshape(ie_network, args.batch_size)
        batch_size = ie_network.batch_size
        logger.info('Network batch size: %s, precision: %s', ie_network.batch_size, ie_network.precision)

        # --------------------- 5. Configuring input of the model ------------------------------------------------------
        next_step()

        config_network_inputs(ie_network)

        # --------------------- 6. Setting device configuration --------------------------------------------------------
        next_step()
        benchmark.set_config(args.number_streams, args.api_type, args.number_threads, args.infer_threads_pinning)

        # --------------------- 7. Loading the model to the device -----------------------------------------------------
        next_step()

        perf_counts = False
        if args.perf_counts or args.exec_graph_path:
            perf_counts = True
        exe_network = benchmark.load_network(ie_network, perf_counts, args.number_infer_requests)

        # --------------------- 8. Setting optimal runtime parameters --------------------------------------------------
        next_step()

        # Number of requests
        infer_requests = exe_network.requests
        benchmark.nireq = len(infer_requests)

        # Iteration limit
        benchmark.niter = get_number_iterations(benchmark.niter, len(exe_network.requests), args.api_type)

        # ------------------------------------ 9. Creating infer requests and filling input blobs ----------------------
        next_step()

        request_queue = InferRequestsQueue(infer_requests)

        path_to_input = os.path.abspath(args.path_to_input) if args.path_to_input else None
        requests_input_data = get_inputs(path_to_input, batch_size, ie_network.inputs, infer_requests)


        # ------------------------------------ 10. Measuring performance -----------------------------------------------

        output_string = process_help_inference_string(benchmark)

        next_step(output_string)
        progress_bar_total_count = 10000
        if benchmark.niter and not benchmark.duration_seconds:
            progress_bar_total_count = benchmark.niter

        progress_bar = ProgressBar(progress_bar_total_count, args.stream_output, args.progress)

        benchmark.infer(request_queue, requests_input_data, batch_size, progress_bar)

        # ------------------------------------ 11. Dumping statistics report -------------------------------------------
        next_step()

        if args.exec_graph_path:
            dump_exec_graph(exe_network, args.exec_graph_path)

        del exe_network

        next_step.step_id = 0
    except Exception as exc:
        logger.exception(exc)
