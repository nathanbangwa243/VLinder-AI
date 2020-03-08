"""
 OpenVINO Profiler
 Class for processing the winograd optimization for a network

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

import ntpath
import os
import shutil
import logging as log

from .benchmark import benchmark_app
from .ir_parser import IRParser
from .benchmark_app_parameters import BenchmarkAppParameters
from .utils import to_time


class WinogradAutotuner:
    def __init__(self, data, output_path: str):
        self.benchmark_app_parameters = BenchmarkAppParameters(data)
        model_xml_file_name, _ = os.path.splitext(ntpath.basename(self.benchmark_app_parameters.path_to_model))
        self.output_model_path = os.path.join(output_path, '{}_winograd.xml'.format(model_xml_file_name))

        self.report_dir = output_path
        self.base_exec_graph_path = os.path.join(self.report_dir, 'base_exec_graph.xml')
        self.wino_exec_graph_path = os.path.join(self.report_dir, 'winograd_exec_graph.xml')
        self.tuned_exec_graph_path = os.path.join(self.report_dir, 'tuned_exec_graph.xml')
        self.wino_threshold = 1.1

    def get_winograd_layers_list(self, base_performance_table, wino_performance_table) -> list:
        winograd_layers = list()

        convolution_layers = tuple(filter(lambda layer: layer['type'] == 'Convolution', base_performance_table))
        for base_item in convolution_layers:
            wino_item = \
                [wino_item for wino_item in wino_performance_table if wino_item['name'] == base_item['name']][0]

            base_item_time = to_time(base_item['execTimeMcs'])
            wino_item_time = to_time(wino_item['execTimeMcs'])

            base_item_orig_layers = base_item['originalLayersNames'].split(',')
            wino_item_orig_layers = wino_item['originalLayersNames'].split(',')

            min_layers_tuple, max_layers_tuple = self.get_order_of_layers_tuples(wino_item_orig_layers,
                                                                                 base_item_orig_layers)
            cumulative_item_time = wino_item_time

            if min_layers_tuple == base_item_orig_layers:
                cumulative_item_time = base_item_time

            for original_layer in min_layers_tuple:
                if original_layer not in max_layers_tuple:
                    orig_layer_time = to_time([orig_layer for orig_layer in min_layers_tuple if
                                               orig_layer['name'] == orig_layer][0]['execTimeMcs'])
                    cumulative_item_time += orig_layer_time
                    base_item_time += orig_layer_time

            log.debug('Layer %s: base_time = %s, wino_time = %s', base_item['name'], base_item_time, wino_item_time)
            if wino_item_time * self.wino_threshold < base_item_time:
                winograd_layers.append(base_item['name'])

        return winograd_layers

    @staticmethod
    def get_order_of_layers_tuples(wino_item_orig_layers, base_item_orig_layers):
        if len(wino_item_orig_layers) < len(base_item_orig_layers):
            return wino_item_orig_layers, base_item_orig_layers
        return base_item_orig_layers, wino_item_orig_layers

    def run(self):
        print('Start preparation step')
        original_bin_path = os.path.splitext(self.benchmark_app_parameters.path_to_model)[0] + '.bin'
        winograd_bin_path = os.path.splitext(self.output_model_path)[0] + '.bin'
        os.makedirs(self.report_dir, exist_ok=True)
        shutil.copy(original_bin_path, winograd_bin_path)

        print('Collect performance statistics from original model')
        self.benchmark_app_parameters.set_exec_graph_path(self.base_exec_graph_path)
        model_parser = IRParser(self.benchmark_app_parameters.path_to_model)
        benchmark_app(self.benchmark_app_parameters)
        base_exec_graph_parser = IRParser(self.base_exec_graph_path)
        base_performance_table = base_exec_graph_parser.get_performance_table()

        print('Enable winograd for all convolution layers')
        model_parser.enable_winograd_for_model()
        model_parser.write_ir(self.output_model_path)
        self.benchmark_app_parameters.set_exec_graph_path(self.wino_exec_graph_path)
        self.benchmark_app_parameters.path_to_model = self.output_model_path
        benchmark_app(self.benchmark_app_parameters)
        wino_exec_graph_parser = IRParser(self.wino_exec_graph_path)
        wino_performance_table = wino_exec_graph_parser.get_performance_table()

        print('Choose winograd primitive for layers that benefit from this optimization')
        wino_layers = self.get_winograd_layers_list(base_performance_table, wino_performance_table)
        model_parser.disable_winograd_for_model()
        model_parser.enable_winograd_for_layers(wino_layers)
        model_parser.write_ir(self.output_model_path)

        self.benchmark_app_parameters.set_exec_graph_path(self.tuned_exec_graph_path)
        self.benchmark_app_parameters.path_to_model = self.output_model_path
        benchmark_app(self.benchmark_app_parameters)
        # Changing of convolution implementation to winograd may
        # lead to inserting new reorder layers into execution graph
        # Currently there is no functionality to correctly take reorder execution time into consideration
        # So in case when reorders overhead becomes bigger
        # than convolution improvements we disable winograd for all layers
        tuned_exec_graph_parser = IRParser(self.tuned_exec_graph_path)
        if os.path.exists(self.base_exec_graph_path):
            os.remove(self.base_exec_graph_path)
        if os.path.exists(self.wino_exec_graph_path):
            os.remove(self.wino_exec_graph_path)
        if os.path.exists(self.tuned_exec_graph_path):
            os.remove(self.tuned_exec_graph_path)
        winograd_layers = model_parser.get_winograd_layers()

        if base_exec_graph_parser.get_total_time() < tuned_exec_graph_parser.get_total_time():
            print('Winograd primitive does not speed up this model')
            if os.path.exists(self.output_model_path):
                os.remove(self.output_model_path)
            return
        model_parser.write_ir(self.output_model_path)
        if winograd_layers:
            print('Winograd enabled for layers:')
            for layer in wino_layers:
                print(layer)
        else:
            print('Winograd disabled for all layers')
        print('Write optimized network to IR file: {}'.format(self.output_model_path))
