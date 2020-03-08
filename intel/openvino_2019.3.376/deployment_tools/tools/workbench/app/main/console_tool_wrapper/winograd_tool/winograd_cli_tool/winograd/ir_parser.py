"""
 OpenVINO Profiler
 Class for parsing IR of a network and set up attributes of layers of the network

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
import defusedxml.ElementTree as et

from .utils import to_time


class IRParser:
    def __init__(self, path):
        self.ir_graph = et.parse(path)

    def write_ir(self, path):
        self.ir_graph.write(path)

    def get_layers(self):
        return self.ir_graph.getroot().findall('layers')[0].findall('layer')

    @staticmethod
    def get_data_for_layer(layer):
        return layer.findall('data')[0]

    @staticmethod
    def enable_winograd_for_layer(layer):
        data = IRParser.get_data_for_layer(layer)
        data.set('PrimitivesPriority', 'cpu:jit_avx512_winograd')

    def enable_winograd_for_model(self):
        for layer in self.get_layers():
            if layer.get('type') == 'Convolution':
                print('Enable Winograd for: {}'.format(layer.get('name')))
                self.enable_winograd_for_layer(layer)

    def enable_winograd_for_layers(self, wino_names):
        for layer in self.get_layers():
            if layer.get('name') in wino_names:
                self.enable_winograd_for_layer(layer)

    @staticmethod
    def disable_winograd_for_layer(layer):
        data = IRParser.get_data_for_layer(layer)
        del data.attrib['PrimitivesPriority']

    def disable_winograd_for_model(self):
        for layer in self.get_layers():
            if layer.get('type') == 'Convolution':
                self.disable_winograd_for_layer(layer)

    def get_performance_table(self):
        performance_table = list()

        for layer in self.get_layers():
            data = self.get_data_for_layer(layer)
            performance = dict()
            performance['name'] = layer.get('name')
            performance['type'] = layer.get('type')
            performance['execTimeMcs'] = data.get('execTimeMcs')
            performance['originalLayersNames'] = data.get('originalLayersNames')
            performance_table.append(performance)

        return performance_table

    def get_total_time(self):
        total_time = 0

        for layer in self.get_layers():
            data = self.get_data_for_layer(layer)
            total_time += to_time(data.get('execTimeMcs'))

        return total_time

    def get_winograd_layers(self):
        res = []
        convolutions = tuple(filter(lambda layer: layer.get('type') == 'Convolution', self.get_layers()))
        for convolution in convolutions:
            data = self.get_data_for_layer(convolution)
            if data.get('PrimitivesPriority') == 'cpu:jit_avx512_winograd':
                res.append(convolution.get('name'))
        return res
