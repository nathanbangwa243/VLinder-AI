"""
 Model Analyzer

 Copyright (c) 2019 Intel Corporation

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

import csv
import operator
import os
import logging as log
import numpy as np
from functools import reduce


class NetworkComputationalComplexity:
    def __init__(self, network, batch):
        self.batch = batch
        self.network = network
        self.computational_complexity = {}
        self.ignored_layers = []

        input_layers = filter(lambda x: x.name in self.network.inputs, self.network.layers.values())
        self.input_layers_names = [i.name for i in input_layers]

        for layer in self.network.layers.values():
            self.computational_complexity[layer.name] = {'layer_type': layer.type, 'layer_name': layer.name}

    def print_network_info(self, output, file_name, complexity, complexity_filename):
        g_flops, g_iops = self.get_total_ops()
        g_flops = '{:.3f}'.format(g_flops)
        g_iops = '{:.3f}'.format(g_iops)
        total_params = '{:.3f}'.format(self.get_total_params() / 1000000.0)
        sparsity = '{:.3f}'.format(self.get_total_sparsity())
        min_mem_consumption = '{:.3f}'.format(self.get_minimum_memory_consumption() / 1000000.0)
        max_mem_consumption = '{:.3f}'.format(self.get_maximum_memory_consumption() / 1000000.0)
        log.info('GFLOPs: %s', g_flops)
        log.info('GIOPs: %s', g_iops)
        log.info('MParams: %s', total_params)
        log.info('Sparsity: %s', sparsity)
        log.info('Minimum memory consumption: %s', min_mem_consumption)
        log.info('Maximum memory consumption: %s', max_mem_consumption)
        export_network_into_csv(g_flops, g_iops, total_params, sparsity, min_mem_consumption, max_mem_consumption, output,
                                file_name)
        if complexity:
            self.export_layers_into_csv(output, complexity_filename)

    def export_layers_into_csv(self, output_dir, file_name):
        if output_dir:
            file_name = os.path.join(output_dir, file_name)
        with open(file_name, mode='w') as info_file:
            info_writer = csv.writer(info_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            info_writer.writerow(['LayerType', 'LayerName', 'GFLOPs', 'GIOPs', 'MParams', 'LayerParams'])
            sorted_layers = sorted(self.computational_complexity.keys())
            core_layers = filter(lambda x: x not in self.input_layers_names, sorted_layers)
            for layer_name in core_layers:
                cur_layer = self.computational_complexity[layer_name]
                info_writer.writerow(
                    [
                        cur_layer['layer_type'],
                        cur_layer['layer_name'],
                        '{:.3f}'.format(float(cur_layer['g_flops'])),
                        '{:.3f}'.format(float(cur_layer['g_iops'])),
                        '{:.3f}'.format(float(cur_layer['m_params'])),
                        cur_layer['layer_params'] if 'layer_params' in cur_layer.keys() else None
                    ]
                )
        log.info('Complexity file name: %s', file_name)

    def get_ops(self, layer, is_int8=False) -> int:
        flops_per_element = 0
        output_size = self.get_blob_size(layer.shape)

        def get_parent(layer):
            parent = layer.parents[0]
            return parent.split('.')[0] if '.' in parent else parent

        if layer.type.lower() == 'convolution':
            self.computational_complexity[layer.name]['layer_params'] = get_conv_params(layer, self.network.layers)
            input_channel = self.network.layers[get_parent(layer)].shape[1]
            kernel_spatial_size = reduce(operator.mul, _get_param_values(layer.params, 'kernel', 'kernel'), 1)
            # (mul + add) x ROI size
            flops_per_element = 2 * (input_channel /
                                     int(layer.params['group'])) * kernel_spatial_size
        elif layer.type.lower() == 'deconvolution':
            self.computational_complexity[layer.name]['layer_params'] = get_conv_params(layer, self.network.layers)
            input_channel = self.network.layers[get_parent(layer)].shape[1]
            kernel_spatial_size = reduce(operator.mul, _get_param_values(layer.params, 'kernel', 'kernel'), 1)
            stride_spatial_size = reduce(operator.mul, _get_param_values(layer.params, 'strides', 'stride'), 1)
            flops_per_element = \
                2 * (input_channel / int(layer.params.get('group', 1))) * kernel_spatial_size / stride_spatial_size
        elif layer.type.lower() == 'relu':
            flops_per_element = 2  # cmp + mul
        elif layer.type.lower() == 'normalize':
            flops_per_element = 4  # cmp + mul
        elif layer.type.lower() == 'norm':
            self.computational_complexity[layer.name]['layer_params'] = get_lrn_params(layer)
            roi_size = 2 * int(layer.params['local-size'])
            if layer.params['region'] == 'across':
                roi_size = roi_size * int(layer.params['local-size'])
            flops_per_element = 2 * roi_size + 1  # (mul + add) x ROI size + div
        elif layer.type.lower() == 'pooling':
            self.computational_complexity[layer.name]['layer_params'] = get_pool_params(layer)
            kernel_spatial_size = reduce(operator.mul, _get_param_values(layer.params, 'kernel', 'kernel'), 1)
            flops_per_element = 1 * kernel_spatial_size  # (max or add) x ROI size
        elif layer.type.lower() == 'fullyconnected':
            parent = get_parent(layer)
            num = self.batch if self.batch != -1 else self.network.layers[parent].shape[0]
            input_blob = self.network.layers[parent]
            input_blob_size = self.get_blob_size(input_blob.shape)
            flops_per_element = 2 * (input_blob_size / num)
        elif layer.type.lower() == 'softmax':
            self.computational_complexity[layer.name]['layer_params'] = get_soft_max_params(layer)
            flops_per_element = 5  # max + sub + exp + sum + div
        elif layer.type.lower() == 'elu':
            flops_per_element = 3
        elif layer.type.lower() == 'eltwise':
            input_blob_count = len(layer.parents)
            flops_per_element = 2 * input_blob_count - 1
        elif layer.type.lower() == 'scaleshift' or layer.type.lower() == 'batchnormalization':
            flops_per_element = 2
        elif layer.type.lower() == 'power':
            power = float(layer.params['power'])
            flops_per_element = 2 + power - 1  # mul + add + (power - 1) x mul
        elif layer.type.lower() == 'clamp':
            flops_per_element = 2  # min + max
        elif layer.type.lower() == 'psroipooling' or layer.type.lower() == 'roipooling':
            in_dims = self.network.layers[get_parent(layer)].shape
            out_dims = layer.shape
            # real kernel sizes are read from input, so approximation is used
            size = 3 if len(in_dims) == 5 else 2
            kernel_spatial_size = 1
            for i in range(0, size):
                kernel_spatial_size *= in_dims[-1 - i] // out_dims[-1 - i]
            flops_per_element = 1 * kernel_spatial_size
        elif layer.type.lower() == 'mvn':
            flops_per_element = 5 if layer.params['normalize_variance'] == '1' else 2
        elif layer.type.lower() == 'grn':
            flops_per_element = 3
        elif layer.type.lower() == 'argmax':
            top_k = int(layer.params['top_k'])
            axis_index = int(layer.params['axis']) if 'axis' in layer.params.keys() else 0
            in_dims = self.network.layers[get_parent(layer)].shape
            roi_size = in_dims[axis_index] if axis_index != 0 else self.get_blob_size(
                self.network.layers[get_parent(layer)])
            flops_per_element = 1 * (roi_size * top_k - top_k * (top_k + 1) / 2)  # cmp x ROI size
        elif layer.type.lower() == 'prelu':
            flops_per_element = 2
        elif layer.type.lower() == 'interp':
            flops_per_element = 9
        elif layer.type.lower() == 'sigmoid':
            flops_per_element = 3
        elif layer.type.lower() == 'gemm':
            in_dims = self.network.layers[get_parent(layer)].shape
            flops_per_element = 2 * in_dims[- 1]

        total_flops = output_size * flops_per_element * pow(10, -9)
        if is_int8:
            self.computational_complexity[layer.name]['g_iops'] = total_flops
            self.computational_complexity[layer.name]['g_flops'] = 0
        else:
            self.computational_complexity[layer.name]['g_flops'] = total_flops
            self.computational_complexity[layer.name]['g_iops'] = 0
        return total_flops

    def get_total_ops(self) -> tuple():
        total_flops = 0
        total_iops = 0
        for layer in self.network.layers.values():
            if layer.params.get('quantization_level') == 'I8':
                total_iops += self.get_ops(layer, is_int8=True)
            else:
                total_flops += self.get_ops(layer)
        return total_flops, total_iops

    def get_total_params(self) -> int:
        total_params = 0
        for layer in self.network.layers.values():
            total_params += self.get_params(layer)
        return total_params

    def get_total_sparsity(self):
        total_params = 0
        zero_params = 0
        for layer in self.network.layers.values():
            if layer.name not in self.ignored_layers and layer.type.lower() in {"convolution", "fullyconnected", "scaleshift"}:
                total_params += self.get_params(layer)
                zero_params += self.get_zero_params(layer)
        return zero_params / total_params

    def get_maximum_memory_consumption(self) -> int:
        total_memory_size = 0
        for layer in self.network.layers.values():
            total_memory_size += self.get_blob_size(layer.shape)
        return total_memory_size

    def get_minimum_memory_consumption(self) -> int:
        is_computed = {}
        all_layers = list(filter(lambda x: x.name not in self.network.inputs, self.network.layers.values()))
        input_layers = list(filter(lambda x: x.name in self.network.inputs, self.network.layers.values()))
        for layer in all_layers:
            is_computed[layer.name] = False

        direct_input_children = []
        for layer in input_layers:
            for child in layer.children:
                direct_input_children.append(child)

        max_memory_size = 0

        for layer in all_layers:
            current_memory_size = 0

            if layer.name in direct_input_children:
                current_memory_size += self.get_input_blobs_total_size(layer)

            current_memory_size += self.get_output_blobs_total_size(layer)

            for prev_layer in all_layers:
                if prev_layer.name == layer.name:
                    break
                memory_not_needed = True

                for child in prev_layer.children:
                    memory_not_needed = memory_not_needed and is_computed[child]

                if not memory_not_needed:
                    current_memory_size += self.get_output_blobs_total_size(prev_layer)

            if max_memory_size < current_memory_size:
                max_memory_size = current_memory_size

            is_computed[layer.name] = True
        return max_memory_size

    def get_output_blobs_total_size(self, layer) -> int:
        for _ in layer.children:
            return self.get_blob_size(layer.shape)
        return 0

    def get_input_blobs_total_size(self, layer) -> int:
        for input_layer in layer.parents:
            return self.get_blob_size(self.network.layers[input_layer].shape)
        return 0

    def get_blob_size(self, shape):
        size = self.batch if self.batch != -1 else shape[0]
        for dim in shape[1:]:
            size *= dim
        return size

    def get_params(self, layer) -> int:
        params = 0
        if not layer.weights:
            self.computational_complexity[layer.name]['m_params'] = params
            return params
        if 'weights' in layer.weights.keys() and layer.weights['weights'] is not None:
            params += layer.weights['weights'].size
        if 'biases' in layer.weights.keys() and layer.weights['biases'] is not None:
            params += layer.weights['biases'].size
        self.computational_complexity[layer.name]['m_params'] = params / 1000000.0
        return params

    def get_zero_params(self, layer):
        zeros = 0
        if not layer.weights:
            return zeros
        if 'weights' in layer.weights.keys() and layer.weights['weights'] is not None:
            zeros += (layer.weights['weights'][np.where(np.isnan(layer.weights['weights']) == False)] == 0).sum()
        if 'biases' in layer.weights.keys() and layer.weights['biases'] is not None:
            zeros += (layer.weights['biases'][np.where(np.isnan(layer.weights['biases']) == False)] == 0).sum()
        return zeros

    def get_ignored_layers(self, ignored_layers, ignore_first_conv, ignore_fc):
        self.ignored_layers.extend(ignored_layers)
        all_convs = []
        all_fcs = []
        all_bns = []
        for l in self.network.layers.values():
            if l.type.lower() == "convolution":
                all_convs.append(l.name)
            elif l.type.lower() == "fullyconnected":
                all_fcs.append(l.name)
            elif l.type.lower() == "scaleshift":
                all_bns.append(l.name)
        self.ignored_layers.extend(all_bns)
        if ignore_first_conv:
            self.ignored_layers.append(all_convs[0])
        if ignore_fc:
            self.ignored_layers.extend(all_fcs)

    def sparsify(self, sparsity):
        data_flat = []
        for layer in self.network.layers.values():
            if layer.name not in self.ignored_layers and layer.type.lower() in {"convolution", "fullyconnected", "scaleshift"}:
                if 'weights' in layer.weights.keys() and layer.weights['weights'] is not None:
                    layer.weights['weights'][np.isnan(layer.weights['weights'])] = 0
                    data_flat.append(layer.weights['weights'].flatten())
                if 'biases' in layer.weights.keys() and layer.weights['biases'] is not None:
                    layer.weights['biases'][np.isnan(layer.weights['biases'])] = 0
                    data_flat.append(layer.weights['biases'].flatten())
        data_flat = np.concatenate(data_flat)
        data_flat = np.absolute(data_flat)
        data_flat.sort()
        sparsity_level = float(sparsity) / 100
        sparsity_threshold = data_flat[int(sparsity_level * len(data_flat))]
        for layer in self.network.layers.values():
            if layer.name not in self.ignored_layers and layer.type.lower() in {"convolution", "fullyconnected", "scaleshift"}:
                if 'weights' in layer.weights.keys() and layer.weights['weights'] is not None:
                    layer.weights['weights'][np.absolute(layer.weights['weights']) < sparsity_threshold] = 0
                if 'biases' in layer.weights.keys() and layer.weights['biases'] is not None:
                    layer.weights['biases'][np.absolute(layer.weights['biases']) < sparsity_threshold] = 0


def export_network_into_csv(g_flops, g_iops, total_params, sparsity, min_mem_consumption, max_mem_consumption, output_dir,
                            file_name):
    if output_dir:
        file_name = os.path.join(output_dir, file_name)
    with open(file_name, mode='w') as info_file:
        info_writer = csv.writer(info_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        info_writer.writerow(['GFLOPs', 'GIOPs', 'MParams', 'MinMem', 'MaxMem', 'Sparsity'])
        info_writer.writerow([g_flops, g_iops, total_params, min_mem_consumption, max_mem_consumption, sparsity])
    log.info('Network status information file name: %s', file_name)


def get_depth_params(layer, all_layers) -> str:
    group_param = int(layer.params.get('group', 1))
    res = ''
    if all_layers[layer.parents[0].split('.')[0]].shape[1] == group_param and layer.shape[1] == group_param:
        res = ' Depthwise'
    elif group_param > 1:
        res = ' group: {}'.format(group_param)

    return res


def get_conv_params(layer, all_layers) -> str:
    dilation_values = _get_param_values(layer.params, 'dilations', 'dilation')
    dilations = 'dil: ({})'.format('x'.join(map(str, dilation_values)))

    kernel_values = _get_param_values(layer.params, 'kernel', 'kernel')
    kernels = 'ker: ({})'.format('x'.join(map(str, kernel_values)))

    stride_values = _get_param_values(layer.params, 'strides', 'stride')
    strides = 'str: ({})'.format('x'.join(map(str, stride_values)))

    depth = get_depth_params(layer, all_layers)

    return '[{} {} {}{}]'.format(kernels, strides, dilations, depth)


def get_pool_params(layer) -> str:
    kernel_values = _get_param_values(layer.params, 'kernel', 'kernel')
    kernels = 'ker: ({})'.format('x'.join(map(str, kernel_values)))

    stride_values = _get_param_values(layer.params, 'strides', 'stride')
    strides = 'str: ({})'.format('x'.join(map(str, stride_values)))

    pool_method = 'pool-method: {}'.format(layer.params['pool-method'])
    return '[{} {} {}]'.format(kernels, strides, pool_method)


def get_soft_max_params(layer):
    axis = layer.params['axis'] if 'axis' in layer.params.keys() else 1
    return '[axis: {}]'.format(axis)


def get_lrn_params(layer):
    return '[local-size: {} region: {}]'.format(layer.params['local-size'], layer.params['region'])


def _get_param_values(params, multiple_form, single_form):
    if multiple_form in params:
        value_per_dimension = [int(i) for i in params[multiple_form].split(',')]
    else:
        x_key = '{}-x'.format(single_form)
        y_key = '{}-y'.format(single_form)
        z_key = '{}-z'.format(single_form)
        x_dim = int(params.get(x_key, 1))
        y_dim = int(params.get(y_key, 1))
        z_dim = int(params.get(z_key, 1))
        value_per_dimension = [x_dim, y_dim, z_dim]
    return value_per_dimension
