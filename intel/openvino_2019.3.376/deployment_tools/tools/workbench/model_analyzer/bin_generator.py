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

import os
import defusedxml.ElementTree


def get_weights_size(xml_file):
    element = defusedxml.ElementTree.parse(xml_file).getroot()
    weights = []
    biases = []
    biggest_weight_offset = {'offset': 0, 'size': 0}
    biggest_bias_offset = {'offset': 0, 'size': 0}
    for child in element.iter():
        if child.tag == 'weights':
            weights.append(child.attrib)
            if int(child.attrib['offset']) > int(biggest_weight_offset['offset']):
                biggest_weight_offset = child.attrib

        if child.tag == 'biases':
            biases.append(child.attrib)
            if int(child.attrib['offset']) > int(biggest_bias_offset['offset']):
                biggest_bias_offset = child.attrib
    end_biases = int(biggest_bias_offset['offset']) + int(biggest_bias_offset['size'])
    end_weights = int(biggest_weight_offset['offset']) + int(biggest_weight_offset['size'])
    return end_biases if end_biases > end_weights else end_weights


def generate_bin(xml_file, path_to_new_bin_file=''):
    size = get_weights_size(xml_file)
    file_size = size*4  # size in bytes
    if not path_to_new_bin_file:
        path_to_new_bin_file = '{}.bin'.format(os.path.splitext(xml_file)[0])
    with open(path_to_new_bin_file, "wb") as file:
        file.write(os.urandom(file_size))
    return path_to_new_bin_file


def remove_generated_file(filename):
    os.remove(filename)
