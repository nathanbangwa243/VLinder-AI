# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os


def create_mo_folder_structure(path):
    path_dict = {
        'ext_path': os.path.join(path, './user_mo_extensions'),
        'ops_path': os.path.join(path, './user_mo_extensions', 'ops'),
        'ext_front_path': os.path.join(path, './user_mo_extensions', 'front'),
        'caffe_extr_path': os.path.join(path, './user_mo_extensions/front', 'caffe'),
        'mxnet_extr_path': os.path.join(path, './user_mo_extensions/front', 'mxnet'),
        'tf_extr_path': os.path.join(path, './user_mo_extensions/front', 'tf')
    }

    for k, val_path in path_dict.items():
        if not os.path.exists(val_path):
            os.makedirs(val_path)
        if not os.path.isfile(os.path.join(val_path, '__init__.py')):
            f = open(os.path.join(val_path, '__init__.py'),"w")
            f.close()

    return path_dict['caffe_extr_path'], path_dict['mxnet_extr_path'], path_dict['tf_extr_path'], path_dict['ops_path']


def create_ie_folder_structure(path):
    ie_cpu_path = os.path.join(path, './user_ie_extensions/cpu')
    ie_gpu_path = os.path.join(path, './user_ie_extensions/gpu')
    if not os.path.exists(ie_cpu_path):
        os.makedirs(ie_cpu_path)

    if not os.path.exists(ie_gpu_path):
        os.makedirs(ie_gpu_path)

    return ie_cpu_path, ie_gpu_path
