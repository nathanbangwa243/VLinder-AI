#!/bin/bash
# Copyright (c) 2018-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

help_message=$"usage install_RPMS.sh \${PATH_TO_UNPACKED_OPENVINO_PACKAGE}

required parameters:

    \${PATH_TO_UNPACKED_OPENVINO_PACKAGE} - path to unpacked OpenVINO package for Ubuntu 16

"
if [ $# -eq 0 ]
  then
    echo "$help_message"
    exit -1
fi

path_to_package=$1

set -ex

declare -a rpms_to_install

ubuntu_version=ubuntu-bionic

rpms_to_install=("intel-openvino-dl-workbench-*64.rpm" \
                 "intel-openvino-gfx-driver-${ubuntu_version}-*64.rpm" \
                 "intel-openvino-ie-bin-3rd-debug-*64.rpm" \
                 "intel-openvino-ie-bin-python-tools-${ubuntu_version}-*64.rpm" \
                 "intel-openvino-ie-rt-20*_64.rpm" \
                 "intel-openvino-ie-rt-core-${ubuntu_version}-*64.rpm" \
                 "intel-openvino-ie-rt-cpu-${ubuntu_version}-*64.rpm" \
                 "intel-openvino-ie-rt-gpu-${ubuntu_version}-*64.rpm" \
                 "intel-openvino-ie-rt-vpu-${ubuntu_version}-*64.rpm" \
                 "intel-openvino-ie-samples-*64.rpm" \
                 "intel-openvino-ie-sdk-${ubuntu_version}-*64.rpm" \
                 "intel-openvino-model-optimizer-*64.rpm" \
                 "intel-openvino-omz-dev-*64.rpm" \
                 "intel-openvino-omz-tools-*64.rpm" \
                 "intel-openvino-opencv-generic-*.noarch.rpm" \
                 "intel-openvino-opencv-lib-${ubuntu_version}-*64.rpm" \
                 "intel-openvino-setupvars-*.noarch.rpm" )

path_to_rpms=$path_to_package/rpm
cd /
for rpm in ${rpms_to_install[@]}; do 
    rpm2cpio $path_to_rpms/${rpm} | sudo cpio -idvm
done
cd -
ln -s /opt/intel/openvino* /opt/intel/openvino
