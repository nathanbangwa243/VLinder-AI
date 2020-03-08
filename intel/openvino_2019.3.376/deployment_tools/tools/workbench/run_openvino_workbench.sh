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

HTTP_PROXY=${http_proxy}
HTTPS_PROXY=${https_proxy}
NO_PROXY=${no_proxy}

help_message=$"usage run_openvino_workbench.sh -PACKAGE_PATH \${PATH_TO_OPENVINO_PACKAGE}
                                              [-HTTP_PROXY \${http_proxy}]
                                              [-HTTPS_PROXY \${https_proxy}]
                                              [-NO_PROXY \${no_proxy}]

required parameters:

    -PACKAGE_PATH - path to OpenVINO package for Ubuntu 16 in tar.gz format

optional parameters:

    -HTTP_PROXY - HTTP proxy in format:  http://<user>:<password>@<proxy-host>:<proxy-port>/
    -HTTPS_PROXY - HTTPS proxy in format:  https://<user>:<password>@<proxy-host>:<proxy-port>/
    -NO_PROXY - URLs that should be excluded from proxying
"

if [ $# -eq 0 ]
  then
    echo "$help_message"
    exit -1
fi

while test $# -gt 0; do
    case "$1" in
        -PACKAGE_PATH)
            shift
            PACKAGE_PATH=$1
            shift
            ;;
        -HTTP_PROXY)
            shift
            HTTP_PROXY=$1
            shift
            ;;
        -HTTPS_PROXY)
            shift
            HTTPS_PROXY=$1
            shift
            ;;
        -NO_PROXY)
            shift
            NO_PROXY=$1
            shift
            ;;
        *)
            echo "$help_message"
            exit -1
            ;;
    esac
done

if [ -z "PACKAGE_PATH" ]; then
    echo "Argument PACKAGE_PATH is required"
    exit 1
fi

if [ "$(docker ps -a | grep workbench)" ]; then
    echo "Stop old workbench docker container"
    docker stop workbench
    echo "Remove old workbench docker container"
    docker rm workbench
fi

set -e

mkdir -p ~/workbench/user_data

./build_docker.sh -PACKAGE_PATH ${PACKAGE_PATH} \
                  -HTTP_PROXY "${HTTP_PROXY}" \
                  -HTTPS_PROXY "${HTTPS_PROXY}" \
                  -NO_PROXY "${NO_PROXY}"

docker run -p 127.0.0.1:5665:5665 \
            --name workbench \
            --privileged \
            -v /dev/bus/usb:/dev/bus/usb \
            -v ~/workbench/user_data:/home/openvino/workbench/usr_data \
            -e PROXY_HOST_ADDRESS=0.0.0.0 \
            -e http_proxy="${HTTP_PROXY}" \
            -e https_proxy="${HTTPS_PROXY}" \
            -e no_proxy="${NO_PROXY}" \
            -it workbench
