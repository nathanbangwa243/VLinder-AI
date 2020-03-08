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

set -e

help_message=$"usage build_docker.sh -PACKAGE_PATH \${PATH_TO_OPENVINO_PACKAGE}
                                              [-IMAGE_NAME \${image_name}]
                                              [-HTTP_PROXY \${http_proxy}]
                                              [-HTTPS_PROXY \${https_proxy}]
                                              [-NO_PROXY \${no_proxy}]

required parameters:

    -PACKAGE_PATH - path to OpenVINO package for Ubuntu 16 in tar.gz format

optional parameters:
    -IMAGE_NAME - name of the new Docker image to be built
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
        -IMAGE_NAME)
            shift
            IMAGE_NAME=$1
            shift
            ;;
        *)
            echo "$1 is not a recognized flag!"
            exit -1
            ;;
    esac
done

if [ -z "PACKAGE_PATH" ]; then
    echo "Argument PACKAGE_PATH is required"
    exit 1
fi

mkdir -p /tmp/workbench
cp -r Dockerfile.package /tmp/workbench/Dockerfile
cp -r install_RPMS.sh /tmp/workbench/install_RPMS.sh
cp -r $PACKAGE_PATH /tmp/workbench/

IMAGE_NAME=${IMAGE_NAME:-'workbench'}

pushd /tmp/workbench

docker build -t ${IMAGE_NAME} . \
       --build-arg https_proxy=${HTTP_PROXY} \
       --build-arg http_proxy=${HTTPS_PROXY} \
       --build-arg no_proxy=${NO_PROXY} \
       --build-arg rabbitmq_password=openvino \
       --build-arg db_password=openvino

popd

rm -rf /tmp/workbench

echo -n "Run command for start the Workbench docker container: "
echo -n "\"docker run "
echo -n "-p 127.0.0.1:5665:5665 "
echo -n "--name workbench "
echo -n "-v /dev/bus/usb:/dev/bus/usb "
echo -n "--privileged "
echo -n "-v data:/home/openvino/workbench/app/data "
echo -n "-v ~/workbench/user_data:/home/openvino/workbench/usr_data "
echo -n "-e PROXY_HOST_ADDRESS=0.0.0.0 "
echo -n "-it ${IMAGE_NAME}\""
echo ""
