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

service rabbitmq-server start
rabbitmqctl add_user openvino ${RABBITMQ_PASSWORD}
rabbitmqctl add_vhost openvino_vhost
rabbitmqctl set_user_tags openvino openvino_tag
rabbitmqctl set_permissions -p openvino_vhost openvino ".*" ".*" ".*"