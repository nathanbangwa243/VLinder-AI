"""
 OpenVINO Profiler
 Accuracy checker's configuration registry

 Copyright (c) 2018-2019 Intel Corporation

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
from app.main.jobs.utils.yml_templates.registry import register_task_method_adapter
from app.main.jobs.utils.yml_abstractions.typed_parameter import Adapter
from app.main.models.enumerates import TaskMethodEnum


@register_task_method_adapter(TaskMethodEnum.ssd)
def get_ssd_adapter() -> Adapter:
    return Adapter(type='ssd', parameters={})


@register_task_method_adapter(TaskMethodEnum.classificator)
def get_classification_adapter() -> Adapter:
    return Adapter(type='classification', parameters={})


@register_task_method_adapter(TaskMethodEnum.yolo_v2)
def get_yolo_v2_adapter() -> Adapter:
    return Adapter(type='yolo_v2', parameters={
        'anchors': 'yolo_v2'
    })


@register_task_method_adapter(TaskMethodEnum.tiny_yolo_v2)
def get_tiny_yolo_v2_adapter() -> Adapter:
    return Adapter(type='yolo_v2', parameters={
        'anchors': 'tiny_yolo_v2'
    })
