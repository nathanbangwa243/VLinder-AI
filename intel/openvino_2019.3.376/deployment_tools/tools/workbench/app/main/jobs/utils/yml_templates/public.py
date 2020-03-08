"""
 OpenVINO Profiler
 Accuracy checker's configuration for public models

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
from app.main.jobs.utils.yml_templates.registry import register_model_configuration
from app.main.jobs.utils.yml_abstractions import Preprocessing, Postprocessing
from app.main.jobs.utils.yml_templates.utils import get_object_detection_template, ModelWithMeta
from app.main.models.enumerates import TaskEnum, TaskMethodEnum


@register_model_configuration('yolo_v2')
def get_yolo_v2_configuration() -> ModelWithMeta:
    template = get_object_detection_template(
        [Preprocessing(type='auto_resize', parameters={}),
         Preprocessing(type='normalization',
                       parameters={'mean': [127.502231289, 127.502231289, 127.502231289],
                                   'std': [127.502231289, 127.502231289, 127.502231289]}),
         ],
        [
            Postprocessing(type='resize_prediction_boxes', parameters={}),
            Postprocessing(type='nms', parameters={'overlap': 0.5}),
        ],
    )

    template.dataset.annotation.annotation_conversion['has_background'] = False

    return ModelWithMeta(TaskEnum.object_detection, TaskMethodEnum.yolo_v2, template)


@register_model_configuration('tiny_yolo_v2')
def get_tiny_yolo_v2_configuration() -> ModelWithMeta:
    template = get_object_detection_template(
        [Preprocessing(type='auto_resize', parameters={}),
         Preprocessing(type='normalization',
                       parameters={'mean': [127.502231289, 127.502231289, 127.502231289],
                                   'std': [127.502231289, 127.502231289, 127.502231289]}),
         ],
        [
            Postprocessing(type='resize_prediction_boxes', parameters={}),
            Postprocessing(type='nms', parameters={'overlap': 0.5}),
        ],
    )

    template.dataset.annotation.annotation_conversion['has_background'] = False

    return ModelWithMeta(TaskEnum.object_detection, TaskMethodEnum.tiny_yolo_v2, template)
