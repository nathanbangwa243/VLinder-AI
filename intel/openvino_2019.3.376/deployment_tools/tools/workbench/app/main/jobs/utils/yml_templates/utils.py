"""
 OpenVINO Profiler
 Yaml configuration utils

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

from app.main.jobs.utils.yml_abstractions import (
    Model, Launcher, Dataset, Metric, Preprocessing, Postprocessing, Annotation)
from app.main.jobs.utils.yml_templates.registry import ConfigRegistry
from app.main.jobs.utils.yml_templates.adapter_templates import get_classification_adapter, get_ssd_adapter
from app.main.models.enumerates import TaskMethodEnum, TaskEnum


class ModelWithMeta:
    def __init__(self, task_type: TaskEnum, task_method: TaskMethodEnum, config: [Model, dict]):
        self.task_type = task_type
        self.task_method = task_method
        self.config = config


def get_classification_template(preprocessings: list) -> Model:
    return Model(
        Launcher(
            framework='dlsdk',
            adapter=get_classification_adapter(),
            device='CPU',
            model='dummy',
            weights='dummy',
            batch=1,
            cpu_extensions=''
        ),
        Dataset(
            data_source=None,
            annotation=Annotation(has_background=False),
            preprocessing=[
                *preprocessings
            ],
            postprocessing=[],
            metrics=[
                Metric(type='accuracy', parameters={'top_k': 1, 'presenter': 'return_value'})
            ]
        )
    )


def get_object_detection_template(preprocessings: list, postprocessings: list) -> Model:
    return Model(
        Launcher(
            framework='dlsdk',
            adapter=get_ssd_adapter(),
            device='CPU',
            model='dummy',
            weights='dummy',
            batch=1,
            cpu_extensions=''
        ),
        Dataset(
            data_source=None,
            annotation=Annotation(has_background=False),
            preprocessing=[
                *preprocessings
            ],
            postprocessing=[
                *postprocessings
            ],
            metrics=[
                Metric(type='map', parameters={
                    'integral': 'max', 'overlap_threshold': 0.5, 'presenter': 'return_value'
                })
            ]
        )
    )


def extract_fields(accuracy_config: Model) -> dict:
    return {
        'preprocessing': [obj.to_dict() for obj in accuracy_config.dataset.preprocessing],
        'postprocessing': [obj.to_dict() for obj in accuracy_config.dataset.postprocessing],
        'metric': [obj.to_dict() for obj in accuracy_config.dataset.metrics],
        'hasBackground': accuracy_config.dataset.annotation.annotation_conversion['has_background']
    }


def get_config_for_topology_model(name: str) -> ModelWithMeta:
    # random config for dldt and coco models
    if name not in ConfigRegistry.model_registry:
        return ModelWithMeta(TaskEnum.object_detection, TaskMethodEnum.ssd, SSD_DEFAULT)
    accuracy_config_with_meta = ConfigRegistry.model_registry[name]()
    truncated_config = {
        **extract_fields(accuracy_config_with_meta.config),
        'taskType': accuracy_config_with_meta.task_type.value,
        'taskMethod': accuracy_config_with_meta.task_method.value
    }
    accuracy_config_with_meta.config = truncated_config
    return accuracy_config_with_meta


def get_default_config_for_classification_topologies() -> dict:
    template_classification = get_classification_template(
        [
            Preprocessing(type='auto_resize', parameters={'size': 'auto_resize'}),
            Preprocessing(type='normalization',
                          parameters={'mean': get_default_normalization(),
                                      'std': get_default_normalization()})
        ],
    )
    return {
        'taskType': TaskEnum.classification.value,
        'taskMethod': TaskMethodEnum.classificator.value,
        **extract_fields(template_classification),
    }


def get_default_config_for_object_detection_topologies() -> tuple:
    template_ssd = get_object_detection_template(
        [
            Preprocessing(type='auto_resize', parameters={'size': 'auto_resize'}),
            Preprocessing(type='normalization',
                          parameters={'mean': get_default_normalization(),
                                      'std': get_default_normalization()})
        ],
        [Postprocessing(type='resize_prediction_boxes', parameters={})],
    )
    template_yolo = get_config_for_topology_model('yolo_v2')
    template_yolo_tiny = get_config_for_topology_model('tiny_yolo_v2')

    return (
        {
            'taskType': TaskEnum.object_detection.value,
            'taskMethod': TaskMethodEnum.ssd.value,
            **extract_fields(template_ssd),
        },
        {
            'taskType': template_yolo.task_method.value,
            'taskMethod': template_yolo.task_type.value,
            **template_yolo.config
        },
        {
            'taskType': TaskEnum.object_detection.value,
            'taskMethod': TaskMethodEnum.tiny_yolo_v2.value,
            **template_yolo_tiny.config
        },
    )


def get_default_normalization() -> tuple:
    return (127.502231289,) * 3


def get_default_config_for_topologies() -> list:
    result = []
    object_detection_defaults = get_default_config_for_object_detection_topologies()
    result.append(get_default_config_for_classification_topologies())
    for config in object_detection_defaults:
        result.append(config)
    result.append({
        'hasBackground': False,
        'taskType': TaskEnum.generic.value,
        'taskMethod': TaskMethodEnum.generic.value,
        'preprocessing': [],
        'postprocessing': [],
        'metric': [],
    })
    return result


SSD_DEFAULT = {
    'taskType': 'object_detection',
    'taskMethod': 'ssd',
    'hasBackground': False,
    'preprocessing': [
        {
            'type': 'auto_resize'
        },
        {
            'type': 'normalization',
            'mean': '127.502231289, 127.502231289, 127.502231289',
            'std': '127.502231289, 127.502231289, 127.502231289'
        }
    ],
    'postprocessing': [
        {
            'type': 'resize_prediction_boxes'
        }
    ],
    'metric': [
        {
            'type': 'map',
            'integral': '11point',
            'presenter': 'return_value',
            'overlap_threshold': 0.5
        }
    ]
}
