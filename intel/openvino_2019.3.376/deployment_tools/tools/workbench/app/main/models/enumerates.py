"""
 OpenVINO Profiler
 Some utilitarian functions and variables for database

 Copyright (c) 2018 Intel Corporation

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
import enum

from sqlalchemy import Enum

from app.main.jobs.job_types_enum import JobTypesEnum


class DevicesEnum(enum.Enum):
    cpu = 'CPU'
    gpu = 'GPU'
    myriad = 'MYRIAD'


DEVICE_ENUM_SCHEMA = Enum(DevicesEnum)


# pylint: disable=assign-to-new-keyword
class ApiEnum(enum.Enum):
    sync_api = 'sync'
    async_api = 'async'


API_ENUM_SCHEMA = Enum(ApiEnum)


class StatusEnum(enum.Enum):
    queued = 'queued'
    running = 'running'
    ready = 'ready'
    error = 'error'
    cancelled = 'cancelled'


STATUS_ENUM_SCHEMA = Enum(StatusEnum)


class TaskEnum(enum.Enum):
    classification = 'classification'
    object_detection = 'object_detection'
    segmentation = 'segmentation'
    generic = 'generic'

    @classmethod
    def has_value(cls, value):
        return any(value == item.value for item in cls)


TASK_ENUM_SCHEMA = Enum(TaskEnum)


class TaskMethodEnum(enum.Enum):
    classificator = 'classificator'
    generic = 'generic'
    ssd = 'ssd'
    tiny_yolo_v2 = 'tiny_yolo_v2'
    yolo_v2 = 'yolo_v2'

    @classmethod
    def has_value(cls, value):
        return any(value == item.value for item in cls)


TASK_METHOD_ENUM_SCHEMA = Enum(TaskMethodEnum)


class DatasetTypesEnum(enum.Enum):
    imagenet = 'imagenet'
    voc_object_detection = 'voc_object_detection'
    voc_segmentation = 'voc_segmentation'


DATASET_TYPES_ENUM_SCHEMA = Enum(DatasetTypesEnum)


class OptimizationTypesEnum(enum.Enum):
    inference = JobTypesEnum.compound_inference_type.value
    int8autotune = JobTypesEnum.int8autotune_type.value
    winograd_autotune = JobTypesEnum.winograd_autotune_type.value


OPTIMIZATION_TYPE_ENUM_SCHEMA = Enum(OptimizationTypesEnum)


class SupportedFrameworksEnum(enum.Enum):
    openvino = 'openvino'
    caffe = 'caffe'
    mxnet = 'mxnet'
    onnx = 'onnx'
    tf = 'tf'
    pytorch = 'pytorch'


SUPPORTED_FRAMEWORKS_ENUM_SCHEMA = Enum(SupportedFrameworksEnum)


class ModelPrecisionEnum(enum.Enum):
    fp32 = 'FP32'
    fp16 = 'FP16'
    i8 = 'INT8'
    i1 = 'INT1'
    mixed = 'MIXED'
    unknown = 'UNKNOWN'


MODEL_PRECISION_ENUM_SCHEMA = Enum(ModelPrecisionEnum)


class ModelSourceEnum(enum.Enum):
    omz = 'omz'
    original = 'original'
    ir = 'ir'


MODEL_SOURCE_ENUM_SCHEMA = Enum(ModelSourceEnum)

STATUS_PRIORITY = {
    StatusEnum.queued: 0,
    StatusEnum.ready: 1,
    StatusEnum.running: 2,
    StatusEnum.cancelled: 3,
    StatusEnum.error: 4
}
