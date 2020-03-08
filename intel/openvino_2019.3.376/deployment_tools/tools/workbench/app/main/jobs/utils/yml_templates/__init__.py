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
from app.main.jobs.utils.yml_templates.adapter_templates import \
    get_classification_adapter, get_ssd_adapter, get_tiny_yolo_v2_adapter, get_yolo_v2_adapter
from app.main.jobs.utils.yml_templates.public import get_yolo_v2_configuration, get_tiny_yolo_v2_configuration
from app.main.jobs.utils.yml_templates.registry import ConfigRegistry
from app.main.jobs.utils.yml_templates.dataset_adapters import (ImagenetDatasetAdapter, VOCDetectionDatasetAdapter)
