"""
 OpenVINO Profiler
 Utils to prepare templates for accuracy checker

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
import json

from app.main.models.datasets_model import DatasetsModel
from app.main.models.enumerates import DevicesEnum
from app.main.models.topologies_model import TopologiesModel
from app.main.jobs.utils.yml_abstractions import Model, Launcher, Dataset, Preprocessing, Postprocessing, Metric
from app.main.jobs.utils.yml_templates import ConfigRegistry
from app.main.utils.utils import find_all_paths
from config.constants import CPU_EXTENSIONS_PATH


def construct_accuracy_tool_config(
        topology_record: TopologiesModel,
        dataset_record: DatasetsModel,
        device: DevicesEnum) -> Model:

    advanced_configuration = json.loads(topology_record.meta.advanced_configuration)

    launcher_config = Launcher(
        adapter=ConfigRegistry.task_method_registry[topology_record.meta.topology_type](),
        device=device.value,
        model=find_all_paths(topology_record.path, ('.xml',))[0],
        weights=find_all_paths(topology_record.path, ('.bin',))[0],
        cpu_extensions=CPU_EXTENSIONS_PATH
    )

    dataset_adapter = ConfigRegistry.dataset_adapter_registry[dataset_record.dataset_type](dataset_record.path)
    annotation_object = dataset_adapter.to_annotation()
    if 'hasBackground' in advanced_configuration:
        annotation_conversion = annotation_object['annotation'].annotation_conversion
        annotation_conversion['has_background'] = advanced_configuration['hasBackground']
    dataset_config = Dataset(
        data_source=annotation_object['data_source'],
        annotation=annotation_object['annotation'],
        preprocessing=Preprocessing.from_list(advanced_configuration['preprocessing']),
        postprocessing=Postprocessing.from_list(advanced_configuration['postprocessing']),
        metrics=Metric.from_list(advanced_configuration['metric'])
    )

    return Model(launcher_config, dataset_config)
