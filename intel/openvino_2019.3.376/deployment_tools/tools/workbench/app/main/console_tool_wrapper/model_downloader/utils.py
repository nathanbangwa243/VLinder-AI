"""
 OpenVINO Profiler
 Utils for work with OMZ tools

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
import logging
from pathlib import Path
from typing import Union

import yaml

from app.extensions_factories.database import get_db
from app.main.console_tool_wrapper.model_downloader.info_dumper_parameters import InfoDumperParameters
from app.main.console_tool_wrapper.model_downloader.info_dumper_parser import InfoDumperParser
from app.main.jobs.tools_runner.runner import run_console_tool
from app.main.jobs.utils.yml_templates import ConfigRegistry
from app.main.models.enumerates import TaskEnum, TaskMethodEnum, ModelPrecisionEnum, SupportedFrameworksEnum
from app.main.models.factory import write_record
from app.main.models.omz_topology_model import OMZTopologyModel
from config.constants import ACCURACY_CHECKER_PATH


def define_topology_task_type(topology):
    if topology['task_type'] == 'classification':
        return TaskEnum.classification
    if topology['task_type'] == 'detection':
        return TaskEnum.object_detection
    return TaskEnum.generic


def extract_single_config(model_name, yaml_file_content) -> Union[dict, None]:
    """
    Find and return a suitable config, if exists.

    There are usually a couple of configs in a YAML file:

    1. for the original model;
    2. for the IR converted from the original model.

    We need the second config.
    It has a launcher with framework property equal to 'dlsdk'.
    """

    ir_configs = [
        config
        for config in yaml_file_content['models']
        if 'dlsdk' in [launcher['framework'] for launcher in config['launchers']]
    ]

    if not ir_configs:
        logging.warning('No sutable configs in %s.yml. Skipping.', model_name)
        return None
    if len(ir_configs) > 1:
        logging.warning(
            'More then one sutable config in %s.yml. Using the first one: %s.', model_name, ir_configs[0]['name'])

    return ir_configs[0]


def extract_dataset_config(model_name, config) -> Union[dict, None]:
    """
    Find and return a dataset config from a given model config, if exists.

    Model config is expected to have "datasets" section,
    which is expected to have one entry with a dataset-related configurations.

    There may be special cases:

    * the section may be absent;
    * the section may be empty;
    * the section may have several enrties.

    These cases should not lead to fatal errors.
    """

    datasets_section = config.get('datasets')
    if datasets_section is None:
        logging.warning('No "datasets" section in %s config from %s.yml. Skipping.', config['name'], model_name)
        return None
    if not datasets_section:
        logging.warning(
            'No entries in the "datasets" section of %s config from %s.yml. Skipping.', config['name'], model_name)
        return None
    if len(datasets_section) > 1:
        logging.warning(
            'More then one sutable entry in the "datasets" section of %s config from %s.yml. Using the first one.',
            config['name'],
            model_name
        )
    return datasets_section[0]


def extract_launcher_config(model_name, config) -> Union[dict, None]:
    """
    Find and return a suitable launcher config from a given model config, if exists.

    Model config is expected to have "launchers" section,
    which usually has a couple of entries,
    having "framework" property equal to "dlsdk"
    and differing only in precision of the IR used.
    We can use any of them.

    There may be special cases:

    * the section may be absent;
    * the section may be empty;
    * the section may have an entry with framework property not equal to "dlsdk".

    These cases should not lead to fatal errors.
    """

    launchers_section = config.get('launchers')
    if launchers_section is None:
        logging.warning('No "launchers" section in %s config from %s.yml. Skipping.', config['name'], model_name)
        return None
    launchers_section = [
        launcher_config
        for launcher_config in launchers_section
        if launcher_config['framework'] == 'dlsdk'
    ]
    if not launchers_section:
        logging.warning(
            'No entries for "dlsdk" framework in the "launchers" section of %s config from %s.yml. Skipping.',
            config['name'],
            model_name
        )
        return None
    return launchers_section[0]


def merge_defaults(defaults: dict, config: dict, identifier_key: str) -> dict:
    identifier = config[identifier_key]
    default = defaults[identifier] if identifier in defaults else {}
    result = dict(**default)
    result.update(config)
    return result


def get_reversed_supported_adapters_map():
    """Return adapter to topology_type map for supported topology_types."""
    return {
        json.dumps(value().to_dict(), sort_keys=True): key
        for key, value in ConfigRegistry.task_method_registry.items()
    }


def get_metadata_for_omz_models():
    accuracy_checker_path = Path(ACCURACY_CHECKER_PATH)
    defaults_file_path = accuracy_checker_path / 'dataset_definitions.yml'
    configs_dir_path = Path(accuracy_checker_path) / 'configs'
    adapter_to_topology_type_map = get_reversed_supported_adapters_map()

    with open(defaults_file_path) as defaults_file:
        defaults_file_contents = yaml.safe_load(defaults_file)
    launchers_defaults = {ld['framework']: ld for ld in defaults_file_contents['launchers']}
    datasets_defaults = {dd['name']: dd for dd in defaults_file_contents['datasets']}

    omz_meta = {}
    for config_file_path in configs_dir_path.glob('*.yml'):
        model_name = config_file_path.stem

        with open(str(config_file_path)) as config_file:
            config_file_content = yaml.safe_load(config_file)

        config = extract_single_config(model_name, config_file_content)
        if not config:
            continue

        dataset_config = extract_dataset_config(model_name, config)
        launcher_config = extract_launcher_config(model_name, config)
        if not dataset_config or not launcher_config:
            continue

        dataset_config = merge_defaults(datasets_defaults, dataset_config, 'name')
        launcher_config = merge_defaults(launchers_defaults, launcher_config, 'framework')
        if isinstance(launcher_config['adapter'], str):
            launcher_config['adapter'] = {'type': launcher_config['adapter']}

        advanced_configuration = {
            'preprocessing': dataset_config.get('preprocessing', []),
            'postprocessing': dataset_config.get('postprocessing', []),
            'metric': dataset_config.get('metrics', []),
        }
        if 'has_background' in dataset_config['annotation_conversion']:
            advanced_configuration['hasBackground'] = dataset_config['annotation_conversion']['has_background']

        omz_meta[model_name] = {
            'topology_type': adapter_to_topology_type_map.get(
                json.dumps(launcher_config['adapter'], sort_keys=True),
                TaskMethodEnum.generic
            ),
            'advanced_configuration': advanced_configuration,
        }
    return omz_meta


def fetch_downloadable_models():
    omz_meta = get_metadata_for_omz_models()
    parameters = InfoDumperParameters()
    parser = InfoDumperParser()
    return_code, _ = run_console_tool(parameters, parser)
    if return_code:
        return
    models = json.loads(parser.stdout)
    for model in models:
        model_meta = omz_meta.get(
            model['name'],
            {'topology_type': TaskMethodEnum.generic, 'advanced_configuration': None}
        )
        for precision in model['precisions']:
            existing_model = (
                OMZTopologyModel.query
                .filter_by(name=model['name'], precision=ModelPrecisionEnum(precision))
                .first()
            )
            if model['framework'] == 'dldt':
                model['framework'] = SupportedFrameworksEnum.openvino.value

            if not existing_model:
                if model_meta['topology_type'] != TaskMethodEnum.generic:
                    task_type = define_topology_task_type(model)
                else:
                    task_type = TaskEnum.generic
                record = OMZTopologyModel(
                    data=model,
                    task_type=task_type,
                    topology_type=model_meta['topology_type'],
                    advanced_configuration=model_meta['advanced_configuration'],
                    precision=ModelPrecisionEnum(precision)
                )
                write_record(record, get_db().session)
