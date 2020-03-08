"""
 OpenVINO Profiler
 Class for storing winograd cli params

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
import os
import sys

from app.main.job_factory.config import CeleryDBAdapter
from app.main.console_tool_wrapper.inference_engine_tool.parameters import Parameters
from app.main.jobs.winograd_autotune.winograd_autotune_config import WinogradAutotuneConfig
from app.main.models.datasets_model import DatasetsModel
from app.main.models.enumerates import DatasetTypesEnum
from app.main.utils.utils import get_images_folder_for_voc, find_all_paths
from config.constants import ALLOWED_EXTENSIONS_IMG, WINOGRAD_CLI_FOLDER


class WinogradParameters(Parameters):
    def __init__(self, config: WinogradAutotuneConfig, path: str = WINOGRAD_CLI_FOLDER):
        super(WinogradParameters, self).__init__(path, 'winograd_tool', config)
        self.path = sys.executable
        self.exe = os.path.join(path, 'winograd_tool.py')
        session = CeleryDBAdapter.session()
        dataset = session.query(DatasetsModel).get(config.dataset_id)
        dataset_path = dataset.path
        if dataset.dataset_type in [DatasetTypesEnum.voc_object_detection, DatasetTypesEnum.voc_segmentation]:
            dataset_path = get_images_folder_for_voc(dataset_path)
        session.close()
        image_path = find_all_paths(dataset_path, tuple(map(lambda ext: '.' + ext, ALLOWED_EXTENSIONS_IMG)))[0]
        self.params['i'] = image_path
        self.params['t'] = 10
        del self.params['d']

    def __str__(self, parameter_prefix='-'):
        exe_path = '{} {}'.format(self.path, self.exe)
        params = ' '.join(
            ['{p}{k} {v}'.format(p=parameter_prefix, k=key, v=value) for key, value in self.params.items()])
        return exe_path + ' ' + params
