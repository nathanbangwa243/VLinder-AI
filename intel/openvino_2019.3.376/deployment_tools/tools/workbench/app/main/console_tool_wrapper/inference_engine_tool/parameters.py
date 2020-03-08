"""
 OpenVINO Profiler
 Interface for storing common parameters for inference engine tools

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

from app.main.job_factory.config import CeleryDBAdapter
from app.main.jobs.interfaces.iconfig import IConfig
from app.main.jobs.tools_runner.runner import ConsoleToolParameters
from app.main.models.datasets_model import DatasetsModel
from app.main.models.enumerates import DatasetTypesEnum
from app.main.models.topologies_model import TopologiesModel
from app.main.utils.utils import find_all_paths, get_images_folder_for_voc
from config.constants import IE_BIN_PATH, LIB_EXTENSION


class Parameters(ConsoleToolParameters):
    def __init__(self, path: str, exe: str, parameters: IConfig):
        super(Parameters, self).__init__()
        self.path = path
        self.exe = exe
        self.params = dict()
        session = CeleryDBAdapter.session()
        dataset = session.query(DatasetsModel).get(parameters.dataset_id)
        dataset_path = dataset.path
        if dataset.dataset_type in [DatasetTypesEnum.voc_object_detection,
                                    DatasetTypesEnum.voc_segmentation]:
            dataset_path = get_images_folder_for_voc(dataset_path)
        model = session.query(TopologiesModel).get(parameters.model_id)
        model_path = model.path
        session.close()

        self.params['i'] = dataset_path
        xml_path = find_all_paths(model_path, ('.xml',))[0]

        self.params['m'] = xml_path
        self.params['d'] = parameters.device
        if self.params['d'] == 'CPU':
            self.params['l'] = os.path.join(IE_BIN_PATH, 'lib', 'libcpu_extension' + LIB_EXTENSION)
