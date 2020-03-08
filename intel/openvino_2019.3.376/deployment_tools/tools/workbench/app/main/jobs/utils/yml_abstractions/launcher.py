"""
 OpenVINO Profiler
 Accuracy checker's configuration field abstraction

 Copyright (c) 2019 Intel Corporation

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
from app.main.jobs.utils.yml_abstractions.typed_parameter import Adapter


#pylint: disable=too-many-arguments
class Launcher:
    def __init__(self,
                 adapter: Adapter,
                 device: str,
                 model: str,
                 weights: str,
                 framework: str = 'dlsdk',
                 batch: int = 1,
                 cpu_extensions: str = 'AUTO'):
        self.framework = framework
        self.adapter = adapter
        self.device = device
        self.model = model
        self.weights = weights
        self.batch = batch
        self.cpu_extensions = cpu_extensions

    def to_dict(self) -> dict:
        return {
            'framework': self.framework,
            'device': self.device,
            'model': self.model,
            'weights': self.weights,
            'adapter': self.adapter.to_dict(),
            'batch': self.batch,
            'cpu_extensions': self.cpu_extensions
        }

    @staticmethod
    def from_dict(data: dict) -> 'Launcher':
        required_fields = {'adapter',
                           'framework',
                           'device',
                           'model',
                           'weights',
                           'batch',
                           'cpu_extensions'}
        if required_fields > set(data):
            raise KeyError('not all of the required fields ({}) are present in the dictionary'
                           .format(required_fields))
        adapter_obj = data['adapter']
        data['adapter'] = Adapter(type=adapter_obj['type'], parameters={key: val for key, val in adapter_obj.items()
                                                                        if key != 'type'})
        return Launcher(**data)
