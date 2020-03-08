"""
 OpenVINO Profiler
 Dataset adapter classes.

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
import re
from pathlib import Path

from app.error.inconsistent_upload_error import InconsistentDatasetError
from app.main.jobs.utils.yml_templates.registry import register_dataset_adapter
from app.main.jobs.utils.yml_abstractions.annotation import Annotation
from app.main.models.enumerates import DatasetTypesEnum
from config.constants import VOC_ROOT_FOLDER


class BaseDatasetAdapter:
    """
    Provides methods for dealing with a dataset.

    Adding a new dataset type:
    1. Create a `BaseDatasetAdapter` subclass
    2. Find the corresponding annotation converter (subclass of `BaseFormatConverter`) in
       http://github.com/opencv/open_model_zoo/tree/master/tools/accuracy_checker/accuracy_checker/annotation_converters
    3. Set the `converter` attribute of the subclass to the name of the corresponding annotation converter
       (The name is contained in the `__provider__` attribute of the annotation converter class).
    4. Implement `recognize()`
    5. Implement `get_data_source()`
    6. Implement `get_specific_params()`
    """

    converter = None

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(self.dataset_path)
        if not self.dataset_path.is_dir():
            raise NotADirectoryError(self.dataset_path)

        self.params = self.get_params()

    @staticmethod
    def recognize(dataset_path: str) -> bool:
        """Check if the `dataset_path` contains a dataset of type related to this subclass."""
        raise NotImplementedError

    def get_data_source(self) -> Path:
        """Return absolute path to the directory containing images."""
        raise NotImplementedError

    def get_specific_params(self) -> dict:
        """
        Return only the annotation conversion params specific for this type of dataset.

        Find the parameters in the `parameters()` method  of the corresponding annotation converter class.
        """
        raise NotImplementedError

    def get_params(self) -> dict:
        """Return all annotation conversion params."""
        params = self.get_specific_params()
        params.update({
            'converter': self.converter,
            'images_dir': self.get_data_source(),  # Used by Accuracy Checker for content checking.
        })
        return params

    def abs_path(self, relative_path: str) -> Path:
        absolute_path = self.dataset_path / relative_path
        if not absolute_path.exists():
            raise InconsistentDatasetError('Cannot find {}'.format(relative_path))
        return absolute_path

    def to_annotation(self) -> dict:
        serializable_params = {key: str(value) for key, value in self.params.items()}
        return {
            'data_source': str(self.get_data_source()),
            'annotation': Annotation(**serializable_params),
        }


@register_dataset_adapter(DatasetTypesEnum.imagenet)
class ImagenetDatasetAdapter(BaseDatasetAdapter):
    converter = 'imagenet'

    @staticmethod
    def recognize(dataset_path: str) -> bool:
        content = list(Path(dataset_path).iterdir())
        no_subdirs = all(path.is_file() for path in content)
        has_txt = any(path.suffix.lower() == '.txt' for path in content)
        return no_subdirs and has_txt

    def get_data_source(self) -> Path:
        return self.dataset_path

    def get_specific_params(self) -> dict:
        return {
            'annotation_file': self.get_annotation_file_path(),
        }

    def get_annotation_file_path(self) -> Path:
        annotation_file_paths = [path for path in self.dataset_path.iterdir() if self.is_imagenet_annotation_file(path)]
        if not annotation_file_paths:
            raise InconsistentDatasetError('Cannot find annotation file.')
        if len(annotation_file_paths) > 1:
            raise InconsistentDatasetError(
                'Too many annotation files: {}.'.format([path.name for path in annotation_file_paths]))
        return annotation_file_paths[0]

    @staticmethod
    def is_imagenet_annotation_file(path: Path):
        if not path.is_file() or path.suffix.lower() != '.txt':
            return False
        with open(str(path)) as file:
            return all(re.match(r'^\S+[ \t]+[0-9]+$', line.rstrip(' \t\r\n')) for line in file if line.strip('\r\n'))


@register_dataset_adapter(DatasetTypesEnum.voc_object_detection)
class VOCDetectionDatasetAdapter(BaseDatasetAdapter):
    converter = 'voc_detection'

    @staticmethod
    def recognize(dataset_path: str) -> bool:
        return (Path(dataset_path) / VOC_ROOT_FOLDER).is_dir()

    def get_data_source(self) -> Path:
        return self.abs_path('VOCdevkit/VOC{}/JPEGImages'.format(self.voc_version))

    def get_specific_params(self) -> dict:
        return {
            'imageset_file': self.get_imageset_file(),
            'annotations_dir': self.abs_path('VOCdevkit/VOC{}/Annotations'.format(self.voc_version)),
        }

    def __init__(self, *args, **kwargs):
        self._voc_version = None
        super().__init__(*args, **kwargs)

    @property
    def voc_version(self):
        if not self._voc_version:
            vocdevkit_dir = self.abs_path('VOCdevkit')
            voc_version_dirnames = [d.name for d in vocdevkit_dir.iterdir() if d.name.startswith('VOC') and d.is_dir()]
            if not voc_version_dirnames:
                raise InconsistentDatasetError(
                    'Cannot find "VOCdevkit/VOC<year>" directory.')
            if len(voc_version_dirnames) > 1:
                raise InconsistentDatasetError(
                    'Too many "VOCdevkit/VOC<year>" directories: {}.'.format(voc_version_dirnames))
            self._voc_version = voc_version_dirnames[0].split('VOC')[1]
        return self._voc_version

    def get_imageset_file(self):
        path_to_dir = self.abs_path('VOCdevkit/VOC{}/ImageSets/Main'.format(self.voc_version))
        for filename in ('test.txt', 'val.txt', 'train.txt'):
            path = path_to_dir / filename
            if path.is_file():
                return path
        raise InconsistentDatasetError('Cannot find an imageset file for this dataset.')
