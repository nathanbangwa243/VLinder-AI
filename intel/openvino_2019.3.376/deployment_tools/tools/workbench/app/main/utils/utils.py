"""
 OpenVINO Profiler
 Profiler utilities functions

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

import math
import os
import shutil
import time

from app.error.inconsistent_upload_error import InconsistentDatasetError
from config.constants import UPLOAD_FOLDER_DATASETS, VOC_ROOT_FOLDER, VOC_IMAGES_FOLDER


def remove_dir(dir_path):
    if dir_path and os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        while True:
            try:
                os.makedirs(dir_path)
            except FileExistsError:
                # If problem is that directory still exists, wait a bit and try again
                # this is the known issue: https://bit.ly/2I5vkZY
                time.sleep(0.01)
            else:
                os.sync()
                break
        shutil.rmtree(dir_path)
        os.sync()


def create_empty_dir(path):
    if os.path.exists(path):
        remove_dir(path)
    os.makedirs(path, exist_ok=True)


def get_dataset_folder(dataset_id: str) -> str:
    return os.path.join(UPLOAD_FOLDER_DATASETS, dataset_id)


def get_images_folder_for_voc(dataset_path):
    root_path = os.path.join(dataset_path, VOC_ROOT_FOLDER)
    second_level_folder = get_second_level_folder_for_voc(root_path)
    return os.path.join(root_path, second_level_folder, VOC_IMAGES_FOLDER)


def get_second_level_folder_for_voc(dataset_root_path):
    subdirs = [os.path.join(dataset_root_path, o) for o in os.listdir(dataset_root_path) if
               os.path.isdir(os.path.join(dataset_root_path, o))]
    for subdir in subdirs:
        if 'VOC2' in subdir:
            return subdir
    raise InconsistentDatasetError('Folder with images was not found')


def find_all_paths(data_dir, ext: tuple) -> list:
    paths = []
    for root, _, files in os.walk(data_dir):
        for file_name in files:
            for extension in ext:
                if extension and file_name.lower().endswith(extension):
                    paths.append(os.path.abspath(os.path.join(root, file_name)))
    return paths


def get_size_of_files(path):
    size = sum(
        os.path.getsize(os.path.join(dirpath, filename)) for dirpath, dirnames, filenames in os.walk(path) for filename
        in filenames)
    return math.ceil(size / (1024 ** 2))


def get_size(archive_size):
    if archive_size is not None:
        return math.ceil(archive_size / (1024 ** 2))
    return None


def get_name_wo_extension(path):
    return os.path.splitext(os.path.basename(path))[0]
