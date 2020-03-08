"""
 OpenVINO Profiler
 Constants variable

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

from pathlib import Path
import os
import platform
import re
from inspect import getfile

try:
    from openvino.tools.accuracy_checker.accuracy_checker.launcher.dlsdk_launcher import DLSDKLauncher
except ImportError:
    from accuracy_checker.launcher.dlsdk_launcher import DLSDKLauncher

from utils.utils import make_canonical_path

SERVER_MODE = os.getenv('SERVER_MODE', 'production')
ROOT_FOLDER = make_canonical_path(os.path.dirname(os.path.dirname(getfile(lambda: 0))))
ROOT_FOLDER = re.sub(r'openvino_20\d\d\.\d\.\d+', 'openvino', ROOT_FOLDER)
DATA_FOLDER = make_canonical_path(os.getenv('OPENVINO_WORKBENCH_DATA_FOLDER', os.path.join(ROOT_FOLDER, 'app', 'data')))
DATA_FOLDER = re.sub(r'openvino_20\d\d\.\d\.\d+', 'openvino', DATA_FOLDER)
STATES_FOLDER = os.path.join(DATA_FOLDER, 'states')
UPLOAD_FOLDER_DATASETS = os.path.join(DATA_FOLDER, 'datasets')
UPLOADS_FOLDER = os.path.join(DATA_FOLDER, 'uploads')
UPLOAD_FOLDER_MODELS = os.path.join(DATA_FOLDER, 'models')
CONSOLE_TOOL_WRAPPER_FOLDER = os.path.join(ROOT_FOLDER, 'app', 'main', 'console_tool_wrapper')
PYTHON_CLI_FOLDER = os.path.join(CONSOLE_TOOL_WRAPPER_FOLDER, 'accuracy_tools')
WINOGRAD_CLI_FOLDER = os.path.join(CONSOLE_TOOL_WRAPPER_FOLDER, 'winograd_tool', 'winograd_cli_tool')
ALLOWED_EXTENSIONS_IMG = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'npy'}
VOC_ROOT_FOLDER = 'VOCdevkit'
VOC_IMAGES_FOLDER = 'JPEGImages'
VOC_ANNOTATIONS_FOLDER = 'Annotations'
VOC_CLASSES_FOLDER = os.path.join(ROOT_FOLDER, 'config/VOC_SSD_Classes.txt')
ORIGINAL_FOLDER = 'original'
DEFAULT_LOG_FILE = os.path.join(ROOT_FOLDER, 'access.log')
BENCHMARK_APP_REPORT_DIR = os.path.join(DATA_FOLDER, 'ba_reports')
MODEL_DOWNLOADS_FOLDER = os.path.join(DATA_FOLDER, 'downloads')
PROFILER_DEV_CPU_EXTENSION = 'libcpu_extension'
DISCOVER_EXTENSION_AUTO = 'AUTO'

LIB_EXTENSION = '.'
if 'darwin' in platform.platform().lower():
    LIB_EXTENSION += 'dylib'
elif 'windows' in platform.platform().lower():
    LIB_EXTENSION += 'dll'
elif 'linux' in platform.platform().lower():
    LIB_EXTENSION += 'so'

PROFILER_DEV_CPU_EXTENSION += LIB_EXTENSION

try:
    IE_BIN_PATH = os.environ['PROFILER_DEV_IE']
    ACCURACY_CHECKER_PATH = os.environ['ACCURACY_CHECKER']
    MODEL_DOWNLOADER_PATH = os.environ['MODEL_DOWNLOADER']
    MODEL_OPTIMIZER_PATH = os.environ['MODEL_OPTIMIZER']
    CPU_EXTENSIONS_PATH = '{}/{}'.format(os.path.join(os.environ['PROFILER_DEV_IE'], 'lib'), PROFILER_DEV_CPU_EXTENSION)
except KeyError:
    try:
        DEPLOYMENT_TOOLS_PATH = os.path.join(os.environ['INTEL_OPENVINO_DIR'], 'deployment_tools')
        INFERENCE_ENGINE_PATH = os.path.join(DEPLOYMENT_TOOLS_PATH, 'inference_engine')
        MODEL_DOWNLOADER_PATH = os.path.join(DEPLOYMENT_TOOLS_PATH, 'open_model_zoo', 'tools', 'downloader')
        ACCURACY_CHECKER_PATH = os.path.join(DEPLOYMENT_TOOLS_PATH, 'open_model_zoo', 'tools', 'accuracy_checker')
        MODEL_OPTIMIZER_PATH = os.path.join(DEPLOYMENT_TOOLS_PATH, 'model_optimizer')
        IE_SAMPLES_BIN_PATH = os.path.join(INFERENCE_ENGINE_PATH, 'samples', 'build', 'intel64')
        OPENVINO_EXTENSION_DIR = os.path.join(INFERENCE_ENGINE_PATH, 'lib', 'intel64')

        CPU_EXTENSIONS_PATH = str(DLSDKLauncher.get_cpu_extension(
            Path('{}/{}'.format(OPENVINO_EXTENSION_DIR, DISCOVER_EXTENSION_AUTO)), None
        ))
        IE_SAMPLES_BIN_PATH_RELEASE = os.path.join(IE_SAMPLES_BIN_PATH, 'Release')
        IE_SAMPLES_BIN_PATH_DEBUG = os.path.join(IE_SAMPLES_BIN_PATH, 'Debug')
        if os.path.isdir(IE_SAMPLES_BIN_PATH_RELEASE) and os.path.exists(IE_SAMPLES_BIN_PATH_RELEASE):
            IE_BIN_PATH = IE_SAMPLES_BIN_PATH_RELEASE
        elif os.path.isdir(IE_SAMPLES_BIN_PATH_DEBUG) and os.path.exists(IE_SAMPLES_BIN_PATH_DEBUG):
            IE_BIN_PATH = IE_SAMPLES_BIN_PATH_DEBUG
        else:
            raise KeyError
    except KeyError:
        raise KeyError('Please set environment variables for OpenVINO Toolkit (run setupvars.sh)')


def find_tf_custom_operations_configs() -> dict:
    configs_dir = Path(MODEL_OPTIMIZER_PATH) / 'extensions' / 'front' / 'tf'
    config_files_paths = [path for path in configs_dir.glob('*.json') if path.is_file()]
    return {path.stem: str(path.resolve()) for path in config_files_paths}


TF_CUSTOM_OPERATIONS_CONFIGS = find_tf_custom_operations_configs()
