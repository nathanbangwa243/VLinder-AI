"""
 OpenVINO Profiler
 Utils functions using in jobs

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
import re
from contextlib import suppress

from app.error.job_error import ModelOptimizerError
from app.main.job_factory.config import CeleryDBAdapter
from app.main.models.enumerates import SupportedFrameworksEnum
from app.main.models.jobs_model import JobsModel


def get_stages_status(job_id: int, session) -> list:
    job = session.query(JobsModel).get(job_id)
    error_message = job.error_message if job else ''
    parent_job = job.parent_job if job else None
    stages = get_stages_status(parent_job, session) if parent_job else []
    stages.append(job_status(job_id))
    if error_message:
        stages[-1]['errorMessage'] = error_message
    return stages


def job_status(job_id: int) -> dict:
    session = CeleryDBAdapter.session()
    job = session.query(JobsModel).get(job_id)
    status = {
        'progress': job.progress,
        'name': job.status.value,
        'stage': job.job_type,
    }
    session.close()
    return status


def resolve_file_args(job_id, config, original_topology):
    framework_to_variants = {
        SupportedFrameworksEnum.caffe: {
            frozenset(['.caffemodel', '.prototxt']): {
                'arg_to_ext': {
                    'input_model': '.caffemodel',
                    'input_proto': '.prototxt',
                },
            },
        },
        SupportedFrameworksEnum.mxnet: {
            frozenset(['.params']): {
                'arg_to_ext': {
                    'input_model': '.params',
                },
            },
        },
        SupportedFrameworksEnum.onnx: {
            frozenset(['.onnx']): {
                'arg_to_ext': {
                    'input_model': '.onnx',
                },
            },
        },
        SupportedFrameworksEnum.tf: {
            frozenset(['.pb']): {
                'arg_to_ext': {
                    'input_model': '.pb',
                    'input_checkpoint': '.ckpt',  # For non-frozen models.
                    'tensorflow_object_detection_api_pipeline_config': '.config',
                },
            },
            frozenset(['.pbtxt']): {
                'arg_to_ext': {
                    'input_model': '.pbtxt',
                    'input_checkpoint': '.ckpt',  # For non-frozen models.
                    'tensorflow_object_detection_api_pipeline_config': '.config',
                },
                'additional_args': {
                    'input_model_is_text': True,
                },
            },
            frozenset(['.meta', '.index', '.data-00000-of-00001']): {
                'arg_to_ext': {
                    'input_meta_graph': '.meta',
                    'tensorflow_object_detection_api_pipeline_config': '.config',
                },
            },
        },
    }

    def extract_ext(path):
        ext = os.path.splitext(path)[1]
        if re.fullmatch(r'\.data-[0-9]{5}-of-[0-9]{5}', ext):
            ext = '.data-00000-of-00001'
        return ext

    extension_to_path = {
        extract_ext(file.path): file.path
        for file in original_topology.files
    }

    for required_extensions, args in framework_to_variants[original_topology.framework].items():
        if required_extensions <= set(extension_to_path.keys()):
            for arg, ext in args['arg_to_ext'].items():
                with suppress(KeyError):  # For optional files, required files already checked.
                    config.mo_args[arg] = extension_to_path[ext]
            if 'additional_args' in args:
                config.mo_args.update(args['additional_args'])
            break
    else:
        ModelOptimizerError('Some files are missing.', job_id)
