"""
 OpenVINO Profiler
 Class for storing weights of jobs in chains

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
from app.main.jobs.job_types_enum import JobTypesEnum


class JobsWeight:
    @staticmethod
    def upload_artifact():
        return {
            JobTypesEnum.iuploader_type: 0.6
        }

    @staticmethod
    def upload_dataset():
        return {
            **JobsWeight.upload_artifact(),
            JobTypesEnum.dataset_extractor_type: 0.15,
            JobTypesEnum.dataset_recognizer_type: 0.10,
            JobTypesEnum.dataset_validator_type: 0.15,
        }

    @staticmethod
    def upload_openvino_model():
        return {
            **JobsWeight.upload_artifact(),
            JobTypesEnum.model_analyzer_type: 0.4,
        }

    @staticmethod
    def upload_source_model():
        return {
            JobTypesEnum.iuploader_type: 1
        }

    @staticmethod
    def upload_and_convert_openvino_model():
        return {
            **JobsWeight.upload_artifact(),
            JobTypesEnum.model_optimizer_scan_type: 0.05,
            JobTypesEnum.model_optimizer_type: 0.3,
            JobTypesEnum.model_analyzer_type: 0.05,
        }

    @staticmethod
    def model_optimizer():
        return {
            JobTypesEnum.model_optimizer_type: 0.9,
            JobTypesEnum.model_analyzer_type: 0.1,
        }

    @staticmethod
    def download_model():
        return {
            JobTypesEnum.model_downloader_type: 0.4
        }

    @staticmethod
    def download_openvino_model():
        return {
            **JobsWeight.download_model(),
            JobTypesEnum.move_model_from_downloader_type: 0.3,
            JobTypesEnum.model_analyzer_type: 0.3,
        }

    @staticmethod
    def download_source_model():
        return {
            **JobsWeight.download_model(),
            JobTypesEnum.model_convert_type: 0.4,
            JobTypesEnum.move_model_from_downloader_type: 0.1,
            JobTypesEnum.model_analyzer_type: 0.1,
        }

    @staticmethod
    def int8_model():
        return {
            JobTypesEnum.int8autotune_type: 0.99,
            JobTypesEnum.model_analyzer_type: 0.01,
        }
