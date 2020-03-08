"""
 OpenVINO Profiler
 Class for storing anything codes

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


class CodeRegistry:
    CODES = {

        'GENERAL_SERVER_ERROR': 1001,

        'DEFECTIVE_CONFIG_ERROR': 2001,

        'REMOVED_UPLOAD': 3001,
        'REMOVED_DATASET': 3002,
        'REMOVED_MODEL': 3003,
        'DATASET_GENERATION_ERROR': 3004,

        # CONSOLE TOOLS ERRORS
        'INT8AUTOTUNE_ERROR': 4001,
        'COMPOUND_INFERENCE_ERROR': 4002,
        'WINOGRAD_AUTOTUNE_ERROR': 4003,
        'MODEL_DOWNLOADER_ERROR': 4004,
        'MODEL_OPTIMIZER_ERROR': 4005,
        'ACCURACY_ERROR': 4006,
        'MODEL_ANALYZER_ERROR': 4007,

        # DATABASE ERRORS
        'DATABASE_ERROR': 5001
    }

    @classmethod
    def get_accuracy_error_code(cls):
        return cls.CODES['ACCURACY_ERROR']

    @classmethod
    def get_remove_upload_code(cls):
        return cls.CODES['REMOVED_UPLOAD']

    @classmethod
    def get_remove_dataset_code(cls):
        return cls.CODES['REMOVED_DATASET']

    @classmethod
    def get_remove_model_code(cls):
        return cls.CODES['REMOVED_MODEL']

    @classmethod
    def get_defective_config_code(cls):
        return cls.CODES['DEFECTIVE_CONFIG_ERROR']

    @classmethod
    def get_general_error_code(cls):
        return cls.CODES['GENERAL_SERVER_ERROR']

    @classmethod
    def get_int8autotune_error_code(cls, ):
        return cls.CODES['INT8AUTOTUNE_ERROR']

    @classmethod
    def get_winograd_autotune_error_code(cls, ):
        return cls.CODES['WINOGRAD_AUTOTUNE_ERROR']

    @classmethod
    def get_compound_inference_error_code(cls, ):
        return cls.CODES['COMPOUND_INFERENCE_ERROR']

    @classmethod
    def get_model_downloader_error_code(cls, ):
        return cls.CODES['MODEL_DOWNLOADER_ERROR']

    @classmethod
    def get_database_error_code(cls):
        return cls.CODES['DATABASE_ERROR']

    @classmethod
    def get_model_optimizer_error_code(cls, ):
        return cls.CODES['MODEL_OPTIMIZER_ERROR']

    @classmethod
    def get_dataset_generation_error_code(cls, ):
        return cls.CODES['DATASET_GENERATION_ERROR']

    @classmethod
    def get_remove_model_analyzer_code(cls):
        return cls.CODES['MODEL_ANALYZER_ERROR']
