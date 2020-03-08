"""
 OpenVINO Profiler
 Job Error class

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

from app.error.general_error import GeneralError
from app.main.jobs.registries.code_registry import CodeRegistry


class JobGeneralError(GeneralError):
    def __init__(self, message, job_id):
        super(JobGeneralError, self).__init__(message)
        self.details['jobId'] = job_id


class Int8AutotuneError(JobGeneralError):
    code = CodeRegistry.get_int8autotune_error_code()


class AccuracyError(JobGeneralError):
    code = CodeRegistry.get_accuracy_error_code()


class CompoundInferenceError(JobGeneralError):
    code = CodeRegistry.get_compound_inference_error_code()


class WinogradAutotuneError(JobGeneralError):
    code = CodeRegistry.get_winograd_autotune_error_code()


class ModelDownloaderError(JobGeneralError):
    code = CodeRegistry.get_model_downloader_error_code()


class ModelOptimizerError(JobGeneralError):
    code = CodeRegistry.get_model_optimizer_error_code()


class DatasetGenerationError(JobGeneralError):
    code = CodeRegistry.get_model_optimizer_error_code()


class ArtifactError(JobGeneralError):
    code = CodeRegistry.get_remove_upload_code()


class ModelAnalyzerError(JobGeneralError):
    code = CodeRegistry.get_remove_model_analyzer_code()
