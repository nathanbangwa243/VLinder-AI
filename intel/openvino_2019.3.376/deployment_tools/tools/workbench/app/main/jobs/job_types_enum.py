"""
 OpenVINO Profiler
 Class for checking type of job

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
import enum


class JobTypesEnum(enum.Enum):
    compound_inference_type = 'CompoundInferenceJob'
    single_inference_type = 'SingleInferenceJob'

    int8autotune_type = 'Int8AutotuneJob'
    accuracy_type = 'AccuracyJob'

    iuploader_type = 'UploaderJob'

    dataset_extractor_type = 'DatasetExtractorJob'
    dataset_recognizer_type = 'DatasetRecognizerJob'
    dataset_validator_type = 'DatasetValidatorJob'
    add_local_dataset_type = 'LocalDatasetCreatorJob'
    add_generated_dataset_type = 'DatasetGeneratorJob'

    model_analyzer_type = 'IRAnalyzerJob'
    model_downloader_type = 'ModelDownloaderJob'
    model_optimizer_type = 'ModelOptimizerJob'
    model_optimizer_scan_type = 'ModelOptimizerScanJob'
    model_convert_type = 'ModelConvertJob'

    winograd_autotune_type = 'WinogradAutotuneJob'
    download_model_type = 'DownloadModelJob'

    move_model_from_downloader_type = 'MoveModelFromDownloaderJob'
