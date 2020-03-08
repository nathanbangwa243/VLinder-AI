"""
 OpenVINO Profiler
 Class for creating Jobs by records from database

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

from app.main.jobs.datasets.dataset_upload_config import DatasetUploadConfig
from app.main.jobs.model_optimizer_scan.model_optimizer_scan_config import ModelOptimizerScanConfig
from app.main.jobs.model_optimizer_scan.model_optimizer_scan_job import ModelOptimizerScanJob
from app.main.jobs.topology_convert.topology_convert_config import TopologyConvertConfig
from app.main.jobs.topology_convert.topology_convert_job import TopologyConvertJob
from app.main.jobs.uploads.models.model_upload_config import ModelUploadConfig
from app.main.models.model_downloader_conversion_jobs_model import ModelDownloaderConversionJobsModel
from app.main.console_tool_wrapper.model_downloader.stages import ModelDownloadStages
from app.main.job_factory.config import CeleryDBAdapter
from app.main.jobs.accuracy.accuracy_config import AccuracyConfig
from app.main.jobs.accuracy.accuracy_job import AccuracyJob
from app.main.jobs.move_model_from_downloader.move_model_from_downloader_config import MoveModelFromDownloaderConfig
from app.main.jobs.move_model_from_downloader.move_model_from_downloader_job import MoveModelFromDownloaderJob
from app.main.jobs.datasets.dataset_extractor_job import DatasetExtractorJob
from app.main.jobs.datasets.dataset_generator_config import DatasetGeneratorConfig
from app.main.jobs.datasets.dataset_generator_job import DatasetGeneratorJob
from app.main.jobs.datasets.dataset_recognizer_job import DatasetRecognizerJob
from app.main.jobs.datasets.dataset_validator_job import DatasetValidatorJob
from app.main.jobs.download_model.download_model_config import DownloadModelConfig
from app.main.jobs.download_model.download_model_job import DownloadModelJob
from app.main.jobs.int8_autotune.int8_autotune_config import Int8AutoTuneConfig
from app.main.jobs.int8_autotune.int8_autotune_job import Int8AutotuneJob
from app.main.jobs.interfaces.ijob import IJob
from app.main.jobs.model_analyzer.model_analyzer_job import ModelAnalyzerJob
from app.main.jobs.model_downloader.model_downloader_config import ModelDownloaderConfig
from app.main.jobs.model_downloader.model_downloader_job import ModelDownloaderJob
from app.main.jobs.model_optimizer.model_optimizer_config import ModelOptimizerConfig
from app.main.jobs.model_optimizer.model_optimizer_job import ModelOptimizerJob
from app.main.jobs.job_types_enum import JobTypesEnum
from app.main.jobs.single_inference.single_inference_config import SingleInferenceConfig
from app.main.jobs.single_inference.single_inference_job import SingleInferenceJob
from app.main.jobs.winograd_autotune.winograd_autotune_config import WinogradAutotuneConfig
from app.main.jobs.winograd_autotune.winograd_autotune_job import WinogradAutotuneJob
from app.main.models.accuracy_model import AccuracyJobsModel
from app.main.models.compound_inference_model import CompoundInferenceJobsModel
from app.main.models.dataset_generation_configs_model import DatasetGenerationConfigsModel
from app.main.models.datasets_model import DatasetsModel
from app.main.models.download_configs_model import DownloadConfigsModel
from app.main.models.enumerates import SupportedFrameworksEnum
from app.main.models.int8_autotune_model import Int8AutotuneJobsModel
from app.main.models.model_downloader_model import ModelDownloaderModel
from app.main.models.model_optimizer_job_model import ModelOptimizerJobModel
from app.main.models.model_optimizer_scan_model import ModelOptimizerScanJobsModel
from app.main.models.omz_topology_model import OMZTopologyModel
from app.main.models.projects_model import ProjectsModel
from app.main.models.topologies_model import TopologiesModel
from app.main.models.winograd_autotune_model import WinogradAutotuneJobsModel
from config.constants import UPLOAD_FOLDER_MODELS, ORIGINAL_FOLDER


class JobFromDBFactory:
    @staticmethod
    def create_job(job_type, job_id: int, socket_io, data, weight, celery_task) -> IJob:
        creator = JobFromDBFactory.job_creators[JobTypesEnum(job_type)]
        job = creator(job_id, data, weight)
        job.emit_message.socket_io = socket_io
        job.celery_task = celery_task
        return job

    @staticmethod
    def inference_job_creator(job_id: int, data, weight):
        session = CeleryDBAdapter.session()
        inference_record = session.query(CompoundInferenceJobsModel).filter_by(job_id=job_id).first()
        project_record = session.query(ProjectsModel).get(inference_record.project_id)
        session.close()
        batch = data['batch']
        nireq = data['nireq']
        config_data = {
            'modelId': project_record.model_id,
            'datasetId': project_record.dataset_id,
            'device': project_record.target.value,
            'previousJobId': inference_record.parent_job,
            'batch': batch,
            'nireq': nireq,
            'inferenceTime': inference_record.inference_time,
        }
        config = SingleInferenceConfig(inference_record.session_id, config_data)
        job = SingleInferenceJob(job_id, config, weight)
        job.emit_message.cumulative_progress = inference_record.progress
        job.emit_message.date = inference_record.creation_timestamp.timestamp()
        return job

    @staticmethod
    def int8autotune_job_creator(job_id, data, weight):
        session = CeleryDBAdapter.session()
        int8_job = session.query(Int8AutotuneJobsModel).filter_by(job_id=job_id).first()
        project = session.query(ProjectsModel).get(int8_job.project_id)
        model = session.query(TopologiesModel).get(project.model_id)
        session.close()
        data = {
            'taskType': model.meta.task_type.value,
            'modelId': project.model_id,
            'datasetId': project.dataset_id,
            'device': project.target.value,
            'previousJobId': int8_job.parent_job,
            'batch': int8_job.batch,
            'threshold': int8_job.threshold
        }

        config = Int8AutoTuneConfig(None, data)
        job = Int8AutotuneJob(job_id, config, weight)
        job.emit_message.date = int8_job.creation_timestamp.timestamp()
        return job

    @staticmethod
    def winograd_autotune_job_creator(job_id, data, weight):
        session = CeleryDBAdapter.session()
        winograd_autotune_job = session.query(WinogradAutotuneJobsModel).filter_by(job_id=job_id).first()
        project = session.query(ProjectsModel).get(winograd_autotune_job.project_id)
        session.close()

        data = {
            'modelId': project.model_id,
            'datasetId': project.dataset_id,
            'device': project.target.value,
            'inferenceTime': winograd_autotune_job.inference_time,
        }
        config = WinogradAutotuneConfig(None, data)
        job = WinogradAutotuneJob(job_id, config, weight)
        job.emit_message.date = winograd_autotune_job.creation_timestamp.timestamp()
        return job

    @staticmethod
    def accuracy_job_creator(job_id, data, weight):
        session = CeleryDBAdapter.session()
        accuracy_job = session.query(AccuracyJobsModel).filter_by(job_id=job_id).first()
        session.close()
        data = {
            'jobId': job_id,
            'projectId': accuracy_job.project_id
        }

        config = AccuracyConfig(None, data)
        job = AccuracyJob(job_id, config, weight)
        job.emit_message.date = accuracy_job.creation_timestamp.timestamp()
        return job

    # pylint: disable=unused-argument
    @staticmethod
    def model_analyzer_job_creator(job_id, data, weight):
        session = CeleryDBAdapter.session()
        record = session.query(TopologiesModel).get(job_id)
        config_data = {
            'name': record.name,
        }
        config = ModelUploadConfig(record.session_id, config_data)
        job = ModelAnalyzerJob(job_id, config, weight)
        job.emit_message.date = record.creation_timestamp.timestamp()
        job.emit_message.previous_progress = record.progress
        session.close()
        return job

    # pylint: disable=unused-argument
    @staticmethod
    def dataset_extractor_job_creator(job_id, data, weight):
        session = CeleryDBAdapter.session()
        record = session.query(DatasetsModel).get(job_id)
        config = DatasetUploadConfig(record.session_id, record.json())
        job = DatasetExtractorJob(job_id, config, weight)
        job.emit_message.date = record.creation_timestamp.timestamp()
        job.emit_message.from_celery = True
        job.emit_message.previous_progress = record.progress
        session.close()
        return job

    # pylint: disable=unused-argument
    @staticmethod
    def dataset_recognizer_job_creator(job_id, data, weight):
        session = CeleryDBAdapter.session()
        record = session.query(DatasetsModel).get(job_id)
        config = DatasetUploadConfig(record.session_id, record.json())
        job = DatasetRecognizerJob(job_id, config, weight)
        job.emit_message.date = record.creation_timestamp.timestamp()
        job.emit_message.from_celery = True
        job.emit_message.previous_progress = record.progress
        session.close()
        return job

    # pylint: disable=unused-argument
    @staticmethod
    def dataset_validator_job_creator(job_id, data, weight):
        session = CeleryDBAdapter.session()
        record = session.query(DatasetsModel).get(job_id)
        config = DatasetUploadConfig(record.session_id, record.json())
        job = DatasetValidatorJob(job_id, config, weight)
        job.emit_message.date = record.creation_timestamp.timestamp()
        job.emit_message.from_celery = True
        job.emit_message.previous_progress = record.progress
        session.close()
        return job

    @staticmethod
    def dataset_generator_job_creator(job_id, data, weight):
        session = CeleryDBAdapter.session()
        dataset = session.query(DatasetsModel).get(job_id)
        dataset_generator_config = session.query(DatasetGenerationConfigsModel).get(job_id)
        data = dataset_generator_config.json()
        data.update({
            **dataset.json()
        })
        config = DatasetGeneratorConfig(dataset.session_id, data)
        job = DatasetGeneratorJob(job_id, config, weight)
        job.emit_message.date = dataset.creation_timestamp.timestamp()
        job.emit_message.from_celery = True
        session.close()
        return job

    @staticmethod
    def download_model_job_creator(job_id, data, weight):
        session = CeleryDBAdapter.session()
        record = session.query(DownloadConfigsModel).get(job_id)
        data = {
            'jobId': record.job_id
        }

        config = DownloadModelConfig(record.session_id, data)
        job = DownloadModelJob(job_id, config, weight)
        job.emit_message.date = record.creation_timestamp.timestamp()
        session.close()
        return job

    # pylint: disable=unused-argument
    @staticmethod
    def model_optimizer_job_creator(job_id, data, weight):
        session = CeleryDBAdapter.session()
        record = session.query(ModelOptimizerJobModel).get(job_id)
        config = ModelOptimizerConfig(record.session_id, record.json())
        job = ModelOptimizerJob(record.result_model_id, config, weight)
        job.emit_message.date = record.creation_timestamp.timestamp()
        job.emit_message.previous_progress = record.result_model.progress
        session.close()
        return job

    # pylint: disable=unused-argument
    @staticmethod
    def model_optimizer_scan_job_creator(job_id, data, weight):
        session = CeleryDBAdapter.session()
        ir_topology = session.query(TopologiesModel).get(job_id)
        original_model = ir_topology.converted_from
        record = session.query(ModelOptimizerScanJobsModel).filter_by(topology_id=original_model).first()
        data = record.json()
        data.update({'name': ir_topology.name})
        config = ModelOptimizerScanConfig(record.session_id, data)
        job = ModelOptimizerScanJob(job_id, config, weight)
        job.emit_message.date = record.creation_timestamp.timestamp()
        job.emit_message.from_celery = True
        job.emit_message.previous_progress = ir_topology.progress
        session.close()
        return job

    @staticmethod
    def model_convert_job_creator(job_id, data, weight):
        session = CeleryDBAdapter.session()
        record = session.query(ModelDownloaderConversionJobsModel).get(job_id)
        topology = session.query(TopologiesModel).get(record.result_model_id)
        data = record.json()
        data['name'] = topology.name
        config = TopologyConvertConfig(record.session_id, record.result_model_id, data)
        config.output = os.path.join(UPLOAD_FOLDER_MODELS, str(record.result_model_id), ORIGINAL_FOLDER)
        job = TopologyConvertJob(job_id, config, weight)
        job.emit_message.date = record.creation_timestamp.timestamp()
        result_model = session.query(TopologiesModel).get(record.result_model_id)
        job.emit_message.previous_progress = result_model.progress
        session.close()
        return job

    # pylint: disable=unused-argument
    @staticmethod
    def model_downloader_job_creator(job_id, data, weight):
        session = CeleryDBAdapter.session()
        record = session.query(ModelDownloaderModel).get(job_id)
        config = ModelDownloaderConfig(record.session_id, record.json())
        omz_topologies = session.query(OMZTopologyModel).filter_by(name=config.name)
        framework = omz_topologies.first().framework
        precisions_count = omz_topologies.count()
        session.close()
        framework_to_files_count = {
            SupportedFrameworksEnum.caffe: 2,
            SupportedFrameworksEnum.mxnet: 2,
            SupportedFrameworksEnum.openvino: 2,
            SupportedFrameworksEnum.pytorch: 2,
            SupportedFrameworksEnum.tf: 1,
        }
        files_count = framework_to_files_count[framework]
        if framework == SupportedFrameworksEnum.openvino:
            files_count *= precisions_count
        stages = ModelDownloadStages(files_count)
        job = ModelDownloaderJob(job_id, config, weight)
        job.emit_message.stages = stages
        job.emit_message.date = record.creation_timestamp.timestamp()
        return job

    @staticmethod
    def move_model_from_downloader_job_creator(job_id, data, weight):
        config = MoveModelFromDownloaderConfig(None, data)
        job = MoveModelFromDownloaderJob(job_id, config, weight)
        return job

    job_creators = {
        JobTypesEnum.single_inference_type: inference_job_creator.__func__,
        JobTypesEnum.compound_inference_type: inference_job_creator.__func__,

        JobTypesEnum.int8autotune_type: int8autotune_job_creator.__func__,
        JobTypesEnum.winograd_autotune_type: winograd_autotune_job_creator.__func__,
        JobTypesEnum.accuracy_type: accuracy_job_creator.__func__,

        # models
        JobTypesEnum.model_analyzer_type: model_analyzer_job_creator.__func__,
        JobTypesEnum.model_downloader_type: model_downloader_job_creator.__func__,
        JobTypesEnum.model_optimizer_type: model_optimizer_job_creator.__func__,
        JobTypesEnum.model_optimizer_scan_type: model_optimizer_scan_job_creator.__func__,
        JobTypesEnum.model_convert_type: model_convert_job_creator.__func__,

        # datasets
        JobTypesEnum.dataset_extractor_type: dataset_extractor_job_creator.__func__,
        JobTypesEnum.dataset_recognizer_type: dataset_recognizer_job_creator.__func__,
        JobTypesEnum.dataset_validator_type: dataset_validator_job_creator.__func__,
        JobTypesEnum.add_generated_dataset_type: dataset_generator_job_creator.__func__,

        # downloads
        JobTypesEnum.download_model_type: download_model_job_creator.__func__,

        JobTypesEnum.move_model_from_downloader_type: move_model_from_downloader_job_creator.__func__
    }
