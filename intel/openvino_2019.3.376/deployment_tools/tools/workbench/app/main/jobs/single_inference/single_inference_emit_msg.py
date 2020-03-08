"""
 OpenVINO Profiler
 Class for inference emit message

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
import logging as log

from app.main.job_factory.config import CeleryDBAdapter
from app.main.jobs.interfaces.exec_info import ExecInfo
from app.main.jobs.interfaces.iemit_message import IEmitMessage, IEmitMessageStage
from app.main.jobs.single_inference.single_inference_config import SingleInferenceConfig
from app.main.jobs.utils.traversal import get_top_level_model_id
from app.main.models.compound_inference_model import CompoundInferenceJobsModel
from app.main.models.enumerates import StatusEnum
from app.main.models.factory import write_record
from app.main.models.inference_results_model import InferenceResultsModel
from app.main.models.jobs_model import JobsModel
from app.main.console_tool_wrapper.benchmark_app.stages import BenchmarkAppStages


class SingleInferenceEmitMessage(IEmitMessage):
    event = 'inference_model'
    namespace = '/{}'.format(event)
    stages = BenchmarkAppStages

    def __init__(self, job, job_id: int, content: SingleInferenceConfig, weight: float):
        super(SingleInferenceEmitMessage, self).__init__(job, job_id, content, weight)
        self.exec_info = ExecInfo(self.config.batch, self.config.nireq)
        session = CeleryDBAdapter.session()
        inference_result = session.query(InferenceResultsModel).filter(InferenceResultsModel.job_id == job_id) \
            .filter(InferenceResultsModel.batch == self.config.batch) \
            .filter(InferenceResultsModel.nireq == self.config.nireq).first()
        self.project_id = session.query(JobsModel).get(job_id).project_id
        session.close()
        self.original_model_id = get_top_level_model_id(self.project_id)
        self.inference_result_record_id = inference_result.id

    def full_json(self):
        session = CeleryDBAdapter.session()
        compound_inference = session.query(CompoundInferenceJobsModel).get(self.job_id)
        status = compound_inference.status.value
        error_message = compound_inference.error_message
        session.close()
        message = {
            **super(SingleInferenceEmitMessage, self).full_json(),
            'result': self.result_to_json(),
            'projectId': self.project_id,
            'originalModelId': self.original_model_id,
            'inferenceResultId': self.inference_result_record_id,
            'status': {
                'progress': self.total_progress,
                'name': status
            }
        }
        if error_message:
            message['status']['errorMessage'] = error_message
        return message

    def short_json(self):
        full_json = self.full_json()
        for result in full_json['result']:
            del result['execGraph']
            del result['pc']
        return full_json

    def update_percent(self, percent):
        log.debug('[ INFERENCE ]: Update progress of stage %s percent: %s', self.get_current_job().name, percent)
        if self.get_current_job().name == BenchmarkAppStages.get_stages()[-1]:
            percent = min(percent, 99)
        self.get_current_job().progress = percent
        self.update_progress_for_inference_result()
        self.emit_progress()

    def update_inference_result(self, results):
        for job in self.jobs:
            job.progress = 100
        self.set_inference_result_to_record(results)
        if self.total_progress >= 100:
            session = CeleryDBAdapter.session()
            job = session.query(CompoundInferenceJobsModel).get(self.job_id)
            infer_results = session.query(InferenceResultsModel).filter_by(job_id=self.job_id).all()
            for infer_result in infer_results:
                infer_result.status = StatusEnum.ready
                write_record(infer_result, session)
            job.status = StatusEnum.ready
            write_record(job, session)
            session.close()
        self.emit_message()

    def set_exec_info(self, data: dict):
        session = CeleryDBAdapter.session()
        infer_result = session.query(InferenceResultsModel).get(self.inference_result_record_id)
        infer_result.update(data)
        session.add(infer_result)
        session.commit()
        session.close()
        self.emit_message()

    def result_to_json(self) -> list:
        res = []
        session = CeleryDBAdapter.session()
        inference_results = session.query(InferenceResultsModel).filter_by(job_id=self.job_id).all()
        for inference_result in inference_results:
            res.append(inference_result.json())
        session.close()
        return res

    def update_progress_for_inference_result(self):
        session = CeleryDBAdapter.session()
        infer_result = session.query(InferenceResultsModel).get(self.inference_result_record_id)
        infer_result.progress = self.local_progress
        infer_result.status = StatusEnum.running
        write_record(infer_result, session)
        session.close()

    def set_inference_result_to_record(self, results):
        session = CeleryDBAdapter.session()
        infer_result = session.query(InferenceResultsModel).get(self.inference_result_record_id)
        infer_result.progress = 100
        infer_result.update(results)
        write_record(infer_result, session)
        session.close()

    def add_stage(self, stage: IEmitMessageStage, silent: bool = False) -> str:
        stage = super().add_stage(stage, silent)
        self.update_progress_for_inference_result()
        return stage

    @property
    def total_progress(self):
        session = CeleryDBAdapter.session()
        infer_results = session.query(InferenceResultsModel).filter_by(job_id=self.job_id).all()
        compound_infer_record = session.query(CompoundInferenceJobsModel).filter_by(job_id=self.job_id).first()
        num_single_inferences = compound_infer_record.num_single_inferences
        progress = 0.0
        for infer_result in infer_results:
            progress += infer_result.progress
        total_progress = progress / num_single_inferences
        compound_infer_record.progress = total_progress
        write_record(compound_infer_record, session)
        session.close()
        return total_progress

    @property
    def local_progress(self):
        return sum([job.progress * job.weight / self.job.total_jobs for job in self.jobs])
