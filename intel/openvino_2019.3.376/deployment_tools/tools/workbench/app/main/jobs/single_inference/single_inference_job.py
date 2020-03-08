"""
 OpenVINO Profiler
 Class for inference job creation

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
import logging as log

from app.error.job_error import CompoundInferenceError

from app.main.console_tool_wrapper.benchmark_app.console_output_parser import BenchmarkConsoleOutputParser
from app.main.console_tool_wrapper.benchmark_app.error_message_processor import BenchmarkErrorMessageProcessor
from app.main.console_tool_wrapper.benchmark_app.parameters import BenchmarkAppParameters
from app.main.console_tool_wrapper.benchmark_app.stages import BenchmarkAppStages
from app.main.console_tool_wrapper.inference_engine_tool.error_message_processor import ErrorMessageProcessor

from app.main.job_factory.config import CeleryDBAdapter
from app.main.jobs.single_inference.single_inference_config import SingleInferenceConfig
from app.main.jobs.single_inference.single_inference_emit_msg import SingleInferenceEmitMessage
from app.main.jobs.interfaces.ijob import IJob
from app.main.jobs.job_types_enum import JobTypesEnum
from app.main.jobs.tools_runner.runner import run_console_tool
from app.main.jobs.utils.database_functions import set_status_in_db
from app.main.models.compound_inference_model import CompoundInferenceJobsModel

from app.main.models.enumerates import StatusEnum
from app.main.models.factory import write_record
from app.main.models.inference_results_model import InferenceResultsModel

from app.main.utils.utils import create_empty_dir
from config.constants import BENCHMARK_APP_REPORT_DIR
from utils.csv_benchmark_report.csv_inference_report import BenchmarkAppReportTypesEnum, CSVBenchmarkReport, \
    CSVNoCountersBenchmarkReport, CSVAverageBenchmarkReport
from utils.runtime_representation_report.runtime_representation_report import RuntimeRepresentationReport


class SingleInferenceJob(IJob):
    db_table = CompoundInferenceJobsModel

    def __init__(self, job_id: int, config: SingleInferenceConfig, weight: float):
        super(SingleInferenceJob, self).__init__(JobTypesEnum.single_inference_type,
                                                 SingleInferenceEmitMessage(self, job_id, config, weight))
        self.infer_queue = []
        self.report_dir = os.path.join(BENCHMARK_APP_REPORT_DIR, str(self.emit_message.job_id))
        self.total_jobs = 2

    def run(self):
        emit_msg = self.emit_message
        config = emit_msg.config
        log.debug('[ INFERENCE ] Setup for model %s and dataset %s', config.model_id, config.dataset_id)
        session = CeleryDBAdapter.session()
        set_status_in_db(CompoundInferenceJobsModel, emit_msg.job_id, StatusEnum.running, session)
        session.close()

        parser = BenchmarkConsoleOutputParser(self.emit_message, BenchmarkAppStages.get_stages())

        create_empty_dir(self.report_dir)

        parameters = self.setup_parameters(self.emit_message.config, perf_counters=True)

        self._run_benchmark_app(parser, parameters)

        perf_report = self._read_report(BenchmarkAppReportTypesEnum.no_counters)
        average_counters_report = self._read_report(BenchmarkAppReportTypesEnum.average_counters)

        exec_graph = None
        if SingleInferenceJob.is_runtime_representation_available(parameters):
            exec_graph_path = parameters.get_parameter('exec_graph_path')
            exec_graph = RuntimeRepresentationReport(exec_graph_path).content

        parameters = self.setup_parameters(self.emit_message.config, perf_counters=False)

        self._run_benchmark_app(parser, parameters)

        second_perf_report = self._read_report(BenchmarkAppReportTypesEnum.no_counters)

        if perf_report.throughput < second_perf_report.throughput:
            perf_report = second_perf_report

        performance_results = {
            'execInfo': {
                'latency': perf_report.latency,
                'throughput': perf_report.throughput,
                'totalExecTime': perf_report.total_exec_time,
            },
            'pc': average_counters_report.performance_counters_results,
        }

        emit_msg.set_exec_info(performance_results)

        emit_msg.update_inference_result({'predictions': None,
                                          'pc': performance_results['pc'],
                                          'execGraph': exec_graph})

    def on_failure(self):
        session = CeleryDBAdapter.session()
        infer_result = session.query(InferenceResultsModel).get(self.emit_message.inference_result_record_id)
        infer_result.status = StatusEnum.error
        compound_inference_job = infer_result.compound_inference_job
        if not compound_inference_job.error_message and self.emit_message.jobs:
            message = ErrorMessageProcessor.general_error(self.emit_message.get_current_job().name)
            self.emit_message.add_error(message)
        write_record(infer_result, session)
        session.close()

    def _run_benchmark_app(self, parser, parameters):

        return_code, message = run_console_tool(parameters, parser, self, measure_performance=True)
        if return_code:
            job_name = self.emit_message.get_current_job().name if self.emit_message.get_current_job() else None
            error = BenchmarkErrorMessageProcessor.recognize_error(message, job_name)
            log.error('[ INFERENCE ] [ ERROR ]: %s', error)
            self.emit_message.add_error('Benchmark app failed: {}'.format(error))
            raise CompoundInferenceError(error, self.emit_message.job_id)

    def _read_report(self, report: type(BenchmarkAppReportTypesEnum)) -> CSVBenchmarkReport:
        try:
            class_report = CSVNoCountersBenchmarkReport if report is BenchmarkAppReportTypesEnum.no_counters \
                else CSVAverageBenchmarkReport
            csv_reader = class_report(os.path.join(self.report_dir, '{}.csv'.format(report.value)))
        except ValueError:
            raise CompoundInferenceError('Inconsistent benchmark app report', self.emit_message.job_id)
        return csv_reader

    def set_task_id(self, task_id: str):
        session = CeleryDBAdapter.session()
        inference_job = session.query(InferenceResultsModel).filter_by(job_id=self.emit_message.job_id).first()
        inference_job.task_id = task_id
        inference_job.status = StatusEnum.running
        write_record(inference_job, session)
        compound_inference_job = session.query(CompoundInferenceJobsModel) \
            .filter_by(job_id=self.emit_message.job_id).first()
        compound_inference_job.task_id = task_id
        compound_inference_job.status = StatusEnum.running
        write_record(compound_inference_job, session)
        session.close()

    @staticmethod
    def is_runtime_representation_available(parameters: BenchmarkAppParameters):
        return parameters.params['d'] in ['CPU', 'GPU', 'MYRIAD']

    def setup_parameters(self, config: SingleInferenceConfig, perf_counters: bool = True) -> BenchmarkAppParameters:
        try:
            parameters = BenchmarkAppParameters(config=config)
        except ValueError as error:
            raise CompoundInferenceError(str(error), self.emit_message.job_id)
        parameters.set_parameter('report_folder', self.report_dir)
        exec_graph_path = os.path.join(self.report_dir, 'exec_graph.xml')
        if perf_counters:
            parameters.set_parameter('report_type', 'average_counters')
            parameters.set_parameter('pc', '')
            if SingleInferenceJob.is_runtime_representation_available(parameters):
                parameters.set_parameter('exec_graph_path', exec_graph_path)
        else:
            parameters.set_parameter('report_type', 'no_counters')
        return parameters
