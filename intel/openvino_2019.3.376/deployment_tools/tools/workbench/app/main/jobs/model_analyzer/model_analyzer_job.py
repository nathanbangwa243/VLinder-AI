"""
 OpenVINO Profiler
 Class for model validation

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

from app.error.job_error import ModelAnalyzerError
from app.main.job_factory.config import CeleryDBAdapter
from app.main.jobs.feed.feed_emit_msg import FeedEmitMessage
from app.main.jobs.interfaces.iemit_message import IEmitMessageStage
from app.main.jobs.interfaces.ijob import IJob
from app.main.jobs.job_types_enum import JobTypesEnum
from app.main.jobs.model_analyzer.model_analyzer_emit_msg import ModelAnalyzerEmitMessage
from app.main.jobs.uploads.models.model_upload_config import ModelUploadConfig
from app.main.models.enumerates import StatusEnum, ModelPrecisionEnum
from app.main.models.factory import write_record
from app.main.models.topologies_model import TopologiesModel
from app.main.models.topology_analysis_jobs_model import TopologyAnalysisJobsModel
from app.main.utils.utils import find_all_paths, get_size_of_files

# pylint: disable=import-error,no-name-in-module
from openvino.inference_engine import IENetwork

PERFORM_ANALYSIS = True
try:
    from model_analyzer.network_complexity import NetworkComputationalComplexity
except ImportError:
    PERFORM_ANALYSIS = False


class ModelAnalyzerJob(IJob):
    """
    The Model Analyzer job fills an analysis data for a model: g_flops, g_iops, max_mem etc.
    and it is only one correct place to fill two model properties: precision and size of a model
    """
    db_table = TopologiesModel
    event = 'model'

    def __init__(self, job_id: int, config: ModelUploadConfig, weight: float):
        super(ModelAnalyzerJob, self).__init__(JobTypesEnum.model_analyzer_type,
                                               ModelAnalyzerEmitMessage(self, job_id, config, weight))
        self.emit_message.event = ModelAnalyzerJob.event

    def run(self):
        self.emit_message.add_stage(IEmitMessageStage('analyzing', weight=1), silent=True)
        session = CeleryDBAdapter.session()
        model = session.query(TopologiesModel).get(self.emit_message.job_id)
        model.size = get_size_of_files(model.path)
        write_record(model, session)
        if model.status in (StatusEnum.cancelled, StatusEnum.queued, StatusEnum.error):
            session.close()
            return
        if not PERFORM_ANALYSIS:
            self.emit_message.update_analyze_progress(100)
            model.progress = 100
            model.status = StatusEnum.ready
            model.size = get_size_of_files(model.path)
            write_record(model, session)
            session.close()
            return
        model_path = model.path
        try:
            analyze_data = self.analyze(model_path)
            session = CeleryDBAdapter.session()
            topology_analysis = (
                session.query(TopologyAnalysisJobsModel)
                .filter_by(model_id=self.emit_message.job_id)
                .first()
            )
            self.emit_message.update_analyze_progress(50)
            topology_analysis.set_analysis_data(analyze_data)
            write_record(topology_analysis, session)
            session.close()
        except ModelAnalyzerError as error:
            FeedEmitMessage.socket_io = self.emit_message.socket_io
            FeedEmitMessage.emit(ModelAnalyzerError.code, str(error))
        model_xml = find_all_paths(model_path, ('.xml',))[0]
        model_bin = os.path.splitext(model_xml)[0] + '.bin'
        try:
            net = IENetwork(model_xml, weights=model_bin)
        except Exception as exc:
            error_message = str(exc)
            self.emit_message.add_error(error_message)
            raise ModelAnalyzerError(error_message, self.emit_message.job_id)
        session.close()
        session = CeleryDBAdapter.session()
        model = session.query(TopologiesModel).get(self.emit_message.job_id)
        if not model.precision:
            model.precision = ModelPrecisionEnum(net.precision)
        model.status = StatusEnum.ready
        model.size = get_size_of_files(model.path)
        write_record(model, session)
        topology_analysis = (
            session.query(TopologyAnalysisJobsModel)
                .filter_by(model_id=self.emit_message.job_id)
                .first()
        )
        topology_analysis.status = StatusEnum.ready
        topology_analysis.progress = 100
        write_record(topology_analysis, session)
        model = session.query(TopologiesModel).get(self.emit_message.job_id)
        model.progress = 100
        model.status = StatusEnum.ready
        write_record(model, session)
        session.close()
        self.emit_message.emit_message()

    def analyze(self, model_path):
        model_xml = find_all_paths(model_path, ('.xml',))[0]
        model_bin = os.path.splitext(model_xml)[0] + '.bin'
        try:
            net = IENetwork(model_xml, weights=model_bin)
            ncc = NetworkComputationalComplexity(net, -1)

            g_flops, g_iops = ncc.get_total_ops()
            g_flops = '{:.3f}'.format(g_flops)
            g_iops = '{:.3f}'.format(g_iops)

            total_params = '{:.3f}'.format(ncc.get_total_params() / 10 ** 6)
            min_mem_consumption = '{:.3f}'.format(ncc.get_minimum_memory_consumption() / 10 ** 6)
            max_mem_consumption = '{:.3f}'.format(ncc.get_maximum_memory_consumption() / 10 ** 6)
            sparsity = '{:.3f}'.format(ncc.get_total_sparsity() * 100)
        except Exception as exc:
            error_message = str(exc)
            raise ModelAnalyzerError(error_message, self.emit_message.job_id)
        return {
            'g_flops': g_flops,
            'g_iops': g_iops,
            'm_params': total_params,
            'min_mem': min_mem_consumption,
            'max_mem': max_mem_consumption,
            'sparsity': sparsity,
        }

    def on_failure(self):
        session = CeleryDBAdapter.session()
        model = session.query(TopologiesModel).get(self.emit_message.job_id)
        model.status = StatusEnum.error
        write_record(model, session)
        session.close()
