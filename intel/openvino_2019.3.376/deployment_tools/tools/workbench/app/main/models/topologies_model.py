"""
 OpenVINO Profiler
 Class for ORM model described a Model

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
import json

from sqlalchemy import Column, ForeignKey, Integer
from sqlalchemy.orm import relationship

from app.main.jobs.utils.yml_templates.config_converter import AccuracyConfigConverter
from app.main.models.artifacts_model import ArtifactsModel
from app.main.models.enumerates import SUPPORTED_FRAMEWORKS_ENUM_SCHEMA, MODEL_PRECISION_ENUM_SCHEMA, \
    MODEL_SOURCE_ENUM_SCHEMA, ModelPrecisionEnum, SupportedFrameworksEnum
from app.main.models.model_optimizer_scan_model import ModelOptimizerScanJobsModel
from app.main.utils.utils import find_all_paths

# pylint: disable=unused-import
from app.main.models.topologies_metadata_model import TopologiesMetaDataModel


class TopologiesModel(ArtifactsModel):
    __tablename__ = 'topologies'

    id = Column(Integer, ForeignKey('artifacts.id'), primary_key=True)
    optimized_from = Column(Integer, ForeignKey('topologies.id'), nullable=True)
    converted_from = Column(Integer, ForeignKey('topologies.id'), nullable=True)
    downloaded_from = Column(Integer, ForeignKey('omz_topologies.id'), nullable=True)
    metadata_id = Column(Integer, ForeignKey('topologies_metadata.id'), nullable=False)
    precision = Column(MODEL_PRECISION_ENUM_SCHEMA, nullable=True)
    framework = Column(SUPPORTED_FRAMEWORKS_ENUM_SCHEMA, nullable=True)
    source = Column(MODEL_SOURCE_ENUM_SCHEMA, nullable=True)

    converted_from_record = relationship('TopologiesModel', foreign_keys=[converted_from], remote_side=[id])
    downloaded_from_record = relationship('OMZTopologyModel', foreign_keys=[downloaded_from], lazy='joined')
    analysis = relationship('TopologyAnalysisJobsModel', back_populates='topology', uselist=False, lazy='joined')
    meta = relationship('TopologiesMetaDataModel', back_populates='topologies', lazy='joined')

    def __init__(self, name, framework, metadata_id, session_id):
        super().__init__(name, session_id)
        self.framework = framework
        self.metadata_id = metadata_id

    def json(self):
        json_message = {
            'xmlContent': self.get_xml_content() if self.framework == SupportedFrameworksEnum.openvino else '',
            **self.short_json()
        }
        original_topology = self.id if not self.converted_from else self.converted_from
        mo_analyzed_job = (
            ModelOptimizerScanJobsModel.query
                .filter_by(topology_id=original_topology)
                .first()
        )
        if mo_analyzed_job:
            mo_scan_data = mo_analyzed_job.json()
            json_message['mo'] = {
                'analyzedParams': mo_scan_data['information']
            }
        return json_message

    def short_json(self):
        if self.converted_from:
            original_framework = self.converted_from_record.framework.value
        elif self.downloaded_from:
            original_framework = self.downloaded_from_record.framework.value
        else:
            original_framework = SupportedFrameworksEnum.openvino.value
        return {
            **super().json(),
            'fileType': 'model',
            'readiness': self.status.value,
            'precision': self.precision.value if self.precision else ModelPrecisionEnum.unknown.value,
            'modelSource': self.source.value if self.source else None,
            'framework': self.framework.value,
            'analysis': self.analysis.json() if self.analysis else {},
            'originalModelFramework': original_framework,
            'advancedConfiguration': AccuracyConfigConverter.from_accuracy_representation(
                {
                    'taskType': self.meta.task_type.value if self.meta.task_type else None,
                    'taskMethod': self.meta.topology_type.value if self.meta.topology_type else None,
                    **(json.loads(self.meta.advanced_configuration) if self.meta.advanced_configuration else {})
                })
        }

    def get_xml_content(self):
        xml_name = find_all_paths(self.path, ('.xml',))
        xml_content = ''
        if xml_name:
            with open(xml_name[0]) as xml_file:
                xml_content = '\n'.join(xml_file.readlines())
        return xml_content
