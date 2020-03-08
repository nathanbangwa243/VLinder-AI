"""
 OpenVINO Profiler
 Class for parsing output of Model Downloader

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

from app.error.job_error import ModelDownloaderError
from app.main.jobs.interfaces.iemit_message import IEmitMessageStage
from app.main.jobs.model_downloader.model_downloader_emit_msg import ModelDownloaderEmitMessage
from app.main.jobs.tools_runner.tool_output_parser import ConsoleToolOutputParser


class ModelDownloaderParser(ConsoleToolOutputParser):

    def __init__(self, emit_message: ModelDownloaderEmitMessage, stage_types: tuple):
        super().__init__(emit_message, stage_types)
        self.buffer = ''

    def parse(self, string: str):
        if not string:
            return
        self.stdout += ('\n' + string)

        # Handle multi-line JSONs.
        try:
            console_status = json.loads(string)
            self.buffer = ''
        except json.decoder.JSONDecodeError:
            self.buffer += string
            try:
                console_status = json.loads(self.buffer)
                self.buffer = ''
            except json.decoder.JSONDecodeError:
                return

        type_stage = console_status['$type']
        if type_stage.startswith('model_'):
            method = getattr(self, type_stage, None)
            if method:
                if 'postprocessing' in type_stage:
                    method()
                else:
                    method(console_status)

    def model_download_begin(self, console_status: dict):
        self.emit_message.total_files = console_status['num_files']
        postprocessing_stage = IEmitMessageStage('Postprocessing', weight=0.01)
        self.emit_message.add_stage(postprocessing_stage, silent=True)

    def model_file_download_begin(self, console_status: dict):
        new_file_name = console_status['model_file']
        stage_type = 'Downloading {}'.format(new_file_name)
        stage = IEmitMessageStage(stage_type, weight=(1 - 0.01) / self.emit_message.total_files)
        stage.file_size = console_status['size']
        self.emit_message.add_stage(stage, silent=True)

    def model_file_download_progress(self, console_status: dict):
        file_name = console_status['model_file']
        stage_name = 'Downloading {}'.format(file_name)
        downloaded_size = console_status['size']
        stage = self.emit_message.find_job_by_name(stage_name)
        total_size = stage.file_size
        self.emit_message.update_progress(stage_name, downloaded_size / total_size * 100)

    def model_file_download_end(self, console_status: dict):
        file_name = console_status['model_file']
        stage_name = 'Downloading {}'.format(file_name)
        self.emit_message.update_progress(stage_name, 100)

    def model_download_end(self, console_status: dict):
        if not console_status['successful']:
            error_message = 'Failed to download: {}'.format(console_status['model'])
            self.emit_message.add_error(error_message)
            raise ModelDownloaderError(error_message, self.emit_message.job_id)
        for stage in self.emit_message.jobs:
            if 'Downloading' in stage.job_type:
                stage.progress = 100
        self.emit_message.emit_progress()

    def model_postprocessing_begin(self):
        self.emit_message.update_progress('Postprocessing', 50)

    def model_postprocessing_end(self):
        self.emit_message.update_progress('Postprocessing', 100)
