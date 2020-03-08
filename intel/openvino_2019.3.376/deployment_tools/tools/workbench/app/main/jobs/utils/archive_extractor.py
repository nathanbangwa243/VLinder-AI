"""
 OpenVINO Profiler
 Class for extracting tar and zip archives

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
import re
from collections import OrderedDict

from app.error.job_error import ArtifactError
from app.main.jobs.interfaces.ijob import IJob
from app.main.jobs.tools_runner.runner import run_console_tool, ConsoleToolParameters
from app.main.jobs.tools_runner.tool_output_parser import ConsoleToolOutputParser
from app.main.utils.utils import create_empty_dir
from config.constants import VOC_ROOT_FOLDER


class TarGzParameters(ConsoleToolParameters):
    def __init__(self, parameters):
        super(TarGzParameters, self).__init__()
        self.exe = 'tar'
        self.params = parameters

    def __str__(self, parameter_prefix=''):
        return ConsoleToolParameters.__str__(self, parameter_prefix)


class ZIPParameters(ConsoleToolParameters):
    def __init__(self, parameters):
        super(ZIPParameters, self).__init__()
        self.exe = 'unzip'
        self.params = parameters


class ExtractParser(ConsoleToolOutputParser):
    progress = 0

    def parse(self, string: str):
        stage = r'inflating:'
        if re.search(stage, string):
            if self.progress < 99:
                self.progress += 3
            if self.progress % 3 * 5:
                self.emit_message.update_extract_progress(self.progress)


class Extractor:
    def __init__(self, file_id, file_name, file_path, job: IJob, extract_path):
        self.file_id = file_id
        self.file_name = file_name
        self.file_path = file_path
        self.job = job
        self.extract_path = extract_path

    def extract_archive(self):
        job_id = self.job.emit_message.job_id
        target_path = os.path.join(self.extract_path, str(job_id))
        _, file_extension = os.path.splitext(self.file_id)
        if file_extension in ['.gz', '.tar', '.tar.gz']:
            _, err = self.unpack_tar(target_path)
        elif file_extension == '.zip':
            _, err = self.unpack_zip(target_path)
        else:
            message = 'Unsupported type of archive. Supported archives extensions: gz, tar, tar.gz, zip'
            self.job.emit_message.add_error(message)
            raise ArtifactError(message, 1)
        self.handle_archived_dir(target_path)
        if err != 'cancelled':
            self.job.emit_message.update_extract_progress(100)

    def handle_archived_dir(self, target_path):
        """Handle archives containing a directory."""
        contents = os.listdir(target_path)
        if len(contents) == 1 and os.path.isdir(os.path.join(target_path, contents[0])):
            if self.job.emit_message.event == 'dataset' and contents[0] == VOC_ROOT_FOLDER:
                return
            temporary_path = os.path.join(self.extract_path, 'temporary_directory')
            os.rename(target_path, temporary_path)
            os.rename(os.path.join(temporary_path, contents[0]), target_path)
            os.rmdir(temporary_path)

    def unpack_tar(self, target_path):
        self.job.emit_message.update_extract_progress(10)
        create_empty_dir(target_path)
        parameters = TarGzParameters(OrderedDict([('xfp', self.file_path), ('-C', target_path)]))
        return run_console_tool(parameters, ExtractParser(self.job.emit_message.job_id), self.job)

    def unpack_zip(self, target_path):
        self.job.emit_message.update_extract_progress(10)
        create_empty_dir(target_path)
        parameters = ZIPParameters(
            OrderedDict([('o', self.file_path),
                         ('d', target_path)]))
        return run_console_tool(parameters, ExtractParser(self.job.emit_message), self.job)
