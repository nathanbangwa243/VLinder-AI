"""
 OpenVINO Profiler
 Class for logging

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
import os

from config.constants import DEFAULT_LOG_FILE


class FileHandler(log.FileHandler):
    def emit(self, record):
        if self.stream:
            self.stream.close()
            self.stream = None
        self.tail_file(self.baseFilename)
        return super(FileHandler, self).emit(record)

    @staticmethod
    def tail_file(file_path: str, lines_number=20000):
        avg_line_length = 100
        file_size = os.path.getsize(file_path)
        needed_size = avg_line_length * lines_number
        if file_size < needed_size:
            return
        try:
            with open(file_path, 'rb') as file_descr:
                file_descr.seek(-needed_size, 2)
                lines = file_descr.read()
            with open(file_path, 'wb') as file_descr:
                file_descr.write(lines)
        except IOError:
            pass


class InitLogger:
    log_file = os.getenv('WB_LOG_FILE', DEFAULT_LOG_FILE)
    handler_num = 0

    @staticmethod
    def init_logger():
        log_level = os.getenv('WB_LOG_LEVEL', 'DEBUG')
        log_file = os.getenv('WB_LOG_FILE', None)
        if log_file is not None:
            handler = FileHandler(log_file)
        else:
            handler = log.StreamHandler()
        logger = log.getLogger()
        logger.setLevel(log_level.upper())
        if InitLogger.handler_num == 0:
            logger.addHandler(handler)
            InitLogger.handler_num += 1
