"""
 OpenVINO Profiler
 Interfaces class for error processing classes

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
import re


class ErrorMessageProcessor:
    match_error = {
        'failed to create engine: clGetPlatformIDs': 'Inference on Integrated Graphics (GPU) is not available',
        'Device with "GPU" name is not registered in the InferenceEngine':
            'Inference on Integrated Graphics (GPU) is not available',
    }

    @staticmethod
    def general_error(stage):
        return 'Failed in stage {}'.format(stage)

    @staticmethod
    def recognize_general_ie_message(error_message):
        return ErrorMessageProcessor.find_message(error_message)

    @classmethod
    def recognize_error(cls, error_message: str, stage) -> str:
        message = cls.find_message(error_message)
        if message:
            return message
        message = ErrorMessageProcessor.recognize_general_ie_message(error_message)
        if message:
            return message
        if error_message:
            errors_strings = set(error_message.split('\n'))
            return '\n'.join(errors_strings)
        message = cls.general_error(stage)
        return message

    @classmethod
    def find_message(cls, error_message):
        for pattern, message in cls.match_error.items():
            pattern = r'.*'.join(pattern.lower().split(' '))
            if re.search(r'.*{s}.*'.format(s=pattern), error_message.lower()):
                return message
        return None
