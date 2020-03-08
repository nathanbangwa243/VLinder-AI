"""
 OpenVINO Profiler
 Class for running cli command

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
import fcntl
import os
import shlex
import subprocess
import logging as log

from app.error.general_error import GeneralError
from app.main.jobs.interfaces.ijob import IJob
from app.main.jobs.tools_runner.console_parameters import ConsoleToolParameters
from app.main.jobs.tools_runner.tool_output_parser import ConsoleToolOutputParser

ALLOWED_CLI_TOOLS = ('calibrate.py',
                     'check_accuracy.py',
                     'winograd_tool.py',
                     'downloader.py',
                     'info_dumper.py',
                     'converter.py',
                     'mo.py',
                     'tar',
                     'unzip',
                     'benchmark.py',
                     'benchmark_app')


def set_non_blocking(stream):
    """
    Without these flags any CLI tool blocks the STDIN/STDOUT and no-one can read it until the process finishes.
    DL Workbench reads the logs in real-time therefore we need those streams to be not blocked.
    """
    flags = fcntl.fcntl(stream, fcntl.F_GETFL)
    flags = flags | os.O_NONBLOCK
    fcntl.fcntl(stream, fcntl.F_SETFL, flags)


def run_console_tool(params: ConsoleToolParameters, parser: ConsoleToolOutputParser, job: IJob = None,
                     measure_performance=False) -> tuple:
    basename = os.path.basename(params.exe)
    if basename not in ALLOWED_CLI_TOOLS:
        raise GeneralError('Unsupported command line tool')
    log.debug('[ RUN SUBPROCESS ] %s', params)
    command = shlex.split(str(params))
    if not measure_performance:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=params.environment)
        set_non_blocking(process.stdout)
        set_non_blocking(process.stderr)
    else:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if job:
        job.subprocess.append(process.pid)

    message = ''
    while True:
        output = process.stdout.readline().decode()
        if not output and process.poll() is not None:
            break
        if output:
            log.debug(output)
        parser.parse(output.strip())
        message += output + '\n\n'
        if not measure_performance:
            # For performance measurements (run benchmark application) stderr forward to stdout
            # error must be recognize and parse in parser
            error = process.stderr.readline().decode()
            if error and 'error' in error.lower():
                message = ''
                while True:
                    message += error
                    error = process.stderr.readline().decode()
                    if not error and process.poll() is not None:
                        log.error(message)
                        return process.poll(), message
    return process.poll(), message
