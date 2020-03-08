"""
 OpenVINO Profiler
 Class for handling reports from benchmark application

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
import enum
import re


class CSVBenchmarkReport:
    separator = ';'

    def __init__(self, path: str = ''):
        self.path_file = path
        try:
            with open(self.path_file) as csv_file:
                self.content = [line.strip() for line in csv_file]
        except OSError as error:
            raise error

    def find_index(self, table_name):
        for i, line in enumerate(self.content):
            if table_name in line:
                return i
        raise ValueError('Inconsistent benchmark app report')

    def read_table(self, head: str):
        try:
            start_index = self.find_index(head)
        except ValueError:
            raise ValueError('Inconsistent benchmark app report does not contain {} table'.format(head))
        table = []
        for i in range(start_index + 1, len(self.content)):
            string = self.content[i]
            if not string:
                break
            table.append(string)
        return table


class CSVNoCountersBenchmarkReport(CSVBenchmarkReport):

    def __init__(self, path: str = ''):
        super().__init__(path)

        self.command_line_parameters = self.read_command_line_parameters()

        self.execution_results = self.read_execution_results()

    def read_command_line_parameters(self):
        head = 'Command line parameters'
        return self.read_table(head)

    @property
    def device(self):
        for parameter in self.command_line_parameters:
            split_line = parameter.split(self.separator)
            name = split_line[0]
            value = split_line[1]
            if name == 'd' or 'target device' in name:
                return value
        raise ValueError('Inconsistent benchmark app report')

    def read_execution_results(self):
        head = 'Execution results'
        return self.read_table(head)

    @property
    def latency(self):
        return self.get_value(self.execution_results, 'latency')

    @property
    def throughput(self):
        return self.get_value(self.execution_results, 'throughput')

    @property
    def total_exec_time(self):
        return self.get_value(self.execution_results, 'total execution time')

    @staticmethod
    def get_value(table, name):
        for line in table:
            if name in line.lower():
                pattern = r'\d*\.\d+|\d+'
                pattern = re.compile(pattern)
                matches = re.findall(pattern, line)
                if len(matches) == 1:
                    return float(matches[0])
                raise ValueError('Found more than one value for {}'.format(name))
        raise ValueError('Did not find a value for  {}'.format(name))


class CSVAverageBenchmarkReport(CSVBenchmarkReport):

    def __init__(self, path: str = '', device: str = 'CPU'):
        super().__init__(path)
        self.performance_counters_results = self.parse_performance_counters(self.content, device)

    @staticmethod
    def parse_performance_counters(table: list, device: str) -> dict:
        head = table[0].split(CSVBenchmarkReport.separator)
        layer_name_index = head.index('layerName')
        exec_status_index = head.index('execStatus')
        layer_type_index = head.index('layerType')
        exec_type_index = head.index('execType')
        real_time_index = head.index('realTime (ms)')
        cpu_time_index = head.index('cpuTime (ms)')

        res = {}
        # Use offset 2. first and second lines are headers
        for row in table[1:]:
            if not row:
                continue
            string = row.split(CSVBenchmarkReport.separator)
            exec_time = float(string[real_time_index]) + float(string[cpu_time_index])
            if device == 'CPU':
                exec_time = float(string[real_time_index])
            res[string[layer_name_index]] = {
                'status': string[exec_status_index],
                'layer_type': string[layer_type_index],
                'exec_type': string[exec_type_index],
                'exec_time': exec_time,
            }
        return res


class BenchmarkAppReportTypesEnum(enum.Enum):
    no_counters = 'benchmark_report'
    average_counters = 'benchmark_average_counters_report'
