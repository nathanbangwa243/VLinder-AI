# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from ext_gen.interactive_module import InteractiveModule


class IEExtensionDescr(InteractiveModule):
    def __init__(self, plugin: str = 'cpu'):
        self.params = {
            'isPythonic': [False, False],
            'plugin': ['', False],
            'opName': ['', False],
            'params_cpu': [[], False],
            'params_gpu': [[], False],
            'supported_cpu_types': [{'int': 'Int', 'float': 'Float', 'bool': 'Bool',
                                     'string': 'String', 'listfloat': 'Floats', 'listint': 'Ints'}]
        }

        self.all_quests = self.get_all_questions()
        self.section_name = "the Inference Engine extension generation"
        super().__init__(self.params, self.all_quests)
        self.check_plugin_name('plugin', plugin)

    @staticmethod
    def is_cpu_and_supported_not_empty():
        return InteractiveModule.get_param('plugin') == 'cpu' and \
               (InteractiveModule.was_param_set('supportedAttrs') and len(InteractiveModule.get_param('supportedAttrs')) != 0 or
                not InteractiveModule.was_param_set('supportedAttrs') and
                (InteractiveModule.was_param_set('customParams') and len(InteractiveModule.get_param('customParams')) != 0 or
                not InteractiveModule.was_param_set('customParams')))

    @staticmethod
    def is_gpu_and_supported_not_empty():
        return InteractiveModule.get_param('plugin') == 'cldnn' and \
               (InteractiveModule.was_param_set('supportedAttrs') and len(InteractiveModule.get_param('supportedAttrs')) != 0 or
                not InteractiveModule.was_param_set('supportedAttrs') and
                (InteractiveModule.was_param_set('customParams') and len(InteractiveModule.get_param('customParams')) != 0 or
                not InteractiveModule.was_param_set('customParams')))

    @staticmethod
    def is_not_set_opname():
        return not InteractiveModule.was_param_set('opName')

    @staticmethod
    def is_layername_set_and_opname_not_set():
        return not InteractiveModule.was_param_set('opName') and InteractiveModule.was_param_set('name')

    @staticmethod
    def set_opname_as_layername(param, answer):
        if param != 'opName':
            log.error("Internal error")
        if answer.lower() == 'y' or answer.lower() == 'yes':
            InteractiveModule.set_answer_to_param_standard('opName', InteractiveModule.get_param('name'))

    @staticmethod
    def check_plugin_name(param_name, answer):
        while answer != 'cldnn' and answer != 'cpu':
            print("Incorrect plugin name, please choose on of [cpu, cldnn]")
            answer = input()

        InteractiveModule.set_answer_to_param_standard(param_name, answer)

    @staticmethod
    def print_ir_attrs():
        s = ''
        if InteractiveModule.was_param_set('supportedAttrs'):
            s = s + "Parameters included in IR: "
            for sup_attr in InteractiveModule.get_param('supportedAttrs'):
                s = s + '{}, '.format(sup_attr[1])
            s = s + '\n'

        return s

    def get_all_questions(self):
        return [
            [
                '\nDo you want to use layer name as operation name? (y/n)    ',
                'opName',
                self.set_opname_as_layername,
                self.is_layername_set_and_opname_not_set,
                'Use layer name as operation name: '
            ],
            [
                "\nEnter operation name:    ",
                'opName',
                InteractiveModule.set_answer_to_param_standard,
                self.is_not_set_opname,
                'Operation name: '
            ],
            [
                ''.join([
                    "\nEnter type for parameters that will be read from IR in format\n",
                    '  <param1> <type>\n',
                    '  <param2> <type>\n',
                    '  ...\n',
                    'Example:\n',
                    '  length int\n',
                    '\n'
                    '  Supported cpu types: %s\n' % ', '.join(self.params['supported_cpu_types'][0].keys()),
                    self.print_ir_attrs()+'\n',
                    'Enter \'q\' when finished:    ',
                ]),
                'params_cpu',
                InteractiveModule.set_answer_to_param_list,
                self.is_cpu_and_supported_not_empty,
                'Parameters types in format <param> <type>: '
            ],
            [
                ''.join([
                    "\nEnter type and default value for parameters that will be read in IR in format\n",
                    '  <param1> <type> <default_value>\n',
                    '  <param2> <type> <default_value>\n',
                    '  ...\n',
                    'Example:\n',
                    '  length int 0\n',
                    '\n',
                    self.print_ir_attrs()+'\n',
                    'Enter \'q\' when finished:    ',
                    ''
                ]),
                'params_gpu',
                InteractiveModule.set_answer_to_param_list,
                self.is_gpu_and_supported_not_empty,
                'Parameters types in format <param> <type>: '
            ]
        ]

    def read_config(self, data):
        # TODO add checks if something omitted in config
        super().set_answer_to_param_standard('opName', data["op"])
        super().set_answer_to_param_standard('isPythonic', data["pythonic"])
        super().set_answer_to_param_standard('plugin', data["plugin"])

        if not data["pythonic"]:
            params = data["params"]
            for p in params:
                if "default" in p.keys():
                    super().params['params_cpu'][0].append([p["name"], p["type"], p["default"]])
                else:
                    super().params['params_cpu'][0].append([p["name"], p["type"]])
            if len(params) != 0:
                super().params['params_cpu'][1] = True
