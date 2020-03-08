# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from ext_gen.interactive_module import InteractiveModule


class MOExtractorDescr(InteractiveModule):
    def __init__(self, op_gen: bool):
        self.params = {
            'isPythonic': ['', False],
            'module_name': ['', False],
            'name': ['', False],
            'paramName': ['', False],
            'allCopy': [True, False],
            'rename': [False, False],
            'customParams': [[], False],
            'opName': ['', False],
            'flatten': [False, False],
            'omitDefault': [True, False]
        }
        self.is_op_gen = op_gen
        self.all_quests = self.get_all_questions()
        self.section_name = "Caffe* extractor generation"
        super().__init__(self.params, self.all_quests)

    def get_all_questions(self):
        return [
            [
                '\nIs your layer Pythonic (y/n)?   ',
                'isPythonic',
                InteractiveModule.set_answer_to_param_bool,
                self.return_true,
                'Pythonic layer:    ',
            ],
            [
                '\nEnter module name:   ',
                'module_name',
                InteractiveModule.set_answer_to_param_standard,
                self.check_is_pythonic,
                'Module name:    ',
            ],
            [
                '\nEnter layer name:   ',
                'name',
                InteractiveModule.set_answer_to_param_standard,
                self.return_true,
                'Layer name:    ',
            ],
            [
                '\nEnter parameters name (as it written in proto file):   ',
                'paramName',
                InteractiveModule.set_answer_to_param_standard,
                lambda: not self.check_is_pythonic(),
                'Parameters name:    ',
            ],
            [
                ''.join([
                    '\nDo you want to copy all parameters from proto file to xml (y/n)\n',
                    '(you can omit training params, for example)    ',
                ]),
                'allCopy',
                InteractiveModule.set_answer_to_param_bool,
                lambda: not self.check_is_pythonic(),
                'Copy all parameters:    ',
            ],
            [
                '\nDo you want to omit optional parameters that was not set for this layer and have not default value? (y/n)    ',
                'omitDefault',
                InteractiveModule.set_answer_to_param_bool,
                lambda: self.check_param_all_copy() and not self.check_is_pythonic(),
                'Omit optional parameters:    ',
            ],
            [
                '\nDo you want to flatten nested parameters? (y/n)    ',
                'flatten',
                InteractiveModule.set_answer_to_param_bool,
                lambda: self.check_param_all_copy() and not self.check_is_pythonic(),
                'Flatten nested parameters:    ',
            ],
            [
                '\nDo you want to rename any parameter or set default value for any parameter: (y/n)    ',
                'rename',
                InteractiveModule.set_answer_to_param_bool,
                lambda: self.check_param_all_copy() and not self.check_is_pythonic(),
                'Rename any parameter or set default value:    ',
            ],
            [
                ''.join([
                    '\nEnter all parameters that you want to rename in format\n',
                    '  <param1> <new name1> <default value>\n',
                    '  <param2> <new name2> <default value>\n',
                    '  ...\n',
                    'Example:\n',
                    '  length attr_length 0\n',
                    '\n'
                    '<default value> is optional\n',
                    'Enter \'q\' when finished:   ',
                ]),
                'customParams',
                InteractiveModule.set_answer_to_param_list,
                self.check_param_rename,
                'Renamed parameters in format <param1> <new name1> <default value>:    ',
            ],
            [
                ''.join([
                    '\nEnter all parameters that you want to copy in format\n',
                    '  <param1> <new name1> <default value>\n',
                    '  <param2> <new name2> <default value>\n',
                    '  ...\n',
                    'Example:\n',
                    '  length attr_length 0\n',
                    '\n'
                    '<new name> and <default value> are optional\n',
                    'Enter \'q\' when finished:    ',
                ]),
                'customParams',
                InteractiveModule.set_answer_to_param_list,
                lambda: not self.check_param_all_copy(),
                'Parameters to copy in format <param1> <new name1> <default value>:    ',
            ],
            [
                '\nEnter operation name to use with this extractor:    ',
                'opName',
                InteractiveModule.set_answer_to_param_standard,
                lambda: not self.is_op_gen,
                'Operation name:    ',
            ]
        ]

    def check_param_all_copy(self):
        return self.get_param('allCopy')

    def check_param_rename(self):
        return self.get_param('rename')

    def check_not_set_param_op_name(self):
        return not self.params['opName'][1]

    def check_is_pythonic(self):
        return self.get_param('isPythonic')

    def read_config(self, data):
        # TODO add checks if something omitted in config
        super().set_answer_to_param_standard('name', data['op'])
        super().set_answer_to_param_standard('opName', data['op'])
        super().set_answer_to_param_standard('isPythonic', data['pythonic'])
        if self.params['isPythonic'][0]:
            super().set_answer_to_param_standard('module_name', data['module'])
        else:
            super().set_answer_to_param_standard('paramName', data['param_name'])
            super().set_answer_to_param_standard('allCopy', data['allCopy'])
            super().set_answer_to_param_standard('rename', data["rename"])
            super().set_answer_to_param_standard('flatten', data['flatten'])
            super().set_answer_to_param_standard('omitDefault', data['omitDefault'])

        if not data['pythonic'] and ((not data['allCopy']) or data['rename']):
            params = data['params']
            for p in params:
                if 'default' in p.keys():
                    super().params['customParams'][0].append([p['old_name'],
                                                              p['new_name'], p['default']])
                else:
                    super().params['customParams'][0].append([p['old_name'], p['new_name']])
            super().params['customParams'][1] = True
        else:
            if not data['pythonic']:
                super().params['customParams'][0] = []
                super().params['customParams'][1] = True
