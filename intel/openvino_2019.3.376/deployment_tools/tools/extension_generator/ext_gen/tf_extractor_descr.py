# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from ext_gen.interactive_module import InteractiveModule


class MOTFExtractorDescr(InteractiveModule):
    def __init__(self, op_gen: bool):
        self.params = {
            'isPythonic': [False, True],
            'name': ['', False],
            'allCopy': [False, False],
            'customParams': [[], False],
            'opName': ['', False],
            'opClassName': ['', False],
            'opClassPath': ['', False],
            'supported_types': [{'b': 'Bool', 'padding': 'Padding type', 'list.b': 'List of bools',
                                 'f': 'Float', 'batch': 'Get batch from dataFormat', 'list.f': 'List of floats',
                                 'i': 'Int', 'channel': 'Get channel from dataFormat', 'list.i': 'List of ints',
                                 's': 'String', 'spatial': 'Get spatial from dataFormat', 'list.s': 'List of strings',
                                 'shape': 'TensorShapeProto', 'list.shape': 'List of TensorShapeProto', 'type': 'DataType',
                                 'list.type': 'List of DataType', }]
        }
        self.is_op_gen = op_gen
        self.all_quests = self.get_all_questions()
        self.section_name = "TensorFlow* extractor generation"
        super().__init__(self.params, self.all_quests)

    def print_types(self):
        s = ""
        i = 0
        for sa in self.params['supported_types'][0].keys():
            if i % 3 == 0:
                s = s + '\n  '
            s = s + "{:40}".format("{} - {},".format(sa, self.params['supported_types'][0][sa]))
            i = i + 1
        return s

    def check_param_all_copy(self):
        return self.get_param('allCopy')

    def get_all_questions(self):
        return [
            [
                '\nEnter layer name:   ',
                'name',
                InteractiveModule.set_answer_to_param_standard,
                self.return_true,
                'Layer name:    ',
            ],
            [
                ''.join([
                    '\nDo you want to automatically parse all parameters from the model file? (y/n)\n',
                    '  Yes means layer parameters will be automatically parsed during Model Optimizer work as is.\n',
                    '  No means you will be prompted for layer parameters in the following section    ',
                ]),
                'allCopy',
                InteractiveModule.set_answer_to_param_bool,
                self.return_true,
                'Automatically parse all parameters from model file:    ',
            ],

            [
                ''.join([
                    '\nEnter all parameters in the following format:\n',
                    '  <param1> <new name1> <type1>\n',
                    '  <param2> <new name2> <type2>\n',
                    '  ...\n',
                    'Where type is one of the following types:\n',
                    ''+self.print_types()+'\n',
                    'Example: \n',
                    '  length attr_length i\n',
                    '\n'
                    'If your attribute type is not shown in the list above, or you want to implement your own attribute parsing,\n',
                    'omit the <type> parameter.\n',
                    'Enter \'q\' when finished:    ',
                ]),
                'customParams',
                InteractiveModule.set_answer_to_param_list,
                lambda: not self.check_param_all_copy(),
                'Parameters entered:  <param1> <new name1> <type1>',
            ],
            [
                '\nEnter operation name to use with this extractor:    ',
                'opName',
                InteractiveModule.set_answer_to_param_standard,
                lambda: not self.is_op_gen,
                'Operation name:    ',
            ],
            [
                '\nEnter class with operation to use with this extractor:    ',
                'opClassName',
                InteractiveModule.set_answer_to_param_standard,
                lambda: not self.is_op_gen,
                'Class with operation:    ',
            ],
            [
                '\nEnter import path to class with operation:    ',
                'opClassPath',
                InteractiveModule.set_answer_to_param_standard,
                lambda: not self.is_op_gen,
                'Import path:    ',
            ]
        ]

    def read_config(self, data):
        # TODO add checks if something omitted in config
        super().set_answer_to_param_standard('name', data['op'])
        super().set_answer_to_param_standard('opName', data['op'])
        super().set_answer_to_param_standard('opClassName', data['op_class_name'])
        super().set_answer_to_param_standard('opClassPath', data['op_class_path'])
        super().set_answer_to_param_standard('allCopy', data['all_copy'])

        if not data['all_copy']:
            params = data['params']
            for p in params:
                if 'type' in p.keys():
                    super().params['customParams'][0].append([p['old_name'],
                                                              p['new_name'], p['type']])
                else:
                    super().params['customParams'][0].append([p['old_name'], p['new_name']])
            super().params['customParams'][1] = True
