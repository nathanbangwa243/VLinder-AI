# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from ext_gen.interactive_module import InteractiveModule


class MOOpDescr(InteractiveModule):

    def __init__(self, is_extr_gen: bool = False):
        self.params = {
            'isPythonic': ['', False],
            'opName': ['', False],
            'supportedAttrs': [[], False],
            'internalAttrs': [[], False],
            'changeShape': [False, False],
            'hasInfer': [True, False],
            'isLayerNameOpName': [False, False],
        }
        self.all_quests = self.get_all_questions()
        self.section_name = "the Model Optimizer operation generation"
        super().__init__(self.params, self.all_quests)
        self.is_extr_gen = is_extr_gen

    def check_is_not_extr_gen(self):
        return not self.is_extr_gen

    @staticmethod
    def check_set_customparams_and_not_pythonic():
        return InteractiveModule.was_param_set('customParams') and \
               len(InteractiveModule.get_param('customParams')) != 0 and \
               not InteractiveModule.get_param('isPythonic')

    @staticmethod
    def check_not_supportedattrs_and_not_pythonic():
        return not InteractiveModule.was_param_set('supportedAttrs') and \
               not InteractiveModule.get_param('isPythonic') and \
               (InteractiveModule.was_param_set('customParams') and len(InteractiveModule.get_param('customParams')) != 0
                or not InteractiveModule.was_param_set('customParams'))

    @staticmethod
    def check_not_internalattrs_and_not_pythonic():
        return not InteractiveModule.was_param_set('internalAttrs') and \
               not InteractiveModule.get_param('isPythonic') and \
               (InteractiveModule.was_param_set('customParams') and len(InteractiveModule.get_param('customParams')) != 0
                or not InteractiveModule.was_param_set('customParams'))

    @staticmethod
    def check_is_not_pythonic():
        return not InteractiveModule.get_param('isPythonic')

    @staticmethod
    def check_is_change_shape():
        return InteractiveModule.get_param('changeShape')

    @staticmethod
    def check_is_not_set_opname():
        return not InteractiveModule.was_param_set('opName')

    @staticmethod
    def check_is_layername_set_and_opname_not_set():
        return not InteractiveModule.was_param_set('opName') and InteractiveModule.was_param_set('name')

    @staticmethod
    def set_opname_as_layername(param, answer):
        if param != 'isLayerNameOpName':
            log.error("Internal error")
        InteractiveModule.set_answer_to_param_bool('isLayerNameOpName', answer)
        if answer.lower() == 'y' or answer.lower() == 'yes':
            InteractiveModule.set_answer_to_param_standard('opName', InteractiveModule.get_param('name'))
        else:
            InteractiveModule.reset_param('opName')

    @staticmethod
    def set_supportedattrs(param, answer):
        if param != 'supportedAttrs':
            log.error("Internal error")
        sup_attrs = []
        int_attrs = []
        attrs = InteractiveModule.get_param('customParams')

        while answer.lower() != 'q':
            sup_attrs.append(attrs[int(answer)-1])
            answer = input()

        InteractiveModule.set_answer_to_param_standard(param, sup_attrs)
        for attr in attrs:
            if attr not in sup_attrs:
                int_attrs.append(attr)

        InteractiveModule.set_answer_to_param_standard('internalAttrs', int_attrs)

    @staticmethod
    def print_attributes():
        s = ""
        if InteractiveModule.was_param_set('customParams'):
            attrs = InteractiveModule.get_param('customParams')
            for i in range(1, len(attrs)+1):
                s = s + "  {}. {}\n".format(i, attrs[i-1][1])
        return s

    def get_all_questions(self):
        return [
            [
                '\nDo you use this operation with Caffe Pythonic layer extractor? (y/n)   ',
                'isPythonic',
                InteractiveModule.set_answer_to_param_bool,
                self.check_is_not_extr_gen,
                'Caffe Pythonic layer: '
            ],
            [
                '\nDo you want to use the layer name as the operation name? (y/n)    ',
                'isLayerNameOpName',
                self.set_opname_as_layername,
                self.check_is_layername_set_and_opname_not_set,
                'Use layer name as operation name? (y/n) '
            ],
            [
                '\nEnter operation name:    ',
                'opName',
                InteractiveModule.set_answer_to_param_standard,
                self.check_is_not_set_opname,
                'Operation name: '
            ],
            [
                ''.join([
                    '\nChoose indexes of all attributes from the list below that should be output in IR or needed for shape calculation:\n',
                    self.print_attributes() + '\n',
                    'Enter \'q\' when finished:    ',
                ]),
                'supportedAttrs',
                self.set_supportedattrs,
                self.check_set_customparams_and_not_pythonic,
                'Attributes included in IR: '
            ],
            [
                ''.join([
                    '\nInput all attributes that should be output in IR or needed for shape calculation in format:\n',
                    '  <attr1>\n',
                    '  <attr2>\n',
                    '  ...\n',
                    'Enter \'q\' when finished:    ',
                ]),
                'supportedAttrs',
                InteractiveModule.set_answer_to_param_list,
                self.check_not_supportedattrs_and_not_pythonic,
                'Attributes included in IR: '
            ],
            [
                ''.join([
                    '\nInput all internal operation attributes, which will be omitted in IR, in format:\n',
                    '  <attr1>\n',
                    '  <attr2>\n',
                    '  ...\n',
                    'Enter \'q\' when finished:    ',
                ]), 'internalAttrs',
                InteractiveModule.set_answer_to_param_list,
                self.check_not_internalattrs_and_not_pythonic,
                'Internal attributes: '
            ],
            [
                '\nDoes your operation change shape? (y/n)    ',
                'changeShape',
                InteractiveModule.set_answer_to_param_bool,
                self.return_true,
                'Operation changes shape? (y/n) ',
            ],
            [
                '\n'.join([
                    '\nDo you want to implement shape calculation? (y/n)',
                    '    If you choose \'n\', framework fallback will be used for shape calculation    ']),
                'hasInfer',
                InteractiveModule.set_answer_to_param_bool,
                self.check_is_change_shape,
                'Implement shape calculation: ',
            ],

        ]

    def read_config(self, data):
        # TODO add checks if something omitted in config
        super().set_answer_to_param_standard('opName', data['op'])
        super().set_answer_to_param_standard('changeShape', data['change_shape'])
        super().set_answer_to_param_standard('hasInfer', data['has_infer'])
        super().set_answer_to_param_standard('isPythonic', data['pythonic'])

        if not data['pythonic']:
            params = data['supported_attrs']
            for p in params:
                super().params['supportedAttrs'][0].append([p, p])
            super().params['supportedAttrs'][1] = True

        super().params['internalAttrs'][1] = True

        if 'internal_attrs' in data.keys():
            params = data['internal_attrs']
            for p in params:
                super().params['internalAttrs'][0].append([p, p])
            super().params['internalAttrs'][1] = True
