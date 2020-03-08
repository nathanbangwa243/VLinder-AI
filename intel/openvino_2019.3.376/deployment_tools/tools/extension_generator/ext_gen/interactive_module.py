# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log


last_index = 1

class InteractiveModule:
    params = {}
    all_quests = []
    asked_questions = []
    section_name = ""

    def __init__(self, params, all_quests):
        for k in params.keys():
            if k not in __class__.params:
                __class__.params[k] = params[k]
        __class__.all_quests = all_quests

    @staticmethod
    def set_answer_to_param_standard(param_name, answer):
        __class__.params[param_name][0] = answer.strip() if isinstance(answer, str) else answer
        __class__.params[param_name][1] = True

    @staticmethod
    def set_answer_to_param_bool(param_name, answer):
        if answer.lower() == 'y' or answer.lower() == 'yes':
            __class__.params[param_name][0] = True
            __class__.params[param_name][1] = True
        else:
            __class__.params[param_name][0] = False
            __class__.params[param_name][1] = True

    @staticmethod
    def set_answer_to_param_list(param_name, answer):
        if len(__class__.params[param_name][0]) != 0:
            __class__.params[param_name][0].clear()
        while answer.lower() != 'q':
            split_answer = answer.split(' ')
            l = len(__class__.params[param_name][0])
            if len(split_answer) != 0:
                __class__.params[param_name][0].append(split_answer)

            if len(split_answer) == 1:
                i = __class__.params[param_name][0][l][0].rfind('.')
                __class__.params[param_name][0][l].append(__class__.params[param_name][0][l][0][(i + 1):])

            answer = input()

        __class__.params[param_name][1] = True

    def ask_question(self, index, check_if=True):
        if index < 0 or index >= len(self.all_quests):
            log.error("There is no question number " + str(index))
            return
        if check_if is False or self.all_quests[index][3]():
            self.asked_questions.append(index)
            answer = input(self.all_quests[index][0])
            self.all_quests[index][2](self.all_quests[index][1], answer)
        return

    @staticmethod
    def get_param(param_name):
        if param_name in __class__.params:
            return __class__.params[param_name][0]
        else:
            log.error("You are trying to get the parameter that was not set " + param_name)
            return

    @staticmethod
    def reset_param(param_name):
        if param_name in __class__.params:
            __class__.params[param_name][1] = False
        else:
            log.error("You are trying to reset the parameter that was not set " + param_name)
            return

    @staticmethod
    def was_param_set(param_name):
        return param_name in __class__.params and __class__.params[param_name][1]

    def final_input_data(self, asked_list):
        final_list = []

        def convert_param_to_string(param):
            if type(param) is bool:
                return "Yes" if param is True else "No"
            else:
                return str(param)

        if len(asked_list) != 0:
            print("\n\n**********************************************************************************************\n" +
                  "Check your answers for {}:\n".format(self.section_name))
            index = last_index
            final_list = []
            for q in asked_list:
                final_list.append(q)
                print(str(index) + ".  {:70} {}".format(self.all_quests[q][4],
                                                        convert_param_to_string(self.get_param(self.all_quests[q][1]))))
                index = index + 1
            print("\n**********************************************************************************************\n")
        return final_list

    def ask_final_check(self):
        global last_index
        fin_list = self.final_input_data(self.asked_questions)
        if len(fin_list) != 0:
            change = input("Do you want to change any answer (y/n) ? Default \'no\' \n")
            while change.lower() == 'yes' or change.lower() == 'y':
                index = input("Input a question number to change your answer:    ")
                index = int(index) - (last_index-1)
                self.ask_question(fin_list[index-1], False)
                for q in range(len(self.all_quests)):
                    len_asked = len(self.asked_questions)
                    if q not in set(fin_list):
                        self.ask_question(q)
                        if len_asked != len(self.asked_questions):
                            fin_list.append(q)
                self.final_input_data(fin_list)
                change = input("Do you want to change any answer (y/n) ? Default \'no\' \n")

            last_index = last_index + len(fin_list)

    def create_extension_description(self):
        self.asked_questions.clear()
        for index in range(0, (len(self.all_quests))):
            self.ask_question(index)

        self.ask_final_check()

    @staticmethod
    def return_true():
        return True

    def get_all_questions(self):
        raise NotImplementedError('Need to override method "get_all_questions" to return all required questions')
