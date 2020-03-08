"""
 OpenVINO Profiler
 Accuracy checker configuration converter

 Copyright (c) 2019 Intel Corporation

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
from app.main.jobs.utils.yml_templates.utils import ModelWithMeta


class AccuracyConfigConverter:
    @staticmethod
    def from_model_with_meta(model: ModelWithMeta) -> dict:
        return AccuracyConfigConverter.from_accuracy_representation({
            'taskType': model.task_type.value,
            'taskMethod': model.task_method.value,
            **model.config
        })

    @staticmethod
    def from_accuracy_representation(data: dict) -> dict:
        result = dict()
        if 'taskType' in data and data['taskType']:
            result['taskType'] = data['taskType']
        if 'taskMethod' in data and data['taskMethod']:
            result['taskMethod'] = data['taskMethod']
        if 'preprocessing' in data and data['preprocessing']:
            result['preprocessing'] = AccuracyConfigConverter.convert_preprocessing(data['preprocessing'],
                                                                                    data.get('hasBackground'))
        if 'postprocessing' in data and data['postprocessing']:
            result['postprocessing'] = AccuracyConfigConverter.convert_postprocessing(data['postprocessing'])
        if 'metric' in data and data['metric']:
            result['metric'] = AccuracyConfigConverter.convert_metric(data['metric'])
        return result

    @staticmethod
    def convert_preprocessing(preprocessing: tuple, has_background: bool = False) -> dict:
        preprocessing_converters = {
            'resize': AccuracyConfigConverter.convert_resize,
            'auto_resize': lambda resize: AccuracyConfigConverter.convert_resize({'size': 'auto_resize'}),
            'normalization': AccuracyConfigConverter.convert_normalization,
            'bgr_to_rgb': AccuracyConfigConverter.convert_bgr_to_rgb
        }
        result = AccuracyConfigConverter.convert_processing(preprocessing, preprocessing_converters)
        if 'bgr_to_rgb' not in result:
            result['bgr_to_rgb'] = False
        result['hasBackground'] = has_background
        return result

    @staticmethod
    def convert_postprocessing(postprocessing: tuple) -> dict:
        if not postprocessing:
            return {
                'resize_prediction_boxes': 'None'
            }
        postprocessing_converters = {
            'resize_prediction_boxes': AccuracyConfigConverter.convert_resize_prediction_boxes,
            'nms': AccuracyConfigConverter.convert_nms
        }
        result = AccuracyConfigConverter.convert_processing(postprocessing, postprocessing_converters)
        has_nms = [key for key in result if 'nms' in key]
        result['resize_prediction_boxes'] = 'ResizeBoxes NMS' if has_nms else 'ResizeBoxes'
        return result

    @staticmethod
    def convert_metric(metrics: tuple):
        results = []
        for metric in metrics:
            results.append({})
            result = results[-1]
            for key, value in metric.items():
                if key == 'type':
                    result['metric'] = value.capitalize()
                    continue
                if key in ('integral', 'overlap_threshold', 'top_k'):
                    result['{}.{}'.format(metric['type'].lower(), key)] = value
                    continue
                result[key] = value
        return results if len(results) != 1 else results[0]

    @staticmethod
    def convert_resize(resize_preprocessing: dict) -> dict:
        return {'resize_size': resize_preprocessing['size']}

    @staticmethod
    def convert_normalization(normalization_preprocessing: dict) -> dict:
        result = dict()
        for key, value in normalization_preprocessing.items():
            if key == 'type':
                continue
            result['normalization.{}'.format(key)] = [float(i) for i in value.split(',')] if isinstance(value,
                                                                                                        str) else value
        return result

    # pylint: disable=unused-argument
    @staticmethod
    def convert_bgr_to_rgb(bgr_to_rgb_preprocessing: dict) -> dict:
        return {'bgr_to_rgb': True}

    @staticmethod
    def convert_resize_prediction_boxes(resize_prediction_boxes: dict) -> dict:
        result = {'resize_prediction_boxes': 'None'}
        for key, value in resize_prediction_boxes.items():
            if key == 'type':
                continue
            result['resize_prediction_boxes'] = value
        return result

    @staticmethod
    def convert_nms(nms: dict) -> dict:
        result = dict()
        for key, value in nms.items():
            if key == 'type':
                continue
            result['nms.{}'.format(key)] = value
        return result

    @staticmethod
    def convert_processing(data: tuple, converters: dict) -> dict:
        result = dict()
        for value in data:
            try:
                result.update(converters[value['type']](value))
            except KeyError:
                # OMZ models contain pre- and postprocessings that are not yet supported in DL Workbench.
                # This method is used only for serializing the accuracy configuration before sending it to the client,
                # therefore, ignoring unsupported routines is OK.
                continue
        return result
