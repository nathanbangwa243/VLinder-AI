"""
 OpenVINO Profiler
 Parameters for Model Optimizer related endpoints

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

from numbers import Real
from pathlib import Path
from typing import Callable, Iterable, Mapping

from app.main.models.enumerates import SupportedFrameworksEnum, ModelPrecisionEnum
from config.constants import TF_CUSTOM_OPERATIONS_CONFIGS


class Param:
    # pylint: disable=too-many-arguments
    def __init__(self, param_name: str, cli_arg_name: [str, None] = None,
                 required: bool = False, scope: str = 'general',
                 validate: Callable = None, to_arg: Callable = None, to_param: Callable = None):
        self.param_name = param_name
        self.cli_arg_name = cli_arg_name
        self.required = required
        self.scope = scope
        self.validate = validate if validate else lambda v: isinstance(v, str)
        self.to_arg = to_arg if to_arg else lambda v: v
        self.to_param = to_param if to_param else lambda v: v


class InputsParam(Param):
    keys_config = {
        'name': {
            'required': True,
            'validate': lambda v: isinstance(v, str),
        },
        'shape': {
            'required': False,
            'validate': lambda v: isinstance(v, list) and all(isinstance(element, int) for element in v),
        },
        'means': {
            'required': False,
            'validate': lambda v: isinstance(v, list) and all(isinstance(element, Real) for element in v),
        },
        'scales': {
            'required': False,
            'validate': lambda v: isinstance(v, list) and all(isinstance(element, Real) for element in v),
        },
    }

    def __init__(self, param_name):
        super().__init__(param_name, validate=self._validate)
        self.to_arg = None
        self.to_param = None

    @classmethod
    def validate_element(cls, element: Mapping) -> [Mapping, None]:
        errors = {
            'unknown': [],
            'missing': [],
            'invalid': {},
        }

        required_keys = set(k for k, p in cls.keys_config.items() if p['required'])
        errors['missing'] = list(required_keys - set(element.keys()))

        for key, value in element.items():
            if key not in cls.keys_config:
                errors['unknown'].append(key)
            if not cls.keys_config[key]['validate'](value):
                errors['invalid'][key] = value

        return errors if any(errors.values()) else None

    @classmethod
    def _validate(cls, value: Iterable[Mapping]) -> bool:
        return bool(value) and isinstance(value, list) and not any(cls.validate_element(e) for e in value)


class MOForm:
    params_config = [
        Param(
            'batch',
            cli_arg_name='batch',
            validate=lambda v: isinstance(v, int) and v > 0,
        ),
        Param(
            'dataType',  # precision
            cli_arg_name='data_type',
            required=True,
            validate=lambda v: v in (ModelPrecisionEnum.fp16.value, ModelPrecisionEnum.fp32.value),
        ),
        Param(
            'originalChannelsOrder',
            cli_arg_name='reverse_input_channels',
            required=True,
            validate=lambda v: v in ('RGB', 'BGR'),
            to_arg=lambda v: v == 'BGR',  # If BGR - reverse, to make it be RGB
            to_param=lambda v: 'BGR' if v else 'RGB',
        ),
        InputsParam('inputs'),
        Param(
            'outputs',
            cli_arg_name='output',
            validate=lambda v: isinstance(v, list) and all(isinstance(element, str) for element in v),
            to_arg=','.join,
            to_param=lambda v: v.split(','),
        ),
        Param(
            'enableSsdGluoncv',
            cli_arg_name='enable_ssd_gluoncv',
            scope=SupportedFrameworksEnum.mxnet.value,
            validate=lambda v: isinstance(v, bool),
        ),
        Param(
            'legacyMxnetModel',
            cli_arg_name='legacy_mxnet_model',
            scope=SupportedFrameworksEnum.mxnet.value,
            validate=lambda v: isinstance(v, bool),
        ),
        Param(
            'useCustomOperationsConfig',
            cli_arg_name='tensorflow_use_custom_operations_config',
            scope=SupportedFrameworksEnum.tf.value,
            validate=lambda v: v in TF_CUSTOM_OPERATIONS_CONFIGS,
            to_arg=lambda v: TF_CUSTOM_OPERATIONS_CONFIGS[v],
            to_param=lambda v: Path(v).stem,
        ),
    ]

    def __init__(self, data: dict, framework: str):
        self.data = {k: v for k, v in data.items() if v is not None}
        self.framework = framework
        self.is_invalid = None
        self.errors = None
        self.validate()

    @classmethod
    def get_param_name_to_param_conf_map(cls) -> dict:
        return {
            param_conf.param_name: param_conf
            for param_conf in cls.params_config
        }

    @classmethod
    def get_cli_arg_name_to_param_conf_map(cls) -> dict:
        return {
            param_conf.cli_arg_name: param_conf
            for param_conf in cls.params_config
            if param_conf.cli_arg_name
        }

    def validate(self) -> [dict, None]:
        errors = {
            'missing': [],
            'unknown': [],
            'unsuitable': [],
            'invalid': {},
        }
        scopes = ('general', self.framework)

        required_params = set(p.param_name for p in self.params_config if p.required and p.scope in scopes)
        errors['missing'] = list(required_params - set(self.data.keys()))

        params_config_map = self.get_param_name_to_param_conf_map()

        for key, value in self.data.items():
            if key not in params_config_map:
                errors['unknown'].append(key)
                continue
            if params_config_map[key].scope not in scopes:
                errors['unsuitable'].append(key)
            if not params_config_map[key].validate(value):
                errors['invalid'][key] = value

        if self.framework == SupportedFrameworksEnum.mxnet.value:
            if 'inputs' not in self.data:
                errors['missing'].append('inputs')

        self.is_invalid = any(errors.values())
        self.errors = errors if self.is_invalid else None
        return self.errors

    def _prepare_channels_order_dependent_values(self, key: str, arg_name: str, args: dict):
        values = {input_['name']: input_[key] for input_ in self.data['inputs'] if key in input_}
        if self.data['originalChannelsOrder'] == 'BGR':
            values = {k: list(reversed(v)) for k, v in values.items()}
        prepared_values = ','.join(
            '{}[{}]'.format(k, ','.join(str(float(e)) for e in v))
            for k, v in values.items()
        )
        if prepared_values:
            args[arg_name] = prepared_values

    def get_args(self) -> dict:
        if self.is_invalid:
            raise ValueError(self.errors)

        params_config_map = self.get_param_name_to_param_conf_map()

        args = {
            params_config_map[key].cli_arg_name: params_config_map[key].to_arg(value)
            for key, value in self.data.items()
            if params_config_map[key].cli_arg_name
        }

        if 'inputs' in self.data:
            inputs = self.data['inputs']

            if 'batch' in self.data:
                del args['batch']

            self._prepare_channels_order_dependent_values('means', 'mean_values', args)
            self._prepare_channels_order_dependent_values('scales', 'scale_values', args)

            input_names = (input_['name'] for input_ in inputs)
            input_shapes = [input_['shape'] for input_ in inputs if 'shape' in input_]

            args['input'] = ','.join(input_names)
            if input_shapes:
                args['input_shape'] = ','.join(
                    '[{}]'.format(','.join(str(int(element)) for element in shape))
                    for shape in input_shapes
                )
        return args

    @classmethod
    def to_params(cls, mo_args: dict):
        arg_to_param_map = cls.get_cli_arg_name_to_param_conf_map()

        params = {
            arg_to_param_map[arg].param_name: arg_to_param_map[arg].to_param(value)
            for arg, value in mo_args.items()
            if arg in arg_to_param_map
        }

        if 'input' in mo_args:
            parsed_values = {arg_name: {} for arg_name in ('mean_values', 'scale_values')}
            for arg_name in parsed_values:
                if arg_name in mo_args:
                    for layer_values in mo_args[arg_name].split('],'):
                        name, vector_str = layer_values.split('[')
                        parsed_values[arg_name][name] = [float(value) for value in vector_str.rstrip(']').split(',')]
                        if params['originalChannelsOrder'] == 'BGR':
                            parsed_values[arg_name][name] = list(reversed(parsed_values[arg_name][name]))

            names = mo_args['input'].split(',')
            shapes = [
                [int(e) for e in shape_string.lstrip('[').rstrip(']').split(',') if e]
                for shape_string in mo_args['input_shape'].split('],')
            ] if 'input_shape' in mo_args else []

            params['inputs'] = []
            for index, name in enumerate(names):
                input_ = {
                    'name': name,
                }
                if index < len(shapes) and shapes[index]:
                    input_['shape'] = shapes[index]
                if name in parsed_values['mean_values']:
                    input_['means'] = parsed_values['mean_values'][name]
                if name in parsed_values['scale_values']:
                    input_['scales'] = parsed_values['scale_values'][name]
                params['inputs'].append(input_)
        return params
