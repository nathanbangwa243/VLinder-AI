"""
 Model Analyzer

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


from argparse import ArgumentParser

import os
import logging as log

import sys
# pylint: disable=import-error,no-name-in-module
from openvino.inference_engine import IENetwork

from bin_generator import generate_bin, remove_generated_file
from network_complexity import NetworkComputationalComplexity


def build_argparser():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    parser = ArgumentParser()

    parser.add_argument('-m', '--model',
                        help='Path to an .xml file of the Intermediate Representation (IR) model',
                        required=True,
                        type=str)

    parser.add_argument('-w', '--weights',
                        help='Path to the .bin file of the Intermediate Representation (IR) model. If not specified'
                             'it is expected that the weights file name is the same as the .xml file passed'
                             'with --model option',
                        type=str)

    parser.add_argument('-b', '--batch',
                        help='Batch size. Default is 1',
                        type=int,
                        default=-1)

    parser.add_argument('-o', '--report-dir',
                        help='Output directory',
                        type=str,
                        default='.')

    parser.add_argument('--model-report',
                        help='Name for the file where theoretical analysis results are stored',
                        type=str,
                        default='model_report.csv')

    parser.add_argument('-g', '--generate-bin',
                        help='Generate bin file for provided xml',
                        action='store_true',
                        default=False)

    parser.add_argument('--per-layer-mode',
                        help='Flag enables collecting per-layer complexity metrics',
                        action='store_true',
                        default=False)

    parser.add_argument('--per-layer-report',
                        help='File name for the per-layer complexity metrics. Should be specified only when -c option' +
                             ' is used',
                        default='per_layer_report.csv')

    parser.add_argument('--sparsity-ignored-layers',
                        help='Comma separated list of ignored layers',
                        default='')

    parser.add_argument('--sparsity-ignore-first-conv',
                        help='Enables ignoring first Convolution layer for sparsity computation',
                        action='store_true',
                        default=False)

    parser.add_argument('--sparsity-ignore-fc',
                        help='Enables ignoring FullyConnected layers for sparsity computation',
                        action='store_true',
                        default=False)

    parser.add_argument('--sparsity-level',
                        help='Desired number of zero parameters in percents',
                        type=int,
                        default=0)

    return parser


def main(cli_args):
    model_xml = cli_args.model
    if cli_args.generate_bin:
        model_bin = generate_bin(model_xml)
    elif not cli_args.weights:
        model_bin = os.path.splitext(model_xml)[0] + '.bin'
    else:
        model_bin = cli_args.weights

    batch = cli_args.batch
    output = cli_args.report_dir
    file_name = cli_args.model_report
    per_layer_mode = cli_args.per_layer_mode
    per_layer_report = cli_args.per_layer_report
    sparsity_ignored_layers = cli_args.sparsity_ignored_layers.split(",")
    sparsity_ignore_first_conv = cli_args.sparsity_ignore_first_conv
    sparsity_ignore_fc = cli_args.sparsity_ignore_fc
    sparsity_level = cli_args.sparsity_level

    log.info('Loading network files:\n\t%s\n\t%s', model_xml, model_bin)

    net = IENetwork(model=model_xml, weights=model_bin)
    ncc = NetworkComputationalComplexity(net, batch)
    ncc.get_ignored_layers(sparsity_ignored_layers, sparsity_ignore_first_conv, sparsity_ignore_fc)
    if cli_args.sparsity_level:
        ncc.sparsify(sparsity_level)
    ncc.print_network_info(output, file_name, per_layer_mode, per_layer_report)

    if cli_args.generate_bin:
        remove_generated_file(model_bin)

    return 0


if __name__ == '__main__':
    ARGS = build_argparser().parse_args()
    sys.exit(main(ARGS))
