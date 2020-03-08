"""
 OpenVINO Profiler
 Entry point for running winograd tool

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

import argparse
import os
import sys
# pylint: disable=import-error
from winograd.winograd_autotuner import WinogradAutotuner


def create_cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--path_to_model',
                        help='Path to an .xml file with a trained model.',
                        type=str,
                        required=True)
    parser.add_argument('-i',
                        '--path_to_input',
                        help='Path to a folder with images and/or binaries or to specific image or binary file.',
                        type=str,
                        required=True)
    parser.add_argument('-l',
                        '--path_to_extension',
                        help='Path to a CPU extension library.',
                        type=str,
                        required=True)
    parser.add_argument('-t',
                        '--inference_time',
                        help='time for one inference in seconds.',
                        type=int,
                        default=10,
                        required=False)
    parser.add_argument('-o',
                        '--output',
                        help='Output name for calibrated model. Default is <original_model_name>_winograd.xml|bin',
                        type=str)
    return parser


def process_output_path(original_path: str):
    return os.path.dirname(original_path)


def main():
    parser = create_cli_parser()
    args = parser.parse_args()
    output_path = args.output if args.output else process_output_path(args.path_to_model)
    try:
        winograd_autotune = WinogradAutotuner(args, output_path)
        winograd_autotune.run()
    except Exception:
        print('Cannot apply winograd primitive for this topology', file=sys.stderr)
        exit(1)


if __name__ == "__main__":
    main()
