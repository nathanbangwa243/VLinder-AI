# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import sys

from os import getcwd

from ext_gen.caffe_extractor_descr import MOExtractorDescr
from ext_gen.tf_extractor_descr import MOTFExtractorDescr
from ext_gen.mxnet_extractor_descr import MOMXNetExtractorDescr
from ext_gen.mo_op_descr import MOOpDescr
from ext_gen.ie_extension_descr import IEExtensionDescr


def read_config(conf):
    f = open(conf, 'r')
    data = json.load(f)
    f.close()

    is_gen_mo_caffe_ext = False
    is_gen_mo_tf_ext = False
    is_gen_mo_mxnet_ext = False
    is_gen_mo_op = False
    is_gen_ie_op = False

    for ds in data:
        if ds == 'mo_caffe_extractor':
            is_gen_mo_caffe_ext = True
        elif ds == 'mo_tf_extractor':
            is_gen_mo_tf_ext = True
        elif ds == 'mo_mxnet_extractor':
            is_gen_mo_mxnet_ext = True
        elif ds == 'mo_op':
            is_gen_mo_op = True
        elif ds == 'ie_op':
            is_gen_ie_op = True

    if is_gen_mo_caffe_ext:
        mo_caffe_extr = MOExtractorDescr(is_gen_mo_op)
        mo_caffe_extr.read_config(data['mo_caffe_extractor'])

    if is_gen_mo_tf_ext:
        mo_tf_extr = MOTFExtractorDescr(is_gen_mo_op)
        mo_tf_extr.read_config(data['mo_tf_extractor'])

    if is_gen_mo_mxnet_ext:
        mo_mxnet_extr = MOMXNetExtractorDescr(is_gen_mo_op)
        mo_mxnet_extr.read_config(data['mo_mxnet_extractor'])

    if is_gen_mo_op:
        mo_op_descr = MOOpDescr(is_gen_mo_caffe_ext)
        mo_op_descr.read_config(data['mo_op'])

    ie_cpu = False
    ie_gpu = False
    if is_gen_ie_op:
        ie_descr = IEExtensionDescr()
        ie_descr.read_config(data['ie_op'])
        if data['ie_op']['plugin'] == 'cpu':
            ie_cpu = True
        elif data['ie_op']['plugin'] == 'cldnn':
            ie_gpu = True
    return is_gen_mo_caffe_ext, is_gen_mo_tf_ext, is_gen_mo_op, ie_cpu, ie_gpu


class ExtGenCLIController:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Generates extension source files with stubs for the core functions",
            usage="\n".join([
                               "Extensions generator has two modes",        
                               "    * Interactive mode - the tool prompts you to input information", 
                               "      extgen command [<args>]",
                               "",
                               "    * Silent mode - the tool reads the input information from a configuration file.",
                               "      extgen config.json",
                               "",
                               "You can run it with the following commands:",
                               "    * to generate Model Optimizer extension (both extractor and operation) for unsupported TensorFlow* layer",
                               "        extgen new --mo-tf-ext --mo-op --output_dir=<output path>",
                               "    * to generate Model Optimizer extension for TensorFlow* layer and Inference Engine CPU extension",
                               "        extgen new --mo-tf-ext --mo-op --ie-cpu-ext --output_dir=<output path>",
                               "    * to generate Model Optimizer extension for Caffe* layer and Inference Engine CPU extension",
                               "        extgen new --mo-caffe-ext --mo-op --ie-cpu-ext --output_dir=<output path>",
                               "    * to generate Model Optimizer extension for MXNet* layer",
                               "        extgen new --mo-mxnet-ext --mo-op --output_dir=<output path>",
                               "",
                               "To get more information about arguments for interactive mode, run:",
                               "  extgen new --help"])
        )
        self.is_mo_mxnet_ext_gen = False
        self.is_mo_tf_ext_gen = False
        self.output_dir = getcwd()
        self.parser.add_argument('command', 
                                 help="The only supported command is new, which generates Model Optimizer and Inference Engine extensions")
        self.subparsers = self.parser.add_subparsers()
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = self.parser.parse_args(sys.argv[1:2])
        if hasattr(self, args.command) and args.command == "new":
            getattr(self, args.command)()
            self.is_from_config = False
        else:
            [self.is_mo_caffe_ext_gen, self.is_mo_tf_ext_gen, 
             self.is_mo_op_gen, 
             self.is_ie_cpu_gen, self.is_ie_gpu_gen] = read_config(sys.argv[1])
            self.is_from_config = True

    def new(self):
        new_parser = argparse.ArgumentParser(
            description='Arguments to configure extension generation in the interactive mode',
            usage='\n'.join(["You can use any combination of the following arguments:"])
        )

        new_parser.add_argument('--mo-caffe-ext', action='store_true', default=False,
                                help="generate a Model Optimizer Caffe* extractor")
        new_parser.add_argument('--mo-mxnet-ext', action='store_true', default=False,
                                help="generate a Model Optimizer MXNet* extractor")
        new_parser.add_argument('--mo-tf-ext', action='store_true', default=False,
                                help="generate a Model Optimizer TensorFlow* extractor")
        new_parser.add_argument("--mo-op", action='store_true', default=False,
                                help="generate a Model Optimizer operation")
        new_parser.add_argument("--ie-cpu-ext", action='store_true', default=False,
                                help="generate an Inference Engine CPU extension")
        new_parser.add_argument("--ie-gpu-ext", action='store_true', default=False,
                                help="generate an Inference Engine GPU extension")
        
        new_parser.add_argument("--output_dir", default=getcwd(),
                                help="set an output directory. If not specified, the current directory is used by default.")

        new_args = ['{}'.format(i) for i in sys.argv[2:]]
        args = new_parser.parse_args(new_args)

        self.is_mo_caffe_ext_gen = args.mo_caffe_ext
        self.is_mo_mxnet_ext_gen = args.mo_mxnet_ext
        self.is_mo_tf_ext_gen = args.mo_tf_ext        
        self.is_mo_op_gen = args.mo_op
        self.is_ie_cpu_gen = args.ie_cpu_ext
        self.is_ie_gpu_gen = args.ie_gpu_ext   

        self.output_dir = args.output_dir        
        
        if not any([self.is_mo_caffe_ext_gen, self.is_mo_mxnet_ext_gen, self.is_mo_tf_ext_gen,
                    self.is_mo_op_gen, self.is_ie_cpu_gen, self.is_ie_gpu_gen]):
            raise Exception('New should take anything or the unknown argument')

    def get_cli_results(self):
        return [self.is_mo_caffe_ext_gen, self.is_mo_mxnet_ext_gen, self.is_mo_tf_ext_gen,
                self.is_mo_op_gen, self.is_ie_cpu_gen, self.is_ie_gpu_gen,
                self.is_from_config, self.output_dir]
