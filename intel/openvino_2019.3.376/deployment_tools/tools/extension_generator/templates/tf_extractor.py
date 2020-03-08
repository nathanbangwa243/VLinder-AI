# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ===============================================================================
# Generated file for TensorFlow layer extractor for Model Optimizer
#
# You need to modify this file if you need several attributes of the layer
# to appear in the IR in different format than the default one. Then you
# need to implement pre-processing logic here.
#
# Refer to the section "Extending Model Optimizer with New Primitives" in
# OpenVINO* documentation (either online or offline in
# <INSTALL_DIR>/deployment_tools/documentation/docs/index.html an then navigate
# to the corresponding section).
# ===============================================================================

import numpy as np
[[[cog
from ext_gen.interactive_module import InteractiveModule


name = InteractiveModule.get_param('name')
op_name = InteractiveModule.get_param('opName')
if InteractiveModule.was_param_set('opClassName'):
    op_class_name = InteractiveModule.get_param('opClassName')
    op_class_path = InteractiveModule.get_param('opClassPath')
else:
    op_class_name = None
    op_class_path = None

custom_params = InteractiveModule.get_param('customParams');
type_to_gen = {'type': "\'{}\':tf_dtype_extractor(param[\"{}\"].type)",
               'shape': "\'{}\':tf_tensor_shape(param[\"{}\"].shape)",
               'padding': "\'{}\':convert_tf_padding_to_str(param[\"{}\"])",
               'spatial': "\'{}\':tf_data_format_spatial(param[\"{}\"])",
               'channel': "\'{}\':tf_data_format_channel(param[\"{}\"])",
               'batch': "\'{}\':tf_data_format_batch(param[\"{}\"])"}

attr_parsers=""
for c in custom_params :
    if len(c) < 3 or c[2] == '':
        attr_parsers +='''
#Parser for attribute {attr}
def convert_tf_{attr_low}({attr}):
    raise Error(\"Attribute parser was not implemented for {attr}\")
    return

'''.format(attr=c[0], attr_low=c[0].lower())

if InteractiveModule.get_param('allCopy'):
    attr_parsing='''
        attrs = collect_tf_attrs(param)
        attrs['op']= __class__.op
'''
else :
    attr_parsing="attrs = {\n"

    for c in custom_params :
        if len(c) < 3 or c[2] == '':
            attr_parsing +="         \'{}\':convert_tf_{}(param[\'{}\']),\n".format(c[1], c[0].lower(), c[0])
        else :
            if c[2] in type_to_gen:
                attr_parsing +=("            "+type_to_gen[c[2]]+",\n").format(c[1], c[0])
            else :
                attr_parsing +="            \'{}\':param[\"{}\"].{},\n".format(c[1], c[0], c[2])
    attr_parsing +="            \'op\': __class__.op\n"
    attr_parsing +="        }"

if op_class_name:
    node_update="{}.update_node_stat(node, attrs)".format(op_class_name)
else :
    node_update="Op.get_op_class_by_name(__class__.op).update_node_stat(node, attrs)"

template='''
from mo.front.extractor import FrontExtractorOp
from mo.ops.op import Op
from mo.front.tf.extractors.utils import *
from mo.front.common.partial_infer.utils import convert_tf_padding_to_str
{import_class}

{attr_parsers}

class {name}FrontExtractor(FrontExtractorOp):
    op = \'{op_name}\' 
    enabled = True

    @staticmethod
    def extract(node):
        proto_layer = node.pb
        param = proto_layer.attr
        # extracting parameters from TensorFlow layer and prepare them for IR
        {attr_parsing}

        # update the attributes of the node
        {node_update}

        return __class__.enabled

'''.format(import_class="from {} import {}".format(op_class_path, op_class_name) if op_class_name else "",
           attr_parsers=attr_parsers, name=name, op_name=op_name,
           attr_parsing=attr_parsing, node_update=node_update)
cog.outl(template)
]]]
[[[end]]]
