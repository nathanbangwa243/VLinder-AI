# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ===============================================================================
# Generated file for Caffe layer extractor for Model Optimizer
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

# [[[cog
from ext_gen.interactive_module import InteractiveModule

is_pythonic = InteractiveModule.get_param('isPythonic')
name = InteractiveModule.get_param('name')
op_name = InteractiveModule.get_param('opName')
all_copy = InteractiveModule.get_param('allCopy')
custom_params = InteractiveModule.get_param('customParams')

attrs = 'mapping_rule = {}'

if len(custom_params):
    attrs = '''
        update_attrs = {{
{params}
        }}
    '''.format(params='\n'.join(['            \'{}\': param.{},'.format(c[0], c[1]) for c in custom_params]))
else:
    attrs = '''
        update_attrs = {}
    '''
if is_pythonic:
    attrs += '''
        mapping_rule = CaffePythonFrontExtractorOp.parse_param_str(param.param_str)
        mapping_rule.update(update_attrs)
    '''
if all_copy and not is_pythonic:
    attrs += '''
        mapping_rule = collect_attributes(param, 
                                          disable_omitting_optional={omit}, 
                                          enable_flattening_nested_params={flatten})
    '''.format(omit=not InteractiveModule.get_param('omitDefault'), flatten=InteractiveModule.get_param('flatten'))
if not is_pythonic and len(custom_params):
    attrs += '''
        mapping_rule.update(update_attrs)
        mapping_rule.update(layout_attrs())
   '''

template = '''
from mo.front.caffe.collect_attributes import merge_attrs, collect_attributes
from mo.front.caffe.extractors.utils import weights_biases
from mo.front.common.extractors.utils import layout_attrs
from mo.front.caffe.extractors.utils import embed_input

from mo.ops.op import Op
from mo.front.extractor import {fe_module}FrontExtractorOp


class {class_name}FrontExtractor({fe_module}FrontExtractorOp):
    op = '{op_mapping_name}'
    enabled = True

    @staticmethod
    def extract(node):
        proto_layer = node.pb
        param = proto_layer.{param_name}

        # extracting parameters from Caffe layer and prepare them for IR
        {extracting_attrs}

        # send parameters blobs from layer to IR
        if node.model_pb and len(node.model_pb.blobs)!=0 :
            if len(node.model_pb.blobs) == 1:
                mapping_rule.update(weights_biases(False, node.model_pb))
            else:
                if len(node.model_pb.blobs) == 2:
                    mapping_rule.update(weights_biases(True, node.model_pb))
                else:
                    for index in range(0, len(node.model_pb.blobs)):
                        embed_input(mapping_rule, index+1, 'custom_{{i}}'.format(i=index), node.model_pb.blobs[index].data)  

        # update the attributes of the node
        Op.get_op_class_by_name('{op_name}').update_node_stat(node, mapping_rule)
        return __class__.enabled

'''.format(fe_module="CaffePython" if is_pythonic else "",
           class_name="{}Python".format(name) if is_pythonic else name,
           op_name=op_name,
           param_name="python_param" if is_pythonic else InteractiveModule.get_param('paramName'),
           extracting_attrs=attrs,
           op_mapping_name='{}.{}'.format(InteractiveModule.get_param('module_name'), name) if is_pythonic else op_name
           )

cog.out(template)

# ]]]
# [[[end]]]
