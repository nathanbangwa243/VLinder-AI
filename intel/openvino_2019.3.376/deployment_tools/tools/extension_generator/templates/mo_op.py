# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ===============================================================================
# Generated file for Model Optimizer Operation extension for a layer
#
# You need to modify this file if you need to
#   1. set default values for several attributes of the layer
#      (do it in __init__() method)
#   2. lessen number of attributes to appear in the IR
#      (specify such a list in backend_attrs() method)
#   3. handle the layer which output blob is different to the input one
#      (implement your own static method infer() and set it as attribute in
#      __init__() dictionary)
#
# Refer to the section "Extending Model Optimizer with New Primitives" in
# OpenVINO* documentation (either online or offline in
# <INSTALL_DIR>/deployment_tools/documentation/docs/index.html an then navigate
# to the corresponding section).
# ===============================================================================

# [[[cog
from ext_gen.interactive_module import InteractiveModule

is_pythonic = InteractiveModule.get_param('isPythonic')
op = InteractiveModule.get_param('opName')
op_name = op + 'Python' if is_pythonic else op
isCustom = InteractiveModule.get_param('changeShape')
hasInfer = InteractiveModule.get_param('hasInfer')

params = ''

if hasattr(InteractiveModule.params, 'customParams') and InteractiveModule.params['customParams'][1]:
    for p in InteractiveModule.get_param('customParams'):
        params += "{}={},\n".format(p[1], str(p[2]))

attrs = InteractiveModule.get_param('supportedAttrs')
if attrs and len(attrs)>0:
    sup_attrs = '''
    def supported_attrs(self):
        # =====================================
        # List all attributes of the layer 
        # all other attributes that are not in 
        # the list are ignored
        # =====================================
        return [\n{attrs}]
    
    '''. format(attrs='        \n'.join(['            \'{}\','.format(a[0]) for a in attrs]))
else:
    sup_attrs = ''

int_attrs = InteractiveModule.get_param('internalAttrs')
if attrs and len(attrs)>0:
    back_attrs = '''
    def backend_attrs(self):
        # =====================================
        # List all attributes of the layer 
        # that should appear in the IR 
        # all other attributes that are not in
        # the list are ignored
        # =====================================
        return [\n{attrs}]
    
    '''. format(attrs='        \n'.join(['            \'{}\','.format(a[0]) for a in attrs if a not in int_attrs]))
else:
    back_attrs = ''

register_as_extractor = ''

if is_pythonic:
    register_as_extractor += '''
# =======================================
# Caffe layers with type 'Python' need
# to be registered in special manner to
# be well distinguished from usual layers
# =======================================
register_caffe_python_extractor({op_name}Op)
'''.format(op_name=op_name)

init_infer=""
infer_func=""
if hasInfer:
    init_infer="infer={op_name}Op.infer".format(op_name=op_name)
    infer_func='''
    @staticmethod
    def infer(node: Node):
        # ==========================================================
        # You should add your shape calculation implementation here
        # If a layer input shape is different to the output one
        # it means that it changes shape and you need to implement
        # it on your own. Otherwise, use copy_shape_infer(node).
        # ==========================================================
        {infer}
'''.format(infer="raise NotImplementedError('{op} should calculate shape')" .format(op=op) if isCustom else "return copy_shape_infer(node)")

template = '''
{importing}
from mo.ops.op import Op
from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.graph.graph import Node


class {op_name}Op(Op):
    op = '{op}'
    
    def __init__(self, graph, attrs):
        mandatory_props = dict(
            type=__class__.op,
            op=__class__.op,
            {params}
            {init_infer}            
        )
        super().__init__(graph, mandatory_props, attrs)
    
    {sup_attrs}
    {back_attrs}
    {infer_func}
{register_as_extractor}
'''.format(importing='from mo.front.caffe.extractor import register_caffe_python_extractor' if is_pythonic else '',
           init_infer=init_infer,
           op_name=op_name, op=op, params=params, register_as_extractor=register_as_extractor, sup_attrs=sup_attrs,
           back_attrs=back_attrs, infer_func=infer_func)
cog.out(template)
# ]]]
# [[[end]]]
