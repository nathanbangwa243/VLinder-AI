/*
 Copyright (c) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

// ===============================================================================
// Generated file for Inference Engine extension for CPU plugin
//
// IMPLEMENT YOUR KERNEL HERE.
//
// You need to edit this file in order to:
//  1. initialize parameters (in constructor)
//  2. implement inference logic (in execute() method)
//
// Refer to the section "Adding Your Own Kernels to the Inference Engine" in
// OpenVINO* documentation (either online or offline in
// <INSTALL_DIR>/deployment_tools/documentation/docs/index.html an then navigate
// to the corresponding section).
// ===============================================================================

#include "ext_list.hpp"
#include "ext_base.hpp"
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

[[[cog
from ext_gen.interactive_module import InteractiveModule

name = InteractiveModule.get_param('opName')
type = name
name = name.replace(".","_")
cog.outl("class %sImpl: public ExtLayerBase {" % (name))
cog.outl("public:")
cog.outl("    explicit %sImpl(const CNNLayer* layer) {" % (name))
]]]
[[[end]]]
        try {
            // LayerSetUp
            // Read parameters from IR and/or initialise them here.
            // Implemented functions for reading parameters are:
            // for single value:
            //     getParamAsFloat, getParamAsInt, getParamsAsBool, getParamAsString
            // for array
            //     getParamAsFloats, getParamAsInts
            // Functions are declared in Inference Engine folder include/ie_layers.h
            [[[cog
               params = []
               cog.outl("//Example of parameters reading is:")
               cog.outl("//   scale_=layer->GetParamAsFloat(\"scale\")\n")
               params = InteractiveModule.get_param('params_cpu')
               cpu_types = InteractiveModule.get_param('supported_cpu_types')
               for p in params:
                   pn = p[0]
                   if p[1] in cpu_types.keys() and p[1]!='bool':
                       pt = cpu_types[p[1]]
                       cog.outl("%s_ = layer->GetParamAs%s(\"%s\");" % (pn, pt, pn))
                   elif p[1] == 'bool':
                       pt = cpu_types[p[1]]
                       cog.outl("%s_ = layer->GetParamsAs%s(\"%s\", <insert default value here>);" % (pn, pt, pn))
                   else:
                       cog.outl("//Unknown parameter type \"%s\", fix function name to correct one!\n" % p[1])
            ]]]
            [[[end]]]
            
            // set configuration: specify data format for layer
            // more information about data formats you can find in "Inference Engine Memory primitives" in OpenVINO* documentation
            // (either online or offline in <INSTALL_DIR>/deployment_tools/documentation/docs/index.html an then navigate
            // to the corresponding section). 
            // addConfig({DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN)}, {DataConfigurator(ConfLayout::PLN)});
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        // Add here implementation for layer inference
        // Examples of implementations you can find in Inerence Engine tool samples/extenstions folder
        return NOT_IMPLEMENTED;
    }

private:
    [[[cog
       for p in params :
           pn = p[0]
           if p[1] == 'int' or p[1]=='float' or p[1]=='bool':
               cog.outl("%s %s_;" % (p[1], pn)) 
           elif p[1] == 'string': 
               cog.outl("std::%s %s_;" % (p[1], pn)) 
           elif p[1] == 'listfloat':
               cog.outl("std::vector<float> %s_;" % (pn))
           elif p[1] == 'listint':
               cog.outl("std::vector<int> %s_;" % (pn))
           else:
               cog.outl("//Unknown parameter type, describe parameter setup here!")
    ]]]
    [[[end]]]
};

[[[cog
cog.outl("REG_FACTORY_FOR(ImplFactory<%sImpl>, %s);" % (name, type))
]]]
[[[end]]]

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
