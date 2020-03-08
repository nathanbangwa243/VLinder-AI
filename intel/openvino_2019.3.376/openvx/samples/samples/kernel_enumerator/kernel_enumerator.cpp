/*
        Copyright 2019 Intel Corporation.
        This software and the related documents are Intel copyrighted materials,
        and your use of them is governed by the express license under which they
        were provided to you (End User License Agreement for the Intel(R) Software
        Development Products (Version May 2017)). Unless the License provides
        otherwise, you may not use, modify, copy, publish, distribute, disclose or
        transmit this software or the related documents without Intel's prior
        written permission.

        This software and the related documents are provided as is, with no
        express or implied warranties, other than those that are expressly
        stated in the License.
*/

#include <VX/vx.h>
#include <VX/vx_intel_volatile.h>


#include <iostream>
#include <iomanip>
#include <sstream>

#include <intel/vx_samples/helper.hpp>


std::string paramDirectionToStr (vx_enum direction)
{
    switch(direction)
    {
        #define VX_ENUM_TO_STR_ENTRY(E) case VX_##E: return #E;
        VX_ENUM_TO_STR_ENTRY(INPUT)
        VX_ENUM_TO_STR_ENTRY(OUTPUT)
        VX_ENUM_TO_STR_ENTRY(BIDIRECTIONAL)
        #undef VX_ENUM_TO_STR_ENTRY
        default: return "UNKNOWN";
    }
}

std::string paramTypeToStr (vx_enum direction)
{
    switch(direction)
    {
        #define VX_ENUM_TO_STR_ENTRY(E) case VX_TYPE_##E: return #E;
        VX_ENUM_TO_STR_ENTRY(INVALID)
        VX_ENUM_TO_STR_ENTRY(CHAR)
        VX_ENUM_TO_STR_ENTRY(INT8)
        VX_ENUM_TO_STR_ENTRY(UINT8)
        VX_ENUM_TO_STR_ENTRY(INT16)
        VX_ENUM_TO_STR_ENTRY(UINT16)
        VX_ENUM_TO_STR_ENTRY(INT32)
        VX_ENUM_TO_STR_ENTRY(UINT32)
        VX_ENUM_TO_STR_ENTRY(INT64)
        VX_ENUM_TO_STR_ENTRY(UINT64)
        VX_ENUM_TO_STR_ENTRY(FLOAT32)
        VX_ENUM_TO_STR_ENTRY(FLOAT64)
        VX_ENUM_TO_STR_ENTRY(ENUM)
        VX_ENUM_TO_STR_ENTRY(SIZE)
        VX_ENUM_TO_STR_ENTRY(DF_IMAGE)
        #if defined(OPENVX_PLATFORM_SUPPORTS_16_FLOAT)
        VX_ENUM_TO_STR_ENTRY(FLOAT16)
        #endif
        VX_ENUM_TO_STR_ENTRY(BOOL)

        VX_ENUM_TO_STR_ENTRY(RECTANGLE)
        VX_ENUM_TO_STR_ENTRY(KEYPOINT)
        VX_ENUM_TO_STR_ENTRY(COORDINATES2D)
        VX_ENUM_TO_STR_ENTRY(COORDINATES3D)
        VX_ENUM_TO_STR_ENTRY(COORDINATES4D_INTEL)
        VX_ENUM_TO_STR_ENTRY(COORDINATES_POLAR_INTEL)
        VX_ENUM_TO_STR_ENTRY(HAAR_WEAK_CLASSIFIER_INTEL)
        VX_ENUM_TO_STR_ENTRY(LBP_WEAK_CLASSIFIER_INTEL)
        VX_ENUM_TO_STR_ENTRY(REFERENCE)
        VX_ENUM_TO_STR_ENTRY(CONTEXT)
        VX_ENUM_TO_STR_ENTRY(GRAPH)
        VX_ENUM_TO_STR_ENTRY(NODE)
        VX_ENUM_TO_STR_ENTRY(KERNEL)
        VX_ENUM_TO_STR_ENTRY(PARAMETER)
        VX_ENUM_TO_STR_ENTRY(DELAY)
        VX_ENUM_TO_STR_ENTRY(LUT)
        VX_ENUM_TO_STR_ENTRY(DISTRIBUTION)
        VX_ENUM_TO_STR_ENTRY(PYRAMID)
        VX_ENUM_TO_STR_ENTRY(THRESHOLD)
        VX_ENUM_TO_STR_ENTRY(MATRIX)
        VX_ENUM_TO_STR_ENTRY(CONVOLUTION)
        VX_ENUM_TO_STR_ENTRY(SCALAR)
        VX_ENUM_TO_STR_ENTRY(ARRAY)
        VX_ENUM_TO_STR_ENTRY(IMAGE)
        VX_ENUM_TO_STR_ENTRY(REMAP)
        VX_ENUM_TO_STR_ENTRY(ERROR)
        VX_ENUM_TO_STR_ENTRY(META_FORMAT)
        VX_ENUM_TO_STR_ENTRY(BG_STATE_INTEL)
        VX_ENUM_TO_STR_ENTRY(SEPFILTER2D_INTEL)

        /*VX_ENUM_TO_STR_ENTRY(REF_ARRAY_INTEL)*/
        VX_ENUM_TO_STR_ENTRY(PARAM_STRUCT_INTEL)

        VX_ENUM_TO_STR_ENTRY(SVM_PARAMS_INTEL)
        #undef VX_ENUM_TO_STR_ENTRY
        default: return "UNKNOWN";
    }
}


std::string paramStateToStr (vx_enum state)
{
    switch(state)
    {
        #define VX_ENUM_TO_STR_ENTRY(E) case VX_PARAMETER_STATE_##E: return #E;
        VX_ENUM_TO_STR_ENTRY(REQUIRED)
        VX_ENUM_TO_STR_ENTRY(OPTIONAL)
        #undef VX_ENUM_TO_STR_ENTRY
        default: return "UNKNOWN";
    }
}


struct vx_kernel_info_less
{
    bool operator() (vx_kernel_info_t kernel1, vx_kernel_info_t kernel2) const
    {
        return std::string(kernel1.name) < std::string(kernel2.name);
    }
};

struct TargetDescription
{
    TargetDescription (const std::string& _name, vx_enum _ref) :
        name(_name),
        ref(_ref)
    {
    }
    
    /// Target name
    std::string name;
    
    /// Target reference
    vx_enum ref;
};


typedef std::vector<TargetDescription> Targets;


void enumerateAllAvailableTargets (vx_context context, Targets& targets)
{
    targets.push_back(TargetDescription("intel.cpu", VX_TARGET_CPU_INTEL));
    targets.push_back(TargetDescription("intel.gpu", VX_TARGET_GPU_INTEL));
    targets.push_back(TargetDescription("intel.ipu", VX_TARGET_IPU_INTEL));
    targets.push_back(TargetDescription("intel.mkldnn", VX_TARGET_STRING));

    std::cout << "Number of supported targets: " << targets.size() << "\n\n";

    for(size_t i = 0; i < targets.size(); ++i)
    {
        std::cout << "Target[" << i << "] name: " << targets[i].name << '\n';
    }
}


bool probeTargetForNode (const TargetDescription& target, vx_node node)
{
    vx_status status = vxSetNodeTarget(node, target.ref, target.name.c_str());

    if(status == VX_SUCCESS)
    {
        return true;
    }
    if (status == VX_ERROR_NOT_SUPPORTED)
    {
        return false;
    }
    else
    {
        CHECK_VX_STATUS(status);
    }

    return false;
}

vx_uint32 maxNumberOfKernelParameters (vx_context context, const std::vector<vx_kernel_info_t>& kernelsTable)
{
    vx_uint32 res = 0;

    for(size_t i = 0; i < kernelsTable.size(); ++i)
    {
        vx_kernel kernel = vxGetKernelByEnum(context, kernelsTable[i].enumeration);
        CHECK_VX_STATUS(vxGetStatus((vx_reference)kernel));
        vx_uint32 paramsNum = 0;
        CHECK_VX_STATUS(vxQueryKernel(kernel, VX_KERNEL_PARAMETERS, &paramsNum, sizeof(paramsNum)));
        if(paramsNum > res)
        {
            res = paramsNum;
        }
        CHECK_VX_STATUS(vxReleaseKernel(&kernel));
    }
    
    return res;
}


std::string getOpenVXExtensions (vx_context context)
{
    vx_size size = 0;
    CHECK_VX_STATUS(vxQueryContext(context, VX_CONTEXT_EXTENSIONS_SIZE, &size, sizeof(size)));
    std::vector<vx_char> extensions(size);
    CHECK_VX_STATUS(vxQueryContext(context, VX_CONTEXT_EXTENSIONS, &extensions[0], size));
    return &extensions[0];
}


void enumerateOpenVXExtensions (vx_context context)
{
    std::istringstream extstr(getOpenVXExtensions(context));
    std::vector<std::string> extensions;
    while(extstr)
    {
        std::string ext;
        extstr >> ext;
        if(extstr)
        {
            extensions.push_back(ext);
        }
    }
    
    std::cout << "Number of extensions: " << extensions.size() << "\n\n";
    for(size_t i = 0; i < extensions.size(); ++i)
    {
        std::cout << extensions[i] << '\n';
    }
    
    std::cout << std::endl;
}


void customConvolutionAttributes (vx_context context)
{
    vx_size maxConvDim = 0;
    CHECK_VX_STATUS(vxQueryContext(context, VX_CONTEXT_CONVOLUTION_MAX_DIMENSION, &maxConvDim, sizeof(maxConvDim)))
    std::cout << "Maximum width or height of a convolution matrix (VX_CONTEXT_CONVOLUTION_MAX_DIMENSION): " << maxConvDim << '\n';
    std::cout << std::endl;
}


int main(int argc, const char** argv)
{
    try
    {
        vx_context context = vxCreateContext();
        
        enumerateOpenVXExtensions(context);
        customConvolutionAttributes(context);
        Targets targets;
        enumerateAllAvailableTargets(context, targets);
        vx_graph graph = vxCreateGraph(context);

        vx_int32 kernelsNum = 0;
        CHECK_VX_STATUS(vxQueryContext(context, VX_CONTEXT_UNIQUE_KERNELS, &kernelsNum, sizeof(kernelsNum)));
        std::cout << "\nNumber of kernels detected: " << kernelsNum << "\n\n";

        std::vector<vx_kernel_info_t> kernelsTable(kernelsNum);
        CHECK_VX_STATUS(vxQueryContext(context, VX_CONTEXT_UNIQUE_KERNEL_TABLE, &kernelsTable[0], kernelsNum*sizeof(vx_kernel_info_t)));
        std::sort(kernelsTable.begin(), kernelsTable.end(), vx_kernel_info_less());
        
        std::cout << "Kernel Enum\tKernel Name";
        for(size_t i = 0; i < targets.size(); ++i)
        {
            std::cout << '\t' << targets[i].name;
        }
        vx_uint32 maxNumOfParams = maxNumberOfKernelParameters(context, kernelsTable);
        
        std::cout << "\t# of parameters";
        
        for(vx_uint32 i = 0; i < maxNumOfParams; ++i)
        {
            std::cout << "\tparameter[" << i << "]";
        }
        
        std::cout << std::endl;

        for(size_t i = 0; i < kernelsNum; ++i)
        {
            vx_kernel kernel = vxGetKernelByEnum(context, kernelsTable[i].enumeration);
            CHECK_VX_STATUS(vxGetStatus((vx_reference)kernel));
            vx_uint32 paramsNum = 0;
            CHECK_VX_STATUS(vxQueryKernel(kernel, VX_KERNEL_PARAMETERS, &paramsNum, sizeof(paramsNum)));
            
            std::cout
                << kernelsTable[i].enumeration << '\t'
                << kernelsTable[i].name << '\t'
            ;

            vx_node node = vxCreateGenericNode(graph, kernel);
            CHECK_VX_STATUS(vxGetStatus((vx_reference)node));
            
            for(size_t j = 0; j < targets.size(); ++j)
            {
                if(probeTargetForNode(targets[j], node))
                {
                    std::cout << targets[j].name;
                }
                else
                {
                    std::cout << "N/A";
                }
                
                std::cout << '\t';
            }
            CHECK_VX_STATUS(vxReleaseNode(&node));

            std::cout
                << paramsNum << '\t'
            ;

            for(size_t j = 0; j < paramsNum; ++j)
            {
                vx_parameter parameter = vxGetKernelParameterByIndex(kernel, j);
                CHECK_VX_STATUS(vxGetStatus((vx_reference)parameter));

                vx_enum direction;
                vx_type_e type;
                vx_enum state;

                CHECK_VX_STATUS(vxQueryParameter(parameter, VX_PARAMETER_DIRECTION, &direction, sizeof(direction)));
                CHECK_VX_STATUS(vxQueryParameter(parameter, VX_PARAMETER_TYPE, &type, sizeof(type)));
                CHECK_VX_STATUS(vxQueryParameter(parameter, VX_PARAMETER_STATE, &state, sizeof(state)));
                
                std::cout
                    << paramDirectionToStr(direction) << "|"
                    << paramTypeToStr(type) << "|"
                    << paramStateToStr(state)
                    << '\t';

                CHECK_VX_STATUS(vxReleaseParameter(&parameter));
            }
            
            std::cout << '\n';
            
            CHECK_VX_STATUS(vxReleaseKernel(&kernel));
        }

        return EXIT_SUCCESS;
    }
    catch(...)
    {
        std::cerr << "[ ERROR ] Unexpected exception happened" << std::endl;
        return EXIT_FAILURE;
    }
}

