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


#include <string>
#include <VX/vx.h>
#include <VX/vx_intel_volatile.h>
#include <intel/vx_samples/helper.hpp>
#include <intel/vx_samples/basic.hpp>
#include "vx_user_pipeline_nodes.h"

typedef enum _symm7x7_params_e {
    SYMM7X7_PARAM_INPUT = 0,
    SYMM7X7_PARAM_OUTPUT
} symm7x7_params_e;

//!*************************************************************************************************************************
//! Function Name        :  Symm7x7Validator
//! Argument 1           :  Handle to the node                      [IN]
//! Argument 2           :  The index of the parameter to validate  [IN]
//! Returns              :  Status
//! Description          :  Input/output parameter validator for the Symm7x7 node
//!                      :  The function, which validates the input and output parameters to this user custom kernel
//!*************************************************************************************************************************
vx_status VX_CALLBACK Symm7x7Validator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    vx_status status = VX_ERROR_INVALID_PARAMETERS;
    if(num!=2)
    {
        return status;
    }

    vx_df_image imageFormat = 0;
    if(vxQueryImage((vx_image)parameters[SYMM7X7_PARAM_INPUT], VX_IMAGE_FORMAT, &imageFormat, sizeof(imageFormat)) == VX_SUCCESS)
    {
        if (imageFormat == VX_DF_IMAGE_U8)
        {
            status = VX_SUCCESS;
        }
        else
        {
            status = VX_ERROR_INVALID_VALUE;
            vxAddLogEntry((vx_reference)node, status, "Symm7x7 Input Validation failed: invalid input image format\n");
            return status;
        }
    }

    vx_uint32 inputWidth = 0;
    vx_uint32 inputHeight = 0;

    //Query the input image
    status = vxQueryImage((vx_image)parameters[SYMM7X7_PARAM_INPUT], VX_IMAGE_WIDTH, &inputWidth, sizeof(inputWidth));
    status |= vxQueryImage((vx_image)parameters[SYMM7X7_PARAM_INPUT], VX_IMAGE_HEIGHT, &inputHeight, sizeof(inputHeight));


    //Input/output is of U8 type 
    //Set width and height for validation as well
    status |= vxSetMetaFormatAttribute(metas[SYMM7X7_PARAM_OUTPUT], VX_IMAGE_WIDTH, &inputWidth, sizeof(inputWidth));
    status |= vxSetMetaFormatAttribute(metas[SYMM7X7_PARAM_OUTPUT], VX_IMAGE_HEIGHT, &inputHeight, sizeof(inputHeight));
    status |= vxSetMetaFormatAttribute(metas[SYMM7X7_PARAM_OUTPUT], VX_IMAGE_FORMAT, &imageFormat, sizeof(imageFormat));

    return status;
}


#if INTEL_SAMPLE_DEEP_DIAGNOSTICS
 
vx_kernel sharedKernel;
#endif


#define MACRO_TO_STR(X) #X


//!**************************************************************************
//! Function Name        :  PublishSymm7x7Kernel
//! Argument 1           :  Context		                        [IN]
//! Returns              :  Status
//! Description          :  This function publishes the user defined kernels
//!**************************************************************************
extern "C" vx_status VX_API_CALL PublishSymm7x7OpenCLKernel(vx_context context)
{
    // First prepare source of an OpenCL program. It is just read from a file.
    // The API that we use below also accepts an OpenCL program in a binary form,
    // here the source form of a program is demonstrated only. Program in a binary form
    // is prepared similarly.
    #if INTEL_SAMPLE_OPENCL_BUG_CVS_1638_EARLY_SOURCE_DELETE
    static
    #endif
    std::string oclSource;
    try
    {
        oclSource = IntelVXSample::readTextFile(exe_dir() + "vxsymm7x7_opencl_impl.cl");
    }
    catch(const SampleError& error)
    {
        vx_status status = VX_FAILURE;
        vxAddLogEntry((vx_reference)context, status, error.what());
        return status;
    }

    // Define how many pixels are processed by one work item in OpenCL C kernel
    #define WORK_ITEM_XSIZE 1
    #define WORK_ITEM_YSIZE 1

    // Form a build string, need to pass some parameters for OpenCL C kernel through macros
    #if INTEL_SAMPLE_OPENCL_BUG_CVS_1638_EARLY_SOURCE_DELETE
    static
    #endif
    std::string compilationFlags =
        #ifndef INTEL_SAMPLE_PRODUCT_BUILD
        // Extra macros to OpenCL program for engenering build
        #if INTEL_SAMPLE_OPENCL_BUG_CVS_1639_S16_SUPPORT
        "-DINTEL_SAMPLE_OPENCL_BUG_CVS_1639_S16_SUPPORT=1 "
        #endif
        #if INTEL_SAMPLE_DEEP_DIAGNOSTICS
        "-DINTEL_SAMPLE_DEEP_DIAGNOSTICS=1 "
        #endif
        #if INTEL_SAMPLE_OPENCL_LOCAL_MEMORY_KERNEL
        "-DTILE_IN_LOCAL_MEMORY=1 "
        #endif
        #endif
        "-DWORK_ITEM_XSIZE=" + to_str(WORK_ITEM_XSIZE) + " "
        "-DWORK_ITEM_YSIZE=" + to_str(WORK_ITEM_YSIZE) + " "
    ;

    std::cout << "[ INFO ] OpenCL build options: " << compilationFlags << "\n";

    // Register OpenCL program as an OpenVX Device Kernel Library. Device kernel library is
    // a container for device kernels. So to be able to call at least one OpenCL device kernel, a device
    // kernel library should be created first with an OpenCL program that includes the kernel(s)
    // implementation.
    //
    // Two types of libraries are supported:
    //   - VX_OPENCL_LIBRARY_SOURCE: created from OpenCL program in source form
    //   - VX_OPENCL_LIBRARY_BINARY: created from OpenCL program in binary form
    //
    vx_device_kernel_library_intel kernelLibrary = vxAddDeviceLibraryIntel(
        context,
        oclSource.length(),               // Size in bytes of the OpenCL source
        oclSource.c_str(),                // Pointer to the program source or binary

        compilationFlags.c_str(),         // Pointer to the compilation flag string
                                          // Convenient way to pass host-side defined constants as macros
                                          // to OpenCL program.
                                          // Ignored if library_type == VX_OPENCL_LIBRARY_BINARY

        VX_OPENCL_LIBRARY_SOURCE_INTEL,   // VX_OPENCL_LIBRARY_SOURCE or VX_OPENCL_LIBRARY_BINARY
        "intel.gpu"                       // Default vx_target name that should run kernels; see OpenVX Target API for reference
    );

    RETURN_VX_OBJ(kernelLibrary);

    // For each OpenVX node parameter, parameter type and direction are defined in
    // the following two arrays. Index in the arrays corresponds to an index of a parameter.
    // So we have a node with two parameters here: both images, the first one is an input parameter
    // and the second one is an output parameter
    vx_enum param_types[] = { VX_TYPE_IMAGE, VX_TYPE_IMAGE };
    vx_enum param_directions[] = { VX_INPUT, VX_OUTPUT };

    // Call vxIntelAddDeviceKernel serves the similar purpose as a regular vxAddKernel function call
    // but accepts OpenCL kernel specific arguments
    vx_kernel oclKernel = vxAddDeviceKernelIntel(
        context,
        VX_KERNEL_NAME_USER_SYMM7X7_OPENCL,  // The name of the kernel in OpenVX nomenclature
        VX_KERNEL_USER_SYMM7X7_OPENCL,       // enum value for OpenVX kernel, formed in the same
                                                     // way as for regular user nodes
        kernelLibrary,                  // kernel library, just created earlier here in the code
        "symm7x7_opt",              // OpenCL kernel name as it appears in OpenCL program
        2,                              // number of OpenVX Kernel parameters
        param_types,                    // Types of parameters: array
        param_directions,               // Directions for each parameter: array
        VX_BORDER_UNDEFINED,       // TODO: it is required, provide a valid value here

        // the following 3 parameters are similar to regular user nodes
        Symm7x7Validator,   // input and output parameters validator
        0,    // node initialization
        0     // node deinitialization
    );

    #if INTEL_SAMPLE_DEEP_DIAGNOSTICS
    std::cerr << "Kernle name: " << VX_KERNEL_NAME_USER_SYMM7X7_OPENCL << '\n';
    sharedKernel = oclKernel; 
    RETURN_VX_OBJ(oclKernel);
    std::cerr << "oclKernel = " << oclKernel << "\n";

    #endif
    RETURN_VX_OBJ(oclKernel);

    // Create a separate OpenVX kernel for tiled OpenCL implementation
    // The kernel OpenCL code implementation may not differ from non-tiled version
    // or may has specific tile-based optimizations (like using async_work_group_tile_copy)
    vx_kernel oclTiledKernel = vxAddDeviceTilingKernelIntel(
        context,
        VX_KERNEL_NAME_USER_SYMM7X7_OPENCL_TILED,  // The name of the kernel in OpenVX nomenclature
        VX_KERNEL_USER_SYMM7X7_OPENCL_TILED,       // enum value for OpenVX kernel, formed in the same
                                                           // way as for regular user nodes
        kernelLibrary,                  // kernel library, just created earlier here in the code
        "symm7x7tiled_opt",              // OpenCL kernel name as it appears in OpenCL program
        2,                              // number of OpenVX Kernel parameters
        param_types,                    // Types of parameters: array
        param_directions,               // Directions for each parameter: array
        VX_BORDER_UNDEFINED,       // TODO: it is required, provide a valid value here

        // the following 3 parameters are similar to regular user nodes
        Symm7x7Validator,   // input and output parameters validator
        0,    // node initialization
        0     // node deinitialization
    );

    RETURN_VX_OBJ(oclTiledKernel);

    // TODO: Set tiling specific attributes here, if their default values are not appropriate

    // Each work-item in the OpenCL kernel process a single pixel by default.
    // All input images should be the same size and this size is used as a global size for
    // the kernel. X image dimension is mapped to 0-th dimension of the NDRange and Y image
    // dimension is mapped to 1-th dimension of the NDRange.

    // Number of pixels processed by a single work item can be modified for each dimension
    // by calling vxSetKernelAttribute with VX_OPENCL_WORK_ITEM_XSIZE and VX_OPENCL_WORK_ITEM_YSIZE. 
    // In case of this example it is set to default values 1x1, because the OpenCL kernel is written
    // to process a single pixel. If values greater than 1 are provided, the global size of NDRange
    // is reduced accordingly: width/xAreaSize x height/yAreaSize, and the kernel should be modified
    // to process multiple pixels in one work item.
    vx_size  xAreaSize = WORK_ITEM_XSIZE;
    vx_size  yAreaSize = WORK_ITEM_YSIZE;
    RETURN_VX_STATUS(vxSetKernelAttribute(oclKernel, VX_OPENCL_WORK_ITEM_XSIZE_INTEL, &xAreaSize, sizeof(xAreaSize)));
    RETURN_VX_STATUS(vxSetKernelAttribute(oclKernel, VX_OPENCL_WORK_ITEM_YSIZE_INTEL, &yAreaSize, sizeof(yAreaSize)));
    RETURN_VX_STATUS(vxFinalizeKernel(oclKernel));

    RETURN_VX_STATUS(vxSetKernelAttribute(oclTiledKernel, VX_OPENCL_WORK_ITEM_XSIZE_INTEL, &xAreaSize, sizeof(xAreaSize)));
    RETURN_VX_STATUS(vxSetKernelAttribute(oclTiledKernel, VX_OPENCL_WORK_ITEM_YSIZE_INTEL, &yAreaSize, sizeof(yAreaSize)));
    // TODO: Set tiling specific attributes here, if their default values are not appropriate
    vx_neighborhood_size_intel_t n;
    n.top = -3;  //requires 3 additional input pixel above
    n.left = -3; //requires 3 additional input pixels to the left
    n.right = 3; //requires 3 additional input pixels to the right
    n.bottom = 3; //requires 3 additional input pixels below

    RETURN_VX_STATUS(vxSetKernelAttribute(oclTiledKernel,
        VX_KERNEL_INPUT_NEIGHBORHOOD_INTEL,
        &n,
        sizeof(vx_neighborhood_size_intel_t)
    ));
    RETURN_VX_STATUS(vxFinalizeKernel(oclTiledKernel));

    return VX_SUCCESS;
}

