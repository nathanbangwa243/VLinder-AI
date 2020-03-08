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


//!*************************************************************************************************************************
//! Function Name        :  vxRemoveFringeValidator
//! Argument 1           :  Handle to the node                      [IN]
//! Argument 2           :  The index of the parameter to validate  [IN]
//! Returns              :  Status
//! Description          :  Input/output parameter validator for the vxRemoveFringeOpenCL node
//!                      :  The function, which validates the input and output parameters to this user custom kernel
//!*************************************************************************************************************************
vx_status VX_CALLBACK vxRemoveFringeOpenCLValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    vx_status status = VX_ERROR_INVALID_PARAMETERS;
    if(num!=6)
    {
        return status;
    }

    vx_df_image format = 0;
    vx_uint32 width = 0;
    vx_uint32 height = 0;
    vx_size num_items = 0;

    if(vxQueryImage((vx_image)parameters[0], VX_IMAGE_FORMAT, &format, sizeof(format)) == VX_SUCCESS)
    {
        if (format == VX_DF_IMAGE_RGBX)
        {
            status = VX_SUCCESS;
        }
        else
        {
            status = VX_ERROR_INVALID_VALUE;
            vxAddLogEntry((vx_reference)node, status, "RemoveFringe Input Validation failed: invalid input image 0 format, it must be RGBX\n");
        }
    }
    status = vxQueryImage((vx_image)parameters[0], VX_IMAGE_WIDTH, &width, sizeof(width));
    if (width%16!=0)
    {
        status = VX_ERROR_INVALID_VALUE;
        vxAddLogEntry((vx_reference)node, status, "RemoveFringe Input Validation failed: input image 0 width must be evenly divisible by 16\n");
    }

    if(vxQueryImage((vx_image)parameters[1], VX_IMAGE_FORMAT, &format, sizeof(format)) == VX_SUCCESS)
    {
        if (format == VX_DF_IMAGE_U8)
        {
            status = VX_SUCCESS;
        }
        else
        {
            status = VX_ERROR_INVALID_VALUE;
            vxAddLogEntry((vx_reference)node, status, "RemoveFringe Input Validation failed: invalid input image 1 format, it must be U8\n");
        }
    }
    status = vxQueryImage((vx_image)parameters[1], VX_IMAGE_WIDTH, &width, sizeof(width));
    if (width%16!=0)
    {
        status = VX_ERROR_INVALID_VALUE;
        vxAddLogEntry((vx_reference)node, status, "RemoveFringe Input Validation failed: input image 1 width must be evenly divisible by 16\n");
    }

    if(vxQueryImage((vx_image)parameters[2], VX_IMAGE_FORMAT, &format, sizeof(format)) == VX_SUCCESS)
    {
        if (format == VX_DF_IMAGE_U8)
        {
            status = VX_SUCCESS;
        }
        else
        {
            status = VX_ERROR_INVALID_VALUE;
            vxAddLogEntry((vx_reference)node, status, "RemoveFringe Input Validation failed: invalid input image 2 format, it must be U8\n");
        }
    }
    status = vxQueryImage((vx_image)parameters[2], VX_IMAGE_WIDTH, &width, sizeof(width));
    if (width%16!=0)
    {
        status = VX_ERROR_INVALID_VALUE;
        vxAddLogEntry((vx_reference)node, status, "RemoveFringe Input Validation failed: input image 2 width must be evenly divisible by 16\n");
    }


    if(vxQueryArray((vx_array)parameters[5], VX_ARRAY_CAPACITY, &num_items, sizeof(num_items)) == VX_SUCCESS)
    {
        if (num_items == 256)
        {
            status = VX_SUCCESS;
        }
        else
        {
            status = VX_ERROR_INVALID_VALUE;
            vxAddLogEntry((vx_reference)node, status, "RemoveFringe Input Validation failed: Input LtoK_nodes array must have 16 entries\n");
        }
    }


    status = vxQueryImage((vx_image)parameters[0], VX_IMAGE_HEIGHT, &height, sizeof(height));
    format = VX_DF_IMAGE_RGBX;
    PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[3], VX_IMAGE_WIDTH, &width, sizeof(width)));
    PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[3], VX_IMAGE_HEIGHT, &height, sizeof(height)));
    PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[3], VX_IMAGE_FORMAT, &format, sizeof(format)));

    format = VX_DF_IMAGE_U8;
    if(metas[4])
    {
        PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[4], VX_IMAGE_WIDTH, &width, sizeof(width)));
        PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[4], VX_IMAGE_HEIGHT, &height, sizeof(height)));
        PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[4], VX_IMAGE_FORMAT, &format, sizeof(format)));
    }

    return status;
}

//!*************************************************************************************************************************
//! Function Name        :  vxRemoveFringeValidator
//! Argument 1           :  Handle to the node                      [IN]
//! Argument 2           :  The index of the parameter to validate  [IN]
//! Returns              :  Status
//! Description          :  Input/output parameter validator for the vxRemoveFringeOpenCL node
//!                      :  The function, which validates the input and output parameters to this user custom kernel
//!*************************************************************************************************************************
vx_status VX_CALLBACK vxRemoveFringePlanarOpenCLValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    vx_status status = VX_ERROR_INVALID_PARAMETERS;
    if(num!=9)
    {
        return status;
    }

    vx_df_image format = 0;
    vx_uint32 width = 0;
    vx_uint32 height = 0;
    vx_size num_items = 0;

    if(vxQueryImage((vx_image)parameters[0], VX_IMAGE_FORMAT, &format, sizeof(format)) == VX_SUCCESS)
    {
        if (format == VX_DF_IMAGE_RGBX)
        {
            status = VX_SUCCESS;
        }
        else
        {
            status = VX_ERROR_INVALID_VALUE;
            vxAddLogEntry((vx_reference)node, status, "RemoveFringe Input Validation failed: invalid input image 0 format, it must be RGBX\n");
        }
    }
    status = vxQueryImage((vx_image)parameters[0], VX_IMAGE_WIDTH, &width, sizeof(width));
    if (width%16!=0)
    {
        status = VX_ERROR_INVALID_VALUE;
        vxAddLogEntry((vx_reference)node, status, "RemoveFringe Input Validation failed: input image 0 width must be evenly divisible by 16\n");
    }

    if(vxQueryImage((vx_image)parameters[1], VX_IMAGE_FORMAT, &format, sizeof(format)) == VX_SUCCESS)
    {
        if (format == VX_DF_IMAGE_U8)
        {
            status = VX_SUCCESS;
        }
        else
        {
            status = VX_ERROR_INVALID_VALUE;
            vxAddLogEntry((vx_reference)node, status, "RemoveFringe Input Validation failed: invalid input image 1 format, it must be U8\n");
        }
    }
    status = vxQueryImage((vx_image)parameters[1], VX_IMAGE_WIDTH, &width, sizeof(width));
    if (width%16!=0)
    {
        status = VX_ERROR_INVALID_VALUE;
        vxAddLogEntry((vx_reference)node, status, "RemoveFringe Input Validation failed: input image 1 width must be evenly divisible by 16\n");
    }

    if(vxQueryImage((vx_image)parameters[2], VX_IMAGE_FORMAT, &format, sizeof(format)) == VX_SUCCESS)
    {
        if (format == VX_DF_IMAGE_U8)
        {
            status = VX_SUCCESS;
        }
        else
        {
            status = VX_ERROR_INVALID_VALUE;
            vxAddLogEntry((vx_reference)node, status, "RemoveFringe Input Validation failed: invalid input image 2 format, it must be U8\n");
        }
    }
    status = vxQueryImage((vx_image)parameters[2], VX_IMAGE_WIDTH, &width, sizeof(width));
    if (width%16!=0)
    {
        status = VX_ERROR_INVALID_VALUE;
        vxAddLogEntry((vx_reference)node, status, "RemoveFringe Input Validation failed: input image 2 width must be evenly divisible by 16\n");
    }


    if(vxQueryArray((vx_array)parameters[5], VX_ARRAY_CAPACITY, &num_items, sizeof(num_items)) == VX_SUCCESS)
    {
        if (num_items == 256)
        {
            status = VX_SUCCESS;
        }
        else
        {
            status = VX_ERROR_INVALID_VALUE;
            vxAddLogEntry((vx_reference)node, status, "RemoveFringe Input Validation failed: Input LtoK_nodes array must have 16 entries\n");
        }
    }


    status = vxQueryImage((vx_image)parameters[0], VX_IMAGE_HEIGHT, &height, sizeof(height));

    format = VX_DF_IMAGE_U8;
    for( int index = 3; index <= 7; index++)
    {
      PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[index], VX_IMAGE_WIDTH, &width, sizeof(width)));
      PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[index], VX_IMAGE_HEIGHT, &height, sizeof(height)));
      PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[index], VX_IMAGE_FORMAT, &format, sizeof(format)));
    }

    return status;
}


//!**************************************************************************
//! Function Name        :  PublishRemoveFringeOpenCLKernel
//! Argument 1           :  Context		                        [IN]
//! Returns              :  Status
//! Description          :  This function publishes the user defined kernels
//!**************************************************************************
extern "C" vx_status VX_API_CALL PublishRemoveFringeOpenCLKernel(vx_context context)
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
        oclSource = IntelVXSample::readTextFile(exe_dir() + "vxremovefringe_opencl_impl.cl");
    }
    catch(const SampleError& error)
    {
        vx_status status = VX_FAILURE;
        vxAddLogEntry((vx_reference)context, status, error.what());
        return status;
    }

    // Define how many pixels are processed by one work item in OpenCL C kernel
    #define WORK_ITEM_XSIZE 4
    #define WORK_ITEM_YSIZE 8

    // Form a build string, need to pass some parameters for OpenCL C kernel through macros
    #if INTEL_SAMPLE_OPENCL_BUG_CVS_1638_EARLY_SOURCE_DELETE
    static
    #endif
    std::string compilationFlags =
        "-DWORK_ITEM_XSIZE=" + to_str(WORK_ITEM_XSIZE) + " "
        "-DWORK_ITEM_YSIZE=" + to_str(WORK_ITEM_YSIZE) + " ";

    std::cout << "[ INFO ] Remove Fringe build options: " << compilationFlags << "\n";

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

    //Publish pixel-interleaved CMYK output RemoveFringe
    {
       // For each OpenVX node parameter, parameter type and direction are defined in
       // the following two arrays. Index in the arrays corresponds to an index of a parameter.
       // So we have a node with two parameters here: both images, the first one is an input parameter
       // and the second one is an output parameter
       vx_enum param_types[] = { VX_TYPE_IMAGE,  //inputCMYK
                                VX_TYPE_IMAGE,  //inputL
                                VX_TYPE_IMAGE,  //inputNeutralEdgeMask
                                VX_TYPE_IMAGE,  //outputCMYK
                                VX_TYPE_IMAGE,  //outputK
                                VX_TYPE_ARRAY   //LtoK_nodes
                               };

       vx_enum param_directions[] = { VX_INPUT, //inputCMYK
                                     VX_INPUT, //inputL
                                     VX_INPUT, //inputNeutralEdgeMask
                                     VX_OUTPUT, //outputCMYK
                                     VX_OUTPUT, //outputK
                                     VX_INPUT   //L-to-K LUT (array)
                                    };

       // Call vxIntelAddDeviceKernel serves the similar purpose as a regular vxAddKernel function call
       // but accepts OpenCL kernel specific arguments
       vx_kernel oclTiledKernel = vxAddDeviceTilingKernelIntel(
          context,
          VX_KERNEL_NAME_USER_REMOVEFRINGE_OPENCL,  // The name of the kernel in OpenVX nomenclature
          VX_KERNEL_USER_REMOVEFRINGE_OPENCL,       // enum value for OpenVX kernel, formed in the same
                                                    // way as for regular user nodes
          kernelLibrary,                  // kernel library, just created earlier here in the code
          "_RemoveFringeKernel",          // OpenCL kernel name as it appears in OpenCL program
          6,                              // number of OpenVX Kernel parameters
          param_types,                    // Types of parameters: array
          param_directions,               // Directions for each parameter: array
          VX_BORDER_UNDEFINED,

          // the following 3 parameters are similar to regular user nodes
          vxRemoveFringeOpenCLValidator,   // input and output parameters validator
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

       RETURN_VX_STATUS(vxSetKernelAttribute(oclTiledKernel, VX_OPENCL_WORK_ITEM_XSIZE_INTEL, &xAreaSize, sizeof(xAreaSize)));
       RETURN_VX_STATUS(vxSetKernelAttribute(oclTiledKernel, VX_OPENCL_WORK_ITEM_YSIZE_INTEL, &yAreaSize, sizeof(yAreaSize)));

       vx_neighborhood_size_intel_t n;
       n.top = 0;  //requires no additional input pixel above
       n.left = 0; //requires no additional input pixels to the left
       n.right = 0; //requires no additional input pixels to the right
       n.bottom = 0; //requires no additional input pixels below

       RETURN_VX_STATUS(vxSetKernelAttribute(oclTiledKernel,
           VX_KERNEL_INPUT_NEIGHBORHOOD_INTEL,
           &n,
           sizeof(vx_neighborhood_size_intel_t)
       ));
       RETURN_VX_STATUS(vxFinalizeKernel(oclTiledKernel));
    }

    //Publish planar-interleaved CMYK output RemoveFringe
    {
       // For each OpenVX node parameter, parameter type and direction are defined in
       // the following two arrays. Index in the arrays corresponds to an index of a parameter.
       // So we have a node with two parameters here: both images, the first one is an input parameter
       // and the second one is an output parameter
       vx_enum param_types[] = { VX_TYPE_IMAGE,  //inputCMYK
                                VX_TYPE_IMAGE,  //inputL
                                VX_TYPE_IMAGE,  //inputNeutralEdgeMask
                                VX_TYPE_IMAGE,  //outputC
                                VX_TYPE_IMAGE,  //outputM
                                VX_TYPE_IMAGE,  //outputY
                                VX_TYPE_IMAGE,  //outputK
                                VX_TYPE_IMAGE,  //outputK_edge
                                VX_TYPE_ARRAY   //LtoK_nodes
                               };

       vx_enum param_directions[] = { VX_INPUT, //inputCMYK
                                     VX_INPUT, //inputL
                                     VX_INPUT, //inputNeutralEdgeMask
                                     VX_OUTPUT, //outputC
                                     VX_OUTPUT, //outputM
                                     VX_OUTPUT, //outputY
                                     VX_OUTPUT, //outputK
                                     VX_OUTPUT, //outputK_edge
                                     VX_INPUT   //L-to-K LUT (array)
                                    };

       // Call vxIntelAddDeviceKernel serves the similar purpose as a regular vxAddKernel function call
       // but accepts OpenCL kernel specific arguments
       vx_kernel oclTiledKernel = vxAddDeviceTilingKernelIntel(
          context,
          VX_KERNEL_NAME_USER_REMOVEFRINGEPLANAR_OPENCL,  // The name of the kernel in OpenVX nomenclature
          VX_KERNEL_USER_REMOVEFRINGEPLANAR_OPENCL,       // enum value for OpenVX kernel, formed in the same
                                                    // way as for regular user nodes
          kernelLibrary,                  // kernel library, just created earlier here in the code
          "_RemoveFringePlanarKernel",          // OpenCL kernel name as it appears in OpenCL program
          9,                              // number of OpenVX Kernel parameters
          param_types,                    // Types of parameters: array
          param_directions,               // Directions for each parameter: array
          VX_BORDER_UNDEFINED,

          // the following 3 parameters are similar to regular user nodes
          vxRemoveFringePlanarOpenCLValidator,   // input and output parameters validator
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

       RETURN_VX_STATUS(vxSetKernelAttribute(oclTiledKernel, VX_OPENCL_WORK_ITEM_XSIZE_INTEL, &xAreaSize, sizeof(xAreaSize)));
       RETURN_VX_STATUS(vxSetKernelAttribute(oclTiledKernel, VX_OPENCL_WORK_ITEM_YSIZE_INTEL, &yAreaSize, sizeof(yAreaSize)));

       vx_neighborhood_size_intel_t n;
       n.top = 0;  //requires no additional input pixel above
       n.left = 0; //requires no additional input pixels to the left
       n.right = 0; //requires no additional input pixels to the right
       n.bottom = 0; //requires no additional input pixels below

       RETURN_VX_STATUS(vxSetKernelAttribute(oclTiledKernel,
           VX_KERNEL_INPUT_NEIGHBORHOOD_INTEL,
           &n,
           sizeof(vx_neighborhood_size_intel_t)
       ));
       RETURN_VX_STATUS(vxFinalizeKernel(oclTiledKernel));
    }

    return VX_SUCCESS;
}

