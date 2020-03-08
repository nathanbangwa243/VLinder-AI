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

#ifndef _VX_USER_PIPELINE_H_
#define _VX_USER_PIPELINE_H_

#include "VX/vx.h"
#include "stdio.h"

#include <intel/vx_samples/helper.hpp>

#define PROCESS_VX_STATUS(NODE, COMMAND)                                \
    {                                                           \
        vx_status __local_status = COMMAND;                     \
        if(__local_status != VX_SUCCESS)                        \
        {                                                       \
            std::string msg = std::string("Code:") + IntelVXSample::vxStatusToStr(__local_status) + std::string(" COMMAND: ") + std::string(#COMMAND);\
            vxAddLogEntry((vx_reference)NODE, __local_status, msg.c_str());\
            return __local_status;                                       \
        }                                                       \
    }

#define SAFE_VX_CALL(STATUS, NODE, COMMAND)                                \
    if(STATUS==VX_SUCCESS){                                                           \
        vx_status __local_status = COMMAND;                     \
        if(__local_status != VX_SUCCESS)                        \
        {                                                       \
            std::string msg = std::string("Code:") + IntelVXSample::vxStatusToStr(__local_status) + std::string(" COMMAND: ") + std::string(#COMMAND);\
            vxAddLogEntry((vx_reference)NODE, __local_status, msg.c_str());\
            (STATUS) = __local_status;                                       \
        }                                                       \
    }

//! User kernel names.
#define VX_KERNEL_NAME_USER_BACKGROUNDSUPPRESS "com.intel.sample.backgroundsuppress"
#define VX_KERNEL_NAME_USER_NEUTRALPIXELDETECTION "com.intel.sample.neutralpixeldetection"
#define VX_KERNEL_NAME_USER_REMOVEFRINGE "com.intel.sample.removefringe"
#define VX_KERNEL_NAME_USER_REMOVEFRINGE_OPENCL "com.intel.sample.clremovefringe"
#define VX_KERNEL_NAME_USER_REMOVEFRINGEPLANAR_OPENCL "com.intel.sample.clremovefringeplanar"
#define VX_KERNEL_NAME_USER_SYMM7X7_OPENCL "com.intel.sample.symm7x7"
#define VX_KERNEL_NAME_USER_SYMM7X7_OPENCL_TILED "com.intel.sample.symm7x7tiled"
#define VX_KERNEL_NAME_USER_IPAHALFTONE "com.intel.sample.ipahalftone"
#define VX_KERNEL_NAME_USER_GAINOFFSET10 "com.intel.sample.gainoffset10"
#define VX_KERNEL_NAME_USER_GAINOFFSET12 "com.intel.sample.gainoffset12"
#define VX_KERNEL_NAME_USER_GENEDGEMASK "com.intel.sample.genedgemask"
#define VX_KERNEL_NAME_USER_GENEDGEK "com.intel.sample.genedgek"

#define VX_LIBRARY_SAMPLE_COLOR_COPY_PIPELINE (0x8)

//! The list of kernels enum.
enum vx_kernel_intel_sample_pipeline_e {
    VX_KERNEL_USER_BACKGROUNDSUPPRESS = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_SAMPLE_COLOR_COPY_PIPELINE) + 0x0,
    VX_KERNEL_USER_NEUTRALPIXELDETECTION = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_SAMPLE_COLOR_COPY_PIPELINE) + 0x1,
    VX_KERNEL_USER_REMOVEFRINGE = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_SAMPLE_COLOR_COPY_PIPELINE) + 0x2,
    VX_KERNEL_USER_REMOVEFRINGE_OPENCL = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_SAMPLE_COLOR_COPY_PIPELINE) + 0x3,
    VX_KERNEL_USER_REMOVEFRINGEPLANAR_OPENCL = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_SAMPLE_COLOR_COPY_PIPELINE) + 0x4,
    VX_KERNEL_USER_SYMM7X7_OPENCL = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_SAMPLE_COLOR_COPY_PIPELINE) + 0x5,
    VX_KERNEL_USER_SYMM7X7_OPENCL_TILED = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_SAMPLE_COLOR_COPY_PIPELINE) + 0x6,
    VX_KERNEL_USER_IPAHALFTONE = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_SAMPLE_COLOR_COPY_PIPELINE) + 0x7,
    VX_KERNEL_USER_GAINOFFSET10 = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_SAMPLE_COLOR_COPY_PIPELINE) + 0x8,
    VX_KERNEL_USER_GAINOFFSET12 = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_SAMPLE_COLOR_COPY_PIPELINE) + 0x9,
    VX_KERNEL_NAME_GENEDGEMASK = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_SAMPLE_COLOR_COPY_PIPELINE) + 0xA,
    VX_KERNEL_USER_GENEDGEK = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_SAMPLE_COLOR_COPY_PIPELINE) + 0xB,
};

#ifdef __cplusplus
extern "C" {
#endif

    //! Functions for user kernels publishing.
#if _WIN32
__declspec(dllexport)
#endif
    vx_status VX_API_CALL vxPublishKernels(vx_context context);
    vx_status VX_API_CALL PublishBackgroundSuppressKernel(vx_context context);
    vx_status VX_API_CALL PublishNeutralPixelDetectionKernel(vx_context context);
    vx_status VX_API_CALL PublishRemoveFringeKernel(vx_context context);
    vx_status VX_API_CALL PublishRemoveFringeOpenCLKernel(vx_context context);
    vx_status VX_API_CALL PublishSymm7x7OpenCLKernel(vx_context context);
    vx_status VX_API_CALL PublishIPAHalftoneKernel(vx_context context);
    vx_status VX_API_CALL PublishGainOffset10Kernel(vx_context context);
    vx_status VX_API_CALL PublishGainOffset12Kernel(vx_context context);
    vx_status VX_API_CALL PublishGenEdgeMaskKernel(vx_context context);
    vx_status VX_API_CALL PublishGenEdgeKKernel(vx_context context);

    //User nodes

    //Background suppression. This kernel will compare the CIELab video
    //against a series of thresholds and determine if each pixel value is
    //"pure white" or "pure black", and if so, adjusts the video accordingly.
    vx_node vxBackgroundSuppressNode(vx_graph graph,
        vx_image inputL,
        vx_image inputA,
        vx_image inputB,
        vx_image outputL,
        vx_image outputA,
        vx_image outputB);

    //Given input a* and b* (from CIELab),
    // produce a 'neutral pixel' mask.
    vx_node vxNeutralPixelDetectionNode(vx_graph graph,
        vx_image inputA,
        vx_image inputB,
        vx_image outputMask);

    //Input:
    //  CMYK: VX_DF_IMAGE_RGBX
    //  L*:   VX_DF_IMAGE_U8
    //  Neutral Edge Mask: VX_DF_IMAGE_U8
    // LtoK_nodes: vx_array 16 vx_uint8 entries.. monotonically decreasing.
    //Output:
    //  CMYK: VX_DF_IMAGE_RGBX
    //  K (Optional): VX_DF_IMAGE_U8
    vx_node vxRemoveFringeNode(vx_graph graph,
        vx_image inputCMYK,
        vx_image inputL,
        vx_image inputNeutralEdgeMask,
        vx_image outputCMYK,
        vx_image outputK,
        vx_array LtoK_nodes);

    //Input:
    //  CMYK: VX_DF_IMAGE_RGBX
    //  L*:   VX_DF_IMAGE_U8
    //  Neutral Edge Mask: VX_DF_IMAGE_U8
    // LtoK_nodes: vx_array 256 vx_uint8 entries.
    //Output:
    //  CMYK: VX_DF_IMAGE_RGBX
    //  K: VX_DF_IMAGE_U8
    vx_node vxRemoveFringeOpenCLTiledNode(vx_graph graph,
        vx_image inputCMYK,
        vx_image inputL,
        vx_image inputNeutralEdgeMask,
        vx_image outputCMYK,
        vx_image outputK,
        vx_array LtoK_nodes);

    //Input:
    //  CMYK: VX_DF_IMAGE_RGBX
    //  L*:   VX_DF_IMAGE_U8
    //  Neutral Edge Mask: VX_DF_IMAGE_U8
    // LtoK_nodes: vx_array 256 vx_uint8 entries.
    //Output:
    //  C: VX_DF_IMAGE_U8
    //  M: VX_DF_IMAGE_U8
    //  Y: VX_DF_IMAGE_U8
    //  K: VX_DF_IMAGE_U8
    //  K_edge: VX_DF_IMAGE_U8
    vx_node vxRemoveFringePlanarOpenCLTiledNode(vx_graph graph,
        vx_image inputCMYK,
        vx_image inputL,
        vx_image inputNeutralEdgeMask,
        vx_image outputC,
        vx_image outputM,
        vx_image outputY,
        vx_image outputK,
        vx_image outputK_edge,
        vx_array LtoK_nodes);

    //!**********************************************************************************************
    //! Function Name      :  vxSymm7x7OpenCLNode
    //! Argument 1         :  The handle to the graph in which to instantiate the node   [IN]
    //! Argument 2         :  Input VX_DF_IMAGE_U8 image 		                         [IN]
    //! Argument 3         :  Output CT VX_DF_IMAGE_U8 image	                         [OUT]
    //! Returns            :  Status
    //! Description        :  Implementation of the census transform node which executes
    //!					   :  in the Graph to invoke the OpenCL symm7x7 kernel.
    //!***********************************************************************************************
    vx_node vxSymm7x7OpenCLNode(vx_graph graph, vx_image input, vx_image output);

    //!**********************************************************************************************
    //! Function Name      :  vxSymm7x7OpenCLTiledNode
    //! Argument 1         :  The handle to the graph in which to instantiate the node   [IN]
    //! Argument 2         :  Input VX_DF_IMAGE_U8 image                                 [IN]
    //! Argument 3         :  Output CT VX_DF_IMAGE_U8 image                             [OUT]
    //! Returns            :  Status
    //! Description        :  Implementation of the census transform node which executes
    //!                    :  in the Graph to invoke the OpenCL symm7x7 tiled kernel.
    //!***********************************************************************************************
    vx_node vxSymm7x7OpenCLTiledNode(vx_graph graph, vx_image input, vx_image output);

    //Input:
    //  input: The contone input image (VX_DF_IMAGE_U8)
    //  screendata: A vx_array of VX_DF_IMAGE_U8 values, representing the
    //              screencell data. The size must be equal to screenwidth * screenheight
    //  screenwidth: The screen width
    //  screenheight: The screen height
    //  screenshift: The x-phase of the screen
    //Output:
    //  output: The bitone (0 & 255 gray level) output image (VX_DF_IMAGE_U8)
    vx_node vxIPAHalftoneNode(vx_graph graph,
                              vx_image input,
                              vx_array screendata,
                              vx_int32 screenwidth,
                              vx_int32 screenheight,
                              vx_int32 screenshift,
                              vx_image output);

    // vxGainOffset10Node: Unpacks input 10-bit image, also applying
    //                     Gain / Offset correction. Correction formula:
    //                     output8[x] = (input10[x] * gain[x] + offset[x])*agc
    //Input:
    //  input: The packed 10-bit input image (VX_DF_IMAGE_U8)
    //  gain: A vx_array of VX_TYPE_FLOAT32 values. The size of
    //        this array must be at least equal to the image width in pixels.
    //  offset: a vx_array of VX_TYPE_FLOAT32 values. The size of
    //        this array must be at least equal to the image width in pixels.
    //  agc: Automatic Gain Control. A vx_scalar holding a VX_TYPE_FLOAT32 value.
    //Output:
    //  output: The unpacked / gain & offset corrected output image (VX_DF_IMAGE_U8)
    vx_node vxGainOffset10Node(vx_graph graph,
                               vx_image input10,
                               vx_image output8,
                               vx_array gain,
                               vx_array offset,
                               vx_scalar agc);

    // vxGainOffset12Node: Unpacks input 12-bit image, also applying
    //                     Gain / Offset correction. Correction formula:
    //                     output8[x] = (input12[x] * gain[x] + offset[x])*agc
    //Input:
    //  input: The packed 12-bit input image (VX_DF_IMAGE_U8)
    //  gain: A vx_array of VX_TYPE_FLOAT32 values. The size of
    //        this array must be at least equal to the image width in pixels.
    //  offset: a vx_array of VX_TYPE_FLOAT32 values. The size of
    //        this array must be at least equal to the image width in pixels.
    //  agc: Automatic Gain Control. A vx_scalar holding a VX_TYPE_FLOAT32 value.
    //Output:
    //  output: The unpacked / gain & offset corrected output image (VX_DF_IMAGE_U8)
    vx_node vxGainOffset12Node(vx_graph graph,
                               vx_image input12,
                               vx_image output8,
                               vx_array gain,
                               vx_array offset,
                               vx_scalar agc);


    // vxGenEdgeMaskNode: Takes input from magnitude and produces the edge mask
    //Input:
    // input:  magnitude image (VX_DF_IMAGE_S16)
    //Output:
    // output: edge mask (1 bpp)
    //           0 = not an edge
    //           1 = edge
    vx_node vxGenEdgeMaskNode(vx_graph graph,
                              vx_image input,
                              vx_image output);

    // vxGenEdgeKNode: Produces final 1bpp K output
    //Input:
    // inputNeutralEdgeMask:  Neutral Edge Mask (1 bpp)
    // inputContoneK: The K output from RemoveFringe (8 bpp)
    // inputRenderedK: The K output from halftone/error diffusion (1bpp)
    // threshLow: low threshold to use for edge enhancement
    // threshHigh: high threshold to use for edge enhancement
    //Output:
    //  outputK: final K output (1 bpp)
    vx_node vxGenEdgeKNode(vx_graph graph,
                           vx_image inputNeutralEdgeMask,
                           vx_image inputContoneK,
                           vx_image inputRenderedK,
                           vx_uint8 threshLow,
                           vx_uint8 threshHigh,
                           vx_image outputK);


#ifdef __cplusplus
}
#endif

#endif /* _VX_USER_PIPELINE_H_ */

