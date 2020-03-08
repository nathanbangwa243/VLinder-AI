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

#include "vx_user_pipeline_nodes.h"

extern "C"
#if _WIN32
__declspec(dllexport)
#endif
vx_status VX_API_CALL vxPublishKernels(vx_context context)
{
    vx_status status = VX_SUCCESS;
    if((status = PublishBackgroundSuppressKernel(context)) != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "BackgroundSuppress kernel publishing failed\n");
    }
    if((status = PublishNeutralPixelDetectionKernel(context)) != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "NeutralPixelDetection kernel publishing failed\n");
    }
    if((status = PublishRemoveFringeKernel(context)) != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "RemoveFringe kernel publishing failed\n");
    }
    if((status = PublishSymm7x7OpenCLKernel(context)) != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "Symm7x7OpenCL kernel publishing failed\n");
    }
    if((status = PublishRemoveFringeOpenCLKernel(context)) != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "RemoveFringeOpenCL kernel publishing failed\n");
    }
    if((status = PublishIPAHalftoneKernel(context)) != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "IPAHalftone kernel publishing failed\n");
    }
    if((status = PublishGainOffset10Kernel(context)) != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "GainOffset10 kernel publishing failed\n");
    }
    if((status = PublishGainOffset12Kernel(context)) != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "GainOffset12 kernel publishing failed\n");
    }
    if((status = PublishGenEdgeMaskKernel(context)) != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "PublishGenEdgeMaskKernel kernel publishing failed\n");
    }
    if((status = PublishGenEdgeKKernel(context)) != VX_SUCCESS)
    {
        vxAddLogEntry((vx_reference)context, status, "PublishGenEdgeMaskKernel kernel publishing failed\n");
    }

    return VX_SUCCESS;
}
