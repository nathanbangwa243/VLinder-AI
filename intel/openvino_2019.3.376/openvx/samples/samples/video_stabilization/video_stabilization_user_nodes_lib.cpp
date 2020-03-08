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
#include "video_stabilization_user_nodes_lib.h"
#include <iostream>
#include <vector>
#include <map>
#include <cassert>
#include <algorithm>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>

#include <intel/vx_samples/helper.hpp>
#include <intel/vx_samples/perfprof.hpp>


// Image stabilization node. Estimate transform matrix by an array of old and new feature points
vx_node vxEstimateTransformNode(
    vx_graph graph,
    vx_array old_points,
    vx_array new_points,
    vx_array rect,
    vx_matrix estimate
)
{
    vx_node node = 0;
    vx_context context = vxGetContext((vx_reference)graph);

    vx_status status = VX_SUCCESS;

    vx_kernel kernel = vxGetKernelByName(context, VX_KERNEL_NAME_INTEL_SAMPLE_ESTIMATE_TRANSFORM);
    if (kernel)
    {
        node = vxCreateGenericNode(graph, kernel);
        if (node)
        {
            // warp rect with vx_scalar

            IntelVXSample::logger(2) << "vxEstimateTransformNode: vxCreateGenericNode is OK\n";
            vx_status statuses[4];
            statuses[0] = vxSetParameterByIndex(node, 0, (vx_reference)old_points);
            statuses[1] = vxSetParameterByIndex(node, 1, (vx_reference)new_points);
            statuses[2] = vxSetParameterByIndex(node, 2, (vx_reference)rect);
            statuses[3] = vxSetParameterByIndex(node, 3, (vx_reference)estimate);

            for (vx_uint32 i = 0; i < sizeof(statuses)/sizeof(statuses[0]); i++)
            {
                if (statuses[i] != VX_SUCCESS)
                {
                    status = VX_ERROR_INVALID_PARAMETERS;
                    vxReleaseNode(&node);
                    vxReleaseKernel(&kernel);
                    node = 0;
                    kernel = 0;
                    IntelVXSample::logger(1) << "Parameter " << i << " for vxEstimateTransformNode wasn't set successfully\n";
                    break;
                }
            }
        }
        else
        {
            vxReleaseKernel(&kernel);
        }
    }

    return node;
}


vx_status vxuEstimateTransform(
    vx_context context,
    vx_array old_points,
    vx_array new_points,
    vx_array rect,
    vx_matrix estimate
)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
        vx_node node = vxEstimateTransformNode(graph, old_points, new_points, rect, estimate);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}


vx_node vxDebugVisualizationNode(
    vx_graph graph,
    vx_image input,
    vx_array oldPoints,
    vx_array newPoints,
    vx_image output
)
{
    vx_node node = 0;
    vx_context context = vxGetContext((vx_reference)graph);

    vx_status status = VX_SUCCESS;

    vx_kernel kernel = vxGetKernelByName(context, VX_KERNEL_NAME_INTEL_SAMPLE_DEBUG_VISUALIZATION);
    if (kernel)
    {
        node = vxCreateGenericNode(graph, kernel);
        if (node)
        {
            // warp rect with vx_scalar

            IntelVXSample::logger(2) << "vxDebugVisualization: vxCreateGenericNode is OK\n";
            vx_status statuses[4];
            statuses[0] = vxSetParameterByIndex(node, 0, (vx_reference)input);
            statuses[1] = vxSetParameterByIndex(node, 1, (vx_reference)oldPoints);
            statuses[2] = vxSetParameterByIndex(node, 2, (vx_reference)newPoints);
            statuses[3] = vxSetParameterByIndex(node, 3, (vx_reference)output);

            for (vx_uint32 i = 0; i < sizeof(statuses)/sizeof(statuses[0]); i++)
            {
                if (statuses[i] != VX_SUCCESS)
                {
                    status = VX_ERROR_INVALID_PARAMETERS;
                    vxReleaseNode(&node);
                    vxReleaseKernel(&kernel);
                    node = 0;
                    kernel = 0;
                    IntelVXSample::logger(1) << "Parameter " << i << " for vxDebugVisualization wasn't set successfully\n";
                    break;
                }
            }
}
        else
        {
            vxReleaseKernel(&kernel);
        }
    }

    return node;
}

vx_status vxuDebugVisualization(
    vx_context context,
    vx_image input,
    vx_array oldPoints,
    vx_array newPoints,
    vx_image output
)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);

    if(graph)
    {
        vx_node node = vxDebugVisualizationNode(graph, input, oldPoints, newPoints, output);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}
