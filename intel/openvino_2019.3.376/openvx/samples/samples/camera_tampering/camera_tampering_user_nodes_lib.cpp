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
#include "camera_tampering_user_nodes_lib.h"
#include <iostream>
#include <vector>
#include <map>
#include <cassert>
#include <algorithm>
#include <cmath>

#include <intel/vx_samples/helper.hpp>
#include <intel/vx_samples/perfprof.hpp>
#include <stdio.h>

vx_node vxConnectedComponentLabelingNode(
    vx_graph graph,
    vx_image input,
    vx_uint32 nThreshold,
    vx_image output,
    vx_array rectangles
    )
{
    vx_node     node        = 0;
    vx_context  context     = vxGetContext((vx_reference)graph);
    vx_scalar   threshold   = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nThreshold);
    vx_status   status      = VX_SUCCESS;
    vx_kernel   kernel      = vxGetKernelByName(context, VX_KERNEL_NAME_INTEL_SAMPLE_CONNECTED_COMPONENT_LABELING);

    if (kernel)
    {
        node = vxCreateGenericNode(graph, kernel);

        if (node)
        {
            vx_uint32 width = 0, height = 0;
            vxQueryImage(input, VX_IMAGE_WIDTH, &width, sizeof(width));
            vxQueryImage(input, VX_IMAGE_HEIGHT, &height, sizeof(height));

            if((width < MINIMUM_IMAGE_WIDTH) || (width > MAXIMUM_IMAGE_WIDTH) || (height < MINIMUM_IMAGE_HEIGHT) || (height > MAXIMUM_IMAGE_HEIGHT))
            {
                std::cerr << "[ ERROR ] Image size not supported" << "\n";
                status = VX_ERROR_INVALID_PARAMETERS;
                return 0;
            }

            if (nThreshold >= (width * height))
            {
                std::cerr << "[ ERROR ] The threshold must locate in [0, width x height) pixel" << "\n";
                status = VX_ERROR_INVALID_PARAMETERS;
                return 0;
            }

            // warp rect with vx_scalar
            IntelVXSample::logger(2) << "vxConnectedComponentLabelingNode: vxCreateGenericNode is OK\n";

            vx_status statuses[4];
            statuses[0] = vxSetParameterByIndex(node, 0, (vx_reference)input);
            statuses[1] = vxSetParameterByIndex(node, 1, (vx_reference)threshold);
            statuses[2] = vxSetParameterByIndex(node, 2, (vx_reference)output);
            statuses[3] = vxSetParameterByIndex(node, 3, (vx_reference)rectangles);

            for (vx_uint32 i = 0; i < sizeof(statuses)/sizeof(statuses[0]); i++)
            {
                if (statuses[i] != VX_SUCCESS)
                {
                    status = VX_ERROR_INVALID_PARAMETERS;
                    vxReleaseNode(&node);
                    vxReleaseKernel(&kernel);
                    node = 0;
                    kernel = 0;
                    IntelVXSample::logger(1) << "Parameter " << i << " for vxConnectedComponentLabelingNode wasn't set successfully\n";
                    break;
                }
            }
        }
        else
        {
            vxReleaseKernel(&kernel);
        }
    }

    vxReleaseScalar(&threshold);

    return node;
}

vx_status vxuConnectedComponentLabeling(
    vx_context context,
    vx_image input,
    vx_uint32 threshold,
    vx_image output,
    vx_array rectangles
    )
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
        vx_node node = vxConnectedComponentLabelingNode(graph, input, threshold, output, rectangles);
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

vx_node vxUserCountNonZeroNode(vx_graph graph, vx_image input, vx_scalar sum_output)
{
    vx_node    node     = 0;
    vx_context context  = vxGetContext( ( vx_reference ) graph );
    vx_status  status   = VX_SUCCESS;
    vx_kernel  kernel   = vxGetKernelByEnum( context, VX_USER_KERNEL_COUNT_NON_ZERO );

    if (kernel)
    {
        node = vxCreateGenericNode( graph, kernel );

        if (node)
        {
            vx_uint32 width = 0, height = 0;
            vxQueryImage(input, VX_IMAGE_WIDTH, &width, sizeof(width));
            vxQueryImage(input, VX_IMAGE_HEIGHT, &height, sizeof(height));

            if((width < MINIMUM_IMAGE_WIDTH) || (width > MAXIMUM_IMAGE_WIDTH) ||
		    (height < MINIMUM_IMAGE_HEIGHT) || (height > MAXIMUM_IMAGE_HEIGHT))
            {
                std::cerr << "[ ERROR ] Image size not supported" << "\n";
                status = VX_ERROR_INVALID_PARAMETERS;
                return 0;
            }

	    vx_status statuses[2];
	    statuses[0] = vxSetParameterByIndex( node, 0, ( vx_reference ) input );
	    statuses[1] = vxSetParameterByIndex( node, 1, ( vx_reference ) sum_output );
            for (vx_uint32 i = 0; i < sizeof(statuses)/sizeof(statuses[0]); i++)
            {
                if (statuses[i] != VX_SUCCESS)
                {
                    status = VX_ERROR_INVALID_PARAMETERS;
                    vxReleaseNode(&node);
                    vxReleaseKernel(&kernel);
                    node = 0;
                    kernel = 0;
                    IntelVXSample::logger(1) << "Parameter " << i <<
					" for vxUserCountNonZeroNode wasn't set successfully\n";
                    break;
                }
            }
        }
    }
    else
    {
        vxReleaseKernel( &kernel );
    }

    return node;
}

vx_status vxuUserCountNonZeroNode(vx_context context, vx_image input, vx_scalar sum_output)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
        vx_node node = vxUserCountNonZeroNode(graph, input, sum_output);
        if (node)
        {
            status = vxVerifyGraph(graph);
	    CHECK_VX_STATUS(status);
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
