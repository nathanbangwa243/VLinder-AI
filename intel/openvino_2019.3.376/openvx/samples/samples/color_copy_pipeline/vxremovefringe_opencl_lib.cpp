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

vx_node vxRemoveFringeOpenCLTiledNode(vx_graph graph,
                                      vx_image inputCMYK,
                                      vx_image inputL,
                                      vx_image inputNeutralEdgeMask,
                                      vx_image outputCMYK,
                                      vx_image outputK,
                                      vx_array LtoK_nodes)
{

    vx_uint32 i;
    vx_node node = 0;
    //get the graph context
    vx_context context = vxGetContext((vx_reference)graph);
    vx_status status = VX_SUCCESS;

    //! Retrieving the remove fringe kernel by name
    vx_kernel kernel = vxGetKernelByName(context, VX_KERNEL_NAME_USER_REMOVEFRINGE_OPENCL);
    if (kernel)
    {
        node = vxCreateGenericNode(graph, kernel);
        if (node)
        {
            vx_status statuses[6];
            statuses[0] = vxSetParameterByIndex(node, 0, (vx_reference)inputCMYK);
            statuses[1] = vxSetParameterByIndex(node, 1, (vx_reference)inputL);
            statuses[2] = vxSetParameterByIndex(node, 2, (vx_reference)inputNeutralEdgeMask);
            statuses[3] = vxSetParameterByIndex(node, 3, (vx_reference)outputCMYK);
            statuses[4] = vxSetParameterByIndex(node, 4, (vx_reference)outputK);
            statuses[5] = vxSetParameterByIndex(node, 5, (vx_reference)LtoK_nodes);

            for (i = 0; i < sizeof(statuses)/sizeof(statuses[0]); i++)
            {
                if (statuses[i] != VX_SUCCESS)
                {
                    status = VX_ERROR_INVALID_PARAMETERS;
                    vxReleaseNode(&node);
                    vxReleaseKernel(&kernel);
                    node = 0;
                    kernel = 0;
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

vx_node vxRemoveFringePlanarOpenCLTiledNode(vx_graph graph,
                                     vx_image inputCMYK,
                                     vx_image inputL,
                                     vx_image inputNeutralEdgeMask,
                                     vx_image outputC,
                                     vx_image outputM,
                                     vx_image outputY,
                                     vx_image outputK,
                                     vx_image outputK_edge,
                                     vx_array LtoK_nodes)
{

    vx_uint32 i;
    vx_node node = 0;
    //get the graph context
    vx_context context = vxGetContext((vx_reference)graph);
    vx_status status = VX_SUCCESS;

    //! Retrieving the remove fringe kernel by name
    vx_kernel kernel = vxGetKernelByName(context, VX_KERNEL_NAME_USER_REMOVEFRINGEPLANAR_OPENCL);
    if (kernel)
    {
        node = vxCreateGenericNode(graph, kernel);
        if (node)
        {
            vx_status statuses[9];
            statuses[0] = vxSetParameterByIndex(node, 0, (vx_reference)inputCMYK);
            statuses[1] = vxSetParameterByIndex(node, 1, (vx_reference)inputL);
            statuses[2] = vxSetParameterByIndex(node, 2, (vx_reference)inputNeutralEdgeMask);
            statuses[3] = vxSetParameterByIndex(node, 3, (vx_reference)outputC);
            statuses[4] = vxSetParameterByIndex(node, 4, (vx_reference)outputM);
            statuses[5] = vxSetParameterByIndex(node, 5, (vx_reference)outputY);
            statuses[6] = vxSetParameterByIndex(node, 6, (vx_reference)outputK);
            statuses[7] = vxSetParameterByIndex(node, 7, (vx_reference)outputK_edge);
            statuses[8] = vxSetParameterByIndex(node, 8, (vx_reference)LtoK_nodes);

            for (i = 0; i < sizeof(statuses)/sizeof(statuses[0]); i++)
            {
                if (statuses[i] != VX_SUCCESS)
                {
                    status = VX_ERROR_INVALID_PARAMETERS;
                    vxReleaseNode(&node);
                    vxReleaseKernel(&kernel);
                    node = 0;
                    kernel = 0;
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
