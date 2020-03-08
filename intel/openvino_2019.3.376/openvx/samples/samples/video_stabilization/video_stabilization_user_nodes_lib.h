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


#ifndef _VX_LIB_SAMPLE_STABILIZATION_USER_NODES_H_
#define _VX_LIB_SAMPLE_STABILIZATION_USER_NODES_H_

#include <VX/vx.h>
#include "video_stabilization_core.h"

#define VX_KERNEL_NAME_INTEL_SAMPLE_ESTIMATE_TRANSFORM "com.intel.sample.estimatetransform"
#define VX_KERNEL_NAME_INTEL_SAMPLE_DEBUG_VISUALIZATION "com.intel.sample.debug_visualization"

#define VX_LIBRARY_SAMPLE_STABILIZATION (0x1)


enum vx_kernel_intel_sample_stabilization_e {
    VX_KERNEL_SAMPLE_STABILIZATION_ESTIMATE_TRANSFORM = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_SAMPLE_STABILIZATION) + 0x0,
    VX_KERNEL_SAMPLE_STABILIZATION_DEBUG_VISUALIZATION = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_SAMPLE_STABILIZATION) + 0x1
}; 


vx_node vxEstimateTransformNode(
    vx_graph graph,
    vx_array old_points,
    vx_array new_points,
    vx_array rect,
    vx_matrix transform
);

vx_status vxuEstimateTransform(
    vx_context context,
    vx_array old_points,
    vx_array new_points,
    vx_array rect,
    TrackingStateOpenVX state,
    vx_matrix transform
);


vx_node vxDebugVisualizationNode(
    vx_graph graph,
    vx_image input,
    vx_array oldPoints,
    vx_array newPoints,
    vx_image output
);


vx_status vxuDebugVisualization(
    vx_context context,
    vx_image input,
    vx_array oldPoints,
    vx_array newPoints,
    vx_image output
);

#endif
