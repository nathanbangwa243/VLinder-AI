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


#ifndef _VX_LIB_SAMPLE_MOTION_DETECTION_USER_NODES_H_
#define _VX_LIB_SAMPLE_MOTION_DETECTION_USER_NODES_H_

#include <VX/vx.h>

#define VX_KERNEL_NAME_INTEL_SAMPLE_CONNECTED_COMPONENT_LABELING    "com.intel.sample.connected_component_labeling"
#define VX_KERNEL_NAME_INTEL_SAMPLE_COUNT_NON_ZERO                  "app.userkernels.count_non_zero"
#define VX_LIBRARY_SAMPLE_CAMERATAMPING                           (0x6)

// Motion detection parameters, users can modify for their application
#define MINIMUM_IMAGE_WIDTH              (16)    // Minimum image width supported by camera tampering
#define MINIMUM_IMAGE_HEIGHT             (16)    // Minimum image height supported by camera tampering
#define MAXIMUM_IMAGE_WIDTH              (8192)  // Maximum image width supported by camera tampering
#define MAXIMUM_IMAGE_HEIGHT             (8192)  // Maximum image height supported by camera tampering

#define MAXIMUM_RECTANGLE_NUMBER                                    (128)   // Maximum number of rectangles drawn on the output image, each rectangle is a distinct moving object.
#define HORIZONTAL_MERGE_OVERLAP_RATIO_THRESHOLD_IN_PERCENTAGE      (80)    // 80%, a threshold to determine when 2 rectangles doesn't vertically overlap, if they horizontally overlap well.
#define VERTICAL_MERGE_OVERLAP_RATIO_THRESHOLD_IN_PERCENTAGE        (80)    // 80%, a threshold to determine when 2 rectangles doesn't horizontally overlap, if they vertically overlap well.
#define HORIZONTAL_MERGE_NEAR_THRESHOLD_DIVISOR                     (100)   // 1%, the threshold to decide if 2 vertically overlapped, horizontally neighboring rectangles can be merged or not.
#define VERTICAL_MERGE_NEAR_THRESHOLD_DIVISOR                       (100)   // 1%, the threshold to decide if 2 horizontally overlapped, vertically neighboring rectangles can be merged or not.
#define SIZE_FILTER_DEFAULT_THRESHOLD_DIVISOR                       (100)   // 1% of overall image pixels, the default size filter threshold when the setting is 0 from input.

enum vx_kernel_intel_sample_camera_tampering_e {
    VX_KERNEL_SAMPLE_CAMERATAMPERING_CONNECTED_COMPONENT_LABELING = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_SAMPLE_CAMERATAMPING) + 0x0,
    VX_USER_KERNEL_COUNT_NON_ZERO = VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_SAMPLE_CAMERATAMPING) + 0x1
};

vx_node vxConnectedComponentLabelingNode(
    vx_graph graph,
    vx_image input,
    vx_uint32 threshold,
    vx_image output,
    vx_array rectangles);

vx_status vxuConnectedComponentLabeling(
    vx_context context,
    vx_image input,
    vx_uint32 threshold,
    vx_image output,
    vx_array rectangles);


vx_node vxUserCountNonZeroNode(
    vx_graph graph,
    vx_image input,
    vx_scalar  sum_output);

#endif
