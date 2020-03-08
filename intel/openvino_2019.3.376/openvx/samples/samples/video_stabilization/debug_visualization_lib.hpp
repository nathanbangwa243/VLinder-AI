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


#ifndef _VX_LIB_SAMPLE_STABILIZATION_DEBUG_VISUALIZATION_HPP_
#define _VX_LIB_SAMPLE_STABILIZATION_DEBUG_VISUALIZATION_HPP_

#include <vector>

#include <VX/vx.h>

#include <opencv2/core.hpp>

#include "video_stabilization_core.h"

/*****************************************************************************

    This file declares various flavors of drawDebugVisualization function
    that draw feature points and motion vectors on images. There are two
    kinds of API are provided: for OpenCV and OpenVX clients.

*****************************************************************************/



/// Draw oldPoints and newPoints and motion lines between corresponding pairs on image
/** A given image should be a 3-channel RGB */
void drawDebugVisualization (
    cv::Mat image,
    const std::vector<cv::Point2f>& oldPoints,
    const std::vector<cv::Point2f>& newPoints,
    const std::vector<unsigned char>& trackingStatus
);


/// Draw oldPoints and newPoints and motion lines between corresponding pairs
/// on inputImage and saves the result to outputImage 
/** An input and output images should be 3-channel RGB images */
void drawDebugVisualization (
    vx_image inputImage,
    vx_array oldPoints,
    vx_array newPoints,
    vx_image outputImage
);



#endif
