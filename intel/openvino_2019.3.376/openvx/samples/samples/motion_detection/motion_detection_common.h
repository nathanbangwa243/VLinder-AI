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

#ifndef __MOTION_DETECTION_COMMON_H__
#define __MOTION_DETECTION_COMMON_H__

#include <stdio.h>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <VX/vx.h>
#include <VX/vxu.h>
#include "motion_detection_user_nodes_lib.h"

#define SCALE_IMAGE_WIDTH_THRESHOLD (1280)
#define SCALE_IMAGE_FACTOR          (2)

bool CheckBoundingBoxes(vx_rectangle_t rect1, vx_rectangle_t rect2, int thresholdX, int thresholdY);
void CombineOverlapBoundingBoxes(vx_rectangle_t &rect1, vx_rectangle_t &rect2);
void MergeBoundingBoxes(std::vector<vx_rectangle_t> &componentBoxes, int thresholdX, int thresholdY);
void SetScaleParameters(bool scaleImage, int width, int height, bool &scaleFlag, int &internalWidth, int &internalHeight, int &scaleFactor);
void CreateScaleImages(vx_graph graph, vx_df_image color, int width, int height, int internalWidth, int internalHeight, vx_image scaleImages[6], vx_image &scaleOutput);

#endif /* __MOTION_DETECTION_COMMON_H__ */
