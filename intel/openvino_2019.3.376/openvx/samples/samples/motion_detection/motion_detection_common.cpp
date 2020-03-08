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

#include "motion_detection_common.h"
#include <intel/vx_samples/helper.hpp>

using namespace std;

// Check if 2 boxes overlaps or not
bool CheckBoundingBoxes(
        vx_rectangle_t  rect1,
        vx_rectangle_t  rect2,
        int             thresholdX,
        int             thresholdY)
{
    bool    bXOverlap   = false;
    bool    bYOverlap   = false;
    bool    bMerge      = false;
    int     overlapRatio;       // Overlap ratio in percentage

    vx_uint32 xMin, xMax, yMin, yMax;

    xMin = min(rect1.start_x, rect2.start_x);
    xMax = max(rect1.end_x, rect2.end_x);

    if (max(rect1.start_x, rect2.start_x) <= min(rect1.end_x, rect2.end_x))
    {
        bXOverlap = true;
    }

    yMin = min(rect1.start_y, rect2.start_y);
    yMax = max(rect1.end_y, rect2.end_y);

    if (max(rect1.start_y, rect2.start_y) <= min(rect1.end_y, rect2.end_y))
    {
        bYOverlap = true;
    }

    // If 2 rectangle overlaps, merge them
    bMerge = bXOverlap && bYOverlap;

    if (bMerge == false)
    {
        // If 2 rectanges don't overlap, check if they're very close to each other. Merge them if they are close enough.
        if ((bXOverlap == true) && (xMax != xMin))
        {
            overlapRatio = (((rect1.end_x - rect1.start_x) + (rect2.end_x - rect2.start_x)) * 100) / ((xMax - xMin) * 2);

            if (overlapRatio > HORIZONTAL_MERGE_OVERLAP_RATIO_THRESHOLD_IN_PERCENTAGE)
            {
                // Check if the boxes are vertically close enough
                if (((yMax - yMin) - ((rect1.end_y - rect1.start_y) + (rect2.end_y - rect2.start_y))) < thresholdY)
                {
                    bMerge = true;
                }
            }
        }
        else if ((bYOverlap == true) && (yMax != yMin))
        {
            overlapRatio = (((rect1.end_y - rect1.start_y) + (rect2.end_y - rect2.start_y)) * 100) / ((yMax - yMin) * 2);

            if (overlapRatio > VERTICAL_MERGE_OVERLAP_RATIO_THRESHOLD_IN_PERCENTAGE)
            {
                // Check if the boxes are horizontally close enough
                if (((xMax - xMin) - ((rect1.end_x - rect1.start_x) + (rect2.end_x - rect2.start_x))) < thresholdX)
                {
                    bMerge = true;
                }
            }
        }
    }

    return bMerge;
}

// Combine 2 overlapped bounding boxes into 1
void CombineOverlapBoundingBoxes(
        vx_rectangle_t &rect1,
        vx_rectangle_t &rect2)
{
    vx_uint32 xMin, xMax, yMin, yMax;

    xMin = min(rect1.start_x, rect2.start_x);
    xMax = max(rect1.end_x, rect2.end_x);
    yMin = min(rect1.start_y, rect2.start_y);
    yMax = max(rect1.end_y, rect2.end_y);

    rect1.start_x   = xMin;
    rect1.end_x     = xMax;
    rect1.start_y   = yMin;
    rect1.end_y     = yMax;
}

// If the bounding boxes have intersection, merge them
void MergeBoundingBoxes(
        vector<vx_rectangle_t>  &componentBoxes,
        int                     thresholdX,
        int                     thresholdY)
{
    int i, j;
    bool bOverlap;

    do
    {
        bOverlap = false;

        for (i=0; i<componentBoxes.size(); i++)
        {
            for (j=i+1; j<componentBoxes.size(); j++)
            {
                bOverlap = CheckBoundingBoxes(componentBoxes[i], componentBoxes[j], thresholdX, thresholdY);

                if (bOverlap == true)
                {
                    CombineOverlapBoundingBoxes(componentBoxes[i], componentBoxes[j]);
                    // After merging overlapped boxes, the second box should be removed from the list
                    componentBoxes.erase(componentBoxes.begin() + j);
                    break;
                }
            }

            if (bOverlap == true)
            {
                break;
            }
        }
    } while (bOverlap == true);
}

void SetScaleParameters(
    bool    scaleImage,
    int     width,
    int     height,
    bool    &scaleFlag,
    int     &internalWidth,
    int     &internalHeight,
    int     &scaleFactor)
{
    if ((scaleImage == true) && (width >= SCALE_IMAGE_WIDTH_THRESHOLD))
    {
        scaleFactor     = SCALE_IMAGE_FACTOR;
        internalWidth   = width / scaleFactor;
        internalHeight  = (height / scaleFactor) & 0xfffffffe;
        scaleFlag       = true;
    }
    else
    {
        scaleFlag       = false;
    }
}

/* This function creates the images used by the scaling nodes */
void CreateScaleImages(
    vx_graph                graph,
    vx_df_image             color,          // input image format: RGB, NV12 or IYUV(I420)
    int                     width,          // input image width
    int                     height,         // input image height
    int                     internalWidth,  // scaled image width
    int                     internalHeight, // scaled image height
    vx_image                scaleImages[6], // array of images used by scaling nodes
    vx_image                &scaleOutput)   // output image after scaling
{
    if (color == VX_DF_IMAGE_RGB)
    {
        for (int i=0; i<3; i++)
        {
            scaleImages[i] = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
            CHECK_VX_OBJ(scaleImages[i]);
        }

        for (int i=3; i<6; i++)
        {
            scaleImages[i] = vxCreateVirtualImage(graph, internalWidth, internalHeight, VX_DF_IMAGE_U8);
            CHECK_VX_OBJ(scaleImages[i]);
        }
    }
    else if ((color == VX_DF_IMAGE_NV12) || (color == VX_DF_IMAGE_IYUV))
    {
        for (int i=0; i<3; i++)
        {
            if (i == 0)
            {
                scaleImages[i] = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
                CHECK_VX_OBJ(scaleImages[i]);
            }
            else
            {
                scaleImages[i] = vxCreateVirtualImage(graph, width / 2, height / 2, VX_DF_IMAGE_U8);
                CHECK_VX_OBJ(scaleImages[i]);
            }
        }

        for (int i=3; i<6; i++)
        {
            if (i == 3)
            {
                scaleImages[i] = vxCreateVirtualImage(graph, internalWidth, internalHeight, VX_DF_IMAGE_U8);
                CHECK_VX_OBJ(scaleImages[i]);
            }
            else 
            {
                scaleImages[i] = vxCreateVirtualImage(graph, internalWidth / 2, internalHeight / 2, VX_DF_IMAGE_U8);
                CHECK_VX_OBJ(scaleImages[i]);
            }
        }
    }

    scaleOutput = vxCreateVirtualImage(graph, internalWidth, internalHeight, color);
    CHECK_VX_OBJ(scaleOutput);
}

