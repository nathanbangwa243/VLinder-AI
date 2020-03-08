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

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include "camera_tampering_core.h"

using namespace std;

// Update global label mapping to make the labels as continuous value, make it easier for size filtering
void UpdateLabels(
        uint32 *globalLabels,
        int32  labelNum)
{
    int32   i, j;
    uint32  globalLabel, globalLabelIdx;
    uint32  oldLabel, newLabel;
    uint8   *updateFlag = (uint8*)malloc(labelNum * sizeof(uint8));

    if (updateFlag == NULL)
    {
        std::cerr << "[ ERROR ] Cannot allocate memory successfully!\n";
        return;
    }
    else
    {
        memset(updateFlag, 0, labelNum * sizeof(uint8));
    }

    // Build the mapping between the neighbor labels
    for (i=1; i<labelNum; i++)
    {
        globalLabelIdx  = i;
        globalLabel     = globalLabels[i];

        while (globalLabelIdx != globalLabel)
        {
            globalLabelIdx = globalLabel;
            globalLabel = globalLabels[globalLabel];
        }

        globalLabels[i] = globalLabel;
    }

    for (i=1; i<labelNum; i++)
    {
        oldLabel = globalLabels[i];
        newLabel = i;

        for (j=i; j<labelNum; j++)
        {
            if ((globalLabels[j] == oldLabel) && (updateFlag[j] == false))
            {
                globalLabels[j] = newLabel;
                updateFlag[j]   =  true;
            }
        }
    }

    free(updateFlag);
}

void SizeFilter(
        uint32                  labelNum,           // The number of labels
        int32                   width,              // Image width
        int32                   height,             // Image height
        int32                   binaryImgLineStep,  // The line step of binary image
        int32                   labelImgLineStep,   // The line step of label image
        uint32                  threshold,          // The threshold to filter out little components(noise), it's the number of pixels, here 0 is the default number
        uint8                   *binaryImg,         // The binary image which is the input of connected component labeling process
        uint32                  *labelImg,          // The label image which is the output of connected component labeling process
        vector<vx_rectangle_t>  &componentBoxes,    // The bounding boxes for components after size filtering
        uint32                  rectListLen)        // The length of rectangle list
{
    int32           i, j, label;
    int32           cols = width - 1;           // number of pixels in a row
    int32           rows = height - 1;          // number of rows in the image
    vx_rectangle_t  rect;
    vector<int32>   labels;

    int32 *pixelCnt = (int32 *)malloc(labelNum * sizeof(int32));   // Index as the label, number of pixels with this label as the value

    if (pixelCnt == NULL)
    {
        std::cerr << "[ ERROR ] allocate memory successfully!\n";
        return;
    }
    else
    {
        memset(pixelCnt, 0, labelNum * sizeof(int32));
    }

    uint8     *pucBinaryRow;
    uint32    binaryRowSize = binaryImgLineStep;    // number of bytes in a row
    uint32    *punLabelRow;
    uint32    labelRowSize = labelImgLineStep / sizeof(uint32);

    // Clear the component boxes list
    componentBoxes.clear();

    // Count the number of pixels for each labels
    for (i=1; i<rows; i++)
    {
        punLabelRow = (uint32*)labelImg + labelRowSize * i;

        for (j=1; j<cols; j++)
        {
            label = punLabelRow[j];

            if (label != 0)
            {
                // Update the count
                pixelCnt[label]++;
            }
        }
    }

    labels.clear();
    for (i=0; i<labelNum; i++)
    {
        if (pixelCnt[i] >= threshold)
        {
            labels.push_back(i);
        }
    }

    // Number of output rectangles shouldn't exceed rectListLen
    if (labels.size() > rectListLen)
    {
        std::sort(labels.begin(), labels.end(), std::greater<int>());
        labels.resize(rectListLen);
    }

    // Set the bounding boxes for the labels as output
    for (int32 k=0; k<labels.size(); k++)
    {
        int32 foundFlag;

        // Rectangle initialization, to fix klocwork issues
        rect.start_x    = 0;
        rect.end_x      = 0;
        rect.start_y    = 0;
        rect.end_y      = 0;

        label = labels[k];

        // Top
        foundFlag = false;
        for (i=1; i<rows; i+=BOUND_RECT_STEP)
        {
            pucBinaryRow = (uint8*)binaryImg + binaryRowSize * i;
            punLabelRow = (uint32*)labelImg + labelRowSize * i;

            for (j=1; j<cols; j+=BOUND_RECT_STEP)
            {
                if (punLabelRow[j] == label)
                {
                    foundFlag = true;
                    break;
                }
            }

            if (foundFlag == true)
            {
                rect.start_y = i;
                break;
            }
        }

        // Bottom
        foundFlag = false;
        for (i=rows; i>0; i-=BOUND_RECT_STEP)
        {
            pucBinaryRow = (uint8*)binaryImg + binaryRowSize * i;
            punLabelRow = (uint32*)labelImg + labelRowSize * i;

            for (j=1; j<cols; j+=BOUND_RECT_STEP)
            {
                if (punLabelRow[j] == label)
                {
                    foundFlag = true;
                    break;
                }
            }

            if (foundFlag == true)
            {
                rect.end_y = i;
                break;
            }
        }

        // Left
        foundFlag = false;
        for (i=1; i<cols; i+=BOUND_RECT_STEP)
        {
            for (j=1; j<rows; j+=BOUND_RECT_STEP)
            {
                pucBinaryRow = (uint8*)binaryImg + binaryRowSize * j;
                punLabelRow = (uint32*)labelImg + labelRowSize * j;

                if (punLabelRow[i] == label)
                {
                    foundFlag = true;
                    break;
                }
            }

            if (foundFlag == true)
            {
                rect.start_x = i;
                break;
            }
        }

        // Right
        foundFlag = false;
        for (i=cols; i>0; i-=BOUND_RECT_STEP)
        {
            for (j=1; j<rows; j+=BOUND_RECT_STEP)
            {
                pucBinaryRow = (uint8*)binaryImg + binaryRowSize * j;
                punLabelRow = (uint32*)labelImg + labelRowSize * j;

                if (punLabelRow[i] == label)
                {
                    foundFlag = true;
                    break;
                }
            }

            if (foundFlag == true)
            {
                rect.end_x = i;
                break;
            }
        }

        // If BOUND_RECT_STEP is larger than 1, not all pixels are checked, make sure all results are valid
        if (BOUND_RECT_STEP > 1)
        {
            rect.end_x  = std::max(rect.start_x, rect.end_x);
            rect.end_y  = std::max(rect.start_y, rect.end_y);
        }

        // Update rectangle coordinates - extend BOUND_RECT_STEP pixels to each direction
        rect.start_x    = (rect.start_x < BOUND_RECT_STEP) ? 0 : (rect.start_x - BOUND_RECT_STEP);
        rect.start_y    = (rect.start_y < BOUND_RECT_STEP) ? 0 : (rect.start_y - BOUND_RECT_STEP);
        rect.end_x      = (rect.end_x > (width - 9)) ? (width - 1) : (rect.end_x + BOUND_RECT_STEP);
        rect.end_y      = (rect.end_y > (height - 9)) ? (height - 1) : (rect.end_y + BOUND_RECT_STEP);

        // Store bounding boxes as output
        componentBoxes.push_back(rect);
    }

    free(pixelCnt);
}

/* Connect component labeling, it's a regular algorithm in motion detection,
 * after background subtraction step. OpenCV may have functions which have
 * similar functionality. This algorithm is adopted as the code is simple, and
 * it's easier for GPU implementation in the future. */
void ConnectedComponentLabeling(
        uint32                  threshold,          // Input threshold to filter out little components(noise), it's in number of pixels
        int32                   width,              // Image width
        int32                   height,             // Image height
        int32                   binaryImgLineStep,  // The line step of binary image
        int32                   labelImgLineStep,   // The line step of label image
        uint8                   *binaryImg,         // Input binary image, with foreground marked as white and background marked as black, each pixel is 8-bit
        uint32                  *labelImg,          // Output label image, with each connected component has its own label number, background pixel remains as 0
        vector<vx_rectangle_t>  &componentBoxes,    // The bounding boxes for components after size filtering
        uint32                  rectListLen)        // The length of rectangle list
{
    int32   i, j, k;
    int32   cols = width - 1;                   // number of pixels in a row
    int32   rows = height - 1;                  // number of rows in the image

    uint8   *pucBinaryRow;
    uint32  *punLabelRow;

    uint32  unLabelLeft, unLabelUp, unLabelUpLeft, unLabelUpRight;

    int32   binaryRowSize   = binaryImgLineStep;                // number of bytes in a row
    int32   labelRowSize    = labelImgLineStep / sizeof(uint32);// number of 4-bytes in a row

    uint32  localLabels[4]; // labels for 4 neighbors: left, up, up-left, up-right
    uint32  *globalLabels = (uint32*)malloc(width * height * sizeof(uint32));

    int32   localLabelNum;
    int32   minLabel;

    if (globalLabels == NULL)
    {
        std::cerr << "[ ERROR ] Cannot allocate memory successfully!\n";
        return;
    }

    uint32 label = 1;
    uint32 localLabel, globalLabel;

    // Initialize global mapping. The index to this vector is the label, and the value is the mapped label with lowest value
    memset(globalLabels, 0, width * height * sizeof(uint32));

    // Initialize label image as 0s
    memset(labelImg, 0, labelImgLineStep * height);

    // First pass
    for (i=1; i<rows; i++)
    {
        pucBinaryRow = (uint8*)binaryImg + binaryRowSize * i;
        punLabelRow  = (uint32*)labelImg + labelRowSize * i;

        for (j=1; j<cols; j++)
        {
            // If current pixel is foreground, check if its neighbors are labelled
            if (pucBinaryRow[j] > 0)
            {
                localLabelNum   = 0;
                minLabel        = 0x7fffffff;

                unLabelLeft     = punLabelRow[j - 1];
                unLabelUp       = punLabelRow[j - labelRowSize];
                unLabelUpLeft   = punLabelRow[j - labelRowSize - 1];
                unLabelUpRight  = punLabelRow[j - labelRowSize + 1];

                // Check 8 connectivity neighbors
                if (unLabelLeft)
                {
                    localLabels[localLabelNum++] = unLabelLeft;
                    minLabel = unLabelLeft;
                }

                if (unLabelUp && (unLabelUp != minLabel))
                {
                    localLabels[localLabelNum++] = unLabelUp;
                    if (unLabelUp < minLabel)
                    {
                        minLabel = unLabelUp;
                    }
                }

                if (unLabelUpLeft && (unLabelUpLeft != minLabel))
                {
                    localLabels[localLabelNum++] = unLabelUpLeft;
                    if (unLabelUpLeft < minLabel)
                    {
                        minLabel = unLabelUpLeft;
                    }
                }

                if (unLabelUpRight && (unLabelUpRight != minLabel))
                {
                    localLabels[localLabelNum++] = unLabelUpRight;
                    if (unLabelUpRight < minLabel)
                    {
                        minLabel = unLabelUpRight;
                    }
                }

                if (localLabelNum == 0)
                {
                    // Mark current pixel
                    punLabelRow[j] = label;

                    // If none of its neighbors are labelled, assign a new label to it
                    globalLabels[label] = label++;
                }
                else
                {
                    // Temporarily mark current pixel
                    punLabelRow[j] = minLabel;

                    // Update mapping of global labels
                    if (localLabelNum > 1)
                    {
                        for (k=0; k<localLabelNum; k++)
                        {
                            localLabel = localLabels[k];
                            if (minLabel < globalLabels[localLabel])
                            {
                                globalLabels[localLabel] = minLabel;
                            }
                        }
                    }
                }
            } // end of if (pucBinaryRow[j] > 0)
        } // end of for (j=1; j<cols; j++)
    } // end of for (i=1; i<rows; i++)

    UpdateLabels(globalLabels, label);

    // Second pass
    for (i=1; i<rows; i++)
    {
        punLabelRow = (uint32*)labelImg + labelRowSize * i;

        for (j=1; j<cols; j++)
        {
            localLabel = punLabelRow[j];
            if (localLabel > 0)
            {
                globalLabel = globalLabels[localLabel];
                punLabelRow[j] = globalLabel;
            }
        }
    }

    free(globalLabels);

    // Size filter
    SizeFilter(label, width, height, binaryImgLineStep, labelImgLineStep, threshold, binaryImg, labelImg, componentBoxes, rectListLen);
}

void ConnectedComponentLabelingClass::init(
    int32   width,
    int32   height,
    int32   srcImgLineStep,
    int32   dstImgLineStep)
{
    m_nWidth            = width;
    m_nHeight           = height;
    m_nSrcImgStep       = srcImgLineStep;
    m_nDstImgStep       = dstImgLineStep;
    m_nBoundingBoxCnt   = 0;

    return;
}

void ConnectedComponentLabelingClass::release()
{
    m_nWidth            = 0;
    m_nHeight           = 0;
    m_nThreshold        = 0;
    m_nSrcImgStep       = 0;
    m_nDstImgStep       = 0;
    m_nBoundingBoxCnt   = 0;

    return;
}

int32 ConnectedComponentLabelingClass::Do(
    uint8           *pSrcImg,
    uint32          *pDstImg,   // TODO: need to decide if this image should be allocated outside, or allocated internally. Currently it's assumed as allocated outside.
    vx_rectangle_t  *rectList,
    uint32          rectListLen)
{
    vector<vx_rectangle_t> componentBoxes;
    m_nBoundingBoxCnt = 0;

    ConnectedComponentLabeling(m_nThreshold, m_nWidth, m_nHeight, m_nSrcImgStep, m_nDstImgStep, pSrcImg, pDstImg, componentBoxes, rectListLen);

    m_nBoundingBoxCnt = (uint32)componentBoxes.size();

    // Copy from vector componentBoxes to array rectList
    if (m_nBoundingBoxCnt > rectListLen)
    {
        std::copy(componentBoxes.begin(), componentBoxes.begin() + rectListLen, rectList);
    }
    else if (m_nBoundingBoxCnt > 0)
    {
        std::copy(componentBoxes.begin(), componentBoxes.end(), rectList);
    }

    return m_nBoundingBoxCnt;
}

