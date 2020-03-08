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


#include "debug_visualization_lib.hpp"
#include <intel/vx_samples/helper.hpp>

#include <opencv2/opencv.hpp>


// Hard-coded radius of circles used to draw debug visualization
const int CIRCLE_RADIOUS = 3;


void drawDebugVisualization (
    cv::Mat image,
    const std::vector<cv::Point2f>& oldPoints,
    const std::vector<cv::Point2f>& newPoints,
    const std::vector<unsigned char>& trackingStatus
)
{
    cv::Scalar oldColor(0, 255, 0);
    cv::Scalar newColor(0, 0, 255);
    if(oldPoints.size() != newPoints.size())
    {
        std::cerr << "[ ERROR ] Number of old and new points do not match\n";
        exit(1);
    }

    size_t numPoints = oldPoints.size();
    IntelVXSample::logger(1) << "Number of corners: " << numPoints << std::endl;

    for(size_t i = 0; i < numPoints; ++i)
    {
        circle(image, cv::Point(oldPoints[i].x, oldPoints[i].y), CIRCLE_RADIOUS, oldColor);

        if(trackingStatus[i])
        {
            line(image, cv::Point(oldPoints[i].x, oldPoints[i].y), cv::Point(newPoints[i].x, newPoints[i].y), newColor);
            circle(image, cv::Point(newPoints[i].x, newPoints[i].y), CIRCLE_RADIOUS - 1, newColor);
        }
    }
}


void drawDebugVisualization (
    cv::Mat imageMat,
    vx_array oldPoints,
    vx_array newPoints
)
{
    cv::Scalar oldColor(0, 255, 0);
    cv::Scalar newColor(0, 0, 255);

    vx_size numOldPoints;
    CHECK_VX_STATUS(vxQueryArray(oldPoints, VX_ARRAY_NUMITEMS, &numOldPoints, sizeof(numOldPoints)));

    vx_size numNewPoints;
    CHECK_VX_STATUS(vxQueryArray(newPoints, VX_ARRAY_NUMITEMS, &numNewPoints, sizeof(numNewPoints)));

    if(numOldPoints != numNewPoints)
    {
        std::cerr << "[ ERROR ] Number of old and new points do not match\n";
        exit(1);
    }

    size_t numPoints = numOldPoints;

    if(numPoints == 0)
    {
        return;
    }

    vx_keypoint_t* oldPointsMapped = 0;
    vx_size oldPointsStride;
    vx_map_id mapIDoldPoints;
    CHECK_VX_STATUS(vxMapArrayRange(oldPoints, 0, numPoints, &mapIDoldPoints, &oldPointsStride, (void**)&oldPointsMapped, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));

    vx_keypoint_t* newPointsMapped = 0;
    vx_size newPointsStride;
    vx_map_id mapIDnewPoints;
    CHECK_VX_STATUS(vxMapArrayRange(newPoints, 0, numPoints, &mapIDnewPoints, &newPointsStride, (void**)&newPointsMapped, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));

    if(newPointsStride < sizeof(vx_keypoint_t) || oldPointsStride < sizeof(vx_keypoint_t))
    {
        std::cerr << "Stride returned by vxAceessArrayRange is lesser than sizeof(vx_keypoint_t)\n";
        exit(1);
    }

    for(size_t i = 0; i < numPoints; ++i)
    {
        circle(imageMat, cv::Point(oldPointsMapped[i].x, oldPointsMapped[i].y), CIRCLE_RADIOUS, oldColor);

        if(newPointsMapped[i].tracking_status)
        {
            line(imageMat, cv::Point(oldPointsMapped[i].x, oldPointsMapped[i].y), cv::Point(newPointsMapped[i].x, newPointsMapped[i].y), newColor);
            circle(imageMat, cv::Point(newPointsMapped[i].x, newPointsMapped[i].y), CIRCLE_RADIOUS - 1, newColor);
        }
    }
    CHECK_VX_STATUS(vxUnmapArrayRange(oldPoints, mapIDoldPoints));
    CHECK_VX_STATUS(vxUnmapArrayRange(newPoints, mapIDnewPoints));
}

void drawDebugVisualization (
    vx_image inputImage,
    vx_array oldPoints,
    vx_array newPoints,
    vx_image outputImage
)
{
    vx_map_id map_id_in;
    vx_map_id map_id_out;
    cv::Mat inputMat = IntelVXSample::mapAsMat(inputImage, VX_READ_ONLY, &map_id_in);
    cv::Mat outputMat = IntelVXSample::mapAsMat(outputImage, VX_WRITE_ONLY, &map_id_out);
    inputMat.copyTo(outputMat);
    IntelVXSample::unmapAsMat(inputImage, inputMat, map_id_in);
    drawDebugVisualization(outputMat, oldPoints, newPoints);
    IntelVXSample::unmapAsMat(outputImage, outputMat, map_id_out);
}

