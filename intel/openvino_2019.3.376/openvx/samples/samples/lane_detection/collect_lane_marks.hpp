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

#pragma once
#include <opencv2/imgproc.hpp>
#include <vector>

using namespace std;

// the output of the Process function are lane marks
// the maximum number that can be collected is defined here
#define MAX_LANE_NUM    4

// class for processing HoughtLinesP result and estimate lane borders parameters
// it is implemented as separate class because
// the same code is used in OpenCV and OpenVX pipeline in final stage
class CollectLaneMarks
{
public:
    CollectLaneMarks()
    {
        // init lane width estimation to mark it as not yet estimated
        m_LaneW = -1;
        m_L0 = -1;
        m_L1 = -1;
    }

    float                   m_LaneW;    // estimation for lane width for debug purposes
    cv::Mat                 m_Edges8U;  // image with filter responce for debug purposes
    cv::Mat                 m_Overlay;  // image for lane drawing for debug purposes
    vector<cv::Vec4i>       m_Lines;    // array of hough transform lines for debug purposes

    cv::Vec2f               m_Lanes[MAX_LANE_NUM];  // estimated lanes border parameters as (X0,K) pair.
                                                    // the lane border equation is y = K * x + X0
    int                     m_L0;                   // index of left lane border
    int                     m_L1;                   // index of right lane border
    std::vector<cv::Point>  m_Points[MAX_LANE_NUM]; // points storage for each lane border
    std::vector<cv::Point>  m_PointsAll;            // points storage for all RANSAC input points

    /*! \brief Process lines estimated by Hough Transform
     * \param [in] image with filter responce that is used to check point to be part of lane border
     * \param [in] threshold to define strong and week edge responce.
     * \param [in] array of line segments from Hough transform as (x0,y0,x1,y1).
     */
    void Process(
        const cv::Mat&          edges,          // image with filter responce that is used to check point to be part of lane border
        const int               edgeThreshold,  // minimal edge value to be part of EstimateLaneBorder
        const std::vector<cv::Vec4i>& lines);   // array of line candidates as (x0,y0,x1,y1)

    /*! \brief get bound of detected lane marks
        * \param [in] index of lane
        */
    cv::Vec4i GetLaneBound(const int lane);

    /*! \brief Draw detected lines over given image using given perspective transform
     * \param [inout] image to draw detected lane marks
     * \param [in] 3x3 perspective tranform matrix that is used to transform detected lane marks into debug image coordinate system.
     * \param [in] thickness of drawing line.
     * \param [in] thickness of drawing line ends.
     */
    void DrawLanes(
        cv::Mat& debugOut,
        const cv::Mat& matPerspectiveTransform,
        const int thicknessLines=1,
        const int thicknessEnds=0, // by default ends are not drawn.
        const bool imageBGR = true);

    /*! \brief Draw detection result over given image using given perspective transform
     * \param [inout] image to draw detected lane marks
     * \param [in] 3x3 perspective tranform matrix that is used to transform detected lane marks into debug image coordinate system.
     * \param [in] 1 means only base result drawing, 2 means additional Hough transform result drawiing.
     */
    void DrawResult(
        cv::Mat& debugOut,
        const cv::Mat& matPerspectiveTransform,
        const int visualization,
        const bool imageBGR=true);

};
