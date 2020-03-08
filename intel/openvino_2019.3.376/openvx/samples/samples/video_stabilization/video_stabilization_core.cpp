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
#include "video_stabilization_core.h"
#include <iostream>
#include <vector>
#include <map>
#include <cassert>
#include <algorithm>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>

#include <intel/vx_samples/helper.hpp>
#include <intel/vx_samples/perfprof.hpp>


namespace
{

// Minimum number of points that should be used for transform matrix reconstruction
#define MIN_NUM_TRACKING_POINTS 5



/// Contains tracking state for dx, dy and da accumulated over previous frames
/** Motion tracking is achieved with simple Kalman filter operating with 3 observabale
    parameters:

        - dx (translation in x dimension),

        - dy (translation in y dimension) and

        - da (rotation).    

    The main method of this class is *update*. It consumes old and new
    coordinates of tracked points on the previous and the current frame correspondingly, and
    gives (dx, dy, da) that should be applied for the current frame to obtain stabilized image.
*/
class TrackingState
{
    cv::KalmanFilter kalmanFilter;
    cv::Mat_<float> measurementDecomposed;

    /// Estimate transform between previous frame and the current frame by a set of tracked points
    void estimateMomentaryTransform (const std::vector<cv::Point2f>& oldPoints, const std::vector<cv::Point2f>& newPoints, float& dx, float& dy, float& da)
    {
        if(oldPoints.size() <= MIN_NUM_TRACKING_POINTS)
        {
            // There are no enough points to estimate transform
            IntelVXSample::logger(1) << "Number of detected/tracked points is not enough (less than " << MIN_NUM_TRACKING_POINTS << ")\n";
            dx = dy = da = 0;
        }
        else
        {
            IntelVXSample::logger(1) << "Number of tracked points: " << oldPoints.size() << "\n";
            cv::Mat_<float> curTransform(2, 3);
            curTransform = cv::estimateRigidTransform(oldPoints, newPoints, false);

            IntelVXSample::logger(1) << "Momentary transform matrix:\n" << curTransform << "\n";

            if(curTransform.rows == 0 || curTransform.cols == 0)
            {
                dx = dy = da = 0;
            }
            else
            {
                dx = curTransform(0, 2);
                dy = curTransform(1, 2);
                da = atan2(curTransform(1, 0), curTransform(0, 0));
            }
        }
    }

public:

    TrackingState () :
        kalmanFilter(3*3, 3, 0),    // 3*3 = 9 elements tracked in the state (translation, rotation and their derivatives, see the matrix below
        measurementDecomposed(3, 1)
    {
        kalmanFilter.transitionMatrix = (cv::Mat_<float>(9, 9) <<
               1,    0,    0,       0,    0,    0,         0,     0,    0,     // directly observed dx
               0,    1,    0,       0,    0,    0,         0,     0,    0,     // directly observed dy
               0,    0,    1,       0,    0,    0,         0,     0,    0,     // directly observed da

               1,    0,    0,    0.95,    0,    0,        -1,     0,    0,     // integrated x
               0,    1,    0,       0, 0.95,    0,         0,    -1,    0,     // integrated y
               0,    0,    1,       0,    0, 0.95,         0,     0,   -1,     // integrated a

            0.05,    0,    0,       0,    0,    0,      0.95,     0,    0,     // accumulated compensation for dx
            0,    0.05,    0,       0,    0,    0,         0,  0.95,    0,     // accumulated compensation for dy
            0,       0, 0.05,       0,    0,    0,         0,     0, 0.95      // accumulated compensation for da
        );

        kalmanFilter.statePre.setTo(cv::Scalar(0));
        cv::setIdentity(kalmanFilter.measurementMatrix);
        cv::setIdentity(kalmanFilter.processNoiseCov, cv::Scalar::all(1e-4));
        cv::setIdentity(kalmanFilter.measurementNoiseCov, cv::Scalar::all(0.01));
    }

    void update (const std::vector<cv::Point2f>& oldPoints, const std::vector<cv::Point2f>& newPoints, float& dx, float& dy, float& da)
    {
        estimateMomentaryTransform(oldPoints, newPoints, dx, dy, da);

        measurementDecomposed(0) = dx;
        measurementDecomposed(1) = dy;
        measurementDecomposed(2) = da;

        IntelVXSample::logger(2) << "kalmanFilter.predict(): " << kalmanFilter.predict() << '\n';

        cv::Mat_<float> decomposedEstimate = kalmanFilter.correct(measurementDecomposed);
        IntelVXSample::logger(2) << "kalmanFilter.correct(...): " << decomposedEstimate << '\n';

        dx = decomposedEstimate(3);
        dy = decomposedEstimate(4);
        da = decomposedEstimate(5);
    }

};


typedef TrackingState* PState;


PERFPROF_REGION_DEFINE(estimateTransform_lib)

}


TrackingStateOpenCV createTrackingStateOpenCV ()
{
    return new TrackingState;
}


void releaseTrackingStateOpenCV (TrackingStateOpenCV* state)
{
    delete (TrackingState*)*state;
    *state = 0;
}


/// Update an instance of TrackingStateOpenCV with point pairs and gives transformation
/** This is the main stabilization method. It updates current tracking state with
    a new set of points pair (oldPoints is for the previous frame, and newPoints is
    for current frame). Then it forms translation (dx, dy) and rotation (da) which
    should be applied to the current frame to keep it stabilized along the whole video. */
void updateTrackingStateOpenCV (
    TrackingStateOpenCV state,
    const std::vector<cv::Point2f>& oldPoints,
    const std::vector<cv::Point2f>& newPoints,
    float& dx,
    float& dy,
    float& da
)
{
    static_cast<TrackingState*>(state)->update(oldPoints, newPoints, dx, dy, da);
}



TrackingStateOpenVX createEstimateTransformState ()
{
    return createTrackingStateOpenCV();
}

void releaseEstimateTransformState (TrackingStateOpenVX* state)
{
    releaseTrackingStateOpenCV((TrackingStateOpenCV*)state);
}



vx_status estimateTransform(
    TrackingStateOpenVX state,
    vx_array old_points,
    vx_array new_points,
    vx_array frameRect,
    vx_matrix estimate
)
{
    PERFPROF_REGION_AUTO(estimateTransform_lib);

    vx_size old_points_num = 0, new_points_num = 0;

    CHECK_VX_STATUS(vxQueryArray(old_points, VX_ARRAY_NUMITEMS, &old_points_num, sizeof(vx_size)));


    CHECK_VX_STATUS(vxQueryArray(new_points, VX_ARRAY_NUMITEMS, &new_points_num, sizeof(vx_size)));

    if(old_points_num != new_points_num)
    {
        return VX_ERROR_INVALID_DIMENSION;
    }

    IntelVXSample::logger(1) << "Number of detected points: " << old_points_num << "\n";
    IntelVXSample::logger(2) << "Start EstimateTransformKernel Real execution\n";

    std::vector<cv::Point2f> oldPoints, newPoints;

    if(old_points_num > MIN_NUM_TRACKING_POINTS)
    {
        vx_keypoint_t* old_keypoints = 0;
        vx_size old_stride = 0;
        vx_map_id map_id_old_points;
        CHECK_VX_STATUS(vxMapArrayRange(old_points, 0, old_points_num, &map_id_old_points, &old_stride, (void**)&old_keypoints, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
        IntelVXSample::AutoCommitArray _old_oints_auto(old_points, 0, old_points_num, old_keypoints, map_id_old_points);
        
        vx_keypoint_t* new_keypoints = 0;
        vx_size new_stride = 0;
        vx_map_id map_id_new_points;
        CHECK_VX_STATUS(vxMapArrayRange(new_points, 0, new_points_num, &map_id_new_points, &new_stride, (void**)&new_keypoints, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
        IntelVXSample::AutoCommitArray _new_points_auto(new_points, 0, new_points_num, new_keypoints, map_id_new_points);
        
        if(old_stride != sizeof(vx_keypoint_t) || new_stride != sizeof(vx_keypoint_t))
        {
            return VX_ERROR_INVALID_PARAMETERS;
        }
        
        oldPoints.reserve(old_points_num);
        newPoints.reserve(old_points_num);

        // To filter out feature points that are not in the valid frame rectangle,
        // the frame size should be obtained. It is packed as one-element vx_array frameRect.
        // Here it is accessed, and the resulting rectangle is stored in the following variable:
        vx_rectangle_t frameRectValue;
        {
            // Here, there is no need to query the size of the array, it should be a single-element array
            // of appropriate type. The input validator function should check all necessary parameters.
            vx_size stride = sizeof(vx_rectangle_t);
            // By providing non-zero pointer to vxMapArrayRange, we obtain a copy of necessary data into
            // frameRectValue.
            vx_rectangle_t* pRectValue = 0;
            vx_map_id map_id;
            CHECK_VX_STATUS(vxMapArrayRange(frameRect, 0, 1, &map_id, &stride, (void**)&pRectValue, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
            frameRectValue = *pRectValue;
            CHECK_VX_STATUS(vxUnmapArrayRange(frameRect, map_id));
        }


        // Filter out untracked points, and points that is originated/moved out of the rect
        for(vx_size i = 0; i < old_points_num; ++i)
        {

            vx_keypoint_t oldPoint = old_keypoints[i];
            vx_keypoint_t newPoint = new_keypoints[i];

            // Filter out bad points 

            if (
                oldPoint.x < frameRectValue.start_x || oldPoint.x >= frameRectValue.end_x ||
                oldPoint.y < frameRectValue.start_y || oldPoint.y >= frameRectValue.end_y
            )
            {
                IntelVXSample::logger(2) << "Old point is out of frameRect: ";
            }
            else if(!new_keypoints[i].tracking_status)
            {
                IntelVXSample::logger(2) << "Point is lost:            ";
            }
            else if (
                newPoint.x < frameRectValue.start_x || newPoint.x >= frameRectValue.end_x ||
                newPoint.y < frameRectValue.start_y || newPoint.y >= frameRectValue.end_y
            )
            {
                IntelVXSample::logger(2) << "New point is out of frameRect: ";
            }
            else
            {
                // consider the current point as good one
                IntelVXSample::logger(2) << "Tracked point:            ";
                oldPoints.push_back(cv::Point2f(oldPoint.x, oldPoint.y));
                newPoints.push_back(cv::Point2f(newPoint.x, newPoint.y));
            }

            IntelVXSample::logger(2) << "([" << oldPoint.x << ", " << oldPoint.y << "] -> [" << newPoint.x << ", " << newPoint.y << "])\n";
        }

        assert(oldPoints.size() == newPoints.size());
    }

    float dx = 0, dy = 0, da = 0;
    ((TrackingState*)state)->update(oldPoints, newPoints, dx, dy, da);
    IntelVXSample::logger(1) << "(dx, dy, da) = (" << dx << ", " << dy << ", " << da << ")" << std::endl;
 

    // x0 = a x + b y + c;
    // y0 = d x + e y + f;
    vx_float32 estimateRaw[3][2] = 
    {
        {   std::cos(da),   std::sin(da)  },   // a, d
        {  -std::sin(da),   std::cos(da)  },   // b, e
        {  dx,              dy            }    // c, f
    };

    CHECK_VX_STATUS(vxCopyMatrix(estimate, &estimateRaw, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));

    IntelVXSample::logger(2) << "End EstimateTransformKernel Real execution\n";

    
    return VX_SUCCESS;
}


void estimateTransformOpenCV(
    std::vector<cv::Point2f> old_points,
    std::vector<cv::Point2f> new_points,
    std::vector<uchar> status,
    cv::Size frameRect,
    TrackingStateOpenCV trackingState,
    cv::Mat_<double> curTransform
)
{
    using namespace cv;

    PERFPROF_REGION_AUTO(estimateTransform_lib);
    // Remove untracked points
    size_t insertPos = 0;
    for (size_t i = 0; i < status.size(); i++)
    {
        Point2f oldPoint = old_points[i];
        Point2f newPoint = new_points[i];

        // Filter out bad points 

        if (
            oldPoint.x < 0 || oldPoint.x >= frameRect.width ||
            oldPoint.y < 0 || oldPoint.y >= frameRect.height
        )
        {
            IntelVXSample::logger(2) << "Old point is out of rect: ";
        }
        else if(!status[i])
        {
            IntelVXSample::logger(2) << "Point is lost:            ";
        }
        else if (
            newPoint.x < 0 || newPoint.x >= frameRect.width ||
            newPoint.y < 0 || newPoint.y >= frameRect.height
        )
        {
            IntelVXSample::logger(2) << "New point is out of rect: ";
        }
        else
        {
            // consider the current point as good one

            IntelVXSample::logger(2) << "Tracked point:            ";

            if (i != insertPos)
            {
                assert(insertPos < i);
                old_points[insertPos] = old_points[i];
                new_points[insertPos] = new_points[i];
            }

            insertPos++;
        }

        IntelVXSample::logger(2) << "(" << oldPoint << " -> " << newPoint << ")\n"; 
    }

    // truncate vectors to hold good points only
    old_points.resize(insertPos);
    new_points.resize(insertPos);

    float dx, dy, da;
    updateTrackingStateOpenCV(trackingState, old_points, new_points, dx, dy, da);
    IntelVXSample::logger(1) << "(dx, dy, da) = (" << dx << ", " << dy << ", " << da << ")" << std::endl;

    curTransform.at<double>(0, 0) = cos(-da);
    curTransform.at<double>(0, 1) = -sin(-da);
    curTransform.at<double>(1, 0) = sin(-da);
    curTransform.at<double>(1, 1) = cos(-da);

    curTransform.at<double>(0, 2) = -dx;
    curTransform.at<double>(1, 2) = -dy;
    IntelVXSample::logger(1) << "curTransform decomposed: dx = " << dx << ", dy = " << dy << ", da = " << da << std::endl;
}
