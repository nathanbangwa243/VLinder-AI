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


#ifndef _VX_LIB_SAMPLE_STABILIZATION_CORE_H_
#define _VX_LIB_SAMPLE_STABILIZATION_CORE_H_

#include <VX/vx.h>

#include <opencv2/core.hpp>


/*****************************************************************************

    This file contains definition of motion tracking state that tracks 
    frame movements along several frames and produces a correction
    transformation that should be applied on the current frame to keep it
    stabilized.

    There are two flavors of API is defined: one is dedicated to be used
    in OpenCV code (if OpenCV is available), and another one is for use in
    OpenVX code.

*****************************************************************************/



// --------------- OpenCV tracing state API -----------------

/// This is an opaque type that holds motion tracking state during video processing
/** See usage of this state in OpenCV version of the video stabilization code */
typedef void* TrackingStateOpenCV;

/// Creates an instate of TrackingStateOpenCV type
TrackingStateOpenCV createTrackingStateOpenCV ();

/// Deletes an instance of TrackingStateOpenCV type
void releaseTrackingStateOpenCV (TrackingStateOpenCV*);

/// Does all necessary work to estimate current transform matrix using OpenCV.
void estimateTransformOpenCV(
    std::vector<cv::Point2f> old_points,
    std::vector<cv::Point2f> new_points,
    std::vector<uchar> status,
    cv::Size frameRect,
    TrackingStateOpenCV trackingState,
    cv::Mat_<double> curTransform
);



// --------------- OpenVX tracing state API -----------------

/// This is an opaque type that holds motion tracking state during video processing
/** This is actually the same structure as OpenCV uses. An instance of this object
    should be passed to estimateTransform function. Then (in the OpenVX user node)
    this object will be used as a node attribute to keep the state with the node instance. */
typedef TrackingStateOpenCV TrackingStateOpenVX;

/// Creates an instate of TrackingStateOpenVX type
TrackingStateOpenVX createEstimateTransformState ();

/// Release TrackingStateOpenVX instance
void releaseEstimateTransformState (TrackingStateOpenVX*);

/// Does all necessary work to estimate current transform matrix using OpenVX.
vx_status estimateTransform(
    TrackingStateOpenVX state,
    vx_array old_points,
    vx_array new_points,
    vx_array rect,
    vx_matrix transform
);


#endif
