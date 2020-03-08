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

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>

#include <intel/vx_samples/helper.hpp>
#include <intel/vx_samples/perfprof.hpp>
#include "video_stabilization_core.h"
#include "cmdoptions.hpp"
#include "video_stabilization.hpp"
#include "debug_visualization_lib.hpp"


void video_stabilization_opencv (const CmdParserVideoStabilization& cmdparser)
{
    std::string fileName = cmdparser.input.getValue();

    cv::VideoCapture cap;
    cv::VideoWriter ocvWriter;  // will be used in case if --output is provided
    cap.set(cv::CAP_PROP_CONVERT_RGB, 1);

    cv::Mat frame;
    cv::Mat grayFrame;

    cap.open(fileName.c_str());
    if (!cap.isOpened())
    {
        std::cerr << "[ ERROR ] Cannot open input file: " << fileName << "\n";
        std::exit(1);
    }

    if(cmdparser.output.isSet())
    {
        IntelVXSample::openVideoWriterByCapture(cap, ocvWriter, cmdparser.output.getValue());
    }

    cv::Mat prevFrame;
    int iFrame = 0;
    int maxFrames = cmdparser.max_frames.getValue();

    TrackingStateOpenCV trackingState = createTrackingStateOpenCV();

    PERFPROF_REGION_DEFINE(Frame);
    PERFPROF_REGION_DEFINE(ReadFrame);
    PERFPROF_REGION_DEFINE(ProcessFrame);
    PERFPROF_REGION_DEFINE(calcOpticalFlowPyrLK);

    for (; !(maxFrames && iFrame >= maxFrames); ++iFrame)
    {
        PERFPROF_REGION_AUTO(Frame);

        {
            PERFPROF_REGION_AUTO(ReadFrame);
            cap >> frame;
            if (frame.empty())
            {
                break;
            }
        }

        PERFPROF_REGION_BEGIN(ProcessFrame);


        cv::Size frameRect = frame.size();

        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        if (prevFrame.empty())
        {
            // This is called for the first time
            prevFrame = grayFrame.clone();
        }

        // vector from prev to cur
        vector<cv::Point2f> prevPoints, curPoints;
        vector<uchar> status;
        vector<float> err;

        cv::goodFeaturesToTrack(
            prevFrame,
            prevPoints, 
            cmdparser.max_corners.getValue(), 
            cmdparser.strength_thresh.getValue(),
            cmdparser.min_distance.getValue(),
            cv::noArray(),
            cmdparser.block_size.getValue(),
            true,   // useHarrisDetector
            cmdparser.sensitivity.getValue()
        );

        unsigned int optical_flow_window = cmdparser.optical_flow_window.getValue();

        PERFPROF_REGION_BEGIN(calcOpticalFlowPyrLK);
        cv::calcOpticalFlowPyrLK(
            prevFrame,
            grayFrame,
            prevPoints,
            curPoints,
            status,
            err,
            cv::Size(optical_flow_window, optical_flow_window),
            cmdparser.pyramid_levels.getValue() - 1,  // OpenCV counts from 0
            cv::TermCriteria(
                cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                cmdparser.optical_flow_iterations.getValue(),
                cmdparser.optical_flow_epsilon.getValue()
            )
        );
        PERFPROF_REGION_END(calcOpticalFlowPyrLK);

        if(cmdparser.debug_output.isSet())
        {
            drawDebugVisualization(frame, prevPoints, curPoints, status);
        }

        cv::Mat_<double> curTransform(2, 3);
        estimateTransformOpenCV(prevPoints, curPoints, status, frameRect, trackingState, curTransform);

        cv::Mat warpedFrame;
        cv::warpAffine(frame, warpedFrame, curTransform, frame.size());

        prevFrame = grayFrame.clone();
        prevFrame = grayFrame.clone();

        PERFPROF_REGION_END(ProcessFrame);

        IntelVXSample::logger(1) << "Number of tracked points: " << prevPoints.size() << std::endl;

        if(!cmdparser.no_show.isSet())
        {
            cv::imshow("warpedFrame", warpedFrame);
            cv::imshow("Input Frame", frame);
        }

        if(ocvWriter.isOpened())
        {
            ocvWriter.write(warpedFrame);
        }

        IntelVXSample::logger(1) << "Frame: " << iFrame << std::endl;

        if (!cmdparser.no_show.isSet())
        {
            int key = cv::waitKey(cmdparser.frame_wait.getValue()) & 0xff;
            if(key == 27)
            {
                break;
            }
        }
    }

    releaseTrackingStateOpenCV(&trackingState);
}

