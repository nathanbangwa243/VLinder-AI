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


#ifndef _INTEL_OPENVX_SAMPLE_VIDEO_STABILIZATION_CMDOPTIONS_HPP_
#define _INTEL_OPENVX_SAMPLE_VIDEO_STABILIZATION_CMDOPTIONS_HPP_

#include <string>

#include <intel/vx_samples/cmdparser.hpp>


// Collection of command-line knobs for video stabilization sample
class CmdParserVideoStabilization : public CmdParserWithHelp
{
public:

    CmdParserVideoStabilization (int argc, const char** argv);
    
    // For detailed description for each option,
    // please refer to the constructor definition in cpp file.

    CmdOption<std::string> input;
    CmdOption<size_t> max_frames;

    CmdOption<std::string> impl;
    CmdEnum<std::string> impl_openvx;
    CmdEnum<std::string> impl_opencv;
    
    CmdOption<bool> no_virtual;
    CmdOptionNoShow no_show;
    CmdOptionOutputVideo output;
    CmdOptionDebugOutput debug_output;

    CmdOption<int> diagnostics;
    CmdOptionFrameWait frame_wait;
    CmdOption<float> strength_thresh;
    CmdOption<unsigned int> min_distance;
    CmdOption<float> sensitivity;

    CmdOption<unsigned int> block_size;
    CmdEnum<unsigned int> block_size_3;
    CmdEnum<unsigned int> block_size_5;
    CmdEnum<unsigned int> block_size_7;

    CmdOption<unsigned int> max_corners;

    CmdOption<unsigned int> optical_flow_window;
    CmdOption<unsigned int> pyramid_levels;
    CmdOption<unsigned int> optical_flow_iterations;
    CmdOption<float> optical_flow_epsilon;
    CmdOption<std::string> hetero_config;
    virtual void parse ();

};


#endif  // end of the include guard
