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


#ifndef _INTEL_OPENVX_SAMPLE_MOTION_DETECTION_CMDOPTIONS_HPP_
#define _INTEL_OPENVX_SAMPLE_MOTION_DETECTION_CMDOPTIONS_HPP_

#include <string>

#include <intel/vx_samples/cmdparser.hpp>


// Collection of command-line knobs for video stabilization sample
class CmdParserMotionDetection : public CmdParserWithHelp
{
public:

    CmdParserMotionDetection (int argc, const char** argv);

    // For detailed description for each option,
    // please refer to the constructor definition in cpp file.

    CmdOption<std::string> input;
    CmdOption<std::string> output;
    CmdOptionNoShow no_show;
    CmdOptionFrameWait frame_wait;
    CmdOption<unsigned int> ct_enable;//enable camera tampering nodes
    CmdOption<float> ct_ratio_threshold;//camera tampering threshold
    CmdOption<float> ct_scale;     //camera tampering image scale
    CmdOption<unsigned int> max_frames;
    CmdOption<unsigned int> threshold;
    CmdOption<unsigned int> merge;
    CmdOption<unsigned int> scale;
    CmdOption<std::string> hetero_config;

    virtual void parse ();
};


#endif
