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

#include "cmdoptions.hpp"

#include <intel/vx_samples/helper.hpp>


#ifdef _MSC_VER
#pragma warning (push)
#pragma warning (disable : 4355)    // 'this': used in base member initializer list
#endif

CmdParserMotionDetection::CmdParserMotionDetection (int argc, const char** argv) :

    CmdParser(argc, argv),
    CmdParserWithHelp(argc, argv),

    input(
        *this,
        'i',
        "input",
        "<file name>",
        "Input file name",
        "VID_640x360.mp4"
    ),

    output(*this),

    no_show(*this),

    frame_wait(*this),

    max_frames(
        *this,
        0,
        "max-frames",
        "<integer>",
        "Maximum number of frames to process from input file. Zero value means all frames from the file are processed.",
        0
    ),

    threshold(
        *this,
        0,
        "threshold",
        "<integer>",
        "Size filtering threshold, value of 0 means 1 / 100 of total pixels in the image is used as threshold",
        0
    ),

    merge(
        *this,
        0,
        "merge",
        "<0 or 1>",
        "A flag to indicate if overlapping rectangles will be merged or not",
        0
    ),
    scene_adaption(
        *this,
        'a',
        "scene_adaption",
        "<integer>",
        "Adaption to scene object change. Recommended [50-200], set small value for fast motion, set large value for smooth motion",
        150
    ),

    scale(
        *this,
        0,
        "scale",
        "<0 or 1>",
        "A flag to indicate if input image can be scaled down to save computation or not",
        0
    ),

    hetero_config(
        *this,
        0,
        "hetero-config",
        "<file name>",
        "Name of config file for fine-grained control over node assignment for targets. "
            "If no file provided, then there will no any assignment made for any node.",
        ""
    )
{
}


void CmdParserMotionDetection::parse ()
{
    CmdParserWithHelp::parse();

    if(input.getValue().empty())
    {
        IntelVXSample::logger(1) << "[ INFO ] Input file name must be specified "<< "\n";
    }
}
