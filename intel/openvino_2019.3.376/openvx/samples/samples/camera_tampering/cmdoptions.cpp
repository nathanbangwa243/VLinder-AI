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
        "ctd_md.mp4"
    ),

    output(
        *this,
        0,
        "output",
        "<file name>",
        "Output file name",
        ""
    ),

    no_show(*this),

    frame_wait(*this),

    ct_enable(
        *this,
        0,
        "ct_enable",
        "<0 or 1>",
        "Enable camera tampering detection",
        1
    ),

    ct_ratio_threshold(
        *this,
        0,
        "ct_ratio_threshold",
        "<float>",
        "> 0 and < 1, tampering ratio threshold",
        0.01
    ),

    ct_scale(
        *this,
        0,
        "ct_scale",
        "<float>",
        " > 0 and <=1, A scale of gray image to original image",
        1
    ),

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
        "hetero.config.default.txt"
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
