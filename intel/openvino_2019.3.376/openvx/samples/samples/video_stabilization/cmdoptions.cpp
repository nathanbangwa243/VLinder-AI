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

CmdParserVideoStabilization::CmdParserVideoStabilization (int argc, const char** argv) :

    CmdParser(argc, argv),
    CmdParserWithHelp(argc, argv),

    input(
        *this,
        'i',
        "input",
        "<file name>",
        "Input video file.",
        "toy_flower.mp4"
    ),

    max_frames(
        *this,
        0,
        "max-frames",
        "<integer>",
        "Number of frames from input video file to be read. May be useful for benchmarking purposes and/or when "
            "input video file is to large to be processed completely. "
            "Zero means that entire files is processed.",
        0
    ),

    impl(
        *this,
        0,
        "impl",
        "",
        "Implementation of the algorithm to be used. Different implementations may give slightly different results."
        ,
        "openvx"
    ),
    impl_openvx(impl, "openvx"),
    impl_opencv(impl, "opencv"),

    no_virtual(
        *this,
        0,
        "no-virtual",
        "",
        "Do not use virtual data objects when OpenVX graph is constructed. Allows to measure "
            "benefits of using virtual data objects for intermediate results in a graph.",
        false
    ),

    no_show(*this),

    debug_output(*this),

    output(*this),

    diagnostics(
        *this,
        0,
        "diagnostics",
        "<non-negative integer>",
        "Level of diagnostics. Zero means minimal diagnostics. "
            "The greater level the more diagnostics is printed. "
            "Diagnostics is printed to stdout stream.",
        0
    ),

    frame_wait(*this),

    strength_thresh(
       *this,
       0,
       "strength-thresh",
       "<float>",
       "OpenVX: Tc value minimum threshold with which to eliminate Harris Corner scores. "
           "OpenCV: Parameter characterizing the minimal accepted quality of image corners. "
           "The parameter value is multiplied by the best corner quality measure. "
           "The corners with the quality measure less than the product are rejected.",
       0.0001
    ),

    min_distance(
       *this,
       0,
       "min-distance",
       "<non-negative float>",
       "Minimum possible Euclidean distance between the returned corners.",
       20
    ),

    sensitivity(
       *this,
       0,
       "sensitivity",
       "<positive float>",
       "Scalar sensitivity threshold k from the Harris-Stephens equation",
       0.04
    ),

    block_size(
       *this,
       0,
       "block-size",
       "",
       "The block window size used to compute the Harris Corner score.",
       3
    ),
    block_size_3(block_size, 3),
    block_size_5(block_size, 5),
    block_size_7(block_size, 7),

    max_corners(
       *this,
       0,
       "max-corners",
       "<positive integer>",
       "Maximum number of corners to return. "
           "If there are more corners than are found, the strongest of them is returned.",
       200
    ),

    optical_flow_window(
       *this,
       0,
       "optical-flow-window",
       "<positive integer>",
       "The size of the window on which to perform Lucas-Kanade optical flow algorithm.",
       21
    ),

    pyramid_levels(
       *this,
       0,
       "pyramid-levels",
       "<positive integer>",
       "Number of levels in pyramid used to run optical flow algorithm on. "
           "One means that no actual pyramid is used, it is just an input image.",
       3
    ),

    optical_flow_iterations(
       *this,
       0,
       "optical-flow-iterations",
       "<positive integer>",
       "Maximum number of iterations to terminate optical flow algorithm.",
       30
    ),

    optical_flow_epsilon(
       *this,
       0,
       "optical-flow-epsilon",
       "<positive float>",
       "Minimum error to terminate optical flow algorithm",
       0.01
    )
    
    ,hetero_config(
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


void CmdParserVideoStabilization::parse ()
{
    CmdParserWithHelp::parse();

    if(input.getValue().empty())
    {
        throw CmdParser::Error("Input file name is required. Use --input FILE to provide input video file name.");
    }

    IntelVXSample::setLoggerGlobalLevel(diagnostics.getValue());
}
