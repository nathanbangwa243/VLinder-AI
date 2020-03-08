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
#include <exception>

#include "video_stabilization.hpp"


int main (int argc, const char* argv[])
{
    try
    {
        //Parse command line arguments.
        // See CmdParserVideoStabilization for command line knobs description.
        CmdParserVideoStabilization cmdparser(argc, argv);
        cmdparser.parse();

        if(cmdparser.help.isSet())
        {
            // Immediately exit if user wanted to see the usage information only.
            return 0;
        }

        if(cmdparser.impl_opencv.isSet())
        {
            video_stabilization_opencv(cmdparser);
        }
    
        if(cmdparser.impl_openvx.isSet())
        {
            return video_stabilization_openvx(cmdparser);
        }

        return 0;
    }
    catch(const CmdParser::Error& error)
    {
        cerr
            << "[ ERROR ] In command line: " << error.what() << "\n"
            << "Run " << argv[0] << " -h for usage info.\n";
        return 1;
    }
    catch(const std::exception& error)
    {
        cerr << "[ ERROR ] " << error.what() << "\n";
        return 1;
    }
    catch(...)
    {
        cerr << "[ ERROR ] Unknown/internal exception happened.\n";
        return 1;
    }
}
