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
#include "pipelinecontrol.h"

#include <intel/vx_samples/helper.hpp>
#include <intel/vx_samples/cmdparser.hpp>
#include <intel/vx_samples/perfprof.hpp>

using namespace std;

int main(int argc, const char *argv[])
{
    try
    {
        // Parse command line arguments.
        // See CmdParserPipeline for command line knobs description.
        CmdParserPipeline cmdparser(argc, argv);
        cmdparser.parse();

        if (cmdparser.help.isSet())
        {
            // Immediatly exit if user wanted to see the usage information only.
            return 0;
        }

        PipelineControl pipelineControl(&cmdparser);

        //Read input image.
        if( pipelineControl.GetInputImage() < 0)
        {
           throw std::runtime_error(std::string("Error in GetInputImage - input file not found"));
        }

        if( pipelineControl.ScanPreProcess() < 0)
        {
           throw std::runtime_error(std::string("Error in ScanPreProcess"));
        }

        if( cmdparser.edpath.isSet() )
        {
           pipelineControl.AssembleErrorDiffusionGraph();
        }
        else
        if( cmdparser.halftonepath.isSet() )
        {
           pipelineControl.AssembleHalftoneGraph();
        }

        pipelineControl.ExecuteGraph();

        pipelineControl.SaveOutputImage();

        return 0;
    }
    catch (const CmdParser::Error& error)
    {
        cerr
            << "[ ERROR ] In command line: " << error.what() << "\n"
            << "Run " << argv[0] << " -h for usage info.\n";
        return 1;
    }
    catch (const std::exception& error)
    {
        cerr << "[ ERROR ] " << error.what() << "\n";
        return 1;
    }
    catch (...)
    {
        cerr << "[ ERROR ] Unknown/internal exception happened.\n";
        return 1;
    }
}
