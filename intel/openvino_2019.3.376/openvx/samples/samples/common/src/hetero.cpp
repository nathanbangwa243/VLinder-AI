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
#include <fstream>
#include <sstream>
#include <cstdlib>

#include <intel/vx_samples/helper.hpp>
#include <intel/vx_samples/hetero.hpp>


namespace IntelVXSample
{

HeteroScheduleConfig::HeteroScheduleConfig (const std::string& fileName)
{
    if(fileName.empty())
    {
        return; // no file provided -- empty configuration
    }

    std::ifstream configFile(fileName);
    
    if(!configFile)
    {
        std::cerr << "[ ERROR ] Cannot open config file for hetero-configuration: " << fileName << "\n";
        std::exit(1);
    }
    
    // Format of the file is simple:
    //  - each line has a configuration for a single node
    //  - each line has a format: <node_name_without_spaces> SPACE <target_name_without_spaces>
    
    int nLine = 0;
    while(configFile)
    {
        nLine++;
        std::string line;
        std::getline(configFile, line);
        if(!configFile)
        {
            break;
        }
        
        if(line.empty())
        {
            continue;
        }
        
        std::istringstream sline(line);
        std::string nodeName, targetName;
        
        sline >> nodeName >> targetName;

        if(!sline)
        {
            std::cerr << "[ ERROR ] Unexpeted format in file " << fileName << " at line " << nLine << "\n";
            std::exit(1);
        }
        
        auto nodeTargetIter = nodeTargets.find(nodeName);
        if(nodeTargetIter != nodeTargets.end())
        {
            std::cerr
                << "[ ERROR ] Node name " << nodeName << " is duplicated in file " << fileName
                << ". The second time is at line " << nLine << "\n";

            std::exit(1);
        }
        
        nodeTargets[nodeName] = targetName;
    }
}


HeteroScheduleConfig::~HeteroScheduleConfig ()
{
}


void HeteroScheduleConfig::addAvailableTarget (const std::string& targetName, vx_enum target)
{
    auto targetIter = targets.find(targetName);
    
    if(targetIter != targets.end())
    {
        std::cerr << "[ ERROR ] Duplicated target name is trying to be registered: " << targetName << "\n";
        std::exit(1);
    }
    
    targets[targetName] = target;
}


vx_enum HeteroScheduleConfig::getTargetByNodeName (const std::string& nodeName)
{
    auto nodeTargetNameIter = nodeTargets.find(nodeName);
    if(nodeTargetNameIter == nodeTargets.end())
    {
        IntelVXSample::logger(0) << "[ INFO ] Node " << nodeName << " not assigned to any specific target (missed in config)\n";
        return 0;
    }
    else
    {
        const std::string& targetName = nodeTargetNameIter->second;
        auto nodeTargetIter = targets.find(targetName);
        if(nodeTargetIter == targets.end())
        {
            IntelVXSample::logger(0) << "[ INFO ] Node " << nodeName << " not assigned to any specific target (no target " << targetName << ")\n";
            return 0;
        }
        else
        {
            IntelVXSample::logger(0) << "[ INFO ] Node " << nodeName << " is assigned to " << targetName << "\n";
            return nodeTargetIter->second;
        }
    }
}


void HeteroScheduleConfig::assignTargetForNode (const std::string& nodeName, vx_node node)
{
    if(vx_enum target = getTargetByNodeName(nodeName))
    {
        CHECK_VX_STATUS(vxSetNodeTarget(node, target, 0));
    }
}


void HeteroScheduleConfig::pupulateSupportedTargets()
{
    // Add all supported targets regardless of actual availability on the current set
    addAvailableTarget("intel.cpu", VX_TARGET_CPU_INTEL);
    addAvailableTarget("intel.gpu", VX_TARGET_GPU_INTEL);
    addAvailableTarget("intel.ipu", VX_TARGET_IPU_INTEL);

    IntelVXSample::logger(0) << "[ INFO ] Number of supported targets: " << targets.size() << "\n";
    int i = 0;
    for(auto p = targets.begin(); p != targets.end(); ++p, ++i)
        IntelVXSample::logger(0) << "[ INFO ]     Target[" << i << "] name: " << p->first << "\n";
}
    
}
