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

#ifndef _VX_INTEL_SAMPLE_HETERO_HPP_
#define _VX_INTEL_SAMPLE_HETERO_HPP_

#include <string>
#include <map>

#include <VX/vx.h>
#include <VX/vx_intel_volatile.h>

namespace IntelVXSample
{

/// Holds desired targets for nodes in a graph
/** Loads configuration from a file. A file describes which target should be
 *  used for each node in a graph. Nodes are differentiated by names.
 *  Usually a separate instance of object of this class should be created
 *  to hold configuration for a particular graph.
 */
class HeteroScheduleConfig
{
    // Available targets; filled by one or several calls to setAvailableTarget
    typedef std::map<std::string, vx_enum> Targets;
    
    Targets targets;

    // Map from user-defined node name to one of target names
    // Name of a target may not be included into targets yet.
    std::map<std::string, std::string> nodeTargets;
    
public:
    
    HeteroScheduleConfig (const std::string& fileName = "");
    
    ~HeteroScheduleConfig ();
    
    /// Add available targets with user-defined names, one by one
    /** The names are user-defined, used in a config file.
        There is alway 'default' target name which doesn't have vx_target_intel defined.
        Using default target means, that tartget will not be set. */
    void addAvailableTarget (const std::string& targetName, vx_enum target);
    
    vx_enum getTargetByNodeName (const std::string& nodeName);
    
    void assignTargetForNode (const std::string& nodeName, vx_node node);
    
    const Targets& getTargets ()
    {
        return targets;
    }
    
    void pupulateSupportedTargets();
};

}


#endif
