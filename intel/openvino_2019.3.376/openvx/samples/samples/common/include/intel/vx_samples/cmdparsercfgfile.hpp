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


#ifndef _INTEL_OPENVX_SAMPLE_CMDPARSER_WITH_CFG_FILE_HPP_
#define _INTEL_OPENVX_SAMPLE_CMDPARSER_WITH_CFG_FILE_HPP_

#include <intel/vx_samples/cmdparser.hpp>

class CmdParserWithConfigFile : public CmdParserWithHelp
{
    vector<string> argvConfig;

    bool m_ignoreUnknownTokens;
    bool m_warnOnUnknownTokens;
    bool m_helpSuppress;    /* suppress help in case when it should be displayed on other parsing layer  */

    //stuff for parser chains
    typedef enum
    {
        AS_BLANK,       /* Noone from parser chain hasn't touched this argument yet */
        AS_PARSED,      /* Someone from parser chain has already parsed this argument - skip it */
        AS_UNKNOWN,     /* Previous parser looked into this argument, but haven't recognized it */
        AS_WRONG        /* This argument has been marked as error by some parser in a chain according to its options and flags */
    } ArgStatus;

    std::vector<ArgStatus> m_as;            /* keeps and passes parsing status for every argument bewteen chained parsers */
    const CmdParserWithConfigFile* m_pPrev; /* pointer to previuos parser in a chain. If NULL, current parser instance is not a part of parser chain */
public:
    CmdParserWithConfigFile(int argc, const char** argv) : CmdParserWithHelp(argc, argv), CmdParser(argc, argv),
        m_ignoreUnknownTokens(false),
        m_warnOnUnknownTokens(false),
        m_helpSuppress(false),
        testCfgDir(*this, 0, "cfgdir", "", "directory containing test configuration files"),
        testCfgFile(*this, 0, "cfg", "", "test configuration file name"),
        masterCfgFile(*this, 0, "master_cfg", "", "run configuration master file name"),
        m_as(argc, AS_BLANK),
        m_pPrev(NULL)
    {}

    CmdParserWithConfigFile(const CmdParserWithConfigFile* pprevParser) :
        CmdParserWithHelp(pprevParser->m_argc, pprevParser->m_argv),
        CmdParser(pprevParser->m_argc, pprevParser->m_argv),
        m_ignoreUnknownTokens(true),
        m_warnOnUnknownTokens(false),
        m_helpSuppress(false),
        testCfgDir(*this, 0, "cfgdir", "", "directory containing test configuration files"),
        testCfgFile(*this, 0, "cfg", "", "test configuration file name"),
        masterCfgFile(*this, 0, "master_cfg", "", "run configuration master file name"),
        m_as(pprevParser->m_as),
        m_pPrev(pprevParser)
    {
    }

    bool loadCfgFromFile(const string& fName);
    bool loadCfgFromLine(const string& line);
    void setConfig(const vector<string>& _cfg);
    virtual void parse();
    virtual void reset();

    void ignoreUnknownTokens(bool val)
    {
        m_ignoreUnknownTokens = val;
    }

    void warnOnUnknownTokens(bool val)
    {
        m_warnOnUnknownTokens = val;
    }

    void helpSuppress(bool val)
    {
        m_helpSuppress = val;
    }

    void setIgnoreOpts()
    {
        m_ignoreUnknownTokens = true;
        m_warnOnUnknownTokens = true;
    }

    CmdOption<string> testCfgDir;       /* path to search test config files when it is set by user */
    CmdOption<string> testCfgFile;      /* full config path for the given test it is set by user */
    CmdOption<string> masterCfgFile;    /* full master config path when it is set by user */

    /* try to locate either master config file
    or test config file, if master isn't set */
    string getFullCfgPath(const string& tname)
    {
        string cfgFname;

        if (testCfgDir.isSet())
            cfgFname = testCfgDir.getValue() + '/';

        if (masterCfgFile.isSet())
            cfgFname += masterCfgFile.getValue();
        else if (testCfgFile.isSet())
            cfgFname += testCfgFile.getValue();
        /* if default workload name is set by caller, take it into account,
        when looking for config file */
        else if (tname.size())
        {
            cfgFname += tname + ".cfg";
        }

        return cfgFname;
    }
};

#endif  /* end of the include guard */