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


#include <cassert>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <iterator>

#include <intel/vx_samples/cmdparsercfgfile.hpp>

void processLine(const string& cfgLine, vector<string>& vConfig)
{
    if (cfgLine.size() < 1)
        return;

    size_t cm_idx = cfgLine.find_first_not_of(" \n\r\t");
    if (cm_idx != string::npos && cfgLine[cm_idx] == '#')       //line is commented out
        return;

    std::istringstream li(cfgLine);
    std::copy(std::istream_iterator<string>(li), std::istream_iterator<string>(), std::back_inserter(vConfig));
}

bool CmdParserWithConfigFile::loadCfgFromFile(const string& cfgFname)
{
    std::ifstream file(cfgFname);

    if (!file.good())
        return false;

    unsigned line_num = 0;
    string cfgLine;
    vector<string> vConfig;

    do
    {
        getline(file, cfgLine);
        line_num++;

        processLine(cfgLine, vConfig);
    } while (!file.eof());

    argvConfig = vConfig;

    return true;
}

bool CmdParserWithConfigFile::loadCfgFromLine(const string& cfgLine)
{
    vector<string> vConfig;

    processLine(cfgLine, vConfig);

    if (vConfig.size())
    {
        argvConfig = vConfig;
        return true;
    }

    return false;
}

void CmdParserWithConfigFile::setConfig(const vector<string>& _cfg)
{
    if(_cfg.size()<1)
    {
        std::cout << "Test arguments are not set." << std::endl;
        return;
    }

    argvConfig = _cfg;
}

void CmdParserWithConfigFile::reset()
{
    CmdParserWithHelp::reset();

    if (m_pPrev)
        m_as = m_pPrev->m_as;
    else
        std::fill(m_as.begin(), m_as.end(), AS_BLANK);
}

void CmdParserWithConfigFile::parse()
{
    const string unknown_opt_err_str("Error - unrecognized option: ");
    const string unknown_opt_warn_str("Warning: Unrecognized option: ");

    //try command line options first
    for (
        int cur_arg_index = 1;
        cur_arg_index < m_argc;
        /* do not increment; incremented in an option parse call */
        )
    {
        int old_cur_arg_index = cur_arg_index;

        if (AS_PARSED == m_as[cur_arg_index])
        {
            cur_arg_index++;
            continue;
        }

        for (CmdOptionBasic* pOption : m_options)
        {
            cur_arg_index = pOption->parse(cur_arg_index, m_argc, m_argv);
            if (cur_arg_index > old_cur_arg_index) //option recognized its argument(s)
            {
                //mark argument(s) as parsed
                if (pOption != &help)
                    std::fill(m_as.begin() + old_cur_arg_index, m_as.begin() + cur_arg_index, AS_PARSED);
                break;
            }
        }

        // no option can recognize the current argument
        if (cur_arg_index == old_cur_arg_index)
        {
            m_as[cur_arg_index] = AS_UNKNOWN;

            if (!m_ignoreUnknownTokens)
            {
                //raise error if ignore flag isn't set
                throw CmdParser::Error(unknown_opt_err_str + m_argv[cur_arg_index] + "\n");
            }
            else
            {
                // WARNING! Temporary feature. To be redesigned in the future.
                //if both ignore and warn flags are set, then print warning
                if (m_warnOnUnknownTokens)
                    std::cerr << unknown_opt_warn_str + m_argv[cur_arg_index] << std::endl;
                cur_arg_index++;
            }
        }
    }

    //init status items for arguments from config file
    if(m_argc + argvConfig.size() > m_as.size())
        m_as.resize(m_argc + argvConfig.size(), AS_BLANK);

    //if config file was loaded, then repeat just the same parsing loop as above for them
    if (argvConfig.size())
    {
        //transform to const char* is required by CmdOption parsing
        std::vector<const char*> argvConfigPtrs(argvConfig.size());
        std::transform(argvConfig.begin(), argvConfig.end(), argvConfigPtrs.begin(), [](const std::string& s) { return s.c_str(); });

        for (
            int cur_arg_index = 0;
            cur_arg_index < argvConfigPtrs.size();
            )
        {
            int old_cur_arg_index = cur_arg_index;

            if (AS_BLANK != m_as[cur_arg_index+m_argc])
            {
                cur_arg_index++;
                continue;
            }

            for (CmdOptionBasic* pOption : m_options)
            {
                cur_arg_index = pOption->parse(cur_arg_index, argvConfigPtrs.size(), argvConfigPtrs.data(), false);

                if ( cur_arg_index > old_cur_arg_index )
                {
                    if(pOption != &help)
                        std::fill(m_as.begin() + old_cur_arg_index + m_argc, m_as.begin() + cur_arg_index + m_argc, AS_PARSED);
                    break;
                }
            }

            if (cur_arg_index == old_cur_arg_index)
            {
                m_as[cur_arg_index + m_argc] = AS_UNKNOWN;

                if (!m_ignoreUnknownTokens)
                    throw CmdParser::Error(unknown_opt_err_str + argvConfig[cur_arg_index] + "\n");
                else
                {
                    if (m_warnOnUnknownTokens)
                        std::cerr << unknown_opt_warn_str + argvConfig[cur_arg_index] << std::endl;
                    cur_arg_index++;
                }
            }
        }
    }

    //complete parsing and mark option alternatives if exist
    for (CmdOptionBasic* pOption : m_options)
        pOption->finishParsing();

    //in case if help option was present and no flag set, print help
    if (help.isSet() && !m_helpSuppress)
    {
        printUsage(std::cout);
    }
}
