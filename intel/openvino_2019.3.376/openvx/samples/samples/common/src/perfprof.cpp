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
#include <algorithm>
#include <vector>
#include <stdio.h>
#include <cmath>

#include <intel/vx_samples/perfprof.hpp>
#include <intel/vx_samples/basic.hpp>

#if INTEL_SAMPLE_USE_OPENCV
//include opencv headers
#include <opencv2/opencv.hpp>
#endif

#if INTEL_SAMPLE_USE_OPENVX && !USE_AMD_OPENVX && !USE_KHRONOS_SAMPLE_IMPL
#endif

namespace IntelVXSample
{

#if defined(INTEL_SAMPLE_USE_OPENVX)


void printNodePerformance (vx_node node, const char* name)
{
    vx_perf_t perf;
    vx_status status = vxQueryNode(node, VX_NODE_PERFORMANCE, &perf, sizeof(perf));
    CHECK_VX_STATUS(status);

    std::cout << "Performance for node: " << name << std::endl;
    std::cout
        << "    tmp = " << perf.tmp << std::endl
        << "    beg = " << perf.beg << std::endl
        << "    end = " << perf.end << std::endl
        << "    sum = " << perf.sum << std::endl
        << "    avg = " << perf.avg << std::endl
#if defined(USE_KHRONOS_SAMPLE_IMPL)
        << "    min = " << perf.min << std::endl
        << "    max = " << perf.max << std::endl
#endif
    ;

}


void printBegOneNodeAtTimeline (vx_node node)
{
    vx_perf_t perf;
    vx_status status = vxQueryNode(node, VX_NODE_PERFORMANCE, &perf, sizeof(perf));
    CHECK_VX_STATUS(status);

    std::cout << perf.beg << '\t';
}


void printEndOneNodeAtTimeline (vx_node node)
{
    vx_perf_t perf;
    vx_status status = vxQueryNode(node, VX_NODE_PERFORMANCE, &perf, sizeof(perf));
    CHECK_VX_STATUS(status);

    std::cout << perf.end << '\t';
}


void printNodesAtTimeline (vx_node* nodes, size_t nNodes)
{
    for(size_t i = 0; i < nNodes; ++i)
        printBegOneNodeAtTimeline(nodes[i]);
    std::cout << '\n';
    for(size_t i = 0; i < nNodes; ++i)
        printEndOneNodeAtTimeline(nodes[i]);
    std::cout << '\n';
}

void drawNodesAtTimeline (vx_node* nodes, size_t nNodes, const std::string* strNames)
{
    std::vector<const char*>  names;
    for(int i=0; i<nNodes && strNames; ++i)
        names.push_back(strNames[i].c_str());
    drawNodesAtTimeline (nodes, nNodes, strNames ? &names[0] : NULL );
}

void drawNodesAtTimeline (vx_node* nodes, size_t nNodes, const char** names)
{
#if INTEL_SAMPLE_USE_OPENCV
    int W = 640;            // width of outout image
    int HBar = 10;          // height of each horizontal bar
    int HText = 15;         // height of place for text (distance between bars)
    int H = HBar+HText;
    std::vector<vx_perf_t>   perfs(nNodes);
    vx_uint64   TMax=0,TMin=0xFFFFFFFFFFFFFFFF;
    double      TDiff;

    for(size_t i = 0; i < nNodes; ++i)
    {
        vx_perf_t perf;
        vx_status status = vxQueryNode(nodes[i], VX_NODE_PERFORMANCE, &(perfs[i]), sizeof(vx_perf_t));
        CHECK_VX_STATUS(status);

        // Work-around for a bug in run-time that sets to 0 beg field
        // for vx_perf_t object queries for the very first node in a graph
        // Check if it is zero and ignore the first node.
        if(perfs[i].beg == 0)
        {
            continue;
        }

        TMax=std::max(TMax,perfs[i].end);
        TMin=std::min(TMin,perfs[i].beg);
    }
    TDiff = (double)(TMax-TMin);
    // create matrix to draw
    cv::Mat out(H*nNodes,W,CV_8UC3);
    out.setTo(cv::Scalar(0,0,0));
    //draw each kernel as horizontal filled rectangle and draw name is exist
    for(size_t i = 0; i < nNodes; ++i)
    {
        // Work-around for a bug in run-time that sets to 0 beg field
        // for vx_perf_t object queries for the very first node in a graph
        // Check if it is zero and ignore the first node.
        if(perfs[i].beg == 0)
        {
            continue;
        }

        double T0 = (double)(perfs[i].beg-TMin);
        double T1 = (double)(perfs[i].end-TMin);
        int x = (int)((T0*W)/TDiff);
        int w = (int)(((T1-T0)*W)/TDiff);
        int y = (int)(H*i);

        cv::putText(
            out,
            names ? names[i] : "Node " + to_str(i),
            cv::Point(x,y+HText),
            cv::FONT_HERSHEY_PLAIN,
            1,
            cv::Scalar(255,255,255)
        );
        cv::rectangle(out,cv::Rect(x,y+HText,w,HBar),cv::Scalar(255,0,0),-1);
        cv::line(out,cv::Point(x,y+2),cv::Point(x,y+H),cv::Scalar(0,0,255));
        cv::line(out,cv::Point(x+w,y+HText),cv::Point(x+w,y+H),cv::Scalar(0,0,255));

    }
    // show image on screen
    cv::imshow("drawNodesAtTimeline",out);
#endif
}
#endif      /* defined(USE_SAMPLE_COMMON_VX_FUNC) */

#if INTEL_SAMPLE_PERFPROF_ITT

__itt_domain* getIttDomain ()
{
    // WARNING! Still not thread safe?
    static __itt_domain* ittDomain = __itt_domain_create("com.intel.openvx-samples");
    return ittDomain;
}

#endif


#if INTEL_SAMPLE_PERFPROF_STDOUT_DEBUG || INTEL_SAMPLE_PERFPROF_ITT

Timer::Timer(const char* pName, const char* pUnits, DefaultSamplingMethod ds, bool printStatsOnDestroy) :
    m_nameBackup(pName),
    m_pName(m_nameBackup.c_str()),
    m_pUnits(pUnits),
    m_bPrintStatsOnDestroy(printStatsOnDestroy)
{
#if INTEL_SAMPLE_PERFPROF_STDOUT_DEBUG
    m_Destroyed = 0;
    m_Sum = 0;
    m_Start = -1;
    m_StopCount = 0;
    m_StartCount = 0;
    m_pTimes = 0;
    m_pStarts = 0;

    if(SampleDetailsSeries ==ds || getenv("INTEL_SAMPLE_PERFPROF_DETAILS_SERIES"))
    {
        m_pTimes = new std::vector<double>;
        m_pStarts = new std::vector<double>;
        timestampOrigin();
    }
    else if(SampleDetails == ds || getenv("INTEL_SAMPLE_PERFPROF_DETAILS"))
    {
        m_pTimes = new std::vector<double>;
    }

#endif
#if INTEL_SAMPLE_PERFPROF_ITT
    m_ITT = __itt_string_handle_create(m_pName);
#endif
};

void Timer::togglePrintStatsOnDestroy(bool bFlag)
{
    m_bPrintStatsOnDestroy = bFlag;
}

struct MetricMS : public Timer::MetricCalc
{
    virtual double operator() (double timeInSeconds) const
    {
        return 1000*timeInSeconds;  // seconds -> ms
    }
};

struct MetricInv : public Timer::MetricCalc
{
    double m_multiplier;

    MetricInv (double multiplier) : m_multiplier(multiplier) {}
    
    virtual double operator() (double timeInSeconds) const
    {
        return m_multiplier/timeInSeconds;
    }
};

void Timer::addUnitsInv(const char* pName, double unitsInvScale)
{
    m_UnitsInvVector.push_back(std::pair<const char*, MetricCalc*>(pName, new MetricInv(unitsInvScale)));
}

void Timer::start()
{
#if INTEL_SAMPLE_PERFPROF_ITT
    __itt_task_begin(IntelVXSample::getIttDomain(), __itt_null, __itt_null, m_ITT);
#endif
#if INTEL_SAMPLE_PERFPROF_STDOUT_DEBUG
    m_Start = time_stamp();
    m_StartCount++;
#endif
}


void Timer::stop()
{
#if INTEL_SAMPLE_PERFPROF_STDOUT_DEBUG
    double stop = time_stamp();
#endif
#if INTEL_SAMPLE_PERFPROF_ITT
    __itt_task_end(IntelVXSample::getIttDomain());
#endif
#if INTEL_SAMPLE_PERFPROF_STDOUT_DEBUG
    if( m_Start < 0 )
    {
        std::cout << "[ WARNING ] stop counter without start counter for " << m_pName << std::endl;
    }
    else
    {
        double T = (double)(stop-m_Start);
        sample(T, m_Start);
        m_Start = -1; // deinit start to avoid wrong times
        m_StopCount++;
    }
#endif
}


// Add a metric value with a timestamp
void Timer::sample (double value, double timestamp)
{
    m_Sum += value;

    if(m_pTimes)
    {
        m_pTimes->push_back(value);
    }
    if(m_pStarts)
    {
        m_pStarts->push_back(timestamp);
    }
}

void Timer::printIterations(std::ostream& out)
{
    // print header for the table
    out << "[ PERFPROF | Metric Name | Iteration | ";
    if (m_pStarts)
            out << "Start Timestamp | ";
    out << "Value ]\n";

    std::streamsize oldPrecision = out.precision();
    out.precision(std::numeric_limits<double>::max_digits10);

    int     N = m_pTimes->size();
    // print all measured times in order of measurements,
    // each time at one line
    for(int i = 0; i < N; ++i)
    {
        out
            << "[ PERFPROF | "
            << m_pName << " | "
            << i + 1 << " | ";

        if (m_pStarts)
            out << 1000 * (*m_pStarts)[i] - 1000 * timestampOrigin() << " ms | ";

        out
            << 1000*(*m_pTimes)[i] << " " << m_pUnits << " "
            << "]\n"
        ;
    }

    out.precision(oldPrecision);
}


void Timer::printStatsCalc (std::ostream& out, const std::string& units, const MetricCalc& metricCalc, bool needUnitsInName)
{
    int     N = m_pTimes->size();
    //calc geomean
    double  logSum = 0;
    int     logCount = 0;
    std::vector<double> values(N);
    for(int i = 0; i < N; ++i)
    {
        double metricValue = metricCalc((*m_pTimes)[i]);
        values[i] = metricValue;
        if(metricValue > 0)
        {
            logSum += std::log(metricValue);
            logCount++;
        }
    }
    
    // Note 'mean' is not always arithmetic mean here for the target metric.
    // It depends on how metricCalc is implemented: it will be arithmetic mean for not inversed
    // time metrics (such as ms), and it would be harmonic mean for inversed metrics (such as FPS).
    out << (N ? metricCalc(m_Sum/N) : 0) << " (mean " << units << "), ";
    out << (logCount ? exp(logSum/logCount) : 0) << " (gmean " << units << "), [";

    // calc different quantiles

    std::sort(values.begin(), values.end());
    const int   quantiles[] = {0,10,25,50,75,90,100};
    for(int i=0; i < sizeof(quantiles)/sizeof(quantiles[0]); ++i)
    {// iterate over quantiles
        double  val;
        int     q = quantiles[i];
        int     ti = (q*N) / 100;
        if( (q*N) % 100 )
        {
            val = values[ti];
        }
        else
        {// quantile is somewhere between 2 values
            val = 0.5 * (values[(ti<N) ? ti : (N-1)] +
                            values[(ti>0) ? (ti-1) : 0]);
        }
        out << val;
        if(q==50) out << " (median)|";
        else if(q==0) out << " (min)|";
        else if(q==100) out << " (max)] for ";
        else out << " (Q" << q << ")|";
    }
    out << m_pName << " collected by "<< N <<" samples" << std::endl;
}


void Timer::printStats (std::ostream& out)
{
    printStatsCalc(out, m_pUnits, MetricMS(), false);

    for(size_t k = 0; k < m_UnitsInvVector.size(); k++)
    {
        const MetricCalc& metricCalc = *m_UnitsInvVector[k].second;
        std::string units = m_UnitsInvVector[k].first;
        printStatsCalc(out, units, metricCalc, true);
    }    
}


void Timer::printShortStatsCalc(std::ostream& out, const std::string& units, const MetricCalc& metricCalc, bool needUnitsInName)
{
    std::streamsize oldPrecision = out.precision();
    out
        << std::fixed << std::setprecision(3)
        << metricCalc(m_Sum/m_StopCount) << " " << units << " by "
        << m_pName + (needUnitsInName ? "(" + units + ")" : std::string()) << " averaged by "
        << m_StopCount << " samples"
        << std::endl;
   out.precision(oldPrecision);
}


void Timer::printShortStats(std::ostream& out)
{
    std::streamsize oldPrecision = out.precision();
    
    printShortStatsCalc(out, m_pUnits, MetricMS(), false);
    
    for(size_t k = 0; k < m_UnitsInvVector.size(); k++)
    {
        const MetricCalc& metricCalc = *m_UnitsInvVector[k].second;
        std::string units = m_UnitsInvVector[k].first;
        printShortStatsCalc(out, units, metricCalc, true);
    }
    
    out.precision(oldPrecision);
}


Timer::~Timer()
{
#if INTEL_SAMPLE_PERFPROF_STDOUT_DEBUG
    if(!m_Destroyed++)
    {//this is workaround to avoid any action in second destruction call
        if(m_StartCount != m_StopCount)
        {
            std::cout
                << "[ WARNING ] start counter (" << m_StartCount << ") != stop counter ("
                << m_StopCount<< ") for " << m_pName
                << std::endl
            ;
        }

        if(m_StopCount || m_pTimes && !m_pTimes->empty())
        {
            if(m_pTimes && m_pTimes->size()>0)
            {
                assert(m_StopCount == m_pTimes->size());

                if(m_pStarts)
                {
                    assert(m_pStarts->size() == m_pTimes->size());

                    if(m_bPrintStatsOnDestroy)
                        printIterations(std::cout);
                }

                if (m_bPrintStatsOnDestroy)
                    printStats(std::cout);

                delete m_pTimes;
                m_pTimes = 0;
                delete m_pStarts;
                m_pStarts = 0;
            }
            else
            {// short statistics
                if (m_bPrintStatsOnDestroy)
                    printShortStats(std::cout);
            }
        }
        
        for(size_t i = 0; i < m_UnitsInvVector.size(); ++i)
        {
            delete m_UnitsInvVector[i].second;
        }
    }
#endif
}

#endif


#if INTEL_SAMPLE_PERFPROF_STDOUT_DEBUG
double Timer::timestampOrigin ()
{
    // WARNING! Still not thread safe?
    static double origin = time_stamp();
    return origin;
}
#endif

#ifdef __linux__
#endif      /* ifdef __linux__ */

}

#if INTEL_SAMPLE_USE_OPENVX && !USE_AMD_OPENVX && !USE_KHRONOS_SAMPLE_IMPL
#endif


