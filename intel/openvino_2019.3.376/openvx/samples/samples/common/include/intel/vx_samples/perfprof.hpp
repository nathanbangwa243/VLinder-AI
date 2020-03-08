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


#ifndef _VX_INTEL_SAMPLE_PERFPROF_HPP_
#define _VX_INTEL_SAMPLE_PERFPROF_HPP_

#if INTEL_SAMPLE_PERFPROF_ITT
#include <ittnotify.h>
#endif
#if INTEL_SAMPLE_PERFPROF_STDOUT_DEBUG
#include "basic.hpp"
#include <vector>
#include <limits>
#endif

#if defined(INTEL_SAMPLE_USE_OPENVX)
#include <intel/vx_samples/helper.hpp>
#endif

#include <iostream>

namespace IntelVXSample
{

#if defined(INTEL_SAMPLE_USE_OPENVX)
void printNodePerformance (vx_node node, const char* name);
void printBegOneNodeAtTimeline (vx_node node);
void printEndOneNodeAtTimeline (vx_node node);
void printNodesAtTimeline (vx_node* nodes, size_t nNodes);
void drawNodesAtTimeline (vx_node* nodes, size_t nNodes, const char** names=NULL);
void drawNodesAtTimeline (vx_node* nodes, size_t nNodes, const std::string* strNames);
#endif

#if INTEL_SAMPLE_PERFPROF_ITT
__itt_domain* getIttDomain ();
#endif

#if INTEL_SAMPLE_PERFPROF_STDOUT_DEBUG || INTEL_SAMPLE_PERFPROF_ITT

class Timer
{
private:
    // declare private copy constructor and = to disable Timer copy
    Timer(const Timer& t);
    Timer& operator=(const Timer& t);
    

public:

    // Functor base for calculation of derived metrics
    struct MetricCalc
    {
        virtual double operator() (double timeInSeconds) const = 0;
    };
    
    const std::string m_nameBackup; // saved metric name
    const char* m_pName;    // m_nameBackup.c_str for C-based APIs and backward compatibility of the code
    const char* m_pUnits;

    typedef enum
    {
        SampleNone = 0,
        SampleDetails,
        SampleDetailsSeries,
    } DefaultSamplingMethod;

    #if INTEL_SAMPLE_PERFPROF_STDOUT_DEBUG
    void addUnitsInv(const char* pName, double unitsInvScale);
    //array for invers units definition (FPS, GOPS, Images/sec)
    std::vector<std::pair<const char*, MetricCalc*> >   m_UnitsInvVector;

    std::vector<double>* m_pTimes;    // array with all measured times
    std::vector<double>* m_pStarts;   // start timestamps for each measured interval
    int         m_Destroyed; //work variable to avoid any action in second destruction
    double      m_Start;
    double      m_Sum;
    int         m_StopCount;
    int         m_StartCount;

    bool        m_bPrintStatsOnDestroy;

    // Timestamps printed as a part of detailed statistics
    // are convenient to have an origin aligned to time
    // when application starts working.
    // The problem is the time_stamp function returns big values,
    // which be adjusted to be meangfull for the application run.
    // The function below initialize timestamp origin when called first time,
    // then always return this fixed value. It is called by Timer::Timer and
    // the first call set the 0-point for all timestamps generated in an application.
    static double timestampOrigin ();

    #ifdef __linux__
    #endif  /*  #ifdef __linux__    */

    #endif
    #if INTEL_SAMPLE_PERFPROF_ITT
    __itt_string_handle* m_ITT;
    #endif
    Timer(const char* pName, const char* pUnits = "ms", DefaultSamplingMethod = SampleNone, bool printStatsOnDestroy = true);
    void start();
    void stop();

    // Add a metric value and the current timestamp
    void sample (double value)
    {
        sample(value, time_stamp());
    }

    // Add a metric value with a timestamp
    void sample (double value, double timestamp);

    void printIterations(std::ostream&);
    void printStats(std::ostream&);
    void printStatsCalc (std::ostream& out, const std::string& units, const MetricCalc& metricCalc, bool needUnitsInName);
    void printShortStatsCalc(std::ostream& out, const std::string& units, const MetricCalc& metricCalc, bool needUnitsInName);
    void printShortStats(std::ostream&);
    void togglePrintStatsOnDestroy(bool);

    ~Timer();
};

#define __PERFPROF_TASK_NAME(NAME)      __sample_perfprof_##NAME
#define __PERFPROF_TASK_TYPE            IntelVXSample::Timer
#define __PERFPROF_TASK_REF_TYPE        IntelVXSample::Timer&
#define __PERFPROF_REGION_BEGIN_TASK(TASK) TASK.start();
#define __PERFPROF_REGION_END_TASK(TASK)   TASK.stop();
#define __PERFPROF_REGION_UNITINV_TASK(TASK, UNITNAME, UNITSCALE)   TASK.addUnitsInv(UNITNAME, UNITSCALE);

#define PERFPROF_REGION_DEFINE(NAME)    __PERFPROF_TASK_TYPE __PERFPROF_TASK_NAME(NAME)(#NAME);


#else


#define PERFPROF_REGION_DEFINE(NAME)
#define __PERFPROF_TASK_NAME(NAME)
#define __PERFPROF_REGION_BEGIN_TASK(TASK)
#define __PERFPROF_REGION_END_TASK(TASK)
#define __PERFPROF_REGION_UNITINV_TASK(TASK, UNITNAME, UNITSCALE)


#endif


#define PERFPROF_REGION_BEGIN(NAME)     __PERFPROF_REGION_BEGIN_TASK(__PERFPROF_TASK_NAME(NAME));
#define PERFPROF_REGION_END(NAME)       __PERFPROF_REGION_END_TASK(__PERFPROF_TASK_NAME(NAME));
#define PERFPROF_REGION_UNITINV(NAME, UNITNAME, UNITSCALE)       __PERFPROF_REGION_UNITINV_TASK(__PERFPROF_TASK_NAME(NAME), UNITNAME, UNITSCALE);


#if INTEL_SAMPLE_PERFPROF_ITT || INTEL_SAMPLE_PERFPROF_STDOUT_DEBUG

class PerfProfRegionAuto
{
    __PERFPROF_TASK_REF_TYPE task;

public:

    PerfProfRegionAuto (__PERFPROF_TASK_REF_TYPE _task) :
        task(_task)
    {
        __PERFPROF_REGION_BEGIN_TASK(task);
    }

    ~PerfProfRegionAuto ()
    {
        __PERFPROF_REGION_END_TASK(task);
    }
};

#define __PERFPROF_HELPER_CAT3(A, B, C)      A##B##C
#define __PERFPROF_AUTO_NAME(NAME, SUFFIX)   __PERFPROF_HELPER_CAT3(NAME, _auto_, SUFFIX)
#define PERFPROF_REGION_AUTO(NAME)           IntelVXSample::PerfProfRegionAuto __PERFPROF_AUTO_NAME(NAME, __LINE__)(__PERFPROF_TASK_NAME(NAME));

#else

#define PERFPROF_REGION_AUTO(NAME)

#endif

}

#if defined(INTEL_SAMPLE_USE_OPENVX) && !USE_AMD_OPENVX && !USE_KHRONOS_SAMPLE_IMPL
#endif


#endif

