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


#ifndef _VX_INTEL_SAMPLE_HELPER_HPP_
#define _VX_INTEL_SAMPLE_HELPER_HPP_

#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <cstring>

#if INTEL_SAMPLE_USE_OPENCV
#include <opencv2/opencv.hpp>
//#include <opencv2/videoio/videoio_c.h>
//#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#endif

#include <VX/vx.h>
#include <VX/vxu.h>

#include <intel/vx_samples/basic.hpp>



namespace IntelVXSample
{

// Returns string that holds textual representation for error code e
// as it is defined in OpenVX headers.
const char* vxStatusToStr (vx_status e);


// Check COMMAND to VX_SUCCESS and if it is not, print error message and
// exits the application.
#define CHECK_VX_STATUS(COMMAND)                                \
    {                                                           \
        vx_status __local_status = COMMAND;                     \
        if(__local_status != VX_SUCCESS)                        \
        {                                                       \
            std::cerr                                           \
                << "[ ERROR ] VX API call failed with "         \
                << IntelVXSample::vxStatusToStr(__local_status) << "\n" \
                << "    expression: " << #COMMAND << "\n"       \
                << "    file:       " << __FILE__ << "\n"       \
                << "    line:       " << __LINE__ << "\n";      \
            std::exit(1);                                       \
        }                                                       \
    }

#define CHECK_VX_OBJ(OBJECT) CHECK_VX_STATUS(vxGetStatus((vx_reference)(OBJECT)))

#define RETURN_VX_STATUS(COMMAND)                                \
    {                                                           \
        vx_status __local_status = COMMAND;                     \
        if(__local_status != VX_SUCCESS)                        \
        {                                                       \
            std::cerr                                           \
                << "[ ERROR ] VX API call failed with "         \
                << IntelVXSample::vxStatusToStr(__local_status) << "\n" \
                << "    expression: " << #COMMAND << "\n"       \
                << "    file:       " << __FILE__ << "\n"       \
                << "    line:       " << __LINE__ << "\n";      \
            return __local_status;                                       \
        }                                                       \
    }

#define RETURN_VX_OBJ(OBJECT) RETURN_VX_STATUS(vxGetStatus((vx_reference)(OBJECT)))

#define TRY_VX_STATUS(COMMAND)                                  \
    {                                                           \
        vx_status __local_status = COMMAND;                     \
        if(__local_status != VX_SUCCESS)                        \
        {                                                       \
            throw SampleError(                                  \
                "[ ERROR ] VX API call failed with "            \
                + to_str(IntelVXSample::vxStatusToStr(__local_status)) + "\n" \
                + "    expression: " + to_str(#COMMAND) + "\n"  \
                + "    file:       " + to_str(__FILE__) + "\n"  \
                + "    line:       " + to_str(__LINE__) + "\n");\
        }                                                       \
    }

#define TRY_VX_OBJ(OBJECT) TRY_VX_STATUS(vxGetStatus((vx_reference)(OBJECT)))

inline void VX_CALLBACK errorReceiver (vx_context context, vx_reference ref, vx_status statux, const vx_char string[])
{
    std::cerr << "[ ERROR ] OpenVX error callback: " << string << "\n";
}


std::ostream& logger (int localLevel);

void setLoggerGlobalLevel (int level);
int getLoggerGlobalLevel ();
bool isLoggerEnabled (int level);
void loggerPause (int level);


#if INTEL_SAMPLE_USE_OPENCV
/// An example of a function that maps vx_image object as OpenCV cv::Mat object.
/** This function is written for illustrative purposes, so it doesnt' support plenty of
    image formats, but only two: VX_DF_IMAGE_U8 and VX_DF_IMAGE_RGB. There are
    other restrictions on image that is valid for this function. */

#if INTEL_SAMPLE_USE_OVX_1_0_1
cv::Mat mapAsMat (vx_image image, vx_enum usage);
#else
cv::Mat mapAsMat (vx_image image, vx_enum usage, vx_map_id *map_id);
#endif

/// Unmaps cv::Mat mapped previously by mapAsMat
/** This function is written for illustrative purposes, so it doesnt' support plenty of
    image formats, but only two: VX_DF_IMAGE_U8 and VX_DF_IMAGE_RGB. There are
    other restrictions on image that is valid for this function. */

#if INTEL_SAMPLE_USE_OVX_1_0_1
void unmapAsMat (vx_image image, cv::Mat mat);
#else
void unmapAsMat (vx_image image, cv::Mat mat, vx_map_id map_id);
#endif
#endif


void testImage1 (vx_image image);


#if INTEL_SAMPLE_USE_OVX_1_0_1
/// Automatically do vxCommitArrayRange when exiting the current scope
/** Helps to correctly track access/commit pairs on vx_arrays.
    The array should be already accessed (locked) before calling this
    class constructor. Pass range infromation and mapped pointer to
    the constructor and vxCommitArrayRange will be called automatically
    when this object is destroyed. */
#else
/// Automatically do vxUnmapArrayRange when exiting the current scope
/** Helps to correctly track access/commit pairs on vx_arrays.
    The array should be already accessed (locked) before calling this
    class constructor. Pass range infromation and mapped pointer to
    the constructor and vxUnmapArrayRange will be called automatically
    when this object is destroyed. */
#endif
class AutoCommitArray
{
    vx_array arr;
    vx_size start;
    vx_size end;
    void* ptr;
    vx_map_id map_id;

public:

    AutoCommitArray (vx_array _arr, vx_size _start, vx_size _end, void* _ptr, vx_map_id _map_id):
        arr(_arr),
        start(_start),
        end(_end),
        ptr(_ptr)
        , map_id(_map_id)
    {
    }

    ~AutoCommitArray ()
    {
        vx_status status = vxUnmapArrayRange(arr, map_id);
        if(status != VX_SUCCESS)
        {
            std::cerr << "[ ERROR ] Cannot vxUnmapArrayRange.\n";
            std::exit(1);
        }
    }
};



/// Reads complete text file as a single std::string
/** In case of an open error, prints error message and throws SampleError error to catch */
std::string readTextFile (const std::string& fileName);

#if INTEL_SAMPLE_USE_OPENCV

/// Initialize cv::VideoWriter object using parameters from cv::VideoCapture object
/** In case of an cv::VideoWriter initialization error, it prints error message and terminates the app */
void openVideoWriterByCapture (cv::VideoCapture& ocvCapture, cv::VideoWriter& ocvWriter, const std::string& fileName, const int in_width = 0, const int in_height = 0);

#endif

}


#endif
