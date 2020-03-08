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
#include <string>
#include <cassert>
#include <fstream>
#include <streambuf>

#include <intel/vx_samples/basic.hpp>
#include <intel/vx_samples/helper.hpp>
#include <intel/vx_samples/perfprof.hpp>


namespace
{

int globalLoggerLevel;

class NullBuffer : public std::streambuf
{
public:

    int overflow (int c)
    {
        return c;
    }
};

NullBuffer nullBuffer;
std::ostream nullStream(&nullBuffer);

}


namespace IntelVXSample
{

const char* vxStatusToStr (vx_status e)
{
#define VX_STATUS_TO_STR_ENTRY(E) case E: return #E;
    switch(e)
    {
        VX_STATUS_TO_STR_ENTRY(VX_STATUS_MIN)
        VX_STATUS_TO_STR_ENTRY(VX_ERROR_REFERENCE_NONZERO)
        VX_STATUS_TO_STR_ENTRY(VX_ERROR_MULTIPLE_WRITERS)
        VX_STATUS_TO_STR_ENTRY(VX_ERROR_GRAPH_ABANDONED)
        VX_STATUS_TO_STR_ENTRY(VX_ERROR_GRAPH_SCHEDULED)
        VX_STATUS_TO_STR_ENTRY(VX_ERROR_INVALID_SCOPE)
        VX_STATUS_TO_STR_ENTRY(VX_ERROR_INVALID_NODE)
        VX_STATUS_TO_STR_ENTRY(VX_ERROR_INVALID_GRAPH)
        VX_STATUS_TO_STR_ENTRY(VX_ERROR_INVALID_TYPE)
        VX_STATUS_TO_STR_ENTRY(VX_ERROR_INVALID_VALUE)
        VX_STATUS_TO_STR_ENTRY(VX_ERROR_INVALID_DIMENSION)
        VX_STATUS_TO_STR_ENTRY(VX_ERROR_INVALID_FORMAT)
        VX_STATUS_TO_STR_ENTRY(VX_ERROR_INVALID_LINK)
        VX_STATUS_TO_STR_ENTRY(VX_ERROR_INVALID_REFERENCE)
        VX_STATUS_TO_STR_ENTRY(VX_ERROR_INVALID_MODULE)
        VX_STATUS_TO_STR_ENTRY(VX_ERROR_INVALID_PARAMETERS)
        VX_STATUS_TO_STR_ENTRY(VX_ERROR_OPTIMIZED_AWAY)
        VX_STATUS_TO_STR_ENTRY(VX_ERROR_NO_MEMORY)
        VX_STATUS_TO_STR_ENTRY(VX_ERROR_NO_RESOURCES)
        VX_STATUS_TO_STR_ENTRY(VX_ERROR_NOT_COMPATIBLE)
        VX_STATUS_TO_STR_ENTRY(VX_ERROR_NOT_ALLOCATED)
        VX_STATUS_TO_STR_ENTRY(VX_ERROR_NOT_SUFFICIENT)
        VX_STATUS_TO_STR_ENTRY(VX_ERROR_NOT_SUPPORTED)
        VX_STATUS_TO_STR_ENTRY(VX_ERROR_NOT_IMPLEMENTED)
        VX_STATUS_TO_STR_ENTRY(VX_FAILURE)
        VX_STATUS_TO_STR_ENTRY(VX_SUCCESS)
        default: return "UNKNOWN VX ERROR CODE";
    }
#undef VX_STATUS_TO_STR_ENTRY
}


bool isLoggerEnabled (int localLevel)
{
   return localLevel <= globalLoggerLevel;
}


std::ostream& logger (int localLevel)
{
    if(isLoggerEnabled(localLevel))
    {
        return std::cout;
    }
    else
    {
        return nullStream;
    }
}


void setLoggerGlobalLevel (int level)
{
    globalLoggerLevel = level;
}


int getLoggerGlobalLevel ()
{
    return globalLoggerLevel;
}


void loggerPause (int localLevel)
{
    if(isLoggerEnabled(localLevel))
    {
        logger(localLevel) << "[ PAUSED ]\n";
        std::cin.get();
    }
}

#if INTEL_SAMPLE_USE_OPENCV
#if INTEL_SAMPLE_USE_OVX_1_0_1
cv::Mat mapAsMat (vx_image image, vx_enum usage)
#else
cv::Mat mapAsMat (vx_image image, vx_enum usage, vx_map_id *map_id)
#endif
{
    // This function is provided for illustrative purposes only,
    // it is not universal for all possible vx_image objects
    vx_status status;

    vx_rectangle_t rect;
    status = vxGetValidRegionImage(image, &rect);
    CHECK_VX_STATUS(status);

    logger(1)
        << "mapAsMat::vxGetValidRegionImage: (rect.start_x = " << rect.start_x
        << ", rect.start_y = " << rect.start_y
        << ", rect.end_x = " << rect.end_x
        << ", rect.end_y = " << rect.end_y << ")\n";

    vx_imagepatch_addressing_t addr;
    void* ptr = 0;
    #if INTEL_SAMPLE_USE_OVX_1_0_1
    status = vxAccessImagePatch(image, &rect, 0, &addr, &ptr, usage);
    #else
    status = vxMapImagePatch(image, &rect, 0, map_id, &addr, &ptr, usage, VX_MEMORY_TYPE_HOST, 0);
    #endif
    CHECK_VX_STATUS(status);

    vx_df_image imageFormat;
    #if INTEL_SAMPLE_USE_OVX_1_0_1
    status = vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &imageFormat, sizeof(imageFormat));
    #else
    status = vxQueryImage(image, VX_IMAGE_FORMAT, &imageFormat, sizeof(imageFormat));
    #endif
    CHECK_VX_STATUS(status);
    int type;

    // This function is written for illustrative purposes, so it doesnt' support plenty of
    // image formats, but only two: VX_DF_IMAGE_U8 and VX_DF_IMAGE_RGB.
    switch(imageFormat)
    {
        case VX_DF_IMAGE_U8:
            assert(addr.stride_x == 1);
            type = CV_8UC1;
            break;
        case VX_DF_IMAGE_RGB:
            assert(addr.stride_x == 3);
            type = CV_8UC3;
            break;
        default:
            cerr << "[ ERROR ] Unsupported image format passed to mapAsMat: " << imageFormat << "\n";
            exit(1);
            break;
    }

    return cv::Mat(rect.end_y - rect.start_y, rect.end_x - rect.start_x, type, ptr, addr.stride_y);
}

#if INTEL_SAMPLE_USE_OVX_1_0_1
void unmapAsMat (vx_image image, cv::Mat mat)
#else
void unmapAsMat (vx_image image, cv::Mat mat, vx_map_id map_id)
#endif
{
#if INTEL_SAMPLE_USE_OVX_1_0_1
    // This function is provided for illustrative purposes only,
    // it is not universal for all possible vx_image objects

    vx_rectangle_t rect;
    vx_status status = vxGetValidRegionImage(image, &rect);
    CHECK_VX_STATUS(status);

    logger(1)
        << "mapAsMat::vxGetValidRegionImage: (rect.start_x = " << rect.start_x
        << ", rect.start_y = " << rect.start_y
        << ", rect.end_x = " << rect.end_x
        << ", rect.end_y = " << rect.end_y << ")\n";

    vx_imagepatch_addressing_t addr;

    // These should be the same values as returned by vxAccessImagePatch
    addr.dim_x = mat.cols;
    addr.dim_y = mat.rows;

    // recognize only those types that are supported in mamAsMat
    switch(mat.type())
    {
        case CV_8UC1:
            addr.stride_x = 1;
            break;
        case CV_8UC3:
            addr.stride_x = 3;
            break;
        default:
            cerr << "[ ERROR ] Unsupported cv::Mat::type passed to unmapAsMat: " << mat.type() << "\n";
            exit(1);
            break;
    }

    addr.stride_y = mat.step;
    addr.scale_x = VX_SCALE_UNITY;
    addr.scale_y = VX_SCALE_UNITY;
    addr.step_x = 1;
    addr.step_y = 1;

    status = vxCommitImagePatch(image, &rect, 0, &addr, mat.data);
#else
    vx_status status = vxUnmapImagePatch(image, map_id);
#endif
    CHECK_VX_STATUS(status);
    mat.data = 0;
}

#endif


void testImage1 (vx_image image)
{
    vx_status status;

    vx_rectangle_t rect;
    status = vxGetValidRegionImage(image, &rect);
    CHECK_VX_STATUS(status);

    logger(1)
        << "testImage1::vxGetValidRegionImage: (rect.start_x = " << rect.start_x
        << ", rect.start_y = " << rect.start_y
        << ", rect.end_x = " << rect.end_x
        << ", rect.end_y = " << rect.end_y << ")\n";

    vx_imagepatch_addressing_t addr;
    void* ptr = 0;
    #if INTEL_SAMPLE_USE_OVX_1_0_1
    status = vxAccessImagePatch(image, &rect, 0, &addr, &ptr, VX_WRITE_ONLY);
    #else
    vx_map_id map_id;
    status = vxMapImagePatch(image, &rect, 0, &map_id, &addr, &ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);
    #endif
    CHECK_VX_STATUS(status);
    assert(rect.start_x == 0 && rect.start_y == 0);

    const unsigned int period = 16;

    for (size_t y = 0; y < addr.dim_y; y += addr.step_y)
    {
        for (size_t x = 0; x < addr.dim_x; x += addr.step_x)
        {
            //vx_uint8 pixel = rand_index(256);//(x%period + y%period)*(256/period/2);
            vx_uint8 pixel = (unsigned(x+y*0.4)%period + unsigned(0.3*y+x)%period)*(256/period/2) - rand_index(50);
            logger(4) << (int)pixel << " ";
            vx_uint8* ptr2 = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &addr);
            for(size_t z = 0; z < addr.stride_x; ++z)
            {
                ptr2[z] = pixel + rand_index(20);
            }
        }
    }

    #if INTEL_SAMPLE_USE_OVX_1_0_1
    status = vxCommitImagePatch(image, &rect, 0, &addr, ptr);
    #else
    status = vxUnmapImagePatch(image, map_id);
    #endif
    CHECK_VX_STATUS(status);
}



std::string readTextFile (const std::string& fileName)
{
    std::ifstream file(fileName, std::ios::binary);
    if(!file)
    {
        std::cerr << "[ ERROR ] Cannot open file " << fileName << "\n";
        throw SampleError("Cannot open file " + fileName);
    }

    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

#if INTEL_SAMPLE_USE_OPENCV

void openVideoWriterByCapture (cv::VideoCapture& ocvCapture, cv::VideoWriter& ocvWriter, const std::string& fileName, const int in_width, const int in_height)
{
    std::cout <<  "Open  file " << fileName << " for writing" << std::endl;

    int width;
    int height;
    if(in_width&&in_height)
    {
        width = in_width;
        height = in_height;
    }
    else
    {
        width = static_cast<int>(ocvCapture.get(cv::CAP_PROP_FRAME_WIDTH));
        height = static_cast<int>(ocvCapture.get(cv::CAP_PROP_FRAME_HEIGHT));
    }

    int fourcc =static_cast<int>(ocvCapture.get(cv::CAP_PROP_FOURCC));
    int fps = static_cast<int>(ocvCapture.get(cv::CAP_PROP_FPS));

    ocvWriter.open(fileName, fourcc, fps, cv::Size(width, height));
    if(!ocvWriter.isOpened())
    {
        std::cerr << "[ ERROR ] Can not open output file: " << fileName << std::endl;
        std::exit(1);
    }
}

#endif

}

