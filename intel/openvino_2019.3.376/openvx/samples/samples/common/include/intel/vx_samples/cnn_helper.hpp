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
#pragma once

#include <iostream>
#include <stdexcept>
#include <sstream>
#include <vector>

#include <limits.h>

#include <VX/vx.h>
#if INTEL_SAMPLE_USE_OVX_1_0_1
#include <VX/vx_ext_intel.h>
#else
#include <VX/vx_khr_nn.h>
#include <VX/vx_intel_volatile.h>
#endif

#if INTEL_SAMPLE_USE_OPENCV
// Include OpenCV headers for input image reading
#include <opencv2/opencv.hpp>
#endif

namespace IntelVXSample
{

// structure to store and sort response for CNN
struct CNNResponse
{
    // 'probability' value from softmax layer
    float m_Prob;
    // index of class
    int m_ID;

    //object bounding box
    float xMin, xMax, yMin, yMax;
};

// structure to store input dimensions of CNN model
struct CNNInputDimensions
{
    int width;
    int height;
    CNNInputDimensions() :
            width(0), height(0)
    {
    }
    CNNInputDimensions(int width, int height)
    {
        // check that dimensions are valid
        if (width <= 0 || height <= 0)
        {
            std::stringstream error;
            error << "input dimensions (" << width << "x" << height << ") are not valid";
            throw std::invalid_argument(error.str());
        }
        this->width = width;
        this->height = height;
    }
};

// CNNResponse comparator to sort in descending order
inline bool compCNNResponse(const CNNResponse& a, const CNNResponse& b)
{
    return a.m_Prob > b.m_Prob;
}

// Function return string with data type name
#if INTEL_SAMPLE_USE_OVX_1_0_1
const char* vxMDDataTypeToStr(vx_df_intel_mddata_e dt);
#else
const char* vxMDDataTypeToStr(vx_type_e dt);
#endif

// Function to convert FP32 into Q78
int16_t float2Q78(float x);

// Function to convert value from Q78 into FP32
float Q782float(int16_t x);

// Function to convert value from vx_intel_md_data data types
// (Q78 or FLOAT32) into float
#if INTEL_SAMPLE_USE_OVX_1_0_1
float mddata2float(char* ptr, vx_df_intel_mddata_e dt);
#else
float mddata2float(char* ptr, vx_type_e dt);
#endif
// Function to convert value from float to vx_intel_md_data data types
#if INTEL_SAMPLE_USE_OVX_1_0_1
void float2mddata(float v, char* ptr, vx_df_intel_mddata_e dt);
#else
void float2mddata(float v, char* ptr, vx_type_e dt);
#endif

#if INTEL_SAMPLE_USE_OVX_1_0_1
vx_df_intel_mddata_e getMDDataType(vx_intel_md_data mddata);
#else
// retrieve data type for vx_intel_md_data
vx_type_e getMDDataType(vx_tensor mddata);
int getMDDataElemSize(vx_type_e dt);
#endif


#if INTEL_SAMPLE_USE_OVX_1_0_1
// this function calculate view for whole mddata
vx_intel_mdview_t getMDDataView(vx_intel_md_data mddata);
#endif

#ifdef DUMP_DATA
#ifdef INTEL_SAMPLE_USE_OVX_1_0_1

// This function damp data into text file.
// It can be used to control internals of vx_intel_md_data
void dumpMDData(vx_intel_md_data mddata, const char* pName);
#else
void dumpMDData(vx_tensor mddata, const char* pName);
#if INTEL_SAMPLE_USE_OPENCV
void dumpTensor(vx_tensor tensorInp, std::string fname);
#endif
#endif
#endif

// This function checks and returns dimensions of vx_intel_md_data
#if INTEL_SAMPLE_USE_OVX_1_0_1
CNNInputDimensions getInputDimensions(vx_intel_md_data mddataInp);
#else
CNNInputDimensions getInputDimensions(vx_tensor tensorInp);
#endif

#if INTEL_SAMPLE_USE_OPENCV
// 1. Convert input image into F32 format.
// 2. Resize result image to CNN input size.
// 3. subtract mean image from resized image. The AlexNet model is supposed
//    that mean image is subtracted from image before training. So the same
//    subtract operation has to be done before classification step.
// 4. Scale result image's values to fit Q78 data range by all activations.
//    For example 1/8 is a good choice for the AlexNet located here
//    http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel
//    But the other trained models may require to choose other scale value
//    to successfully run CNN using Q78 data. This good value can be found experimentally
cv::Mat preProcessImage(cv::Mat image, CNNInputDimensions dims, cv::Mat mean, float scale = 1.0f);
#endif // USE_OPENCV

#if INTEL_SAMPLE_USE_OPENCV
// This function takes OpenCV image as input and initializes vx_intel_md_data from it.
// Convert and store result F32 image values into vx_intel_md_data storage.
#if INTEL_SAMPLE_USE_OVX_1_0_1
void image2IntelMDData(vx_intel_md_data mddataInp, cv::Mat inp32F);
#else
void image2Tensor(vx_tensor tensorInp, std::vector<cv::Mat> inp32F);
#endif
#endif // USE_OPENCV

typedef std::vector<std::vector<CNNResponse>> ResponseVector;
// This function takes vx_intel_md_data and extracts vector of responses for classification CNN model
#if INTEL_SAMPLE_USE_OVX_1_0_1
std::vector<CNNResponse> mdData2Responses(vx_intel_md_data output);
#else
typedef std::vector<std::vector<CNNResponse>> ResponseVector;
ResponseVector tensor2Responses(vx_tensor tensorOut, unsigned int classDimId = 0);
#endif

const int SingleImageBatch = 1;
}
