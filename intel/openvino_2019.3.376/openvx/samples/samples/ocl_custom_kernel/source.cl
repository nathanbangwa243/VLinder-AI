// Copyright (2016) Intel Corporation.
//
// The source code, information and material ("Material") contained herein is
// owned by Intel Corporation or its suppliers or licensors, and title to
// such Material remains with Intel Corporation or its suppliers or licensors.
// The Material contains proprietary information of Intel or its suppliers and
// licensors. The Material is protected by worldwide copyright laws and treaty
// provisions. No part of the Material may be used, copied, reproduced, modified,
// published, uploaded, posted, transmitted, distributed or disclosed in any
// way without Intel's prior express written permission. No license under any
// patent, copyright or other intellectual property rights in the Material is
// granted to or conferred upon you, either expressly, by implication,
// inducement, estoppel or otherwise. Any license under such intellectual
// property rights must be express and approved by Intel in writing.
//
// Unless otherwise agreed by Intel in writing, you may not remove or alter
// this notice or any other notice embedded in Materials by Intel or Intel's
// suppliers or licensors in any way.


kernel void oclKernel (

    // The kernel signature is completelly defined by param_types array passed to
    // vxIntelAddDeviceKernel function in host C code. Each OpenVX parameter is translated
    // to one or multiple arguments of OpenCL kernel.
    // Even if the kernel's body doesn't use all these arguments, they should be defined here
    // anyway, because OpenVX run-time relies on the order and specific number of parameters,
    // to set them correctly when calling this kernel and traslating OpenVX parameters.

    // OpenVX kernel 0-th parameter has type vx_image, it is mapped to these 5 OpenCL kernel arguments
    // This is input RGB image

    global const uchar* inImgPtr,
    unsigned int        widthInImg,        // width of the input image
    unsigned int        heightInImg,       // height of the input image
    unsigned int        pixelStrideInImg,  // pixel stride in bytes
    unsigned int        rowPitchInImg,     // row stride in bytes

    // OpenVX kernel 1-st parameter has type vx_image, it is mapped to these 5 OpenCL kernel arguments
    // This is output RGB image

    global uchar*       outImgPtr,
    unsigned int        widthOutImg,       // width of the output image
    unsigned int        heightOutImg,      // height of the output image
    unsigned int        pixelStrideOutImg, // pixel stride in bytes
    unsigned int        rowPitchOutImg     // row stride in bytes
)
{
    int x = get_global_id(0);   // 0..(widthInImg/VX_OPENCL_WORK_ITEM_XSIZE-1)
    int y = get_global_id(1);   // 0..(heightInImg/VX_OPENCL_WORK_ITEM_YSIZE-1)

    // to prepare pixelate image we duplicate one of the source pixels from a square,
    // coordinates of this pixel are calculated as a truncation of x and y
    int srcOffset = (y&~0xF)*rowPitchInImg + (x&~0xF)*pixelStrideInImg;
    int dstOffset = y*rowPitchOutImg + x*pixelStrideOutImg;

    // Assume that inImgPtr and outImgPtr are pointers to 3-bytes per pixel RGB images

    uchar b = inImgPtr[srcOffset + 0];
    uchar g = inImgPtr[srcOffset + 1];
    uchar r = inImgPtr[srcOffset + 2];

    outImgPtr[dstOffset + 0] = b;
    outImgPtr[dstOffset + 1] = g;
    outImgPtr[dstOffset + 2] = r;
}

