/*
        Copyright 2018 Intel Corporation.
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


kernel void censustransform (

    // The kernel signature is completely defined by param_types array passed to
    // vxAddDeviceKernelIntel function in host C code. Each OpenVX parameter is translated
    // to one or multiple arguments of OpenCL kernel.
    // Even if the kernel's body doesn't use all these arguments, they should be defined here
    // anyway, because OpenVX run-time relies on the order and specific number of parameters,
    // to set them correctly when calling this kernel and translating OpenVX parameters.

    // OpenVX kernel 0-th parameter has type vx_image, it is mapped to these 5 OpenCL kernel arguments
    // This is input VX_DF_IMAGE_S16 image

    global const uchar* inImgPtr,
    unsigned int        widthInImg,        // width of the input image
    unsigned int        heightInImg,       // height of the input image
    unsigned int        pixelStrideInImg,  // pixel stride in bytes
    unsigned int        rowPitchInImg,     // row stride in bytes

    // OpenVX kernel 1-st parameter has type vx_image, it is mapped to these 5 OpenCL kernel arguments
    // This is output VX_DF_IMAGE_U8 image

    global uchar*       outImgPtr,
    unsigned int        widthOutImg,       // width of the output image
    unsigned int        heightOutImg,      // height of the output image
    unsigned int        pixelStrideOutImg, // pixel stride in bytes
    unsigned int        rowPitchOutImg     // row stride in bytes
)
{
    int x = get_global_id(0);   // 0..(widthInImg/VX_OPENCL_WORK_ITEM_XSIZE-1)
    int y = get_global_id(1);   // 0..(heightInImg/VX_OPENCL_WORK_ITEM_YSIZE-1)

    // coordinates of a top-left pixel in output image area for processing by one WI
    // some of the pixels from WORK_ITEM_XSIZE x WORK_ITEM_YSIZE area are not valid on the border
    // they will not be processed (look at if's in the code below)
    x *= WORK_ITEM_XSIZE;
    y *= WORK_ITEM_YSIZE;

    int xSize = WORK_ITEM_XSIZE;
    int ySize = WORK_ITEM_YSIZE;

    // partial processing can happen on the right and/or bottom image border
    // the following conditions adjust an image area size dedicated for processing by one WI
    // default area processed by each WI is WORK_ITEM_XSIZE x WORK_ITEM_YSIZE set by the host
    // but when the whole output image size is not evenly divided by WORK_ITEM_XSIZE x WORK_ITEM_YSIZE tiles
    // some WIs on the border will process lesser number of pixels
    if(x + WORK_ITEM_XSIZE > widthOutImg)
    {
        xSize = (int)widthOutImg - x;
    }

    if(y + WORK_ITEM_YSIZE > heightOutImg)
    {
        ySize = (int)heightOutImg - y;
    }

    // if xSize or ySize is equal to zero or less then 0, it means that current WI isn't assigned to
    // any real pixels on the image; so it won't make any contribution to the output image

    int srcOffset = (y+1)*rowPitchInImg + (x+1)*pixelStrideInImg;
    int dstOffset = y*rowPitchOutImg + x*pixelStrideOutImg;

    global const uchar* src = inImgPtr + srcOffset;
    #define readShort(ADDRESS) (*(global const short*)(ADDRESS))


    // handy macro to read source values from global memory by output pixel coordinates YI, XI
    #define readSource(YI, XI) readShort(src + (int)rowPitchInImg*(yi - 1) + (int)pixelStrideInImg*(xi - 1))


    // cached area with all input pixels needed to produce the output pixels in area WORK_ITEM_YSIZE x WORK_ITEM_XSIZE
    short neighborsArea[WORK_ITEM_YSIZE+2][WORK_ITEM_XSIZE+2];

    if(WORK_ITEM_YSIZE == ySize && WORK_ITEM_XSIZE == xSize)
    {
        // 'fast' kernel version for a WI where all pixels from WORK_ITEM_YSIZE x WORK_ITEM_XSIZE area are processed

        // populate neighborArea from source (WORK_ITEM_YSIZE+2) x (WORK_ITEM_XSIZE+2) pixels are read
        for(int yi = 0; yi < WORK_ITEM_YSIZE+2; ++yi)
            for(int xi = 0; xi < WORK_ITEM_XSIZE+2; ++xi)
            {
                neighborsArea[yi][xi] = readSource(yi, xi);
            }

        // core census transform filter, loop over WORK_ITEM_YSIZE x WORK_ITEM_XSIZE area
        for(int yi = 1; yi < WORK_ITEM_YSIZE+1; ++yi)
            for(int xi = 1; xi < WORK_ITEM_XSIZE+1; ++xi)
            {
                short central = neighborsArea[yi][xi];
                uchar result = 0;

                if(neighborsArea[yi-1][xi-1] >= central)result |= (1u << 7);
                if(neighborsArea[yi-1][  xi] >= central)result |= (1u << 6);
                if(neighborsArea[yi-1][xi+1] >= central)result |= (1u << 5);
                if(neighborsArea[  yi][xi-1] >= central)result |= (1u << 4);
                if(neighborsArea[  yi][xi+1] >= central)result |= (1u << 3);
                if(neighborsArea[yi+1][xi-1] >= central)result |= (1u << 2);
                if(neighborsArea[yi+1][  xi] >= central)result |= (1u << 1);
                if(neighborsArea[yi+1][xi+1] >= central)result |= (1u << 0);

                outImgPtr[dstOffset + (int)rowPitchOutImg*(yi-1) + (int)pixelStrideOutImg*(xi-1)] = result;
            }
    }
    else
    {
        // 'flexible' kernel version for WIs where not all pixels from WORK_ITEM_YSIZE x WORK_ITEM_XSIZE area are processed
        // it differs from 'fast' version in the loop limits: they use variables xSize and ySize instead of constants WORK_ITEM_XSIZE
        // WORK_ITEM_YSIZE.

        // populate neighborArea from source (ySize+2) x (xSize+2) pixels are read
        for(int yi = 0; yi < ySize+2; ++yi)
            for(int xi = 0; xi < xSize+2; ++xi)
            {
                neighborsArea[yi][xi] = readSource(yi, xi);
            }

        // core census transform filter, loop over ySize x xSize area
        for(int yi = 1; yi < ySize+1; ++yi)
            for(int xi = 1; xi < xSize+1; ++xi)
            {
                short central = neighborsArea[yi][xi];
                uchar result = 0;

                if(neighborsArea[yi-1][xi-1] >= central)result |= (1u << 7);
                if(neighborsArea[yi-1][  xi] >= central)result |= (1u << 6);
                if(neighborsArea[yi-1][xi+1] >= central)result |= (1u << 5);
                if(neighborsArea[  yi][xi-1] >= central)result |= (1u << 4);
                if(neighborsArea[  yi][xi+1] >= central)result |= (1u << 3);
                if(neighborsArea[yi+1][xi-1] >= central)result |= (1u << 2);
                if(neighborsArea[yi+1][  xi] >= central)result |= (1u << 1);
                if(neighborsArea[yi+1][xi+1] >= central)result |= (1u << 0);

                outImgPtr[dstOffset + (int)rowPitchOutImg*(yi-1) + (int)pixelStrideOutImg*(xi-1)] = result;
            }
    }
}

