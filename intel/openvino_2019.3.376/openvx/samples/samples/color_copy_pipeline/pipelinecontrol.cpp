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

#include <stdio.h>
#include <cassert>
#include <vector>
#include <malloc.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include "pipelinecontrol.h"
#include "lab2cmykparams_17x17x17.h"
#include "lab2cmykparams_33x33x33.h"
#include "cmykhalftone.h"
#include "rgb2labparams_17x17x17.h"
#include "lightnessdarknesscontrast_params.h"

//user vx nodes
#include "vx_user_pipeline_nodes.h"


using namespace std;

//Utility function to save a vx_image to disk in raw format (no header, just contiguous bytestream)
static void SaveImageToDisk(const char *filename, vx_image img)
{
   vx_uint8 *pImage = NULL;
   vx_imagepatch_addressing_t imagepatch;
   vx_uint32 height, width;
   vx_df_image format;
   CHECK_VX_STATUS(vxQueryImage(img, VX_IMAGE_WIDTH, &width, sizeof(width)));
   CHECK_VX_STATUS(vxQueryImage(img, VX_IMAGE_HEIGHT, &height, sizeof(height)));
   CHECK_VX_STATUS(vxQueryImage(img, VX_IMAGE_FORMAT, &format, sizeof(format)));
   vx_rectangle_t rectFullImage = {0, 0, width, height};
   vx_map_id map_id;

   CHECK_VX_STATUS(vxMapImagePatch(img, &rectFullImage, 0, &map_id, &imagepatch, (void **)&pImage, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
   std::ofstream out(filename, std::ios::out | std::ios::binary);

   int valid_bytes_per_line;
   switch(format)
   {
      case VX_DF_IMAGE_U8:
         valid_bytes_per_line = width;
      break;

      case VX_DF_IMAGE_RGB:
         valid_bytes_per_line = width*3;
      break;

      case VX_DF_IMAGE_RGBX:
         valid_bytes_per_line = width*4;
      break;

      default:
         std::cout << "Unsupported image format for writing" << std::endl;
         return;
      break;
   }
   for( int ss = 0; ss < height; ss++)
   {
      out.write( (char *)pImage, valid_bytes_per_line);
      pImage += imagepatch.stride_y;
   }
   out.close();

   CHECK_VX_STATUS(vxUnmapImagePatch(img, map_id));
}

PipelineControl::PipelineControl(CmdParserPipeline *cmdparser)
  : m_cmdparser(cmdparser)
{

    //setup environment variables to improve overall performance
    if( !m_cmdparser->clnontiled.isSet() )
    {
#ifdef _WIN32
      const char* env = "VX_CL_TILED_MODE = 1";
      if(_putenv_s("VX_CL_TILED_MODE=1", "1")!=0)
#else//LINUX
      if (setenv("VX_CL_TILED_MODE", "1", 1) == -1)
#endif
      {
            fprintf(stderr, "Error setup env. VX_CL_TILED_MODE to 1, error = %s\n", strerror(errno));
            exit(1);
        }
    }

    m_inputImageWidth = m_cmdparser->width.getValue();
    m_inputImageHeight = m_cmdparser->height.getValue();

    if( !m_inputImageWidth )
    {
       fprintf(stderr, "Error! Input image width specified with --width is 0\n");
       exit(1);
    }

    if( m_inputImageWidth % 16 )
    {
       fprintf(stderr, "Error! Input image width must be divisible by 16 for this sample\n");
       exit(1);
    }

    if( !m_inputImageHeight )
    {
       fprintf(stderr, "Error! Input image height specified with --height is 0\n");
       exit(1);
    }

    if( !m_cmdparser->staticmempoolsize.getValue() )
    {
       fprintf(stderr, "Error! Static memory pool size specified by --staticmempoolsize is 0\n");
       exit(1);
    }

    switch(m_cmdparser->skewinterptype.getValue() )
    {
       case 0:
          m_warpInterpType = VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR;
       break;

       case 1:
          m_warpInterpType = VX_INTERPOLATION_TYPE_BILINEAR;
       break;

       case 2:
          m_warpInterpType = VX_INTERPOLATION_BICUBIC_INTEL;
       break;

       default:
          fprintf(stderr, "Error! Invalid value specified by --skewinterptype\n");
          fprintf(stderr, "Supported Values:\n");
          fprintf(stderr, "0 - Nearest Neighbor Interpolation\n");
          fprintf(stderr, "1 - BiLinear Interpolation\n");
          fprintf(stderr, "2 - BiCubic Interpolation\n");
          exit(1);
       break;
    }

    m_context = vxCreateContext();
    CHECK_VX_OBJ(m_context);

    //register our logging callback
    vxRegisterLogCallback(m_context, IntelVXSample::errorReceiver, vx_true_e);

    //Load kernels.
    CHECK_VX_STATUS(vxLoadKernels(m_context, "color_copy_pipeline_lib"));

    m_graph = vxCreateGraph(m_context);
    CHECK_VX_OBJ(m_graph);

    //create containers for the 3D LUT parameters
    m_rgb2lab_nodevals17x17x17 = vxCreateArray(m_context, VX_TYPE_UINT8, 17*17*17*3);
    CHECK_VX_OBJ(m_rgb2lab_nodevals17x17x17);

    vx_uint8 pNodeVals17x17x17[17*17*17*3];
    for( int i = 0; i < 17*17*17; i++)
    {
        pNodeVals17x17x17[i*3 + 0] = L_NODE_VALUES_17[i];
        pNodeVals17x17x17[i*3 + 1] = A_NODE_VALUES_17[i];
        pNodeVals17x17x17[i*3 + 2] = B_NODE_VALUES_17[i];
    }

    //add these elements to our vx_array. A copy will happen here.
    CHECK_VX_STATUS(vxAddArrayItems(m_rgb2lab_nodevals17x17x17, 17*17*17*3, pNodeVals17x17x17, 0));

    m_3dlut_nlatticepoints_lab2cmyk = 17;
    const vx_uint8 *PreComputedNodeValsC = C_NODE_VALUES17;
    const vx_uint8 *PreComputedNodeValsM = M_NODE_VALUES17;
    const vx_uint8 *PreComputedNodeValsY = Y_NODE_VALUES17;
    const vx_uint8 *PreComputedNodeValsK = K_NODE_VALUES17;

    if( m_cmdparser->nlatticepoints33.isSet() )
    {
       m_3dlut_nlatticepoints_lab2cmyk = 33;
       PreComputedNodeValsC = C_NODE_VALUES33;
       PreComputedNodeValsM = M_NODE_VALUES33;
       PreComputedNodeValsY = Y_NODE_VALUES33;
       PreComputedNodeValsK = K_NODE_VALUES33;
    }

    int ntableentries = m_3dlut_nlatticepoints_lab2cmyk*m_3dlut_nlatticepoints_lab2cmyk*m_3dlut_nlatticepoints_lab2cmyk;
    m_lab2cmyk_nodevals = vxCreateArray(m_context, VX_TYPE_UINT8, ntableentries*4);
    CHECK_VX_OBJ(m_lab2cmyk_nodevals);
    vx_uint8 *pNodeVals_lab2cmyk = new vx_uint8[ntableentries*4];

    //combine separate C, M, Y, K (planar-interleaved) tables into a
    // single CMYK (pixel-interleaved) table
    for( int i = 0; i < ntableentries; i++)
    {
        pNodeVals_lab2cmyk[i*4 + 0] = PreComputedNodeValsC[i];
        pNodeVals_lab2cmyk[i*4 + 1] = PreComputedNodeValsM[i];
        pNodeVals_lab2cmyk[i*4 + 2] = PreComputedNodeValsY[i];
        pNodeVals_lab2cmyk[i*4 + 3] = PreComputedNodeValsK[i];
    }

    //add these elements to our vx_array. A copy will happen here.
    CHECK_VX_STATUS(vxAddArrayItems(m_lab2cmyk_nodevals, ntableentries*4, pNodeVals_lab2cmyk, 0));

    delete [] pNodeVals_lab2cmyk;

    //Create our input / output images... the ones that are accessible by the user
    // (i.e. non-virtual)
    m_srcImageR = vxCreateImage(m_context, m_inputImageWidth, m_inputImageHeight, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(m_srcImageR);
    m_srcImageG = vxCreateImage(m_context, m_inputImageWidth, m_inputImageHeight, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(m_srcImageG);
    m_srcImageB = vxCreateImage(m_context, m_inputImageWidth, m_inputImageHeight, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(m_srcImageB);

    //Output image is 1bpp CMYK image. There is no support for such image format in OpenVX right now.
    //Will store result in 8bpp VX_DF_IMAGE_RGBX image with size equal to input image size divided by 8.
    vx_uint32 output_width = m_inputImageWidth / 8;

    m_dstImageC = vxCreateImage(m_context, output_width, m_inputImageHeight, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(m_dstImageC);
    m_dstImageM = vxCreateImage(m_context, output_width, m_inputImageHeight, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(m_dstImageM);
    m_dstImageY = vxCreateImage(m_context, output_width, m_inputImageHeight, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(m_dstImageY);
    m_dstImageK = vxCreateImage(m_context, output_width, m_inputImageHeight, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(m_dstImageK);

    //Create a virtual image for the output of Background Suppression.
    m_iBackgroundSuppress_0 = vxCreateVirtualImage(m_graph, m_inputImageWidth, m_inputImageHeight, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(m_iBackgroundSuppress_0);
    m_iBackgroundSuppress_1 = vxCreateVirtualImage(m_graph, m_inputImageWidth, m_inputImageHeight, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(m_iBackgroundSuppress_1);
    m_iBackgroundSuppress_2 = vxCreateVirtualImage(m_graph, m_inputImageWidth, m_inputImageHeight, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(m_iBackgroundSuppress_2);

    //Create a virtual image for the output of Lab2CMYK.
    // The output is CMYK (4-channels), but there is no VX interface enum
    // for CMYK, so we use RGBX instead...
    m_iLab2CMYK = vxCreateVirtualImage(m_graph, m_inputImageWidth, m_inputImageHeight, VX_DF_IMAGE_RGBX);
    CHECK_VX_OBJ(m_iLab2CMYK);

    //Create a virtual image for the output of lightness / darkness / contrast (table lookup)
    m_iLightnessDarknessContrast = vxCreateVirtualImage(m_graph, m_inputImageWidth, m_inputImageHeight, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(m_iLightnessDarknessContrast);

    //Allocate large enough user buffer and provide it to runtime in custom allocator
    m_pMemoryPool = (unsigned char *)malloc(m_cmdparser->staticmempoolsize.getValue());
    if( !m_pMemoryPool )
    {
        fprintf(stderr, "Error Allocating static memory pool of %u bytes\n", m_cmdparser->staticmempoolsize.getValue());
        exit(1);
    }

    //touch the memory to force the OS to actually acquire it.
    // Remember, Linux doesn't actually attach DRAM at malloc time,
    // only once it is used.
    memset(m_pMemoryPool, 0, m_cmdparser->staticmempoolsize.getValue());

    m_currentOffset = 0;

    m_customAllocator.opaque = this;
    m_customAllocator.alloc_ptr = Allocate;
    m_customAllocator.free_ptr = Free;

    m_3dlut_interp_type_lab2cmyk = VX_INTERPOLATION_TRILINEAR_INTEL;
    if( m_cmdparser->tetrainterp.isSet() )
    {
       m_3dlut_interp_type_lab2cmyk = VX_INTERPOLATION_TETRAHEDRAL_INTEL;
    }

    m_correctionMatrix = vxCreateMatrix(m_context, VX_TYPE_FLOAT32, 2, 3);
    CHECK_VX_OBJ(m_correctionMatrix);

    //"Warm" the GPU kernels
    WarmGPUKernels();
}


void *PipelineControl::Allocate(void* opaque, vx_size size)
{
    PipelineControl *pThis = (PipelineControl *)opaque;
    if( pThis->m_currentOffset + size > pThis->m_cmdparser->staticmempoolsize.getValue() )
    {
        fprintf(stderr, "PipelineControl::Allocate: Ran out of dedicated memory. Returning 0\n");
        fprintf(stderr, "  Try increasing the static memory pool size using --staticmempoolsize\n");
        return 0;
    }

    unsigned char *pBuf = &(pThis->m_pMemoryPool[pThis->m_currentOffset]);
    pThis->m_currentOffset += size;

    return pBuf;
}

void PipelineControl::Free(void* opaque, void* data)
{
    //do nothing - m_pMemoryPool will be released in destructor
}

PipelineControl::~PipelineControl()
{
    //release our references
    CHECK_VX_STATUS(vxReleaseImage(&m_srcImageR));
    CHECK_VX_STATUS(vxReleaseImage(&m_srcImageG));
    CHECK_VX_STATUS(vxReleaseImage(&m_srcImageB));
    CHECK_VX_STATUS(vxReleaseArray(&m_lab2cmyk_nodevals));
    CHECK_VX_STATUS(vxReleaseArray(&m_rgb2lab_nodevals17x17x17));
    CHECK_VX_STATUS(vxReleaseGraph(&m_graph));
    CHECK_VX_STATUS(vxReleaseImage(&m_dstImageC));
    CHECK_VX_STATUS(vxReleaseImage(&m_dstImageM));
    CHECK_VX_STATUS(vxReleaseImage(&m_dstImageY));
    CHECK_VX_STATUS(vxReleaseImage(&m_dstImageK));
    CHECK_VX_STATUS(vxReleaseImage(&m_iBackgroundSuppress_0));
    CHECK_VX_STATUS(vxReleaseImage(&m_iBackgroundSuppress_1));
    CHECK_VX_STATUS(vxReleaseImage(&m_iBackgroundSuppress_2));
    CHECK_VX_STATUS(vxReleaseImage(&m_iLab2CMYK));
    CHECK_VX_STATUS(vxReleaseImage(&m_iLightnessDarknessContrast));
    CHECK_VX_STATUS(vxReleaseMatrix(&m_correctionMatrix));
    CHECK_VX_STATUS(vxReleaseContext(&m_context));

    if( m_pMemoryPool ) free(m_pMemoryPool);
}

int PipelineControl::GetInputImage()
{
    vx_image srcImageRGB = vxCreateImage(m_context, m_inputImageWidth, m_inputImageHeight, VX_DF_IMAGE_RGB);
    CHECK_VX_OBJ(srcImageRGB);

    vx_uint8 *pRGB = NULL;
    vx_imagepatch_addressing_t imagepatchRGB;
    //create a vx rectangle object which corresponds to the full image
    vx_rectangle_t rectFullImage = {0, 0, m_inputImageWidth, m_inputImageHeight};
    vx_map_id map_id;
    CHECK_VX_STATUS(vxMapImagePatch(srcImageRGB, &rectFullImage, 0, &map_id, &imagepatchRGB, (void **)&pRGB, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0));

    //open the desired input image
    ifstream inRGB(m_cmdparser->input.getValue(), ios::in | ios::binary);
    if(!inRGB)
    {
        return -1;
    }

    for( int ss = 0; ss < m_inputImageHeight; ss++)
    {
        unsigned char *pTmpRGB = pRGB + ss*imagepatchRGB.stride_y;

        //read 8 bit raw RGB image from disk
        inRGB.read( (char *)pTmpRGB, m_inputImageWidth*3);
    }

    inRGB.close();

    //once we're done reading the input image into our src VX image,
    // 'commit' the patch, telling the engine that we're done writing
    // to the image.
    CHECK_VX_STATUS(vxUnmapImagePatch(srcImageRGB, map_id));

    //Convert the pixel-interleaved RGB input into separate R, G, B planes.
    CHECK_VX_STATUS(vxuChannelSeparateIntel(m_context, srcImageRGB, m_srcImageR, m_srcImageG, m_srcImageB, 0));

    vxReleaseImage(&srcImageRGB);

    return 0;
}

//Common SubGraph - common RGB2LAB and preprocessing for High1, High3 and High6
void PipelineControl::AttachCommonSubGraph()
{
    //create 'virtual' images for output of RGB2Lab
    // This means that we (as the user) are acknowledging that we don't need
    // access to these images, and they can be kept internal to the graph
    // by the VX implementation.
    vx_image iRGB2Lab_0 = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(iRGB2Lab_0);
    vx_image iRGB2Lab_1 = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(iRGB2Lab_1);
    vx_image iRGB2Lab_2 = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(iRGB2Lab_2);

    //Attach the RGB-to-CIELab subgraph
    AttachRGB2LabSubGraph(m_copyInputR, m_copyInputG, m_copyInputB, iRGB2Lab_0, iRGB2Lab_1, iRGB2Lab_2);

    //we are going to apply a symmetrical 7x7 filter to the L* output of RGB2Lab.
    //create a virtual image for the output of the 7x7 filter
    vx_image i7x7Filter = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(i7x7Filter);

    vx_node n7x7Filter;
    if( m_cmdparser->gpusymm7x7_custom.isSet() )
    {
        //Create the actual 7x7 filter node
        if( !m_cmdparser->clnontiled.isSet() )
        {
            //Use tiled version
            n7x7Filter = vxSymm7x7OpenCLTiledNode(m_graph, iRGB2Lab_0, i7x7Filter);
        }
        else
        {   //Use non-tiled version
            n7x7Filter = vxSymm7x7OpenCLNode(m_graph, iRGB2Lab_0, i7x7Filter);
        }
        CHECK_VX_OBJ(n7x7Filter);
    }
    else
    {
        //create / setup the parameters that we will use for 7x7 filter
        vx_array coefficients_array = vxCreateArray(m_context, VX_TYPE_INT32, 10);
        CHECK_VX_OBJ(coefficients_array);
        vx_int32 coefficients[10] = {1140, -118, 526, 290, -236, 64, -128, -5, -87, -7};

        CHECK_VX_STATUS(vxAddArrayItems(coefficients_array, 10, coefficients, 0));

        vx_int32 shift = 10;

        //Create the actual 7x7 filter node
        n7x7Filter = vxSymmetrical7x7FilterNodeIntel(m_graph, iRGB2Lab_0, coefficients_array, shift, i7x7Filter);
        CHECK_VX_OBJ(n7x7Filter);

        //The Symm7x7 node will create it's own border.
        //Set it to a constant value of 255 (white in the case of L*)
        vx_border_t symm7x7borderMode;
        symm7x7borderMode.mode = VX_BORDER_CONSTANT;
        symm7x7borderMode.constant_value.U32 = 255;

        //set the node attribute
        CHECK_VX_STATUS(vxSetNodeAttribute(n7x7Filter, VX_NODE_BORDER, &symm7x7borderMode, sizeof(symm7x7borderMode)));

        if( m_cmdparser->gpusymm7x7.isSet() )
        {
            CHECK_VX_STATUS(vxSetNodeTarget(n7x7Filter, VX_TARGET_GPU_INTEL, 0));
        }
        else if( m_cmdparser->ipusymm7x7.isSet() )
        {
            CHECK_VX_STATUS(vxSetNodeTarget(n7x7Filter, VX_TARGET_IPU_INTEL, 0));
        }
    }

    //Create the actual Background Suppress node
    vx_node nBackgroundSuppress = vxBackgroundSuppressNode(m_graph, i7x7Filter, iRGB2Lab_1, iRGB2Lab_2, m_iBackgroundSuppress_0, m_iBackgroundSuppress_1, m_iBackgroundSuppress_2);
    CHECK_VX_OBJ(nBackgroundSuppress);


    //get the lightness and contrast parameters from the user
    int lightness = m_cmdparser->lightness.getValue();
    int contrast = m_cmdparser->contrast.getValue();

    //Create a vx_lut object to be used within the vxTableLookup node
    vx_lut lightnessdarknesslut = vxCreateLUT(m_context, VX_TYPE_UINT8, 256);
    CHECK_VX_OBJ(lightnessdarknesslut);

    vx_uint8 *pLUT = 0;

    //Access the contents of the created LUT, so that we can write to it.
    vx_map_id map_id;
    CHECK_VX_STATUS(vxMapLUT(lightnessdarknesslut, &map_id, (void **)&pLUT, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0));

    //Write to the accessed LUT, given the lightness and contrast settings.
    //If the user doesn't specify (lightness = 0, contrast = 0),
    // the LUT will default to identity (0:255)
    InitializeLightnessDarknessContrastLUT(lightness, contrast, pLUT);

    //Let the VX system know that we're done writing to the LUT
    CHECK_VX_STATUS(vxUnmapLUT(lightnessdarknesslut, map_id));

    vx_node nLightnessDarknessContrast = vxTableLookupNode(m_graph, m_iBackgroundSuppress_0, lightnessdarknesslut, m_iLightnessDarknessContrast);
    CHECK_VX_OBJ(nLightnessDarknessContrast);

    if( m_cmdparser->gpulut.isSet() )
    {
        CHECK_VX_STATUS(vxSetNodeTarget(nLightnessDarknessContrast, VX_TARGET_GPU_INTEL, 0));
    }

    //Attach the Lab-to-CMYK subgraph
    AttachLab2CMYKSubGraph(m_iLightnessDarknessContrast, m_iBackgroundSuppress_1, m_iBackgroundSuppress_2, m_iLab2CMYK);

    CHECK_VX_STATUS(vxReleaseNode(&nLightnessDarknessContrast));
    CHECK_VX_STATUS(vxReleaseLUT(&lightnessdarknesslut));
    CHECK_VX_STATUS(vxReleaseNode(&nBackgroundSuppress));
    CHECK_VX_STATUS(vxReleaseNode(&n7x7Filter));
    CHECK_VX_STATUS(vxReleaseImage(&i7x7Filter));
    CHECK_VX_STATUS(vxReleaseImage(&iRGB2Lab_2));
    CHECK_VX_STATUS(vxReleaseImage(&iRGB2Lab_1));
    CHECK_VX_STATUS(vxReleaseImage(&iRGB2Lab_0));

    return;
}

//Given a center x,y point, angle, & scale factor, calculate a warp affine transformation matrix
void GetRotationMatrix(vx_float64 center_x, vx_float64 center_y, vx_float64 angle, vx_float64 scale, vx_float32 Matrix[3][2])
{
  const vx_float64 PI = 3.1415926535897932384626433832795;
  angle *= PI/180;

  vx_float64 alpha = cos(angle)*scale;
  vx_float64 beta = sin(angle)*scale;

  Matrix[0][0] = (vx_float32)alpha;
  Matrix[0][1] = (vx_float32)-beta;
  Matrix[1][0] = (vx_float32)beta;
  Matrix[1][1] = (vx_float32)alpha;
  Matrix[2][0] = (vx_float32)((1-alpha)*center_x - beta*center_y);
  Matrix[2][1] = (vx_float32)(beta*center_x + (1-alpha)*center_y);
}

//Given a warp affine transformation matrix, invert it.
// For example, if the passed in transform represents a rotation of 3 degrees,
//  this function will invert the matrix, producing a transformation matrx
//  which would represent a rotation of -3 degrees.
void InvertMatrix(vx_float32 Matrix[3][2])
{
   vx_float32 alpha = Matrix[0][0];
   vx_float32 beta = Matrix[0][1];
   vx_float32 Tx = Matrix[2][0];
   vx_float32 Ty = Matrix[2][1];

   Matrix[1][0] = beta;
   Matrix[0][1] = -beta;

   Matrix[2][0] = -1.0f*Tx*alpha - Ty*beta;
   Matrix[2][1] = -1.0f*Ty*alpha + Tx*beta;
}

//Given input image width/height and rotation (skew) angle,
// calculate the skewed width & height, as well as the
// rotation matrix to use for warpAffine
void GetRotateBound(vx_float64 angle,
                    vx_int32 input_width,
                    vx_int32 input_height,
                    vx_float32 Matrix[3][2],
                    vx_int32 &output_width,
                    vx_int32 &output_height)
{
   vx_float32 center_x = input_width / 2.0f;
   vx_float32 center_y = input_height / 2.0f;

   GetRotationMatrix(center_x, center_y, angle, 1.0f, Matrix);

   vx_float32 cos = fabs(Matrix[0][0]);
   vx_float32 sin = fabs(Matrix[1][0]);

   output_width = (int)(input_height*sin + input_width*cos);
   output_height = (int)(input_height*cos + input_width*sin);

   Matrix[2][0] += (output_width / 2.0f) - center_x;
   Matrix[2][1] += (output_height / 2.0f) - center_y;

   InvertMatrix(Matrix);

}

//Generate a random floating point value in the range: min <= rand <= max
static inline vx_float32 RandFloat(vx_float32 min, vx_float32 max)
{
  return min + (max - min) * rand_uniform_01<vx_float32>();
}

//Utility function to convert an 8bpp image (U8) to a "raw" image of 'bits' (10 or 12) bits,
// while randomizing gain & offset for each pixel, as well as agc
static void Convert8bppImageToRaw(vx_context context, vx_image image8, vx_image imageraw, unsigned int bits,
                                  vx_scalar &vxagc, vx_array &vxgain, vx_array &vxoffset)
{
   vx_uint32 image8height, image8width;
   CHECK_VX_STATUS(vxQueryImage(image8, VX_IMAGE_WIDTH, &image8width, sizeof(image8width)));
   CHECK_VX_STATUS(vxQueryImage(image8, VX_IMAGE_HEIGHT, &image8height, sizeof(image8height)));
   vx_rectangle_t rectImage8 = {0, 0, image8width, image8height};
   vx_imagepatch_addressing_t imagepatch8;
   vx_uint8 *p8 = NULL;
   vx_map_id map_id8;
   CHECK_VX_STATUS(vxMapImagePatch(image8, &rectImage8, 0, &map_id8, &imagepatch8, (void **)&p8, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));

   vx_uint32 rawheight, rawwidth;
   CHECK_VX_STATUS(vxQueryImage(imageraw, VX_IMAGE_WIDTH, &rawwidth, sizeof(rawwidth)));
   CHECK_VX_STATUS(vxQueryImage(imageraw, VX_IMAGE_HEIGHT, &rawheight, sizeof(rawheight)));
   vx_rectangle_t rectRawImage = { 0, 0, rawwidth, rawheight};
   vx_imagepatch_addressing_t imagepatchraw;
   vx_uint8 *pRaw = NULL;
   vx_map_id map_idraw;

   CHECK_VX_STATUS(vxMapImagePatch(imageraw, &rectRawImage, 0, &map_idraw, &imagepatchraw, (void **)&pRaw, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));

   //Generate a random agc (Automatic Gain Control) value
   vx_float32 acgf = RandFloat(0.9, 1.1);
   vxagc = vxCreateScalar(context, VX_TYPE_FLOAT32, &acgf);

   //Create a random gain & offset value per pixel
   vx_float32 *pGain = new vx_float32[image8width];
   vx_float32 *pOffset = new vx_float32[image8width];
   for( int i = 0; i < image8width; i++ )
   {
      vx_float32 randGain = RandFloat(0.9, 3.0);
      pGain[i] = randGain;

      vx_float32 randOffset = RandFloat(-64.0f, 0.0f);
      pOffset[i] = randOffset;
   }

   for( int y = 0; y < image8height; y++)
   {
      unsigned char *pUnpackedLine = p8 + y*imagepatch8.stride_y;
      unsigned char *pPackedLine = pRaw + y*imagepatchraw.stride_y;

      //we will pack 4 values per loop iteration
      for( int x = 0; x < image8width/4; x++ )
      {
         //Create the 10bpp / 12bpp input pixel by taking the "real" 8bpp pixel value, and applying the reverse gain/offset/agc transform
         // The Gain/Offset/AGC node will apply:
         // output8[x] = (input10[x] * gain[x] + offset[x])*agc
         // So we reverse that to create our input10 / input12
         // input10[x] = (output8[x]*1/agc - offset[x]) * (1/gain[x])
         float p0 = (((float)pUnpackedLine[x*4 + 0] * (1.0f/acgf) - pOffset[x*4 + 0]) * (1.0f/pGain[x*4 + 0]));
         float p1 = (((float)pUnpackedLine[x*4 + 1] * (1.0f/acgf) - pOffset[x*4 + 1]) * (1.0f/pGain[x*4 + 1]));
         float p2 = (((float)pUnpackedLine[x*4 + 2] * (1.0f/acgf) - pOffset[x*4 + 2]) * (1.0f/pGain[x*4 + 2]));
         float p3 = (((float)pUnpackedLine[x*4 + 3] * (1.0f/acgf) - pOffset[x*4 + 3]) * (1.0f/pGain[x*4 + 3]));

         if( bits == 10 )
         {
            p0 = (p0 > 1023.0f) ? 1023.0f : p0;
            p1 = (p1 > 1023.0f) ? 1023.0f : p1;
            p2 = (p2 > 1023.0f) ? 1023.0f : p2;
            p3 = (p3 > 1023.0f) ? 1023.0f : p3;
         }
         else
         {
            p0 = (p0 > 4095.0f) ? 4095.0f : p0;
            p1 = (p1 > 4095.0f) ? 4095.0f : p1;
            p2 = (p2 > 4095.0f) ? 4095.0f : p2;
            p3 = (p3 > 4095.0f) ? 4095.0f : p3;
         }

         p0 = (p0 < 0.0f) ? 0.0f : p0;
         p1 = (p1 < 0.0f) ? 0.0f : p1;
         p2 = (p2 < 0.0f) ? 0.0f : p2;
         p3 = (p3 < 0.0f) ? 0.0f : p3;


         unsigned short topack0 = (unsigned short)p0;
         unsigned short topack1 = (unsigned short)p1;
         unsigned short topack2 = (unsigned short)p2;
         unsigned short topack3 = (unsigned short)p3;

         if( bits == 10 )
         {
            //The 10-bit packing scheme we use here is to pack
            // every 4 10-bit pixels into 5 bytes.
            // Byte0 = [Pixel0 0:7]
            // Byte1 = [Pixel1 0:7]
            // Byte2 = [Pixel2 0:7]
            // Byte3 = [Pixel3 0:7]
            // Byte4 = MSB [Pixel0 8:9][Pixel1 8:9][Pixel2 8:9][Pixel3 8:9]

            unsigned char byte0;
            unsigned char byte1;
            unsigned char byte2;
            unsigned char byte3;
            unsigned char byte4;

            byte0 = topack0 & 0xff;
            byte1 = topack1 & 0xff;
            byte2 = topack2 & 0xff;
            byte3 = topack3 & 0xff;
            byte4 = (topack0 & 0x300) >> 2;
            byte4 = byte4 | (topack1 & 0x300) >> 4;
            byte4 = byte4 | (topack2 & 0x300) >> 6;
            byte4 = byte4 | (topack3 & 0x300) >> 8;

            *pPackedLine++ = byte0;
            *pPackedLine++ = byte1;
            *pPackedLine++ = byte2;
            *pPackedLine++ = byte3;
            *pPackedLine++ = byte4;
         }
         else
         if( bits == 12)
         {
            //The 12-bit packing scheme we use here is to pack
            // every 2 12-bit pixels into 3 bytes.
            // Byte0 = [Pixel0 0:7]
            // Byte1 = MSB [Pixel1 8:11][Pixel0 0:3]
            // Byte2 = [Pixel1 4:11]
            unsigned char byte0;
            unsigned char byte1;
            unsigned char byte2;
            unsigned char byte3;
            unsigned char byte4;
            unsigned char byte5;

            byte0 = topack0 & 0x0ff;
            byte1 = (topack0 & 0xf00) >> 8;
            byte1 = byte1 | ((topack1 & 0x00f) << 4);
            byte2 = (topack1 & 0xff0) >> 4;
            byte3 = topack2 & 0x0ff;
            byte4 = (topack2 & 0xf00) >> 8;
            byte4 = byte4 | ((topack3 & 0x00f) << 4);
            byte5 = (topack3 & 0xff0) >> 4;

            *pPackedLine++ = byte0;
            *pPackedLine++ = byte1;
            *pPackedLine++ = byte2;
            *pPackedLine++ = byte3;
            *pPackedLine++ = byte4;
            *pPackedLine++ = byte5;
         }
      }
   }

   CHECK_VX_STATUS(vxUnmapImagePatch(image8, map_id8));
   CHECK_VX_STATUS(vxUnmapImagePatch(imageraw, map_idraw));

   vxgain = vxCreateArray(context, VX_TYPE_FLOAT32, image8width);
   vxAddArrayItems(vxgain, image8width, pGain, 0);

   vxoffset = vxCreateArray(context, VX_TYPE_FLOAT32, image8width);
   vxAddArrayItems(vxoffset, image8width, pOffset, 0);

   delete [] pGain;
   delete [] pOffset;

}

//Setup / Run the ScanPreProcess (Gain / Offset + Skew Correction) graph
int PipelineControl::ScanPreProcess()
{
   //if an SPP option is set
   if( m_cmdparser->sppseparate.isSet() || m_cmdparser->sppconnected.isSet() )
   {
      //make sure both options are not set
      if( m_cmdparser->sppseparate.isSet() && m_cmdparser->sppconnected.isSet() )
      {
         std::cout << "Error! Command line options --sppseparate and --sppconnected are both set" << std::endl;
         return -1;
      }

      //make sure --sppskew or --sppbits is set
      if( !(m_cmdparser->sppskew.isSet() || m_cmdparser->sppbits.isSet()) )
      {
         std::cout << "Error! --sppseparate or --sppconnected is set, but neither option --sppskew or --sppbits is set" << std::endl;
         return -1;
      }

      if( m_cmdparser->sppbits.isSet() &&
         ((m_cmdparser->sppbits.getValue() != 10) && (m_cmdparser->sppbits.getValue() != 12)))
      {
         std::cout << "Error! If --sppbits is set, it must be set to 10 or 12" << std::endl;
         return -1;
      }

      vx_image skewed8bppImageR;
      vx_image skewed8bppImageG;
      vx_image skewed8bppImageB;
      vx_int32 skewedWidth, skewedHeight;
      vx_float32 Matrix[3][2];
      if( m_cmdparser->sppskew.isSet() )
      {
         //If --sppskew is set, we want to take the RGB input image provided by the user,
         // and skew it (rotate it) by the user specified number of degrees, provided by --sppskew
         vx_uint32 inputWidth = m_inputImageWidth;
         vx_uint32 inputHeight = m_inputImageHeight;

         double skewAngle = (double)m_cmdparser->sppskew.getValue();

         //Given input image width/height and rotation (skew) angle,
         // calculate the skewed width & height, as well as the
         // rotation matrix to use for warpAffine
         GetRotateBound(skewAngle, inputWidth, inputHeight, Matrix, skewedWidth, skewedHeight);

         //ensure that the skewed width is divisible by 64, this will increase performance.
         if( (skewedWidth % 64) != 0 )
         {
            skewedWidth = ((skewedWidth/64)+1)*64;
         }

         std::cout << "Skewed Image Size = " << skewedWidth << "x" << skewedHeight << std::endl;

         vx_matrix warpMatrix = vxCreateMatrix(m_context, VX_TYPE_FLOAT32, 2, 3);
         vxWriteMatrix(warpMatrix, Matrix);

         //create non-virtual outputs for the R, G, B skewed output
         skewed8bppImageR = vxCreateImage(m_context, skewedWidth, skewedHeight, VX_DF_IMAGE_U8);
         skewed8bppImageG = vxCreateImage(m_context, skewedWidth, skewedHeight, VX_DF_IMAGE_U8);
         skewed8bppImageB = vxCreateImage(m_context, skewedWidth, skewedHeight, VX_DF_IMAGE_U8);

         vx_graph skewGraph = vxCreateGraph(m_context);
         CHECK_VX_OBJ(skewGraph);

         //warpAffine only supports pixel-interleaved RGB images, so we want to first
         // channel combine the R, G, B planes into a single RGB image
         vx_image iRGB = vxCreateVirtualImage(skewGraph, 0, 0, VX_DF_IMAGE_RGB);
         CHECK_VX_OBJ(iRGB);
         vx_node nCombine = vxChannelCombineNode(skewGraph, m_srcImageR, m_srcImageG, m_srcImageB, 0, iRGB);
         CHECK_VX_OBJ(nCombine);

         //add the skew creation (via warp affine) to the graph
         //Note that it's important that the virtual image output of vxWarpAffine has defined it's specific output
         // width & height.
         vx_image iRGB_skewed = vxCreateVirtualImage(skewGraph, skewedWidth, skewedHeight, VX_DF_IMAGE_RGB);
         CHECK_VX_OBJ(iRGB_skewed);
         vx_node nWarp = vxWarpAffineNode(skewGraph, iRGB, warpMatrix, m_warpInterpType, iRGB_skewed);

         //By default, as per the OpenVX specification, the warp affine implementation has to assume
         // that the user may change the transformation matrix AFTER vxVerifyGraph and/or in-between
         // multiple calls to vxProcessGraph. Since during vxVerifyGraph() is where the output-to-input
         // tile dependencies are set, and the input tile required to produce a given output is totally
         // dependent on the transformation matrix, the default implementation is forced to require the
         // entire input image to produce any output tile. This of course creates a "synchronization"
         // point in the graph where the vxWarpAffine node has to wait for the entire input image to
         // be produced before starting any tasks. To work around this issue, there is an intel extension
         // that can be used to give a hint to the OpenVX compiler/runtime that all parameters set
         // for a given node are final, and won't change post-vxVerifyGraph(). This allows the implementation
         // to set "true" input tile dependencies, which enables pipelining across vxWarpAffine (removes the
         // synchronization point).
         vx_bool static_optimization = vx_true_e;
         CHECK_VX_STATUS( vxSetNodeAttribute(nWarp, VX_NODE_ENABLE_STATIC_PARAMETER_OPTIMIZATION, &static_optimization, sizeof(static_optimization)));

         //we now need to separate the pixel-interleave RGB output from vxWarpAffine into individual
         // R, G, B planes
         vx_node nSeparate = vxChannelSeparateNodeIntel(skewGraph, iRGB_skewed, skewed8bppImageR, skewed8bppImageG, skewed8bppImageB, 0);
         CHECK_VX_OBJ(nSeparate);
         if( m_cmdparser->gpuskew.isSet() )
         {
            CHECK_VX_STATUS(vxSetNodeTarget(nCombine, VX_TARGET_GPU_INTEL, 0));
            CHECK_VX_STATUS(vxSetNodeTarget(nWarp, VX_TARGET_GPU_INTEL, 0));
            CHECK_VX_STATUS(vxSetNodeTarget(nSeparate, VX_TARGET_GPU_INTEL, 0));
         }

         //compile & run the graph producing the skewed 8bpp R, G, B images
         CHECK_VX_STATUS(vxVerifyGraph(skewGraph));
         CHECK_VX_STATUS(vxProcessGraph(skewGraph));
         vxReleaseNode(&nSeparate);
         vxReleaseNode(&nWarp);
         vxReleaseNode(&nCombine);
         vxReleaseImage(&iRGB);
         vxReleaseImage(&iRGB_skewed);
         vxReleaseMatrix(&warpMatrix);


         vxReleaseGraph(&skewGraph);

         //uncomment to save the produced skewed images to disk
      }
      else
      {
         skewed8bppImageR = m_srcImageR;
         skewed8bppImageG = m_srcImageG;
         skewed8bppImageB = m_srcImageB;

         skewedWidth = m_inputImageWidth;
         skewedHeight = m_inputImageHeight;
      }

      vx_graph sppgraph;
      //if the spp subgraph is connected to the front of the copy graph,
      // the images linking the two subgraphs (spp & copy graph) can be virtual
      if( m_cmdparser->sppconnected.isSet() )
      {
         m_copyInputR = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_U8);
         m_copyInputG = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_U8);
         m_copyInputB = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_U8);
         sppgraph = m_graph;
      }
      else
      {
         //otherwise, it needs to be non-virtual. In this case,
         // we can write back into the 8bpp R, G, B images.
         m_copyInputR = m_srcImageR;
         m_copyInputG = m_srcImageG;
         m_copyInputB = m_srcImageB;
         sppgraph = vxCreateGraph(m_context);
      }

      vx_image correctedGainOffsetR;
      vx_image correctedGainOffsetG;
      vx_image correctedGainOffsetB;
      if( m_cmdparser->sppbits.isSet() && ((m_cmdparser->sppbits.getValue() == 10) || (m_cmdparser->sppbits.getValue() == 12)))
      {
         unsigned int bits = m_cmdparser->sppbits.getValue();

         //calcucate the "valid bytes" per line which we will use to create our "raw" image
         vx_uint32 rawwidthbits = skewedWidth * bits;
         vx_uint32 validbytes = rawwidthbits / 8;

         //create "raw" 10 or 12 bit image
         m_rawGainOffsetInputR = vxCreateImage(m_context, validbytes, skewedHeight, VX_DF_IMAGE_U8);
         CHECK_VX_OBJ(m_rawGainOffsetInputR);
         m_rawGainOffsetInputG = vxCreateImage(m_context, validbytes, skewedHeight, VX_DF_IMAGE_U8);
         CHECK_VX_OBJ(m_rawGainOffsetInputG);
         m_rawGainOffsetInputB = vxCreateImage(m_context, validbytes, skewedHeight, VX_DF_IMAGE_U8);
         CHECK_VX_OBJ(m_rawGainOffsetInputB);

         vx_scalar sAgcR, sAgcG, sAgcB;
         vx_array gain_arrayR, gain_arrayG, gain_arrayB;
         vx_array offset_arrayR, offset_arrayG, offset_arrayB;

         Convert8bppImageToRaw(m_context, skewed8bppImageR, m_rawGainOffsetInputR, bits,
                               sAgcR, gain_arrayR, offset_arrayR);
         Convert8bppImageToRaw(m_context, skewed8bppImageG, m_rawGainOffsetInputG, bits,
                               sAgcG, gain_arrayG, offset_arrayG);
         Convert8bppImageToRaw(m_context, skewed8bppImageB, m_rawGainOffsetInputB, bits,
                               sAgcB, gain_arrayB, offset_arrayB);

         vx_node nGainOffsetR;
         vx_node nGainOffsetG;
         vx_node nGainOffsetB;

         if( m_cmdparser->sppskew.isSet() )
         {
            correctedGainOffsetR = vxCreateVirtualImage(sppgraph, 0, 0, VX_DF_IMAGE_U8);
            correctedGainOffsetG = vxCreateVirtualImage(sppgraph, 0, 0, VX_DF_IMAGE_U8);
            correctedGainOffsetB = vxCreateVirtualImage(sppgraph, 0, 0, VX_DF_IMAGE_U8);
         }
         else
         {
            correctedGainOffsetR = m_copyInputR;
            correctedGainOffsetG = m_copyInputG;
            correctedGainOffsetB = m_copyInputB;
         }

         if( bits == 10 )
         {
            nGainOffsetR = vxGainOffset10Node(sppgraph,
                                              m_rawGainOffsetInputR,
                                              correctedGainOffsetR,
                                              gain_arrayR,
                                              offset_arrayR,
                                              sAgcR);
            nGainOffsetG = vxGainOffset10Node(sppgraph,
                                              m_rawGainOffsetInputG,
                                              correctedGainOffsetG,
                                              gain_arrayG,
                                              offset_arrayG,
                                              sAgcG);
            nGainOffsetB = vxGainOffset10Node(sppgraph,
                                              m_rawGainOffsetInputB,
                                              correctedGainOffsetB,
                                              gain_arrayB,
                                              offset_arrayB,
                                              sAgcB);
         }
         else if( bits == 12 )
         {
            nGainOffsetR = vxGainOffset12Node(sppgraph,
                                              m_rawGainOffsetInputR,
                                              correctedGainOffsetR,
                                              gain_arrayR,
                                              offset_arrayR,
                                              sAgcR);
            nGainOffsetG = vxGainOffset12Node(sppgraph,
                                              m_rawGainOffsetInputG,
                                              correctedGainOffsetG,
                                              gain_arrayG,
                                              offset_arrayG,
                                              sAgcG);
            nGainOffsetB = vxGainOffset12Node(sppgraph,
                                              m_rawGainOffsetInputB,
                                              correctedGainOffsetB,
                                              gain_arrayB,
                                              offset_arrayB,
                                              sAgcB);
         }

      }
      else
      {
         correctedGainOffsetR = skewed8bppImageR;
         correctedGainOffsetG = skewed8bppImageG;
         correctedGainOffsetB = skewed8bppImageB;
      }

      m_correctionMatrix = vxCreateMatrix(m_context, VX_TYPE_FLOAT32, 2, 3);
      if( m_cmdparser->sppskew.isSet() )
      {
         //the warpaffine matrix to correct for the skewed angle can be obtained by
         // inverting the one we used to skew the image to begin with.
         InvertMatrix(Matrix);

         //create a vx_matrix from that
         CHECK_VX_STATUS(vxWriteMatrix(m_correctionMatrix, Matrix));

         //warpAffine only supports pixel-interleaved RGB images, so we want to first
         // channel combine the R, G, B planes into a single RGB image
         vx_image iRGB = vxCreateVirtualImage(sppgraph, 0, 0, VX_DF_IMAGE_RGB);
         CHECK_VX_OBJ(iRGB);
         vx_node nCombine = vxChannelCombineNode(sppgraph, correctedGainOffsetR, correctedGainOffsetG, correctedGainOffsetB, 0, iRGB);
         CHECK_VX_OBJ(nCombine);

         //add the skew creation (via warp affine) to the graph
         //Note that it's important that the virtual image output of vxWarpAffine has defined it's specific output
         // width & height.
         vx_image iRGB_corrected = vxCreateVirtualImage(sppgraph, m_inputImageWidth, m_inputImageHeight, VX_DF_IMAGE_RGB);
         CHECK_VX_OBJ(iRGB_corrected);
         vx_node nWarp = vxWarpAffineNode(sppgraph, iRGB, m_correctionMatrix, m_warpInterpType, iRGB_corrected);

         vx_bool static_optimization = vx_true_e;
         CHECK_VX_STATUS( vxSetNodeAttribute(nWarp, VX_NODE_ENABLE_STATIC_PARAMETER_OPTIMIZATION, &static_optimization, sizeof(static_optimization)));

         //we now need to separate the pixel-interleave RGB output from vxWarpAffine into individual
         // R, G, B planes
         vx_node nSeparate = vxChannelSeparateNodeIntel(sppgraph, iRGB_corrected, m_copyInputR, m_copyInputG, m_copyInputB, 0);
         CHECK_VX_OBJ(nSeparate);

         if( m_cmdparser->gpuskew.isSet() )
         {
            CHECK_VX_STATUS(vxSetNodeTarget(nCombine, VX_TARGET_GPU_INTEL, 0));
            CHECK_VX_STATUS(vxSetNodeTarget(nWarp, VX_TARGET_GPU_INTEL, 0));
            CHECK_VX_STATUS(vxSetNodeTarget(nSeparate, VX_TARGET_GPU_INTEL, 0));

         }

         vxReleaseNode(&nSeparate);
         vxReleaseNode(&nWarp);
         vxReleaseNode(&nCombine);
         vxReleaseImage(&iRGB_corrected);
         vxReleaseImage(&iRGB);
      }

      if( m_cmdparser->sppseparate.isSet() )
      {
         int tilewidth = skewedWidth;
         int tileheight = m_cmdparser->tileheight.getValue();

         CHECK_VX_STATUS(vxSetGraphAttribute(sppgraph, VX_GRAPH_TILE_WIDTH_INTEL, (void *)&tilewidth, sizeof(tilewidth)));
         CHECK_VX_STATUS(vxSetGraphAttribute(sppgraph, VX_GRAPH_TILE_HEIGHT_INTEL, (void *)&tileheight, sizeof(tileheight)));

         //Set our custom allocator
         //Reserve own memory pool. Speedups vxVerifyGraph and first call to vxProcessGraph
         CHECK_VX_STATUS(vxSetGraphAttribute(sppgraph, VX_GRAPH_ALLOCATOR_INTEL, (void *)&m_customAllocator, sizeof(m_customAllocator)));

         //Set memory optimization level (performance/memory consumption tradeoff)
         int memfactor_value = 4;
         CHECK_VX_STATUS(vxSetGraphAttribute(sppgraph, VX_GRAPH_MEM_OPTIMIZATION_LEVEL_INTEL, (void *)&memfactor_value, sizeof(memfactor_value)));

         // Validate the Graph
         double startv = time_stamp()*1000.0;
         CHECK_VX_STATUS(vxVerifyGraph(sppgraph));
         double endv = time_stamp()*1000.0;
         std::cout << "Scan Pre-Process Graph: vxVerifyGraph time = " << endv - startv << " ms." << std::endl;

         for (unsigned int i = 0; i < 10; i++)
         {
            double starte = time_stamp()*1000.0;
            CHECK_VX_STATUS(vxProcessGraph(sppgraph));
            double ende = time_stamp()*1000.0;
            std::cout << "Scan Pre-Process Graph: vxProcessGraph time = " << ende - starte << " ms." << std::endl;
         }

         //After vxVerifyGraph is run, memory statistics which show
         // the amount of 'intermediate buffer' storage that
         // was allocated can be gathered.
         vx_size memory_usage_bytes = 0;
         CHECK_VX_STATUS(vxQueryGraph(sppgraph, VX_GRAPH_PROFILE_MEMORY_SIZE_INTEL, &memory_usage_bytes, sizeof(memory_usage_bytes)));
         std::cout << "Scan Pre-Process Graph: Internal Buffers Allocated = " << memory_usage_bytes << " bytes. " << std::endl;

         //Reset the offset of our memory pool back to 0
         m_currentOffset = 0;

         //uncomment to save the output images of this graph (corrected R, G, B) to disk

         vxReleaseGraph(&sppgraph);
      }

      m_tileWidth = skewedWidth;

   }
   else
   {
      m_copyInputR = m_srcImageR;
      m_copyInputG = m_srcImageG;
      m_copyInputB = m_srcImageB;

      m_tileWidth = m_inputImageWidth;
   }
   return 0;
}

void PipelineControl::AssembleHalftoneGraph()
{
    //Assemble common part for high1, high3 and high6
    AttachCommonSubGraph();

    //Create a virtual image which represents our neutral edge mask
    vx_image iNeutralEdgeMask = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(iNeutralEdgeMask);

    //Attach the 'neutral edge detect' sub-graph
    AttachNeutralEdgeDetectSubGraph(m_iLightnessDarknessContrast, m_iBackgroundSuppress_1, m_iBackgroundSuppress_2, iNeutralEdgeMask);

    vx_image iNeutralEdgeMaskUnpacked = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(iNeutralEdgeMaskUnpacked);

    vx_node nUnpackNEdge = vxUnpack1to8NodeIntel(m_graph, iNeutralEdgeMask, iNeutralEdgeMaskUnpacked);
    CHECK_VX_OBJ(nUnpackNEdge);

    vx_image iRemoveFringeEdgeK = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(iRemoveFringeEdgeK);

    vx_image iRemoveFringeC = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(iRemoveFringeC);
    vx_image iRemoveFringeM = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(iRemoveFringeM);
    vx_image iRemoveFringeY = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(iRemoveFringeY);
    vx_image iRemoveFringeK = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(iRemoveFringeK);

    vx_node nRemoveFringe;
    vx_array LtoK_array;
    if(m_cmdparser->gpuremovefringe.isSet())
    {
       //Create the L-to-K 'knots' array that we will use as a parameter for RemoveFringe
       LtoK_array = vxCreateArray(m_context, VX_TYPE_UINT8, 256);


       vx_uint8 LtoK_array_values[256] =
       {
          0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
          0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
          0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
          0xff, 0xff, 0xfe, 0xfd, 0xfc, 0xfc, 0xfb, 0xfa, 0xf9, 0xf9, 0xf8, 0xf7, 0xf6, 0xf6, 0xf5, 0xf4,
          0xf3, 0xf3, 0xf2, 0xf1, 0xf0, 0xf0, 0xef, 0xee, 0xed, 0xed, 0xec, 0xeb, 0xea, 0xea, 0xe9, 0xe8,
          0xe7, 0xe7, 0xe6, 0xe5, 0xe4, 0xe4, 0xe3, 0xe2, 0xe1, 0xe1, 0xe0, 0xdf, 0xde, 0xde, 0xdd, 0xdc,
          0xdb, 0xdb, 0xda, 0xd9, 0xd8, 0xd8, 0xd7, 0xd6, 0xd5, 0xd5, 0xd4, 0xd3, 0xd2, 0xd2, 0xd1, 0xd0,
          0xcf, 0xcf, 0xce, 0xcd, 0xcc, 0xcc, 0xcb, 0xca, 0xc9, 0xc9, 0xc8, 0xc7, 0xc6, 0xc6, 0xc5, 0xc4,
          0xc3, 0xc3, 0xc2, 0xc1, 0xc0, 0xc0, 0xbf, 0xbe, 0xbd, 0xbd, 0xbc, 0xbb, 0xba, 0xba, 0xb9, 0xb8,
          0xb7, 0xb7, 0xb6, 0xb5, 0xb4, 0xb4, 0xb3, 0xb2, 0xb1, 0xb1, 0xb0, 0xaf, 0xae, 0xae, 0xad, 0xac,
          0xab, 0xa9, 0xa6, 0xa3, 0xa1, 0x9e, 0x9b, 0x99, 0x96, 0x93, 0x91, 0x8e, 0x8b, 0x89, 0x86, 0x83,
          0x80, 0x78, 0x70, 0x68, 0x60, 0x58, 0x50, 0x48, 0x40, 0x38, 0x30, 0x28, 0x20, 0x18, 0x10, 0x08,
          0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
          0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
          0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
          0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
       };

       CHECK_VX_STATUS(vxAddArrayItems(LtoK_array, 256, LtoK_array_values, 0));

       //create an instance of the RemoveFringe node
       nRemoveFringe = vxRemoveFringePlanarOpenCLTiledNode(m_graph,
              m_iLab2CMYK,
              m_iLightnessDarknessContrast,
              iNeutralEdgeMaskUnpacked,
              iRemoveFringeC,
              iRemoveFringeM,
              iRemoveFringeY,
              iRemoveFringeK,
              iRemoveFringeEdgeK,
              LtoK_array);
       CHECK_VX_OBJ(nRemoveFringe);
    }
    else
    {
       //Create virtual images for the output of RemoveFringe
       // The output is CMYK (4-channels), but there is no VX interface enum
       // for CMYK, so we use RGBX instead...
       vx_image iRemoveFringeCMYK = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_RGBX);
       CHECK_VX_OBJ(iRemoveFringeCMYK);

       //Create the L-to-K 'knots' array that we will use as a parameter for CPU RemoveFringe
       LtoK_array = vxCreateArray(m_context, VX_TYPE_UINT8, 16);
       CHECK_VX_OBJ(LtoK_array);

       vx_uint8 LtoK_knots[16] = {255, 255, 255, 255, 243, 231, 219, 207, 195, 183, 171, 128, 0, 0, 0, 0};

       CHECK_VX_STATUS(vxAddArrayItems(LtoK_array, 16, LtoK_knots, 0));

       //create an instance of the RemoveFringe node
       nRemoveFringe = vxRemoveFringeNode(m_graph,
           m_iLab2CMYK,
           m_iLightnessDarknessContrast,
           iNeutralEdgeMaskUnpacked,
           iRemoveFringeCMYK,
           iRemoveFringeEdgeK,
           LtoK_array);
       CHECK_VX_OBJ(nRemoveFringe);



       vx_node nPixelToPlanar = vxChannelSeparateNodeIntel(m_graph,
                                                           iRemoveFringeCMYK,
                                                           iRemoveFringeC,
                                                           iRemoveFringeM,
                                                           iRemoveFringeY,
                                                           iRemoveFringeK);
       CHECK_VX_OBJ(nPixelToPlanar);
       CHECK_VX_STATUS(vxReleaseNode(&nPixelToPlanar));
       CHECK_VX_STATUS(vxReleaseImage(&iRemoveFringeCMYK));

    }

    //Create virtual images for the planar representation of the output of Error Diffusion
    vx_image ibitoneC = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(ibitoneC);
    vx_image ibitoneM = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(ibitoneM);
    vx_image ibitoneY = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(ibitoneY);
    vx_image ibitoneK = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(ibitoneK);

    vx_image iPackedRenderedK = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_U8);

    AttachHalftoneSubGraph(iRemoveFringeC, iRemoveFringeM, iRemoveFringeY, iRemoveFringeK, ibitoneC, ibitoneM, ibitoneY, ibitoneK);

    //finally, pack the C, M, Y, K output
    vx_node nPack8to1C = vxPack8to1NodeIntel(m_graph, ibitoneC, m_dstImageC);
    CHECK_VX_OBJ(nPack8to1C);
    vx_node nPack8to1M = vxPack8to1NodeIntel(m_graph, ibitoneM, m_dstImageM);
    CHECK_VX_OBJ(nPack8to1M);
    vx_node nPack8to1Y = vxPack8to1NodeIntel(m_graph, ibitoneY, m_dstImageY);
    CHECK_VX_OBJ(nPack8to1Y);

    vx_node nPack8to1K = vxPack8to1NodeIntel(m_graph, ibitoneK, iPackedRenderedK);
    CHECK_VX_OBJ(nPack8to1K);

    CHECK_VX_STATUS(vxReleaseNode(&nPack8to1K));
    CHECK_VX_STATUS(vxReleaseNode(&nPack8to1M));
    CHECK_VX_STATUS(vxReleaseNode(&nPack8to1Y));
    CHECK_VX_STATUS(vxReleaseNode(&nPack8to1C));

    vx_node nGenEdgeK = vxGenEdgeKNode(m_graph, iNeutralEdgeMask,
                                                iRemoveFringeEdgeK,
                                                iPackedRenderedK,
                                                96,
                                                130,
                                                m_dstImageK);

    CHECK_VX_STATUS(vxReleaseImage(&ibitoneK));
    CHECK_VX_STATUS(vxReleaseImage(&ibitoneY));
    CHECK_VX_STATUS(vxReleaseImage(&ibitoneM));
    CHECK_VX_STATUS(vxReleaseImage(&ibitoneC));
    CHECK_VX_STATUS(vxReleaseNode(&nRemoveFringe));
    CHECK_VX_STATUS(vxReleaseImage(&iRemoveFringeEdgeK));

    CHECK_VX_STATUS(vxReleaseArray(&LtoK_array));
    CHECK_VX_STATUS(vxReleaseImage(&iNeutralEdgeMask));

    return;
}

void PipelineControl::AssembleErrorDiffusionGraph()
{
    //Assemble common part for high1, high3 and high6
    AttachCommonSubGraph();

    //Create a virtual image which represents our neutral edge mask
    vx_image iNeutralEdgeMask = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(iNeutralEdgeMask);

    //Attach the 'neutral edge detect' sub-graph
    AttachNeutralEdgeDetectSubGraph(m_iLightnessDarknessContrast, m_iBackgroundSuppress_1, m_iBackgroundSuppress_2, iNeutralEdgeMask);

    vx_image iNeutralEdgeMaskUnpacked = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(iNeutralEdgeMaskUnpacked);

    vx_node nUnpackNEdge = vxUnpack1to8NodeIntel(m_graph, iNeutralEdgeMask, iNeutralEdgeMaskUnpacked);
    CHECK_VX_OBJ(nUnpackNEdge);

    //Create virtual images for the output of RemoveFringe
    // The output is CMYK (4-channels), but there is no VX interface enum
    // for CMYK, so we use RGBX instead...
    vx_image iRemoveFringeCMYK = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_RGBX);
    CHECK_VX_OBJ(iRemoveFringeCMYK);
    vx_image iRemoveFringeEdgeK = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(iRemoveFringeEdgeK);

    vx_image iRemoveFringeC = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(iRemoveFringeC);
    vx_image iRemoveFringeM = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(iRemoveFringeM);
    vx_image iRemoveFringeY = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(iRemoveFringeY);
    vx_image iRemoveFringeK = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(iRemoveFringeK);

    vx_node nRemoveFringe;
    vx_array LtoK_array;
    if(m_cmdparser->gpuremovefringe.isSet())
    {
       //Create the L-to-K 'knots' array that we will use as a parameter for RemoveFringe
       LtoK_array = vxCreateArray(m_context, VX_TYPE_UINT8, 256);

       vx_uint8 LtoK_array_values[256] =
       {
          0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
          0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
          0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
          0xff, 0xff, 0xfe, 0xfd, 0xfc, 0xfc, 0xfb, 0xfa, 0xf9, 0xf9, 0xf8, 0xf7, 0xf6, 0xf6, 0xf5, 0xf4,
          0xf3, 0xf3, 0xf2, 0xf1, 0xf0, 0xf0, 0xef, 0xee, 0xed, 0xed, 0xec, 0xeb, 0xea, 0xea, 0xe9, 0xe8,
          0xe7, 0xe7, 0xe6, 0xe5, 0xe4, 0xe4, 0xe3, 0xe2, 0xe1, 0xe1, 0xe0, 0xdf, 0xde, 0xde, 0xdd, 0xdc,
          0xdb, 0xdb, 0xda, 0xd9, 0xd8, 0xd8, 0xd7, 0xd6, 0xd5, 0xd5, 0xd4, 0xd3, 0xd2, 0xd2, 0xd1, 0xd0,
          0xcf, 0xcf, 0xce, 0xcd, 0xcc, 0xcc, 0xcb, 0xca, 0xc9, 0xc9, 0xc8, 0xc7, 0xc6, 0xc6, 0xc5, 0xc4,
          0xc3, 0xc3, 0xc2, 0xc1, 0xc0, 0xc0, 0xbf, 0xbe, 0xbd, 0xbd, 0xbc, 0xbb, 0xba, 0xba, 0xb9, 0xb8,
          0xb7, 0xb7, 0xb6, 0xb5, 0xb4, 0xb4, 0xb3, 0xb2, 0xb1, 0xb1, 0xb0, 0xaf, 0xae, 0xae, 0xad, 0xac,
          0xab, 0xa9, 0xa6, 0xa3, 0xa1, 0x9e, 0x9b, 0x99, 0x96, 0x93, 0x91, 0x8e, 0x8b, 0x89, 0x86, 0x83,
          0x80, 0x78, 0x70, 0x68, 0x60, 0x58, 0x50, 0x48, 0x40, 0x38, 0x30, 0x28, 0x20, 0x18, 0x10, 0x08,
          0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
          0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
          0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
          0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
       };

       CHECK_VX_STATUS(vxAddArrayItems(LtoK_array, 256, LtoK_array_values, 0));

       //create an instance of the RemoveFringe node
       nRemoveFringe = vxRemoveFringeOpenCLTiledNode(m_graph,
              m_iLab2CMYK,
              m_iLightnessDarknessContrast,
              iNeutralEdgeMaskUnpacked,
              iRemoveFringeCMYK,
              iRemoveFringeEdgeK,
              LtoK_array);
       CHECK_VX_OBJ(nRemoveFringe);

    }
    else
    {
       //Create the L-to-K 'knots' array that we will use as a parameter for CPU RemoveFringe
       LtoK_array = vxCreateArray(m_context, VX_TYPE_UINT8, 16);
       CHECK_VX_OBJ(LtoK_array);

       vx_uint8 LtoK_knots[16] = {255, 255, 255, 255, 243, 231, 219, 207, 195, 183, 171, 128, 0, 0, 0, 0};

       CHECK_VX_STATUS(vxAddArrayItems(LtoK_array, 16, LtoK_knots, 0));

       //create an instance of the RemoveFringe node
       nRemoveFringe = vxRemoveFringeNode(m_graph,
           m_iLab2CMYK,
           m_iLightnessDarknessContrast,
           iNeutralEdgeMaskUnpacked,
           iRemoveFringeCMYK,
           iRemoveFringeEdgeK,
           LtoK_array);
       CHECK_VX_OBJ(nRemoveFringe);
    }

    //Create virtual images for the planar representation of the output of Error Diffusion
    vx_image ibitoneC = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(ibitoneC);
    vx_image ibitoneM = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(ibitoneM);
    vx_image ibitoneY = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(ibitoneY);
    vx_image ibitoneK = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(ibitoneK);

    vx_image iPackedRenderedK = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_U8);

    AttachErrorDiffusionSubGraph(iRemoveFringeCMYK, ibitoneC, ibitoneM, ibitoneY, ibitoneK);

    //finally, pack the C, M, Y, K output
     vx_node nPack8to1C = vxPack8to1NodeIntel(m_graph, ibitoneC, m_dstImageC);
     CHECK_VX_OBJ(nPack8to1C);
     vx_node nPack8to1M = vxPack8to1NodeIntel(m_graph, ibitoneM, m_dstImageM);
     CHECK_VX_OBJ(nPack8to1M);
     vx_node nPack8to1Y = vxPack8to1NodeIntel(m_graph, ibitoneY, m_dstImageY);
     CHECK_VX_OBJ(nPack8to1Y);

     vx_node nPack8to1K = vxPack8to1NodeIntel(m_graph, ibitoneK, iPackedRenderedK);
     CHECK_VX_OBJ(nPack8to1K);

     CHECK_VX_STATUS(vxReleaseNode(&nPack8to1K));
     CHECK_VX_STATUS(vxReleaseNode(&nPack8to1M));
     CHECK_VX_STATUS(vxReleaseNode(&nPack8to1Y));
     CHECK_VX_STATUS(vxReleaseNode(&nPack8to1C));

    vx_node nGenEdgeK = vxGenEdgeKNode(m_graph, iNeutralEdgeMask,
                                                iRemoveFringeEdgeK,
                                                iPackedRenderedK,
                                                96,
                                                130,
                                                m_dstImageK);



    CHECK_VX_STATUS(vxReleaseImage(&ibitoneK));
    CHECK_VX_STATUS(vxReleaseImage(&ibitoneY));
    CHECK_VX_STATUS(vxReleaseImage(&ibitoneM));
    CHECK_VX_STATUS(vxReleaseImage(&ibitoneC));
    CHECK_VX_STATUS(vxReleaseNode(&nRemoveFringe));
    CHECK_VX_STATUS(vxReleaseImage(&iRemoveFringeEdgeK));
    CHECK_VX_STATUS(vxReleaseImage(&iRemoveFringeCMYK));
    CHECK_VX_STATUS(vxReleaseArray(&LtoK_array));
    CHECK_VX_STATUS(vxReleaseImage(&iNeutralEdgeMask));

    return;
}


PERFPROF_REGION_DEFINE(vxProcessGraph);
PERFPROF_REGION_DEFINE(vxVerifyGraph);
void PipelineControl::ExecuteGraph()
{
    int tilewidth = m_tileWidth;
    int tileheight = m_cmdparser->tileheight.getValue();

    CHECK_VX_STATUS(vxSetGraphAttribute(m_graph, VX_GRAPH_TILE_WIDTH_INTEL, (void *)&tilewidth, sizeof(tilewidth)));
    CHECK_VX_STATUS(vxSetGraphAttribute(m_graph, VX_GRAPH_TILE_HEIGHT_INTEL, (void *)&tileheight, sizeof(tileheight)));

    //default the number of execution threads to 0 (0 means let runtime decide)
    //get the number of desired threads from the command line (if present)
    int nthreads = m_cmdparser->nthreads.getValue();
    if( nthreads != 0 )
    {
        //set the number of CPU worker threads that this context is allowed to use.
        CHECK_VX_STATUS(vxSetContextAttribute(m_context, VX_CONTEXT_POOL_THREAD_COUNT_INTEL, (void *)&nthreads, sizeof(nthreads)));
    }

    //Set our custom allocator
    //Reserve own memory pool. Speedups vxVerifyGraph and first call to vxProcessGraph
    CHECK_VX_STATUS(vxSetGraphAttribute(m_graph, VX_GRAPH_ALLOCATOR_INTEL, (void *)&m_customAllocator, sizeof(m_customAllocator)));

    //Set memory optimization level (performance/memory consumption tradeoff)
    int memfactor_value = 4;
    CHECK_VX_STATUS(vxSetGraphAttribute(m_graph, VX_GRAPH_MEM_OPTIMIZATION_LEVEL_INTEL, (void *)&memfactor_value, sizeof(memfactor_value)));


    //verify the graph (prep for execution)
    {
        PERFPROF_REGION_BEGIN(vxVerifyGraph);
        CHECK_VX_STATUS(vxVerifyGraph(m_graph));
        PERFPROF_REGION_END(vxVerifyGraph);
    }

    //Warm up runs
    for( int i = 0; i < 5; i++)
    {
        double start = time_stamp()*1000.0;
        //process the graph (execute the graph)
        CHECK_VX_STATUS(vxProcessGraph(m_graph));
        double end = time_stamp()*1000.0;
        std::cout << "Main Graph: vxProcessGraph time = " << end - start << " ms." << std::endl;
    }


    unsigned int runs = m_cmdparser->frames.getValue();
    std::vector<double> runtimes;
    //Main performance measurement runs
    double start = time_stamp()*1000.0;
    for(unsigned int i = 0; i < runs; i++)
    {
        //process the graph (execute the graph)
        PERFPROF_REGION_BEGIN(vxProcessGraph);
        double starte = time_stamp()*1000.0;
        CHECK_VX_STATUS(vxProcessGraph(m_graph));
        double ende = time_stamp()*1000.0;
        PERFPROF_REGION_END(vxProcessGraph);
        std::cout << "Main Graph: vxProcessGraph time = " << ende - starte << " ms." << std::endl;
        runtimes.push_back(ende - starte);
    }
    double mean=0.0;
    for( int i = 0; i < runtimes.size(); i++)
    {
      mean += runtimes[i];
    }

    mean/=runtimes.size();

    //After vxVerifyGraph is run, memory statistics which show
    // the amount of 'intermediate buffer' storage that
    // was allocated can be gathered.
    vx_size memory_usage_bytes = 0;
    CHECK_VX_STATUS(vxQueryGraph(m_graph, VX_GRAPH_PROFILE_MEMORY_SIZE_INTEL, &memory_usage_bytes, sizeof(memory_usage_bytes)));
    std::cout << "Main Graph: Internal Buffers Allocated = " << memory_usage_bytes << " bytes. " << std::endl;

    //Calculate Pages Per Minute (PPM) performance metric
    int PPM = 60000.0 / mean; //60000 milliseconds in one minute
    std::cout << "PPM = " << PPM << std::endl;

    return;
}

void PipelineControl::SaveOutputImage()
{
    if( m_cmdparser->output.isSet())
    {
        fprintf(stderr, "Reformatting output to save to disk...\n");

        vx_uint32 height, width;
        CHECK_VX_STATUS(vxQueryImage(m_dstImageC, VX_IMAGE_WIDTH, &width, sizeof(width)));
        CHECK_VX_STATUS(vxQueryImage(m_dstImageC, VX_IMAGE_HEIGHT, &height, sizeof(height)));

        vx_image dstImageCMYK = vxCreateImage(m_context, width, height, VX_DF_IMAGE_RGBX);
        CHECK_VX_OBJ(dstImageCMYK);

        //if we were in 'High 6' mode, the output image
        // is stored as planar C, M, Y, K.
        //The format that we want to save to disk is
        // pixel interleaved CMYK, so set up a VX
        // graph to do the conversion.
        vx_graph planar2pixelGraph = vxCreateGraph(m_context);
        CHECK_VX_OBJ(planar2pixelGraph);

        //unpack the 1bpp planar interleaved images
        vx_image iUnpackedC = vxCreateVirtualImage(planar2pixelGraph, 0, 0, VX_DF_IMAGE_U8);
        CHECK_VX_OBJ(iUnpackedC);
        vx_image iUnpackedM = vxCreateVirtualImage(planar2pixelGraph, 0, 0, VX_DF_IMAGE_U8);
        CHECK_VX_OBJ(iUnpackedM);
        vx_image iUnpackedY = vxCreateVirtualImage(planar2pixelGraph, 0, 0, VX_DF_IMAGE_U8);
        CHECK_VX_OBJ(iUnpackedY);
        vx_image iUnpackedK = vxCreateVirtualImage(planar2pixelGraph, 0, 0, VX_DF_IMAGE_U8);
        CHECK_VX_OBJ(iUnpackedK);
        vx_node nUnpackedC = vxUnpack1to8NodeIntel(planar2pixelGraph, m_dstImageC, iUnpackedC);
        CHECK_VX_OBJ(nUnpackedC);
        vx_node nUnpackedM = vxUnpack1to8NodeIntel(planar2pixelGraph, m_dstImageM, iUnpackedM);
        CHECK_VX_OBJ(nUnpackedM);
        vx_node nUnpackedY = vxUnpack1to8NodeIntel(planar2pixelGraph, m_dstImageY, iUnpackedY);
        CHECK_VX_OBJ(nUnpackedY);
        vx_node nUnpackedK = vxUnpack1to8NodeIntel(planar2pixelGraph, m_dstImageK, iUnpackedK);
        CHECK_VX_OBJ(nUnpackedK);

        vx_image iUnpackedCMYK = vxCreateVirtualImage(planar2pixelGraph, 0, 0, VX_DF_IMAGE_RGBX);
        CHECK_VX_OBJ(iUnpackedCMYK);

        //combine the planar channels into pixel interleaved
        vx_node nUnpackedCMYK = vxChannelCombineNode(planar2pixelGraph,
            iUnpackedC,
            iUnpackedM,
            iUnpackedY,
            iUnpackedK,
            iUnpackedCMYK);
        CHECK_VX_OBJ(nUnpackedCMYK);

        //pack it
        vx_node nPackCMYK = vxPack8to1NodeIntel(planar2pixelGraph, iUnpackedCMYK, dstImageCMYK);
        CHECK_VX_OBJ(nPackCMYK);

        CHECK_VX_STATUS(vxVerifyGraph(planar2pixelGraph));

        CHECK_VX_STATUS(vxProcessGraph(planar2pixelGraph));

        CHECK_VX_STATUS(vxReleaseImage(&iUnpackedC));
        CHECK_VX_STATUS(vxReleaseImage(&iUnpackedM));
        CHECK_VX_STATUS(vxReleaseImage(&iUnpackedY));
        CHECK_VX_STATUS(vxReleaseImage(&iUnpackedK));
        CHECK_VX_STATUS(vxReleaseNode(&nUnpackedC));
        CHECK_VX_STATUS(vxReleaseNode(&nUnpackedM));
        CHECK_VX_STATUS(vxReleaseNode(&nUnpackedY));
        CHECK_VX_STATUS(vxReleaseNode(&nUnpackedK));
        CHECK_VX_STATUS(vxReleaseNode(&nUnpackedCMYK));
        CHECK_VX_STATUS(vxReleaseNode(&nPackCMYK));
        CHECK_VX_STATUS(vxReleaseGraph(&planar2pixelGraph));

        SaveImageToDisk(m_cmdparser->output.getValue().c_str(), dstImageCMYK);

        vxReleaseImage(&dstImageCMYK);
    }

    return;
}

//Attaches a neutral edge detection sub-graph to m_graph... pass in the input links (CIELab channels),
// will pass back the neutral edge graph
void PipelineControl::AttachNeutralEdgeDetectSubGraph(vx_image l_in, vx_image a_in, vx_image b_in, vx_image neutral_edge_out)
{

    //check to make sure that the input images to this sub-graph are valid
    CHECK_VX_OBJ(l_in);
    CHECK_VX_OBJ(a_in);
    CHECK_VX_OBJ(b_in);
    CHECK_VX_OBJ(neutral_edge_out);

    //Create 'virtual' images for the output of the box filters.
    // This means that we (as the user) are acknowledging that we don't need
    // access to these images, and they can be kept internal to the graph
    // by the VX implementation.
    vx_image iBoxFilterA = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(iBoxFilterA);
    vx_image iBoxFilterB = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(iBoxFilterB);

    //Create instances of 3x3 Box Filter nodes.
    vx_node nBoxFilterA = vxBox3x3Node(m_graph, a_in, iBoxFilterA);
    CHECK_VX_OBJ(nBoxFilterA);
    vx_node nBoxFilterB = vxBox3x3Node(m_graph, b_in, iBoxFilterB);
    CHECK_VX_OBJ(nBoxFilterB);

    //The box filters will create their own border.
    //Set it to a constant value of 128 ('gray' in the case of a*/b*)
    vx_border_t boxborderMode;
    boxborderMode.mode = VX_BORDER_CONSTANT;
    boxborderMode.constant_value.U32 = 128;

    //set the 'border mode' node attribute
    CHECK_VX_STATUS(vxSetNodeAttribute(nBoxFilterA, VX_NODE_BORDER, &boxborderMode, sizeof(boxborderMode)));

    //set the node attribute
    CHECK_VX_STATUS(vxSetNodeAttribute(nBoxFilterB, VX_NODE_BORDER, &boxborderMode, sizeof(boxborderMode)));

    if( m_cmdparser->gpuboxfilter.isSet() )
    {
        CHECK_VX_STATUS(vxSetNodeTarget(nBoxFilterA, VX_TARGET_GPU_INTEL, 0));
        CHECK_VX_STATUS(vxSetNodeTarget(nBoxFilterB, VX_TARGET_GPU_INTEL, 0));
    }

    //Create 'virtual' images for the output of the Sobel filters
    //horizontal Sobel filter
    vx_image iSobelFilterX  = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_S16);
    CHECK_VX_OBJ(iSobelFilterX);
    //vertical Sobel filter
    vx_image iSobelFilterY  = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_S16);
    CHECK_VX_OBJ(iSobelFilterY);

    //Create an instance of a Sobel 3x3 Node
    vx_node nSobelFilter = vxSobel3x3Node(m_graph, l_in, iSobelFilterX, iSobelFilterY);
    CHECK_VX_OBJ(nSobelFilter);

    //The sobel filter will create it's own border.
    //Set it to a constant value of 255 (white in the case of L*)
    vx_border_t sobelborderMode;
    sobelborderMode.mode = VX_BORDER_CONSTANT;
    sobelborderMode.constant_value.U32 = 255;

    //set the 'border mode' node attribute
    CHECK_VX_STATUS(vxSetNodeAttribute(nSobelFilter, VX_NODE_BORDER, &sobelborderMode, sizeof(sobelborderMode)));

    if( m_cmdparser->gpusobelfilter.isSet() )
    {
        CHECK_VX_STATUS(vxSetNodeTarget(nSobelFilter, VX_TARGET_GPU_INTEL, 0));
    }

    //Create a 'virtual' image for the output of Neutral Pixel Detection node
    vx_image iNeutralPixelDetect  = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(iNeutralPixelDetect);

    //Create an instance of a Neutral Pixel Detection node
    vx_node nNeutralPixelDetect = vxNeutralPixelDetectionNode(m_graph, iBoxFilterA, iBoxFilterB, iNeutralPixelDetect);
    CHECK_VX_OBJ(nNeutralPixelDetect);

    vx_image iMagnitude = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_S16);
    CHECK_VX_OBJ(iMagnitude);
    vx_node nMagnitude = vxMagnitudeNode(m_graph, iSobelFilterX, iSobelFilterY, iMagnitude);
    CHECK_VX_OBJ(nMagnitude);

    if( m_cmdparser->gpusobelfilter.isSet() )
    {
        CHECK_VX_STATUS(vxSetNodeTarget(nMagnitude, VX_TARGET_GPU_INTEL, 0));
    }

    vx_image iEdgeMask = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_U8);
    CHECK_VX_OBJ(iEdgeMask);
    vx_node nGenEdgeMask = vxGenEdgeMaskNode(m_graph, iMagnitude, iEdgeMask);
    CHECK_VX_OBJ(nGenEdgeMask);

    //Now, 'And' together the output of Neutral Pixel Detection, and the output of Edge Detection.
    // This will create the 'Neutral Edge' mask
    vx_node nAnd1 = vxAndNode(m_graph, iNeutralPixelDetect, iEdgeMask, neutral_edge_out);
    CHECK_VX_OBJ(nAnd1);

    CHECK_VX_STATUS(vxReleaseNode(&nAnd1));
    CHECK_VX_STATUS(vxReleaseNode(&nGenEdgeMask));
    CHECK_VX_STATUS(vxReleaseImage(&iEdgeMask));
    CHECK_VX_STATUS(vxReleaseNode(&nMagnitude));
    CHECK_VX_STATUS(vxReleaseNode(&nNeutralPixelDetect));
    CHECK_VX_STATUS(vxReleaseImage(&iNeutralPixelDetect));
    CHECK_VX_STATUS(vxReleaseNode(&nSobelFilter));
    CHECK_VX_STATUS(vxReleaseImage(&iSobelFilterY));
    CHECK_VX_STATUS(vxReleaseImage(&iSobelFilterX));
    CHECK_VX_STATUS(vxReleaseNode(&nBoxFilterA));
    CHECK_VX_STATUS(vxReleaseNode(&nBoxFilterB));
    CHECK_VX_STATUS(vxReleaseImage(&iBoxFilterA));
    CHECK_VX_STATUS(vxReleaseImage(&iBoxFilterB));
    return;

}

void PipelineControl::InitializeLightnessDarknessContrastLUT( int lightness, int contrast, vx_uint8 lut[256])
{
    if( (lightness > 3) || (lightness < -3) )
        lightness = 0;

    if( (contrast > 2) || (contrast < -2) )
        contrast = 0;

    //normalize the parameters
    lightness += 3;
    contrast += 2;

    //calculate the start index into our params table
    int start_index = (lightness*5*256) + contrast*256;

    for( int i = 0; i < 256; i++ )
    {
        lut[i] = LIGHT_DARK_CONSTRAST[i + start_index];
    }
}

void PipelineControl::AttachRGB2LabSubGraph(vx_image r_in,
                                            vx_image g_in,
                                            vx_image b_in,
                                            vx_image l_out,
                                            vx_image a_out,
                                            vx_image b_out)
{

    if( m_cmdparser->ipurgb2lab.isSet() || m_cmdparser->gpurgb2lab.isSet() )
    {
      vx_float32 coefficients[9] = {0.412453, 0.357580, 0.180423, 0.212671, 0.715160, 0.072169, 0.019334, 0.119193, 0.950227};
      vx_float32 recip[3] = {0.950455,1.000000,1.088753};

      vx_array coefficients_array = vxCreateArray(m_context, VX_TYPE_FLOAT32, 9);
      CHECK_VX_OBJ(coefficients_array);
      CHECK_VX_STATUS(vxAddArrayItems(coefficients_array, 9, coefficients, 0));

      vx_array recip_array = vxCreateArray(m_context, VX_TYPE_FLOAT32, 3);
      CHECK_VX_OBJ(recip_array);
      CHECK_VX_STATUS(vxAddArrayItems(recip_array, 3, recip, 0));

      vx_node nRGB2Lab = vxRgbToLabNodeIntel(m_graph,
                                r_in,
                                g_in,
                                b_in,
                                coefficients_array,
                                recip_array,
                                l_out,
                                a_out,
                                b_out);

      CHECK_VX_OBJ(nRGB2Lab)

      if( m_cmdparser->gpurgb2lab.isSet() )
      {
        CHECK_VX_STATUS(vxSetNodeTarget(nRGB2Lab, VX_TARGET_GPU_INTEL, 0));
      }
      else if( m_cmdparser->ipurgb2lab.isSet() )
      {

        CHECK_VX_STATUS(vxSetNodeTarget(nRGB2Lab, VX_TARGET_IPU_INTEL, 0));
      }

      CHECK_VX_STATUS(vxReleaseArray(&coefficients_array));
      CHECK_VX_STATUS(vxReleaseArray(&recip_array));
      CHECK_VX_STATUS(vxReleaseNode(&nRGB2Lab));
    }
    else //use CPU optimized LUT3D
    {
      vx_image iRGBPixelInterleaved = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_RGB);
      CHECK_VX_OBJ(iRGBPixelInterleaved);

      //combine the planar channels into pixel interleaved
      vx_node nRGBPixel = vxChannelCombineNode(m_graph,
          r_in,
          g_in,
          b_in,
          0,
          iRGBPixelInterleaved);
      CHECK_VX_OBJ(nRGBPixel);

      vx_image iLABPixelInterleaved = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_RGB);
      CHECK_VX_OBJ(iLABPixelInterleaved);


      vx_node nRGB2Lab = vxLUT3DNodeIntel(m_graph,
          iRGBPixelInterleaved,
          VX_INTERPOLATION_TRILINEAR_INTEL, 17, 0, 0, 0,
          m_rgb2lab_nodevals17x17x17,
          iLABPixelInterleaved);
      CHECK_VX_OBJ(nRGB2Lab);

      vx_node nLabSeparate = vxChannelSeparateNodeIntel(m_graph, iLABPixelInterleaved, l_out, a_out, b_out, 0);
      CHECK_VX_OBJ(nLabSeparate);

      CHECK_VX_STATUS(vxReleaseNode(&nLabSeparate));
      CHECK_VX_STATUS(vxReleaseNode(&nRGB2Lab));
      CHECK_VX_STATUS(vxReleaseImage(&iLABPixelInterleaved));
      CHECK_VX_STATUS(vxReleaseNode(&nRGBPixel));
      CHECK_VX_STATUS(vxReleaseImage(&iRGBPixelInterleaved));
    }

    return;
}

void PipelineControl::AttachLab2CMYKSubGraph(vx_image l_in,
                                             vx_image a_in,
                                             vx_image b_in,
                                             vx_image cmyk_out)
{
    vx_image iLABPixelInterleaved = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_RGB);
    CHECK_VX_OBJ(iLABPixelInterleaved);

    //combine the planar channels into pixel interleaved
    vx_node nLabPixel = vxChannelCombineNode(m_graph,
        l_in,
        a_in,
        b_in,
        0,
        iLABPixelInterleaved);
    CHECK_VX_OBJ(nLabPixel);

    vx_node nLab2CMYK = vxLUT3DNodeIntel(m_graph,
        iLABPixelInterleaved,
        m_3dlut_interp_type_lab2cmyk, m_3dlut_nlatticepoints_lab2cmyk, 0, 0, 0,
        m_lab2cmyk_nodevals,
        cmyk_out);
    CHECK_VX_OBJ(nLab2CMYK);

    if( m_cmdparser->gpulab2cmyk.isSet() )
    {
        CHECK_VX_STATUS(vxSetNodeTarget(nLabPixel, VX_TARGET_GPU_INTEL, 0));
        CHECK_VX_STATUS(vxSetNodeTarget(nLab2CMYK, VX_TARGET_GPU_INTEL, 0));
    }

    CHECK_VX_STATUS(vxReleaseNode(&nLab2CMYK));
    CHECK_VX_STATUS(vxReleaseNode(&nLabPixel));
    CHECK_VX_STATUS(vxReleaseImage(&iLABPixelInterleaved));

    return;
}

void PipelineControl::AttachHalftoneSubGraph(vx_image c_contone,
                                             vx_image m_contone,
                                             vx_image y_contone,
                                             vx_image k_contone,
                                             vx_image c_bitone,
                                             vx_image m_bitone,
                                             vx_image y_bitone,
                                             vx_image k_bitone)
{
    if( m_cmdparser->ipahalftone.isSet() )
    {
       //Create vx_array's for each of the C, M, Y, and K screen's that are defined in
       // cmykhalftone.h
       vx_array screenDataC = vxCreateArray(m_context, VX_TYPE_UINT8, C_WIDTH * C_HEIGHT);
       vxAddArrayItems(screenDataC, C_WIDTH * C_HEIGHT, C_SCREEN_DATA, 0);

       vx_array screenDataM = vxCreateArray(m_context, VX_TYPE_UINT8, M_WIDTH * M_HEIGHT);
       vxAddArrayItems(screenDataM, M_WIDTH * M_HEIGHT, M_SCREEN_DATA, 0);

       vx_array screenDataY = vxCreateArray(m_context, VX_TYPE_UINT8, Y_WIDTH * Y_HEIGHT);
       vxAddArrayItems(screenDataY, Y_WIDTH * Y_HEIGHT, Y_SCREEN_DATA, 0);

       vx_array screenDataK = vxCreateArray(m_context, VX_TYPE_UINT8, K_WIDTH * K_HEIGHT);
       vxAddArrayItems(screenDataK, K_WIDTH * K_HEIGHT, K_SCREEN_DATA, 0);

       vx_node nHalftoneC = vxIPAHalftoneNode(m_graph,
                                              c_contone,
                                              screenDataC,
                                              C_WIDTH,
                                              C_HEIGHT,
                                              C_SHIFT,
                                              c_bitone);

       vx_node nHalftoneM = vxIPAHalftoneNode(m_graph,
                                              m_contone,
                                              screenDataM,
                                              M_WIDTH,
                                              M_HEIGHT,
                                              M_SHIFT,
                                              m_bitone);

       vx_node nHalftoneY = vxIPAHalftoneNode(m_graph,
                                              y_contone,
                                              screenDataY,
                                              Y_WIDTH,
                                              Y_HEIGHT,
                                              Y_SHIFT,
                                              y_bitone);

       vx_node nHalftoneK = vxIPAHalftoneNode(m_graph,
                                              k_contone,
                                              screenDataK,
                                              K_WIDTH,
                                              K_HEIGHT,
                                              K_SHIFT,
                                              k_bitone);

       CHECK_VX_STATUS(vxReleaseNode(&nHalftoneK));
       CHECK_VX_STATUS(vxReleaseNode(&nHalftoneY));
       CHECK_VX_STATUS(vxReleaseNode(&nHalftoneM));
       CHECK_VX_STATUS(vxReleaseNode(&nHalftoneC));
       CHECK_VX_STATUS(vxReleaseArray(&screenDataK));
       CHECK_VX_STATUS(vxReleaseArray(&screenDataY));
       CHECK_VX_STATUS(vxReleaseArray(&screenDataM));
       CHECK_VX_STATUS(vxReleaseArray(&screenDataC));
    }
    else
    {

       vx_image halftoneC = vxCreateImage(m_context, C_WIDTH, C_HEIGHT, VX_DF_IMAGE_U8);
       {
          vx_rectangle_t recthalftone = {0, 0, C_WIDTH, C_HEIGHT};
          vx_map_id mapId;
          vx_uint8 *pHalftone = NULL;
          vx_imagepatch_addressing_t imagepatch;
          vxMapImagePatch(halftoneC, &recthalftone, 0, &mapId, &imagepatch, (void **)&pHalftone, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X);

          for( int y = 0; y < C_HEIGHT; y++)
          {
             for( int x = 0; x < C_WIDTH; x++)
             {
                pHalftone[x] = C_SCREEN_DATA[y*C_HEIGHT + x];
             }
             pHalftone += imagepatch.stride_y;
          }

          vxUnmapImagePatch(halftoneC, mapId);
       }

       vx_image halftoneM = vxCreateImage(m_context, M_WIDTH, M_HEIGHT, VX_DF_IMAGE_U8);
       {
          vx_rectangle_t recthalftone = {0, 0, M_WIDTH, M_HEIGHT};
          vx_map_id mapId;
          vx_uint8 *pHalftone = NULL;
          vx_imagepatch_addressing_t imagepatch;
          vxMapImagePatch(halftoneM, &recthalftone, 0, &mapId, &imagepatch, (void **)&pHalftone, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X);

          for( int y = 0; y < M_HEIGHT; y++)
          {
             for( int x = 0; x < M_WIDTH; x++)
             {
                pHalftone[x] = M_SCREEN_DATA[y*M_HEIGHT + x];
             }
             pHalftone += imagepatch.stride_y;
          }

          vxUnmapImagePatch(halftoneM, mapId);
       }

       vx_image halftoneY = vxCreateImage(m_context, Y_WIDTH, Y_HEIGHT, VX_DF_IMAGE_U8);
       {
          vx_rectangle_t recthalftone = {0, 0, Y_WIDTH, Y_HEIGHT};
          vx_map_id mapId;
          vx_uint8 *pHalftone = NULL;
          vx_imagepatch_addressing_t imagepatch;
          vxMapImagePatch(halftoneY, &recthalftone, 0, &mapId, &imagepatch, (void **)&pHalftone, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X);

          for( int y = 0; y < Y_HEIGHT; y++)
          {
             for( int x = 0; x < Y_WIDTH; x++)
             {
                pHalftone[x] = Y_SCREEN_DATA[y*Y_HEIGHT + x];
             }
             pHalftone += imagepatch.stride_y;
          }

          vxUnmapImagePatch(halftoneY, mapId);
       }

       vx_image halftoneK = vxCreateImage(m_context, K_WIDTH, K_HEIGHT, VX_DF_IMAGE_U8);
       {
          vx_rectangle_t recthalftone = {0, 0, K_WIDTH, K_HEIGHT};
          vx_map_id mapId;
          vx_uint8 *pHalftone = NULL;
          vx_imagepatch_addressing_t imagepatch;
          vxMapImagePatch(halftoneK, &recthalftone, 0, &mapId, &imagepatch, (void **)&pHalftone, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X);

          for( int y = 0; y < K_HEIGHT; y++)
          {
             for( int x = 0; x < K_WIDTH; x++)
             {
                pHalftone[x] = K_SCREEN_DATA[y*K_HEIGHT + x];
             }
             pHalftone += imagepatch.stride_y;
          }

          vxUnmapImagePatch(halftoneK, mapId);
       }


       vx_node nHalftoneC = vxHalftoneNodeIntel(m_graph, c_contone, halftoneC, 0, c_bitone);
       CHECK_VX_OBJ(nHalftoneC);

       vx_node nHalftoneM = vxHalftoneNodeIntel(m_graph, m_contone, halftoneM, 0, m_bitone);
       CHECK_VX_OBJ(nHalftoneM);

       vx_node nHalftoneY = vxHalftoneNodeIntel(m_graph, y_contone, halftoneY, 0, y_bitone);
       CHECK_VX_OBJ(nHalftoneY);

       vx_node nHalftoneK = vxHalftoneNodeIntel(m_graph, k_contone, halftoneK, 0, k_bitone);
       CHECK_VX_OBJ(nHalftoneK);

       CHECK_VX_STATUS(vxReleaseNode(&nHalftoneK));
       CHECK_VX_STATUS(vxReleaseNode(&nHalftoneY));
       CHECK_VX_STATUS(vxReleaseNode(&nHalftoneM));
       CHECK_VX_STATUS(vxReleaseNode(&nHalftoneC));
       CHECK_VX_STATUS(vxReleaseImage(&halftoneK));
       CHECK_VX_STATUS(vxReleaseImage(&halftoneY));
       CHECK_VX_STATUS(vxReleaseImage(&halftoneM));
       CHECK_VX_STATUS(vxReleaseImage(&halftoneC));
    }
}

void PipelineControl::AttachErrorDiffusionSubGraph(vx_image cmyk_contone,
                                                   vx_image c_bitone,
                                                   vx_image m_bitone,
                                                   vx_image y_bitone,
                                                   vx_image k_bitone)
{
    //Create a virtual image for the output of Error Diffusion
    // The output is CMYK (4-channels), but there is no VX interface enum
    // for CMYK, so we use RGBX instead...
    vx_image iEDCMYK = vxCreateVirtualImage(m_graph, 0, 0, VX_DF_IMAGE_RGBX);
    CHECK_VX_OBJ(iEDCMYK);

    //Create an error diffusion node.
    // Note, input is the output of Lab2CMYK (Tetrahedral Interpolation)
    vx_node nErrorDiffusion = vxErrorDiffusionCMYKNodeIntel(m_graph, cmyk_contone, iEDCMYK);
    CHECK_VX_OBJ(nErrorDiffusion);


    //Convert output of Error Diffusion from 'Pixel-interleaved' to 'Planar-interleaved'
    vx_node nPixelToPlanar = vxChannelSeparateNodeIntel(m_graph,
        iEDCMYK,
        c_bitone,
        m_bitone,
        y_bitone,
        k_bitone);
    CHECK_VX_OBJ(nPixelToPlanar);

    CHECK_VX_STATUS(vxReleaseNode(&nPixelToPlanar));
    CHECK_VX_STATUS(vxReleaseNode(&nErrorDiffusion));
    CHECK_VX_STATUS(vxReleaseImage(&iEDCMYK));


}

void PipelineControl::WarmGPUKernels()
{
    vx_target_intel targetGPU = vxGetTargetByNameIntel(m_context, "intel.gpu");
    if( vxGetStatus((vx_reference)targetGPU) == VX_SUCCESS )
    {
       //change the default target for immediate mode calls to GPU
       CHECK_VX_STATUS(vxSetImmediateModeTarget(m_context, VX_TARGET_GPU_INTEL, 0));

       if( m_cmdparser->gpulab2cmyk.isSet() )
       {
          vx_image iIn0 = vxCreateImage(m_context, 32, 32, VX_DF_IMAGE_U8);
          CHECK_VX_OBJ(iIn0);

          vx_image iIn1 = vxCreateImage(m_context, 32, 32, VX_DF_IMAGE_U8);
          CHECK_VX_OBJ(iIn1);

          vx_image iIn2 = vxCreateImage(m_context, 32, 32, VX_DF_IMAGE_U8);
          CHECK_VX_OBJ(iIn2);

          vx_graph graph = vxCreateGraph(m_context);
          CHECK_VX_OBJ(graph);

          vx_image iCombined = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_RGB);
          CHECK_VX_OBJ(iCombined);

          vx_node nCombine = vxChannelCombineNode(graph, iIn0, iIn1, iIn2, 0, iCombined);
          CHECK_VX_OBJ(nCombine);

          vx_image iOut = vxCreateImage(m_context, 32, 32, VX_DF_IMAGE_RGBX);
          CHECK_VX_OBJ(iOut);


          vx_node nlut3d = vxLUT3DNodeIntel(graph,
                                        iCombined,
                                        m_3dlut_interp_type_lab2cmyk, m_3dlut_nlatticepoints_lab2cmyk, 0, 0, 0,
                                        m_lab2cmyk_nodevals,
                                        iOut);

        CHECK_VX_STATUS(vxSetNodeTarget(nlut3d, VX_TARGET_GPU_INTEL, 0));
        CHECK_VX_STATUS(vxSetNodeTarget(nCombine, VX_TARGET_GPU_INTEL, 0));

          CHECK_VX_STATUS(vxVerifyGraph(graph));

          vxReleaseGraph(&graph);
          CHECK_VX_STATUS(vxReleaseImage(&iOut));
          CHECK_VX_STATUS(vxReleaseImage(&iIn0));
          CHECK_VX_STATUS(vxReleaseImage(&iIn1));
          CHECK_VX_STATUS(vxReleaseImage(&iIn2));
       }

       if( m_cmdparser->gpusymm7x7.isSet() )
       {
          vx_image iIn0 = vxCreateImage(m_context, 32, 32, VX_DF_IMAGE_U8);
          CHECK_VX_OBJ(iIn0);

          vx_image iOut0 = vxCreateImage(m_context, 32, 32, VX_DF_IMAGE_U8);
          CHECK_VX_OBJ(iOut0);

          vx_array coefficients_array = vxCreateArray(m_context, VX_TYPE_INT32, 10);
          CHECK_VX_OBJ(coefficients_array);
          vx_int32 coefficients[10] = {1140, -118, 526, 290, -236, 64, -128, -5, -87, -7};

          CHECK_VX_STATUS(vxAddArrayItems(coefficients_array, 10, coefficients, 0));

          vx_int32 shift = 10;

          vx_border_t symm7x7borderMode;
          symm7x7borderMode.mode = VX_BORDER_CONSTANT;
          symm7x7borderMode.constant_value.U32 = 255;

          CHECK_VX_STATUS(vxSetContextAttribute(m_context, VX_CONTEXT_IMMEDIATE_BORDER, &symm7x7borderMode, sizeof(symm7x7borderMode)));

          CHECK_VX_STATUS(vxuSymmetrical7x7FilterIntel(m_context, iIn0, coefficients_array, shift, iOut0));

          vxReleaseImage(&iIn0);
          vxReleaseImage(&iOut0);
          vxReleaseArray(&coefficients_array);
       }

       if( m_cmdparser->gpurgb2lab.isSet() )
       {
          vx_image r_in = vxCreateImage(m_context, 32, 32, VX_DF_IMAGE_U8);
          CHECK_VX_OBJ(r_in);
          vx_image g_in = vxCreateImage(m_context, 32, 32, VX_DF_IMAGE_U8);
          CHECK_VX_OBJ(g_in);
          vx_image b_in = vxCreateImage(m_context, 32, 32, VX_DF_IMAGE_U8);
          CHECK_VX_OBJ(b_in);
          vx_image l_out = vxCreateImage(m_context, 32, 32, VX_DF_IMAGE_U8);
          CHECK_VX_OBJ(l_out);
          vx_image a_out = vxCreateImage(m_context, 32, 32, VX_DF_IMAGE_U8);
          CHECK_VX_OBJ(a_out);
          vx_image b_out = vxCreateImage(m_context, 32, 32, VX_DF_IMAGE_U8);
          CHECK_VX_OBJ(b_out);

          vx_float32 coefficients[9] = {0.412453, 0.357580, 0.180423, 0.212671, 0.715160, 0.072169, 0.019334, 0.119193, 0.950227};
          vx_float32 recip[3] = {0.950455,1.000000,1.088753};

          vx_array coefficients_array = vxCreateArray(m_context, VX_TYPE_FLOAT32, 9);
          CHECK_VX_OBJ(coefficients_array);
          CHECK_VX_STATUS(vxAddArrayItems(coefficients_array, 9, coefficients, 0));

          vx_array recip_array = vxCreateArray(m_context, VX_TYPE_FLOAT32, 3);
          CHECK_VX_OBJ(recip_array);
          CHECK_VX_STATUS(vxAddArrayItems(recip_array, 3, recip, 0));

          CHECK_VX_STATUS(vxuRgbToLabIntel(m_context,
                                r_in,
                                g_in,
                                b_in,
                                coefficients_array,
                                recip_array,
                                l_out,
                                a_out,
                                b_out));

          CHECK_VX_STATUS(vxReleaseArray(&coefficients_array));
          CHECK_VX_STATUS(vxReleaseArray(&recip_array));
          CHECK_VX_STATUS(vxReleaseImage(&r_in));
          CHECK_VX_STATUS(vxReleaseImage(&g_in));
          CHECK_VX_STATUS(vxReleaseImage(&b_in));
          CHECK_VX_STATUS(vxReleaseImage(&l_out));
          CHECK_VX_STATUS(vxReleaseImage(&a_out));
          CHECK_VX_STATUS(vxReleaseImage(&b_out));
       }

       if( m_cmdparser->gpuboxfilter.isSet() )
       {
          vx_image iIn = vxCreateImage(m_context, 32, 32, VX_DF_IMAGE_U8);
          CHECK_VX_OBJ(iIn);

          vx_image iOut = vxCreateImage(m_context, 32, 32, VX_DF_IMAGE_U8);
          CHECK_VX_OBJ(iOut);

          vx_border_t boxborderMode;
          boxborderMode.mode = VX_BORDER_CONSTANT;
          boxborderMode.constant_value.U32 = 128;
          CHECK_VX_STATUS(vxSetContextAttribute(m_context, VX_CONTEXT_IMMEDIATE_BORDER, &boxborderMode, sizeof(boxborderMode)));

          CHECK_VX_STATUS(vxuBox3x3(m_context, iIn, iOut));

          CHECK_VX_STATUS(vxReleaseImage(&iOut));
          CHECK_VX_STATUS(vxReleaseImage(&iIn));
       }

       if( m_cmdparser->gpusobelfilter.isSet() )
       {
          vx_image iIn = vxCreateImage(m_context, 32, 32, VX_DF_IMAGE_U8);
          CHECK_VX_OBJ(iIn);
          vx_image iOut = vxCreateImage(m_context, 32, 32, VX_DF_IMAGE_S16);
          CHECK_VX_OBJ(iIn);

          vx_graph graph = vxCreateGraph(m_context);

          vx_image iOutX = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_S16);
          CHECK_VX_OBJ(iOutX);
          vx_image iOutY = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_S16);
          CHECK_VX_OBJ(iOutY);

          vx_border_t sobelborderMode;
          sobelborderMode.mode = VX_BORDER_CONSTANT;
          sobelborderMode.constant_value.U32 = 255;

          vx_node nSobel = vxSobel3x3Node(graph, iIn, iOutX, iOutY);
          CHECK_VX_OBJ(nSobel);

          CHECK_VX_STATUS(vxSetNodeAttribute(nSobel, VX_NODE_BORDER, &sobelborderMode, sizeof(sobelborderMode)));

          vx_node mag = vxMagnitudeNode(graph, iOutX, iOutY, iOut);
          CHECK_VX_OBJ(mag);

        CHECK_VX_STATUS(vxSetNodeTarget(mag, VX_TARGET_GPU_INTEL, 0));
        CHECK_VX_STATUS(vxSetNodeTarget(nSobel, VX_TARGET_GPU_INTEL, 0));

          CHECK_VX_STATUS(vxVerifyGraph(graph));

          CHECK_VX_STATUS(vxReleaseImage(&iOutY));
          CHECK_VX_STATUS(vxReleaseImage(&iOutX));
          CHECK_VX_STATUS(vxReleaseGraph(&graph));
           CHECK_VX_STATUS(vxReleaseImage(&iOut));
          CHECK_VX_STATUS(vxReleaseImage(&iIn));
       }

       if( m_cmdparser->gpulut.isSet() )
       {
           vx_image in = vxCreateImage(m_context, 32, 32, VX_DF_IMAGE_U8);
           vx_image out = vxCreateImage(m_context, 32, 32, VX_DF_IMAGE_U8);

           vx_lut lightnessdarknesslut = vxCreateLUT(m_context, VX_TYPE_UINT8, 256);
           CHECK_VX_OBJ(lightnessdarknesslut);

           vx_map_id map_id;
           vx_uint8 *pLUT = 0;
           CHECK_VX_STATUS(vxMapLUT(lightnessdarknesslut, &map_id, (void **)&pLUT, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0));
           for( int i = 0; i < 256; i++)
              pLUT[i] = 0;

          CHECK_VX_STATUS(vxUnmapLUT(lightnessdarknesslut, map_id));

          CHECK_VX_STATUS(vxuTableLookup(m_context, in, lightnessdarknesslut, out));

          CHECK_VX_STATUS(vxReleaseImage(&out));
          CHECK_VX_STATUS(vxReleaseImage(&in));
       }

       if( m_cmdparser->gpuremovefringe.isSet() )
       {
          //Create the L-to-K 'knots' array that we will use as a parameter for RemoveFringe
          vx_array LtoK_array = vxCreateArray(m_context, VX_TYPE_UINT8, 256);

          vx_uint8 LtoK_array_values[256] =
          {
             0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
             0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
             0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
             0xff, 0xff, 0xfe, 0xfd, 0xfc, 0xfc, 0xfb, 0xfa, 0xf9, 0xf9, 0xf8, 0xf7, 0xf6, 0xf6, 0xf5, 0xf4,
             0xf3, 0xf3, 0xf2, 0xf1, 0xf0, 0xf0, 0xef, 0xee, 0xed, 0xed, 0xec, 0xeb, 0xea, 0xea, 0xe9, 0xe8,
             0xe7, 0xe7, 0xe6, 0xe5, 0xe4, 0xe4, 0xe3, 0xe2, 0xe1, 0xe1, 0xe0, 0xdf, 0xde, 0xde, 0xdd, 0xdc,
             0xdb, 0xdb, 0xda, 0xd9, 0xd8, 0xd8, 0xd7, 0xd6, 0xd5, 0xd5, 0xd4, 0xd3, 0xd2, 0xd2, 0xd1, 0xd0,
             0xcf, 0xcf, 0xce, 0xcd, 0xcc, 0xcc, 0xcb, 0xca, 0xc9, 0xc9, 0xc8, 0xc7, 0xc6, 0xc6, 0xc5, 0xc4,
             0xc3, 0xc3, 0xc2, 0xc1, 0xc0, 0xc0, 0xbf, 0xbe, 0xbd, 0xbd, 0xbc, 0xbb, 0xba, 0xba, 0xb9, 0xb8,
             0xb7, 0xb7, 0xb6, 0xb5, 0xb4, 0xb4, 0xb3, 0xb2, 0xb1, 0xb1, 0xb0, 0xaf, 0xae, 0xae, 0xad, 0xac,
             0xab, 0xa9, 0xa6, 0xa3, 0xa1, 0x9e, 0x9b, 0x99, 0x96, 0x93, 0x91, 0x8e, 0x8b, 0x89, 0x86, 0x83,
             0x80, 0x78, 0x70, 0x68, 0x60, 0x58, 0x50, 0x48, 0x40, 0x38, 0x30, 0x28, 0x20, 0x18, 0x10, 0x08,
             0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
             0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
             0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
             0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
          };

          CHECK_VX_STATUS(vxAddArrayItems(LtoK_array, 256, LtoK_array_values, 0));

          {
             vx_image iIn0 = vxCreateImage(m_context, 32, 32, VX_DF_IMAGE_RGBX);
             CHECK_VX_OBJ(iIn0);
             vx_image iIn1 = vxCreateImage(m_context, 32, 32, VX_DF_IMAGE_U8);
             CHECK_VX_OBJ(iIn1);
             vx_image iIn2 = vxCreateImage(m_context, 32, 32, VX_DF_IMAGE_U8);
             CHECK_VX_OBJ(iIn1);
             vx_image iOut0 = vxCreateImage(m_context, 32, 32, VX_DF_IMAGE_RGBX);
             CHECK_VX_OBJ(iOut0);
             vx_image iOut1 = vxCreateImage(m_context, 32, 32, VX_DF_IMAGE_U8);
             CHECK_VX_OBJ(iOut1);
             vx_graph graph = vxCreateGraph(m_context);
             vx_node nRemoveFringe = vxRemoveFringeOpenCLTiledNode(graph,
               iIn0,
               iIn1,
               iIn2,
               iOut0,
               iOut1,
               LtoK_array);
             CHECK_VX_OBJ(nRemoveFringe);
             CHECK_VX_STATUS(vxReleaseNode(&nRemoveFringe));

             CHECK_VX_STATUS(vxVerifyGraph(graph));

             CHECK_VX_STATUS(vxReleaseGraph(&graph));
             CHECK_VX_STATUS(vxReleaseImage(&iOut1));
             CHECK_VX_STATUS(vxReleaseImage(&iOut0));
             CHECK_VX_STATUS(vxReleaseImage(&iIn2));
             CHECK_VX_STATUS(vxReleaseImage(&iIn1));
             CHECK_VX_STATUS(vxReleaseImage(&iIn0));
          }

          {
             vx_image iIn0 = vxCreateImage(m_context, 32, 32, VX_DF_IMAGE_RGBX);
             CHECK_VX_OBJ(iIn0);
             vx_image iIn1 = vxCreateImage(m_context, 32, 32, VX_DF_IMAGE_U8);
             CHECK_VX_OBJ(iIn1);
             vx_image iIn2 = vxCreateImage(m_context, 32, 32, VX_DF_IMAGE_U8);
             CHECK_VX_OBJ(iIn1);
             vx_image iOut0 = vxCreateImage(m_context, 32, 32, VX_DF_IMAGE_U8);
             CHECK_VX_OBJ(iOut0);
             vx_image iOut1 = vxCreateImage(m_context, 32, 32, VX_DF_IMAGE_U8);
             CHECK_VX_OBJ(iOut1);
             vx_image iOut2 = vxCreateImage(m_context, 32, 32, VX_DF_IMAGE_U8);
             CHECK_VX_OBJ(iOut2);
             vx_image iOut3 = vxCreateImage(m_context, 32, 32, VX_DF_IMAGE_U8);
             CHECK_VX_OBJ(iOut3);
             vx_image iOut4 = vxCreateImage(m_context, 32, 32, VX_DF_IMAGE_U8);
             CHECK_VX_OBJ(iOut4);
             vx_graph graph = vxCreateGraph(m_context);
             vx_node nRemoveFringe = vxRemoveFringePlanarOpenCLTiledNode(graph,
               iIn0,
               iIn1,
               iIn2,
               iOut0,
               iOut1,
               iOut2,
               iOut3,
               iOut4,
               LtoK_array);
             CHECK_VX_OBJ(nRemoveFringe);
             CHECK_VX_STATUS(vxReleaseNode(&nRemoveFringe));

             CHECK_VX_STATUS(vxVerifyGraph(graph));

             CHECK_VX_STATUS(vxReleaseGraph(&graph));
             CHECK_VX_STATUS(vxReleaseImage(&iOut4));
             CHECK_VX_STATUS(vxReleaseImage(&iOut3));
             CHECK_VX_STATUS(vxReleaseImage(&iOut2));
             CHECK_VX_STATUS(vxReleaseImage(&iOut1));
             CHECK_VX_STATUS(vxReleaseImage(&iOut0));
             CHECK_VX_STATUS(vxReleaseImage(&iIn2));
             CHECK_VX_STATUS(vxReleaseImage(&iIn1));
             CHECK_VX_STATUS(vxReleaseImage(&iIn0));
          }
          CHECK_VX_STATUS(vxReleaseArray(&LtoK_array));
       }

       //change the default target for immediate mode back to CPU
       CHECK_VX_STATUS(vxSetImmediateModeTarget(m_context, VX_TARGET_CPU_INTEL, 0));
    }
}
