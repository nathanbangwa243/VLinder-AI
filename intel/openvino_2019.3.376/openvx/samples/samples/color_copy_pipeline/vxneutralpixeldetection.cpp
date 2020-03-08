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

#include <VX/vx_intel_volatile.h>
#include <stdio.h>
#include <immintrin.h>
#include "vx_user_pipeline_nodes.h"


static vx_status vxNeutralPixelDetectionKernel(vx_node node,
                                           void *   parameters[],
                                           vx_uint32 num,
                                           void *   tile_memory,
                                           vx_size tile_memory_size)
{
    vx_tile_intel_t *pInTileA = (vx_tile_intel_t *)parameters[0];
    vx_tile_intel_t *pInTileB = (vx_tile_intel_t *)parameters[1];
    vx_tile_intel_t *pOutTileMask = (vx_tile_intel_t *)parameters[2];

    vx_uint8 *paSrc = pInTileA->base[0];
    vx_uint8 *pbSrc = pInTileB->base[0];
    vx_uint8 *pMaskDst = pOutTileMask->base[0];

    int srcaStep = pInTileA->addr[0].stride_y;
    int srcbStep = pInTileB->addr[0].stride_y;
    int dstMaskStep = pOutTileMask->addr[0].stride_y;

    int tile_width = pOutTileMask->addr[0].dim_x*8;
    int tile_height = pOutTileMask->addr[0].dim_y;

    //we'll unroll by 16
    int loop_unrolled = tile_width / 16;
    int remaining_pixels = tile_width % 16;

    //the thresholds are hard-coded for now.
    //We could always have them be passed in as
    // a parameter.

    const unsigned char threshLow = 54;
    const unsigned char threshHigh = 74;

    const __m128i vReversemask = _mm_setr_epi8(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8);
    __m128i vThreshlow = _mm_set1_epi8(threshLow);
    __m128i vThreshhigh = _mm_set1_epi8(threshHigh);
    __m128i vBias = _mm_set1_epi8(64);

    unsigned char *output;
    unsigned char *avgA;
    unsigned char *avgB;

    for( int scanline = 0; scanline < tile_height; scanline++ )
    {
        output = pMaskDst + scanline*dstMaskStep;
        avgA = paSrc + scanline*srcaStep;
        avgB = pbSrc + scanline*srcbStep;

        vx_uint16 *pDst = (vx_uint16 *)output;

        for( int outer_unroll = 0; outer_unroll < loop_unrolled; outer_unroll++ )
        {
            //load two vectors, one for a, one for b
            __m128i aVec = _mm_load_si128( (__m128i *)avgA);
            __m128i bVec = _mm_load_si128( (__m128i *)avgB);

            //subtract the bias
            aVec = _mm_sub_epi8( aVec, vBias);
            bVec = _mm_sub_epi8( bVec, vBias);


            //is a/b greater than low threshold?
            __m128i agtVec = _mm_cmpgt_epi8(aVec, vThreshlow);
            __m128i bgtVec = _mm_cmpgt_epi8(bVec, vThreshlow);

            //is a/b less than high threshold?
            __m128i altVec = _mm_cmplt_epi8(aVec, vThreshhigh);
            __m128i bltVec = _mm_cmplt_epi8(bVec, vThreshhigh);

            //is a/b higher than low AND less than high?
            __m128i anVec = _mm_and_si128(agtVec, altVec);
            __m128i bnVec = _mm_and_si128(bgtVec, bltVec);

            //and both a and b for result
            __m128i resultv = _mm_and_si128(anVec, bnVec);

            //store it
           // _mm_store_si128((__m128i *)output, resultv);
            //_mm_store_si128((__m128i *)output, _mm_set1_epi8(255));
            resultv = _mm_shuffle_epi8(resultv, vReversemask);

            *pDst++ = _mm_movemask_epi8(resultv);

            avgA += 16;
            avgB += 16;
        }

    }
    return VX_SUCCESS;
}

vx_status VX_CALLBACK vxNeutralPixelDetectionValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{

    vx_status status = VX_ERROR_INVALID_PARAMETERS;
    if(num!=3)
    {
        return status;
    }

    vx_df_image format = 0;
    vx_uint32 width = 0;
    vx_uint32 height = 0;

    if(vxQueryImage((vx_image)parameters[0], VX_IMAGE_FORMAT, &format, sizeof(format)) == VX_SUCCESS)
    {
        if (format == VX_DF_IMAGE_U8)
        {
            status = VX_SUCCESS;
        }
        else
        {
            status = VX_ERROR_INVALID_VALUE;
            vxAddLogEntry((vx_reference)node, status, "NeutralPixelDetection Input Validation failed: invalid input image 0 format, it must be U8\n");
            return status;
        }
    }

    if(vxQueryImage((vx_image)parameters[1], VX_IMAGE_FORMAT, &format, sizeof(format)) == VX_SUCCESS)
    {
        if (format == VX_DF_IMAGE_U8)
        {
            status = VX_SUCCESS;
        }
        else
        {
            status = VX_ERROR_INVALID_VALUE;
            vxAddLogEntry((vx_reference)node, status, "NeutralPixelDetection Input Validation failed: invalid input 1 image format, it must be U8\n");
            return status;
        }
    }

    status = vxQueryImage((vx_image)parameters[0], VX_IMAGE_WIDTH, &width, sizeof(width));


    status = vxQueryImage((vx_image)parameters[0], VX_IMAGE_HEIGHT, &height, sizeof(height));

    width/=8;

    PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[2], VX_IMAGE_WIDTH, &width, sizeof(width)));
    PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[2], VX_IMAGE_HEIGHT, &height, sizeof(height)));
    PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[2], VX_IMAGE_FORMAT, &format, sizeof(format)));

    return status;
}

vx_status vxNeutralPixelDetectionTileMapping (vx_node node,
                                         vx_reference parameters[],
                                         const vx_tile_t_attributes_intel_t* dstRectIn,
                                         vx_tile_t_attributes_intel_t* srcRectOut,
                                         vx_uint32 param_num)
{
   srcRectOut->x = dstRectIn->x*8;
   srcRectOut->y = dstRectIn->y;
   srcRectOut->tile_block.width = dstRectIn->tile_block.width*8;
   srcRectOut->tile_block.height = dstRectIn->tile_block.height;

   return VX_SUCCESS;
}


vx_status VX_API_CALL PublishNeutralPixelDetectionKernel(vx_context context)
{
    vx_status status = VX_SUCCESS;
    //create a kernel via the vxAddAdvancedTilingKernelIntel interface
    vx_kernel kernel = vxAddAdvancedTilingKernelIntel(context,
        (char*)VX_KERNEL_NAME_USER_NEUTRALPIXELDETECTION,
        VX_KERNEL_USER_NEUTRALPIXELDETECTION,
        vxNeutralPixelDetectionKernel,
        vxNeutralPixelDetectionTileMapping,
        3,
        vxNeutralPixelDetectionValidator,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL);

    PROCESS_VX_STATUS(context, vxGetStatus((vx_reference)kernel));
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));


    vx_tile_block_size_intel_t blockSize;

    //Since we are using SSE Intrinsics, we generally want the tile
    // width to be divisible by chunks of 16 pixels.
    blockSize.width = 16;
    blockSize.height = 1;
    SAFE_VX_CALL(status, context, vxSetKernelAttribute(kernel, VX_KERNEL_OUTPUT_TILE_BLOCK_SIZE_INTEL, &blockSize, sizeof(vx_tile_block_size_intel_t)));
    SAFE_VX_CALL(status, context, vxFinalizeKernel(kernel));

    if( VX_SUCCESS != status )
        vxRemoveKernel( kernel );

    return status;
}
