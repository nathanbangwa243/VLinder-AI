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


static vx_status vxBackgroundSuppressKernel(vx_node node,
                                           void *   parameters[],
                                           vx_uint32 num,
                                           void *   tile_memory,
                                           vx_size tile_memory_size)
{
    vx_tile_intel_t *pInTileL = (vx_tile_intel_t *)parameters[0];
    vx_tile_intel_t *pInTileA = (vx_tile_intel_t *)parameters[1];
    vx_tile_intel_t *pInTileB = (vx_tile_intel_t *)parameters[2];
    vx_tile_intel_t *pOutTileL = (vx_tile_intel_t *)parameters[3];
    vx_tile_intel_t *pOutTileA = (vx_tile_intel_t *)parameters[4];
    vx_tile_intel_t *pOutTileB = (vx_tile_intel_t *)parameters[5];


    vx_uint8 *pLSrc = pInTileL->base[0];
    vx_uint8 *paSrc = pInTileA->base[0];
    vx_uint8 *pbSrc = pInTileB->base[0];
    vx_uint8 *pLDst = pOutTileL->base[0];
    vx_uint8 *paDst = pOutTileA->base[0];
    vx_uint8 *pbDst = pOutTileB->base[0];

    int srcLStep = pInTileL->addr[0].stride_y;
    int srcaStep = pInTileA->addr[0].stride_y;
    int srcbStep = pInTileB->addr[0].stride_y;
    int dstLStep = pOutTileL->addr[0].stride_y;
    int dstaStep = pOutTileA->addr[0].stride_y;
    int dstbStep = pOutTileB->addr[0].stride_y;

    __m128i L_thresh = _mm_set1_epi8(220);

    __m128i a_low = _mm_set1_epi8(116);
    __m128i a_high = _mm_set1_epi8(140);

    __m128i b_low = _mm_set1_epi8(116);
    __m128i b_high = _mm_set1_epi8(140);

    __m128i whitepointL = _mm_set1_epi8(255);
    __m128i whitepointa = _mm_set1_epi8(127);
    __m128i whitepointb = _mm_set1_epi8(127);

    __m128i L_low_thresh = _mm_set1_epi8(70);
    __m128i neutral = _mm_set1_epi8(128);
    __m128i plusminus = _mm_set1_epi8(10);

    __m128i neutral_plus_plusminus = _mm_adds_epu8(neutral, plusminus);
    __m128i neutral_minus_plusminus = _mm_subs_epu8(neutral, plusminus);

    int tile_width = pOutTileL->addr[0].dim_x;
    int tile_height = pOutTileL->addr[0].dim_y;

    //we will process 16 pixels per iteration
    int loop_unrolled = tile_width / 16;

    for( int scanline = 0; scanline < tile_height; scanline++ )
    {
        __m128i *pinputL = (__m128i *)(pLSrc + scanline*srcLStep);
        __m128i *pinputa = (__m128i *)(paSrc + scanline*srcaStep);
        __m128i *pinputb = (__m128i *)(pbSrc + scanline*srcbStep);

        __m128i *poutputL = (__m128i *)(pLDst + scanline*dstLStep);
        __m128i *poutputa = (__m128i *)(paDst + scanline*dstaStep);
        __m128i *poutputb = (__m128i *)(pbDst + scanline*dstbStep);

        for( int output_pixel = 0; output_pixel < loop_unrolled; output_pixel++)
        {
            //L
            __m128i inputL = _mm_load_si128(pinputL);
            pinputL++;

            __m128i tmp = _mm_max_epu8(inputL, L_thresh);
            __m128i block0 = _mm_cmpeq_epi8(inputL, tmp); //inputL[fs] >= L_thresh

            tmp = _mm_min_epu8(inputL, L_low_thresh);
            __m128i block1 = _mm_cmpeq_epi8(inputL, tmp); //inputL[fs] <= L_low_thresh


            //A
            __m128i inputA = _mm_load_si128(pinputa);
            pinputa++;

            tmp = _mm_min_epu8(inputA, a_high);
            block0 = _mm_and_si128( block0, _mm_cmpeq_epi8(inputA, tmp)); //inputa[fs] <= a_high

            tmp = _mm_max_epu8(inputA, a_low);
            block0 = _mm_and_si128( block0, _mm_cmpeq_epi8(inputA, tmp)); //inputa[fs] >= a_low

            tmp = _mm_min_epu8(inputA, neutral_plus_plusminus);
            block1 = _mm_and_si128(block1, _mm_cmpeq_epi8(inputA, tmp)); //(inputa[fs] <= (neutral + plusminus))

            tmp = _mm_max_epu8(inputA, neutral_minus_plusminus);
            block1 = _mm_and_si128(block1, _mm_cmpeq_epi8(inputA, tmp)); //(inputa[fs] >= (neutral - plusminus)

            //B
            __m128i inputB = _mm_load_si128(pinputb);
            pinputb++;

            tmp = _mm_min_epu8(inputB, b_high);
            block0 = _mm_and_si128( block0, _mm_cmpeq_epi8(inputB, tmp)); //inputb[fs] <= b_high
            tmp = _mm_max_epu8(inputB, b_low);
            block0 = _mm_and_si128( block0, _mm_cmpeq_epi8(inputB, tmp)); //inputb[fs] <= b_high

            tmp = _mm_min_epu8(inputB, neutral_plus_plusminus);
            block1 = _mm_and_si128(block1, _mm_cmpeq_epi8(inputB, tmp)); //(inputb[fs] <= (neutral + plusminus))

            tmp = _mm_max_epu8(inputB, neutral_minus_plusminus);
            block1 = _mm_and_si128(block1, _mm_cmpeq_epi8(inputB, tmp)); //(inputb[fs] >= (neutral - plusminus)

            //__m128i outputL = _mm_or_si128(_mm_and_si128(whitepointL, block0), _mm_and_si128(inputL, block1));
            __m128i outputL = _mm_blendv_epi8(inputL, whitepointL, block0);

            __m128i outputA = _mm_or_si128(_mm_and_si128(whitepointa, block0), _mm_and_si128(neutral, block1));
            __m128i outputB = _mm_or_si128(_mm_and_si128(whitepointb, block0), _mm_and_si128(neutral, block1));

            __m128i block0_or_block1 = _mm_or_si128(block0, block1);

            outputA = _mm_or_si128(outputA, _mm_andnot_si128(block0_or_block1, inputA));
            outputB = _mm_or_si128(outputB, _mm_andnot_si128(block0_or_block1, inputB));

            //store
            _mm_store_si128(poutputL, outputL);
            poutputL++;

            _mm_store_si128(poutputa, outputA);
            poutputa++;

            _mm_store_si128(poutputb, outputB);
            poutputb++;


        }

    }
    return VX_SUCCESS;

}

vx_status VX_CALLBACK vxBackgroundSuppressValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{

    vx_status status = VX_ERROR_INVALID_PARAMETERS;
    if(num!=6)
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
            vxAddLogEntry((vx_reference)node, status, "BackgroundSuppress Input Validation failed: invalid input image 0 format, it must be U8\n");
            return status;
        }
    }
    status = vxQueryImage((vx_image)parameters[0], VX_IMAGE_WIDTH, &width, sizeof(width));
    if (width%16!=0)
    {
        status = VX_ERROR_INVALID_VALUE;
        vxAddLogEntry((vx_reference)node, status, "BackgroundSuppress Input Validation failed: input image 0 width must be evenly divisible by 16\n");
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
            vxAddLogEntry((vx_reference)node, status, "BackgroundSuppress Input Validation failed: invalid input 1 image format, it must be U8\n");
            return status;
        }
    }
    status = vxQueryImage((vx_image)parameters[1], VX_IMAGE_WIDTH, &width, sizeof(width));
    if (width%16!=0)
    {
        status = VX_ERROR_INVALID_VALUE;
        vxAddLogEntry((vx_reference)node, status, "BackgroundSuppress Input Validation failed: input image 1 width must be evenly divisible by 16\n");
        return status;
    }

    if(vxQueryImage((vx_image)parameters[2], VX_IMAGE_FORMAT, &format, sizeof(format)) == VX_SUCCESS)
    {
        if (format == VX_DF_IMAGE_U8)
        {
            status = VX_SUCCESS;
        }
        else
        {
            status = VX_ERROR_INVALID_VALUE;
            vxAddLogEntry((vx_reference)node, status, "BackgroundSuppress Input Validation failed: invalid input image 2 format, it must be U8\n");
            return status;
        }
    }
    status = vxQueryImage((vx_image)parameters[2], VX_IMAGE_WIDTH, &width, sizeof(width));
    if (width%16!=0)
    {
        status = VX_ERROR_INVALID_VALUE;
        vxAddLogEntry((vx_reference)node, status, "BackgroundSuppress Input Validation failed: input image 2 width must be evenly divisible by 16\n");
        return status;
    }


    status = vxQueryImage((vx_image)parameters[0], VX_IMAGE_HEIGHT, &height, sizeof(height));


    PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[3], VX_IMAGE_WIDTH, &width, sizeof(width)));
    PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[3], VX_IMAGE_HEIGHT, &height, sizeof(height)));
    PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[3], VX_IMAGE_FORMAT, &format, sizeof(format)));
    PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[4], VX_IMAGE_WIDTH, &width, sizeof(width)));
    PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[4], VX_IMAGE_HEIGHT, &height, sizeof(height)));
    PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[4], VX_IMAGE_FORMAT, &format, sizeof(format)));
    PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[5], VX_IMAGE_WIDTH, &width, sizeof(width)));
    PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[5], VX_IMAGE_HEIGHT, &height, sizeof(height)));
    PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[5], VX_IMAGE_FORMAT, &format, sizeof(format)));

    return status;
}


vx_status VX_API_CALL PublishBackgroundSuppressKernel(vx_context context)
{
    vx_status status = VX_SUCCESS;
    //create a kernel via the vxAddAdvancedTilingKernelIntel interface
    vx_kernel kernel = vxAddAdvancedTilingKernelIntel(context,
        (char*)VX_KERNEL_NAME_USER_BACKGROUNDSUPPRESS,
        VX_KERNEL_USER_BACKGROUNDSUPPRESS,
        vxBackgroundSuppressKernel,
        NULL,
        6,
        vxBackgroundSuppressValidator,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL);

    PROCESS_VX_STATUS(context, vxGetStatus((vx_reference)kernel));
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, 3, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, 4, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, 5, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));

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
