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


#define _mm_srli_epi8(_A, _Imm) _mm_and_si128( _mm_set1_epi8((int8_t)(0xFF >> _Imm)), _mm_srli_epi32( _A, _Imm ) )


static inline void RemoveFringeCMYK_GenerateK(vx_uint8 *pSrcCMYK,
                                              vx_uint8 *pSrcL,
                                              vx_uint8 *pSrcNeutralEdgeMask,
                                              vx_uint8 *pDstCMYK,
                                              vx_uint8 *pDstK,
                                              int srcCMYKStep,
                                              int srcLStep,
                                              int srcNeutralEdgeStep,
                                              int dstCMYKStep,
                                              int dstKStep,
                                              int tile_width,
                                              int tile_height,
                                              __m128i vknots)
{
    __m128i vlowermask = _mm_set1_epi8(0x0F);
    __m128i v0x1 = _mm_set1_epi8(0x01);
    __m128i v0xff = _mm_set1_epi8(0xff);

    int unrolled_iterations = tile_width / 16;
#if defined(__linux__)
    __attribute__ ((aligned(16)))
#else
    __declspec(align(16))
#endif

    __m128i CMYK_Mask_Shuffle[4];
    CMYK_Mask_Shuffle[0] = _mm_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
    CMYK_Mask_Shuffle[1] = _mm_setr_epi8(4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7);
    CMYK_Mask_Shuffle[2] = _mm_setr_epi8(8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11);
    CMYK_Mask_Shuffle[3] = _mm_setr_epi8(12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15);


    __m128i CMYK_NEOutput_Shuffle[4];
    CMYK_NEOutput_Shuffle[0] = _mm_setr_epi8(0x80, 0x80, 0x80, 0, 0x80, 0x80, 0x80, 1, 0x80, 0x80, 0x80, 2, 0x80, 0x80, 0x80, 3);
    CMYK_NEOutput_Shuffle[1] = _mm_setr_epi8(0x80, 0x80, 0x80, 4, 0x80, 0x80, 0x80, 5, 0x80, 0x80, 0x80, 6, 0x80, 0x80, 0x80, 7);
    CMYK_NEOutput_Shuffle[2] = _mm_setr_epi8(0x80, 0x80, 0x80, 8, 0x80, 0x80, 0x80, 9, 0x80, 0x80, 0x80, 10, 0x80, 0x80, 0x80, 11);
    CMYK_NEOutput_Shuffle[3] = _mm_setr_epi8(0x80, 0x80, 0x80, 12, 0x80, 0x80, 0x80, 13, 0x80, 0x80, 0x80, 14, 0x80, 0x80, 0x80, 15);


    for( int ss = 0; ss < tile_height; ss++ )
    {
        __m128i  *mask128  = (__m128i *)(pSrcNeutralEdgeMask + ss*srcNeutralEdgeStep);
        __m128i  *L128 = (__m128i *)(pSrcL + ss*srcLStep);
        __m128i  *output128 = (__m128i *)(pDstCMYK + ss*dstCMYKStep);
        __m128i  *outputK = (__m128i *)(pDstK + ss*dstKStep);
        __m128i  *input128 = (__m128i *)(pSrcCMYK + ss*srcCMYKStep);

        for( int pixel = 0; pixel < unrolled_iterations; pixel++)
        {
            __m128i vinput8 = _mm_load_si128(L128);
            L128++;

            //The lower bound index is input / 16
            __m128i vlowerboundindex = _mm_srli_epi8(vinput8, 4);

            //obtain the lower bound value by passing
            __m128i vlowerboundvalue = _mm_shuffle_epi8(vknots, vlowerboundindex);

            //The upper bound index
            __m128i vupperboundindex = _mm_add_epi8(vlowerboundindex, v0x1);

            //obtain the upper bound value
            __m128i vupperboundvalue = _mm_shuffle_epi8(vknots, vupperboundindex);

            //if the upper bound index was 15, set the value to 255. Otherwise leave it alone
            vupperboundvalue = _mm_blendv_epi8(vupperboundvalue, v0xff, _mm_cmpeq_epi8(vlowermask, vupperboundvalue));

            //get the lower minus upper value
            __m128i vupper_minus_lower8 = _mm_sub_epi8(vlowerboundvalue, vupperboundvalue);

            //get where the point resides between the lower and upper bound (range 0 to 15)
            __m128i vbetween8 = _mm_and_si128( vinput8, vlowermask );

            //convert to 16 bits for interpolation
            __m128i vupper_minus_lower16 = _mm_cvtepu8_epi16(vupper_minus_lower8);
            __m128i vbetween16 = _mm_cvtepu8_epi16(vbetween8);

            //multiply them
            __m128i vmul0 = _mm_mullo_epi16(vupper_minus_lower16, vbetween16);
            //divide by 16
            vmul0 = _mm_srli_epi16(vmul0, 4);

            vupper_minus_lower8 = _mm_srli_si128(vupper_minus_lower8, 8);
            vbetween8 = _mm_srli_si128(vbetween8, 8);


            //convert to 16 bits for interpolation
            vupper_minus_lower16 = _mm_cvtepu8_epi16(vupper_minus_lower8);
            vbetween16 = _mm_cvtepu8_epi16(vbetween8);

            //multiply them
            __m128i vmul1 = _mm_mullo_epi16(vupper_minus_lower16, vbetween16);
            //divide by 16
            vmul1 = _mm_srli_epi16(vmul1, 4);

            __m128i mappedv = _mm_packus_epi16(vmul0, vmul1);

            mappedv = _mm_subs_epu8(vlowerboundvalue, mappedv );

            //load up 16 pixels of our mask
            __m128i maskv = _mm_load_si128(mask128);
            mask128++;

            //write the K output
            _mm_store_si128(outputK, mappedv);
            outputK++;

            //do 4 pixels at a time
#pragma unroll(4)
            for( int i = 0; i < 4; i++)
            {
                //load up 4 pixels worth of cmyk input

                __m128i CMYK_Mask = _mm_shuffle_epi8(maskv, CMYK_Mask_Shuffle[i]);
                __m128i CMYK_NEOutput = _mm_shuffle_epi8(mappedv, CMYK_NEOutput_Shuffle[i]);
                //__m128i result = _mm_setzero_si128();
                __m128i result = _mm_and_si128(CMYK_Mask, CMYK_NEOutput);
                __m128i CMYK_Input = _mm_load_si128(input128);
                result = _mm_or_si128(result, _mm_andnot_si128(CMYK_Mask, CMYK_Input));

                //store the output
                _mm_store_si128(output128, result);
                input128++;
                output128++;
            }

        }
    }
}

static inline void RemoveFringeCMYKOnly(vx_uint8 *pSrcCMYK,
                                        vx_uint8 *pSrcL,
                                        vx_uint8 *pSrcNeutralEdgeMask,
                                        vx_uint8 *pDstCMYK,
                                        int srcCMYKStep,
                                        int srcLStep,
                                        int srcNeutralEdgeStep,
                                        int dstCMYKStep,
                                        int tile_width,
                                        int tile_height,
                                        __m128i vknots)
{
    __m128i vlowermask = _mm_set1_epi8(0x0F);
    __m128i v0x1 = _mm_set1_epi8(0x01);
    __m128i v0xff = _mm_set1_epi8(0xff);

    int unrolled_iterations = tile_width / 16;
#if defined (__linux__)
    __attribute__ ((aligned(16)))
#else
    __declspec(align(16))
#endif

    __m128i CMYK_Mask_Shuffle[4];
    CMYK_Mask_Shuffle[0] = _mm_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
    CMYK_Mask_Shuffle[1] = _mm_setr_epi8(4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7);
    CMYK_Mask_Shuffle[2] = _mm_setr_epi8(8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11);
    CMYK_Mask_Shuffle[3] = _mm_setr_epi8(12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15);


    __m128i CMYK_NEOutput_Shuffle[4];
    CMYK_NEOutput_Shuffle[0] = _mm_setr_epi8(0x80, 0x80, 0x80, 0, 0x80, 0x80, 0x80, 1, 0x80, 0x80, 0x80, 2, 0x80, 0x80, 0x80, 3);
    CMYK_NEOutput_Shuffle[1] = _mm_setr_epi8(0x80, 0x80, 0x80, 4, 0x80, 0x80, 0x80, 5, 0x80, 0x80, 0x80, 6, 0x80, 0x80, 0x80, 7);
    CMYK_NEOutput_Shuffle[2] = _mm_setr_epi8(0x80, 0x80, 0x80, 8, 0x80, 0x80, 0x80, 9, 0x80, 0x80, 0x80, 10, 0x80, 0x80, 0x80, 11);
    CMYK_NEOutput_Shuffle[3] = _mm_setr_epi8(0x80, 0x80, 0x80, 12, 0x80, 0x80, 0x80, 13, 0x80, 0x80, 0x80, 14, 0x80, 0x80, 0x80, 15);


    for( int ss = 0; ss < tile_height; ss++ )
    {
        __m128i  *mask128  = (__m128i *)(pSrcNeutralEdgeMask + ss*srcNeutralEdgeStep);
        __m128i  *L128 = (__m128i *)(pSrcL + ss*srcLStep);
        __m128i  *output128 = (__m128i *)(pDstCMYK + ss*dstCMYKStep);
        __m128i  *input128 = (__m128i *)(pSrcCMYK + ss*srcCMYKStep);

        for( int pixel = 0; pixel < unrolled_iterations; pixel++)
        {
            __m128i vinput8 = _mm_load_si128(L128);
            L128++;

            //The lower bound index is input / 16
            __m128i vlowerboundindex = _mm_srli_epi8(vinput8, 4);

            //obtain the lower bound value by passing
            __m128i vlowerboundvalue = _mm_shuffle_epi8(vknots, vlowerboundindex);

            //The upper bound index
            __m128i vupperboundindex = _mm_add_epi8(vlowerboundindex, v0x1);

            //obtain the upper bound value
            __m128i vupperboundvalue = _mm_shuffle_epi8(vknots, vupperboundindex);

            //if the upper bound index was 15, set the value to 255. Otherwise leave it alone
            vupperboundvalue = _mm_blendv_epi8(vupperboundvalue, v0xff, _mm_cmpeq_epi8(vlowermask, vupperboundvalue));

            //get the lower minus upper value
            __m128i vupper_minus_lower8 = _mm_sub_epi8(vlowerboundvalue, vupperboundvalue);

            //get where the point resides between the lower and upper bound (range 0 to 15)
            __m128i vbetween8 = _mm_and_si128( vinput8, vlowermask );

            //convert to 16 bits for interpolation
            __m128i vupper_minus_lower16 = _mm_cvtepu8_epi16(vupper_minus_lower8);
            __m128i vbetween16 = _mm_cvtepu8_epi16(vbetween8);

            //multiply them
            __m128i vmul0 = _mm_mullo_epi16(vupper_minus_lower16, vbetween16);
            //divide by 16
            vmul0 = _mm_srli_epi16(vmul0, 4);

            vupper_minus_lower8 = _mm_srli_si128(vupper_minus_lower8, 8);
            vbetween8 = _mm_srli_si128(vbetween8, 8);


            //convert to 16 bits for interpolation
            vupper_minus_lower16 = _mm_cvtepu8_epi16(vupper_minus_lower8);
            vbetween16 = _mm_cvtepu8_epi16(vbetween8);

            //multiply them
            __m128i vmul1 = _mm_mullo_epi16(vupper_minus_lower16, vbetween16);
            //divide by 16
            vmul1 = _mm_srli_epi16(vmul1, 4);

            __m128i mappedv = _mm_packus_epi16(vmul0, vmul1);

            mappedv = _mm_subs_epu8(vlowerboundvalue, mappedv );

            //load up 16 pixels of our mask
            __m128i maskv = _mm_load_si128(mask128);
            mask128++;

            //do 4 pixels at a time
#pragma unroll(4)
            for( int i = 0; i < 4; i++)
            {
                //load up 4 pixels worth of cmyk input

                __m128i CMYK_Mask = _mm_shuffle_epi8(maskv, CMYK_Mask_Shuffle[i]);
                __m128i CMYK_NEOutput = _mm_shuffle_epi8(mappedv, CMYK_NEOutput_Shuffle[i]);
                //__m128i result = _mm_setzero_si128();
                __m128i result = _mm_and_si128(CMYK_Mask, CMYK_NEOutput);
                __m128i CMYK_Input = _mm_load_si128(input128);
                result = _mm_or_si128(result, _mm_andnot_si128(CMYK_Mask, CMYK_Input));

                //store the output
                _mm_store_si128(output128, result);
                input128++;
                output128++;
            }

        }
    }
}


static vx_status vxRemoveFringeKernel(vx_node node,
                                           void *   parameters[],
                                           vx_uint32 num,
                                           void *   tile_memory,
                                           vx_size tile_memory_size)
{
    vx_tile_intel_t *pInTileCMYK = (vx_tile_intel_t *)parameters[0];
    vx_tile_intel_t *pInTileL = (vx_tile_intel_t *)parameters[1];
    vx_tile_intel_t *pInTileNeutralEdgeMask = (vx_tile_intel_t *)parameters[2];
    vx_tile_intel_t *pOutTileCMYK = (vx_tile_intel_t *)parameters[3];
    vx_tile_intel_t *pOutTileK = (vx_tile_intel_t *)parameters[4];


    vx_uint8 *pSrcCMYK = pInTileCMYK->base[0];
    vx_uint8 *pSrcL = pInTileL->base[0];
    vx_uint8 *pSrcNeutralEdgeMask = pInTileNeutralEdgeMask->base[0];
    vx_uint8 *pDstCMYK = pOutTileCMYK->base[0];
    vx_uint8 *pDstK;

    if( pOutTileK )
    {
        pDstK = pOutTileK->base[0];
    }

    int srcCMYKStep = pInTileCMYK->addr[0].stride_y;
    int srcLStep = pInTileL->addr[0].stride_y;
    int srcNeutralEdgeStep = pInTileNeutralEdgeMask->addr[0].stride_y;
    int dstCMYKStep = pOutTileCMYK->addr[0].stride_y;
    int dstKStep;

    if( pOutTileK )
    {
        dstKStep = pOutTileK->addr[0].stride_y;
    }

    int tile_width = pOutTileCMYK->addr[0].dim_x;
    int tile_height = pOutTileCMYK->addr[0].dim_y;

    vx_array LtoK_knot_array = (vx_array)parameters[5];

    //grab the elements from the array
    vx_uint8 *pKnotsArray = 0;
    vx_size stride = 0;
    vx_size num_items = 0;

    vxQueryArray(LtoK_knot_array, VX_ARRAY_CAPACITY, &num_items, sizeof(num_items));
    vx_map_id map_id;

	PROCESS_VX_STATUS(node, vxMapArrayRange(LtoK_knot_array, 0, num_items, &map_id, &stride, reinterpret_cast<void**>(&pKnotsArray), VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));

    __m128i vknots = _mm_setr_epi8(pKnotsArray[0],
        pKnotsArray[1],
        pKnotsArray[2],
        pKnotsArray[3],
        pKnotsArray[4],
        pKnotsArray[5],
        pKnotsArray[6],
        pKnotsArray[7],
        pKnotsArray[8],
        pKnotsArray[9],
        pKnotsArray[10],
        pKnotsArray[11],
        pKnotsArray[12],
        pKnotsArray[13],
        pKnotsArray[14],
        pKnotsArray[15]);

    //let the VX system know that we're done reading from these elements
    PROCESS_VX_STATUS(node, vxUnmapArrayRange(LtoK_knot_array, map_id));

    if( pOutTileK )
    {
        RemoveFringeCMYK_GenerateK(pSrcCMYK,
            pSrcL,
            pSrcNeutralEdgeMask,
            pDstCMYK,
            pDstK,
            srcCMYKStep,
            srcLStep,
            srcNeutralEdgeStep,
            dstCMYKStep,
            dstKStep,
            tile_width,
            tile_height,
            vknots);
    }
    else
    {
        RemoveFringeCMYKOnly(pSrcCMYK,
            pSrcL,
            pSrcNeutralEdgeMask,
            pDstCMYK,
            srcCMYKStep,
            srcLStep,
            srcNeutralEdgeStep,
            dstCMYKStep,
            tile_width,
            tile_height,
            vknots);
    }
    return VX_SUCCESS;
}

vx_status VX_CALLBACK vxRemoveFringeValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{

    vx_status status = VX_ERROR_INVALID_PARAMETERS;
    if(num!=6)
    {
        return status;
    }

    vx_df_image format = 0;
    vx_uint32 width = 0;
    vx_uint32 height = 0;
    vx_size num_items = 0;

    if(vxQueryImage((vx_image)parameters[0], VX_IMAGE_FORMAT, &format, sizeof(format)) == VX_SUCCESS)
    {
        if (format == VX_DF_IMAGE_RGBX)
        {
            status = VX_SUCCESS;
        }
        else
        {
            status = VX_ERROR_INVALID_VALUE;
            vxAddLogEntry((vx_reference)node, status, "RemoveFringe Input Validation failed: invalid input image 0 format, it must be RGBX\n");
            return status;
        }
    }
    status = vxQueryImage((vx_image)parameters[0], VX_IMAGE_WIDTH, &width, sizeof(width));
    if (width%16!=0)
    {
        status = VX_ERROR_INVALID_VALUE;
        vxAddLogEntry((vx_reference)node, status, "RemoveFringe Input Validation failed: input image 0 width must be evenly divisible by 16\n");
        return status;
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
            vxAddLogEntry((vx_reference)node, status, "RemoveFringe Input Validation failed: invalid input image 1 format, it must be U8\n");
            return status;
        }
    }
    status = vxQueryImage((vx_image)parameters[1], VX_IMAGE_WIDTH, &width, sizeof(width));
    if (width%16!=0)
    {
        status = VX_ERROR_INVALID_VALUE;
        vxAddLogEntry((vx_reference)node, status, "RemoveFringe Input Validation failed: input image 1 width must be evenly divisible by 16\n");
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
            vxAddLogEntry((vx_reference)node, status, "RemoveFringe Input Validation failed: invalid input image 2 format, it must be U8\n");
            return status;
        }
    }
    status = vxQueryImage((vx_image)parameters[2], VX_IMAGE_WIDTH, &width, sizeof(width));
    if (width%16!=0)
    {
        status = VX_ERROR_INVALID_VALUE;
        vxAddLogEntry((vx_reference)node, status, "RemoveFringe Input Validation failed: input image 2 width must be evenly divisible by 16\n");
        return status;
    }


    if(vxQueryArray((vx_array)parameters[5], VX_ARRAY_CAPACITY, &num_items, sizeof(num_items)) == VX_SUCCESS)
    {
        if (num_items == 16)
        {
            status = VX_SUCCESS;
        }
        else
        {
            status = VX_ERROR_INVALID_VALUE;
            vxAddLogEntry((vx_reference)node, status, "RemoveFringe Input Validation failed: Input LtoK_nodes array must have 16 entries\n");
            return status;
        }
    }


    status = vxQueryImage((vx_image)parameters[0], VX_IMAGE_HEIGHT, &height, sizeof(height));
    format = VX_DF_IMAGE_RGBX;
    PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[3], VX_IMAGE_WIDTH, &width, sizeof(width)));
    PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[3], VX_IMAGE_HEIGHT, &height, sizeof(height)));
    PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[3], VX_IMAGE_FORMAT, &format, sizeof(format)));

    format = VX_DF_IMAGE_U8;
    if(metas[4])
    {
        PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[4], VX_IMAGE_WIDTH, &width, sizeof(width)));
        PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[4], VX_IMAGE_HEIGHT, &height, sizeof(height)));
        PROCESS_VX_STATUS(node, vxSetMetaFormatAttribute(metas[4], VX_IMAGE_FORMAT, &format, sizeof(format)));
    }

    return status;
}


vx_status VX_API_CALL PublishRemoveFringeKernel(vx_context context)
{
    vx_status status = VX_SUCCESS;
    //create a kernel via the vxAddAdvancedTilingKernelIntel interface
    vx_kernel kernel = vxAddAdvancedTilingKernelIntel(context,
        (char*)VX_KERNEL_NAME_USER_REMOVEFRINGE,
        VX_KERNEL_USER_REMOVEFRINGE,
        vxRemoveFringeKernel,
        NULL,
        6,
        vxRemoveFringeValidator,
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
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, 4, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_OPTIONAL));
    SAFE_VX_CALL(status, context, vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));

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
