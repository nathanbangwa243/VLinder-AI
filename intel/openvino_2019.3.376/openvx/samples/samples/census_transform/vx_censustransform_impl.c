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

#include "vx_user_census_nodes.h"
#include <stdio.h>

#include <immintrin.h>


//!************************************************************************************************
//! Function Name      :  censustransform
//! Argument 1         :  Pointer to source data tile	                            [IN]
//! Argument 2         :  Source data tile stride  	                                [IN]
//! Argument 3         :  Pointer to destination data tile                          [IN/OUT]
//! Argument 4         :  Destination data tile stride  	                        [IN]
//! Argument 5         :  Deestination data tile width                              [IN]
//! Argument 6         :  Destination data tile height  	                        [IN]
//! Returns            :  Status
//! Description        :  SSE optimized implementation of Census Transform
//!************************************************************************************************
vx_status censustransform(vx_int16 *pSrc,
                          vx_int32 srcStride,
                          vx_uint8 *pDst,
                          vx_int32 dstStride,
                          vx_uint32 dstWidth,
                          vx_uint32 dstHeight)
{
    vx_uint8 *pSrc8 = (vx_uint8 *)pSrc;
    vx_uint8 *pDst8 = pDst;
    vx_uint8* PrevRowPtrBase   = pSrc8 + sizeof(vx_int16);
    vx_uint8* CurrRowPtrBase   = PrevRowPtrBase + srcStride;
    vx_uint8* NextRowPtrBase   = CurrRowPtrBase + srcStride;
    vx_uint8* DstPtrBase   = pDst8;

    //Masks
    __m128i mask0 = _mm_set1_epi32(0x80);
    __m128i mask1 = _mm_set1_epi32(0x40);
    __m128i mask2 = _mm_set1_epi32(0x20);
    __m128i mask3 = _mm_set1_epi32(0x10);
    __m128i mask5 = _mm_set1_epi32(0x08);
    __m128i mask6 = _mm_set1_epi32(0x04);
    __m128i mask7 = _mm_set1_epi32(0x02);
    __m128i mask8 = _mm_set1_epi32(0x01);
    __m128i ff    =  _mm_set1_epi32(0xFF);

    //4 pixels per iteration
    int x_iterations = dstWidth / 4;

    vx_int32 x, y;

    //for each destination row
    for(y = 0; y <  dstHeight; y++)
    {
        vx_int16* PrevRowPtr   = (vx_int16* )PrevRowPtrBase;
        vx_int16* CurrRowPtr   = (vx_int16* )CurrRowPtrBase;
        vx_int16* NextRowPtr   = (vx_int16* )NextRowPtrBase;
        vx_int32 *dst     = (vx_int32 *)DstPtrBase;

        //process 4 pixels per iteration using SSE intrinsics
        for( x = 0; x < x_iterations; x++)
        {
            __m128i image_00 = _mm_set_epi32 ((vx_int32)(((vx_int16*)(PrevRowPtr-1))[3]), (vx_int32)(((vx_int16*)(PrevRowPtr-1))[2]), (vx_int32)(((vx_int16*)(PrevRowPtr-1))[1]), (vx_int32)(((vx_int16*)(PrevRowPtr-1))[0]));
            __m128i image_01 = _mm_set_epi32 ((vx_int32)(((vx_int16*)(PrevRowPtr))[3]), (vx_int32)(((vx_int16*)(PrevRowPtr))[2]), (vx_int32)(((vx_int16*)(PrevRowPtr))[1]), (vx_int32)(((vx_int16*)(PrevRowPtr-1))[0]));
            __m128i image_02 = _mm_set_epi32 ((vx_int32)(((vx_int16*)(PrevRowPtr+1))[3]), (vx_int32)(((vx_int16*)(PrevRowPtr+1))[2]), (vx_int32)(((vx_int16*)(PrevRowPtr+1))[1]), (vx_int32)(((vx_int16*)(PrevRowPtr+1))[0]));
            __m128i image_10 = _mm_set_epi32 ((vx_int32)(((vx_int16*)(CurrRowPtr-1))[3]), (vx_int32)(((vx_int16*)(CurrRowPtr-1))[2]), (vx_int32)(((vx_int16*)(CurrRowPtr-1))[1]), (vx_int32)(((vx_int16*)(CurrRowPtr-1))[0]));
            __m128i image_12 = _mm_set_epi32 ((vx_int32)(((vx_int16*)(CurrRowPtr+1))[3]), (vx_int32)(((vx_int16*)(CurrRowPtr+1))[2]), (vx_int32)(((vx_int16*)(CurrRowPtr+1))[1]), (vx_int32)(((vx_int16*)(CurrRowPtr+1))[0]));
            __m128i image_20 = _mm_set_epi32 ((vx_int32)(((vx_int16*)(NextRowPtr-1))[3]), (vx_int32)(((vx_int16*)(NextRowPtr-1))[2]), (vx_int32)(((vx_int16*)(NextRowPtr-1))[1]), (vx_int32)(((vx_int16*)(NextRowPtr-1))[0]));
            __m128i image_21 = _mm_set_epi32 ((vx_int32)(((vx_int16*)(NextRowPtr))[3]), (vx_int32)(((vx_int16*)(NextRowPtr))[2]), (vx_int32)(((vx_int16*)(NextRowPtr))[1]), (vx_int32)(((vx_int16*)(NextRowPtr))[0]));
            __m128i image_22 = _mm_set_epi32 ((vx_int32)(((vx_int16*)(NextRowPtr+1))[3]), (vx_int32)(((vx_int16*)(NextRowPtr+1))[2]), (vx_int32)(((vx_int16*)(NextRowPtr+1))[1]), (vx_int32)(((vx_int16*)(NextRowPtr+1))[0]));
            __m128i testPix = _mm_set_epi32 ((vx_int32)(((vx_int16*)(CurrRowPtr))[3]), (vx_int32)(((vx_int16*)(CurrRowPtr))[2]), (vx_int32)(((vx_int16*)(CurrRowPtr))[1]), (vx_int32)(((vx_int16*)(CurrRowPtr))[0]));


            __m128i res0 = _mm_and_si128(_mm_xor_si128(ff, _mm_cmplt_epi32(image_00, testPix)), mask0);
            __m128i res1 = _mm_and_si128(_mm_xor_si128(ff, _mm_cmplt_epi32(image_01, testPix)), mask1);
            __m128i res2 = _mm_and_si128(_mm_xor_si128(ff, _mm_cmplt_epi32(image_02, testPix)), mask2);
            __m128i res3 = _mm_and_si128(_mm_xor_si128(ff, _mm_cmplt_epi32(image_10, testPix)), mask3);
            __m128i res5 = _mm_and_si128(_mm_xor_si128(ff, _mm_cmplt_epi32(image_12, testPix)), mask5);
            __m128i res6 = _mm_and_si128(_mm_xor_si128(ff, _mm_cmplt_epi32(image_20, testPix)), mask6);
            __m128i res7 = _mm_and_si128(_mm_xor_si128(ff, _mm_cmplt_epi32(image_21, testPix)), mask7);
            __m128i res8 = _mm_and_si128(_mm_xor_si128(ff, _mm_cmplt_epi32(image_22, testPix)), mask8);

            __m128i result0 = _mm_or_si128(res0, _mm_or_si128(res1, _mm_or_si128(res2, _mm_or_si128(res3, _mm_or_si128(res5,
                _mm_or_si128(res6, _mm_or_si128(res7, res8)))))));

            result0 = _mm_packs_epi32(result0, result0);
            result0 = _mm_packus_epi16(result0, result0);
            *dst = _mm_cvtsi128_si32(result0);

            PrevRowPtr += 4;
            CurrRowPtr += 4;
            NextRowPtr += 4;
            dst++;
        }

        vx_int32 i32ccols;
        //if the width is not a multiple of 4, process the remainder serially
        for( i32ccols = x * 4; i32ccols < dstWidth;i32ccols ++)
        {
            vx_int16* prev_src_ptr = (vx_int16 *)PrevRowPtrBase + i32ccols;
            vx_int16* curr_src_ptr = (vx_int16 *)CurrRowPtrBase + i32ccols;
            vx_int16* next_src_ptr = (vx_int16 *)NextRowPtrBase + i32ccols;
            vx_uint8 ui8Index = 0;
            if (*curr_src_ptr <= *(prev_src_ptr - 1)) ui8Index += 0x80;
            if (*curr_src_ptr <= *prev_src_ptr)       ui8Index += 0x40;
            if (*curr_src_ptr <= *(prev_src_ptr + 1)) ui8Index += 0x20;
            if (*curr_src_ptr <= *(curr_src_ptr - 1)) ui8Index += 0x10;
            if (*curr_src_ptr <= *(curr_src_ptr + 1)) ui8Index += 0x08;
            if (*curr_src_ptr <= *(next_src_ptr - 1)) ui8Index += 0x04;
            if (*curr_src_ptr <= *next_src_ptr  )     ui8Index += 0x02;
            if (*curr_src_ptr <= *(next_src_ptr + 1)) ui8Index += 0x01;
            DstPtrBase[i32ccols] = ui8Index;
        }

        PrevRowPtrBase += srcStride;
        CurrRowPtrBase += srcStride;
        NextRowPtrBase += srcStride;
        DstPtrBase += dstStride;
    }

    return VX_SUCCESS;
}
