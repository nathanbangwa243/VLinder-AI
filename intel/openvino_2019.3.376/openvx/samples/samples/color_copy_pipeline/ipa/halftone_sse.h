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
#include <emmintrin.h>
#include <smmintrin.h>

static void
core_halftone_sse(uint8_t *dst, const uint8_t *contone, const uint8_t *screen, int w)
{
    __m128i mm_signfix = _mm_set1_epi8(0x80);
    __m128i mm_shuffle = _mm_set_epi8(8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7);

    w += 31;
    for (w >>= 5; w > 0; w--)
    {
        __m128i mm0 = _mm_load_si128((const __m128i *)contone);
        __m128i mm1 = _mm_load_si128((const __m128i *)screen);
        contone += 16;
        screen += 16;
        mm0 = _mm_xor_si128(mm0, mm_signfix);
        //mm1 = _mm_xor_si128(mm1, mm_signfix);
        mm0 = _mm_subs_epi8(mm0, mm1);
        mm0 = _mm_shuffle_epi8(mm0, mm_shuffle);
        *(int16_t *)dst = _mm_movemask_epi8(mm0);
        dst += 2;
        mm0 = _mm_load_si128((const __m128i *)contone);
        mm1 = _mm_load_si128((const __m128i *)screen);
        contone += 16;
        screen += 16;
        mm0 = _mm_xor_si128(mm0, mm_signfix);
        //mm1 = _mm_xor_si128(mm1, mm_signfix);
        mm0 = _mm_subs_epi8(mm0, mm1);
        mm0 = _mm_shuffle_epi8(mm0, mm_shuffle);
        *(int16_t *)dst = _mm_movemask_epi8(mm0);
        dst += 2;
    }
}
