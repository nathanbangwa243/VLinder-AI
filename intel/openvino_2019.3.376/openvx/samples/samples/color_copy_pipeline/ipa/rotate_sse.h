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
zoom_x1_sse(uint8_t       * ipa_restrict tmp,
            const uint8_t * ipa_restrict src,
            const index_t * ipa_restrict index,
            const int32_t * ipa_restrict weights,
            uint32_t                     dst_w,
            uint32_t                     src_w,
            uint32_t                     channels,
            const uint8_t * ipa_restrict bg)
{
    __m128i round = _mm_set1_epi32(WEIGHT_ROUND);

    if (0)
slow:
    {
        /* Do any where we might index off the edge of the source */
        int            pix_num = index->first_pixel;
        const uint8_t *s = &src[pix_num];
        const int32_t *w = &weights[index->index];
        uint32_t       j = index->n;
        int32_t        pixel0 = WEIGHT_ROUND;
        if (pix_num < 0)
        {
            int32_t wt = *w++;
            assert(pix_num == -1);
            pixel0 += bg[0] * wt;
            s++;
            j--;
            pix_num = 0;
        }
        pix_num = (int)src_w - pix_num;
        if (pix_num > (int)j)
            pix_num = j;
        j -= pix_num;
        while (pix_num > 0)
        {
            pixel0 += *s++ * *w++;
            pix_num--;
        }
        if (j > 0)
        {
            assert(j == 1);
            pixel0 += bg[0] * *w;
        }
        pixel0 >>= WEIGHT_SHIFT;
        *tmp++ = CLAMP(pixel0, 0, 255);
        index++;
        dst_w--;
    }

    while (dst_w > 0)
    {
        const uint8_t *s;
        uint32_t       j;
        const int32_t *w;

        /* Jump out of band to do the (rare) slow (edge) pixels */
        if (index->slow)
            goto slow;

        s = &src[index->first_pixel];
        j = index->n;
        w = &weights[index->index];
        if (j <= 4)
        {
            __m128i mw0, mm0;
            mw0 = _mm_loadu_si128((const __m128i *)w);
            mm0 = _mm_loadu_si128((const __m128i *)s);
                                              // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
            mm0 = _mm_cvtepu8_epi32(mm0);     // mm0 = 000000dd000000cc000000bb000000aa SSE4.1
            mm0 = _mm_mullo_epi32(mm0,mw0);   // mm0 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
            mm0 = _mm_hadd_epi32(mm0,mm0);
            mm0 = _mm_hadd_epi32(mm0,mm0);
            mm0 = _mm_add_epi32(mm0, round);  // Add round
            mm0 = _mm_srai_epi32(mm0, WEIGHT_SHIFT-8); // Shift down
            mm0 = _mm_packus_epi32(mm0,mm0);  // Clamp to 0 to 65535 range.
            *tmp++ = _mm_extract_epi8(mm0,1);
        }
        else if (j <= 8)
        {
            __m128i mw0, mw1, mm0, mm1;
            mw0 = _mm_loadu_si128((const __m128i *)w);
            mm0 = _mm_loadu_si128((const __m128i *)s);
                                              // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
            mw1 = _mm_loadu_si128((const __m128i *)(w+4));
            mm1 = _mm_cvtepu8_epi32(mm0);     // mm1 = 000000dd000000cc000000bb000000aa SSE4.1
            mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 000000ppoonnmmllkkjjiihhggffeedd SSE2
            mm1 = _mm_mullo_epi32(mm1,mw0);   // mm1 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
            mm0 = _mm_cvtepu8_epi32(mm0);     // mm0 = 000000hh000000gg000000ff000000ee SSE4.1
            mm0 = _mm_mullo_epi32(mm0,mw1);   // mm1 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
            mm1 = _mm_add_epi32(mm1, mm0);
            mm1 = _mm_hadd_epi32(mm1,mm1);
            mm1 = _mm_hadd_epi32(mm1,mm1);
            mm1 = _mm_add_epi32(mm1, round);  // Add round
            mm1 = _mm_srai_epi32(mm1, WEIGHT_SHIFT-8); // Shift down
            mm1 = _mm_packus_epi32(mm1,mm1);  // Clamp to 0 to 65535 range.
            *tmp++ = _mm_extract_epi8(mm1,1);
        }
        else
        {
            int32_t        pixel0 = WEIGHT_ROUND;
            for (j = index->n; j > 0; j--)
            {
                pixel0 += *s++ * *w++;
            }
            pixel0 >>= WEIGHT_SHIFT;
            *tmp++ = CLAMP(pixel0, 0, 255);
        }
        index++;
        dst_w--;
    }
}

static void
zoom_x3_sse(uint8_t       * ipa_restrict tmp,
            const uint8_t * ipa_restrict src,
            const index_t * ipa_restrict index,
            const int32_t * ipa_restrict weights,
            uint32_t                     dst_w,
            uint32_t                     src_w,
            uint32_t                     channels,
            const uint8_t * ipa_restrict bg)
{
    __m128i round = _mm_set1_epi32(WEIGHT_ROUND);

    if (0)
slow:
    {
        /* Do any where we might index off the edge of the source */
        int            pix_num = index->first_pixel;
        const uint8_t *s = &src[pix_num * 3];
        const int32_t *w = &weights[index->index];
        uint32_t       j = index->n;
        int32_t        pixel0 = WEIGHT_ROUND;
        int32_t        pixel1 = WEIGHT_ROUND;
        int32_t        pixel2 = WEIGHT_ROUND;
        if (pix_num < 0)
        {
            int32_t wt = *w++;
            assert(pix_num == -1);
            pixel0 += bg[0] * wt;
            pixel1 += bg[1] * wt;
            pixel2 += bg[2] * wt;
            s += 3;
            j--;
            pix_num = 0;
        }
        pix_num = (int)src_w - pix_num;
        if (pix_num > (int)j)
            pix_num = j;
        j -= pix_num;
        while (pix_num > 0)
        {
            int32_t wt = *w++;
            pixel0 += *s++ * wt;
            pixel1 += *s++ * wt;
            pixel2 += *s++ * wt;
            pix_num--;
        }
        if (j > 0)
        {
            int32_t wt = *w++;
            assert(j == 1);
            pixel0 += bg[0] * wt;
            pixel1 += bg[1] * wt;
            pixel2 += bg[2] * wt;
        }
        pixel0 >>= WEIGHT_SHIFT;
        pixel1 >>= WEIGHT_SHIFT;
        pixel2 >>= WEIGHT_SHIFT;
        *tmp++ = CLAMP(pixel0, 0, 255);
        *tmp++ = CLAMP(pixel1, 0, 255);
        *tmp++ = CLAMP(pixel2, 0, 255);
        index++;
        dst_w--;
    }

    while (dst_w > 0)
    {
        const uint8_t *s;
        int            j;
        const int32_t *w;
        __m128i mm0, mm1, mm4, mw0, mw1;

        /* Jump out of band to do the (rare) slow (edge) pixels */
        if (index->slow)
            goto slow;

        s = &src[index->first_pixel * 3];
        j = (int)index->n;
        w = &weights[index->index];

        mm4 = round;
        mm0 = _mm_loadu_si128((const __m128i *)s); // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        if (j == 4)
        {
            mw0 = _mm_loadu_si128((const __m128i *)w);
            mw1 = _mm_shuffle_epi32(mw0, 0 + (0 << 2) + (0 << 4) + (0 << 6));
            mm1 = _mm_cvtepu8_epi32(mm0);      // mm1 = 000000dd000000cc000000bb000000aa SSE4.1
            mm1 = _mm_mullo_epi32(mm1, mw1);   // mm1 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
            mm4 = _mm_add_epi32(mm4, mm1);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
            mm0 = _mm_srli_si128(mm0, 3);      // mm0 = 000000ppoonnmmllkkjjiihhggffeedd SSE2
            mw1 = _mm_shuffle_epi32(mw0, 1 + (1 << 2) + (1 << 4) + (1 << 6));
            mm1 = _mm_cvtepu8_epi32(mm0);      // mm1 = 000000dd000000cc000000bb000000aa SSE4.1
            mm1 = _mm_mullo_epi32(mm1, mw1);   // mm1 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
            mm4 = _mm_add_epi32(mm4, mm1);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
            mm0 = _mm_srli_si128(mm0, 3);      // mm0 = 000000ppoonnmmllkkjjiihhggffeedd SSE2
            mw1 = _mm_shuffle_epi32(mw0, 2 + (2 << 2) + (2 << 4) + (2 << 6));
            mm1 = _mm_cvtepu8_epi32(mm0);      // mm1 = 000000dd000000cc000000bb000000aa SSE4.1
            mm1 = _mm_mullo_epi32(mm1, mw1);   // mm1 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
            mm4 = _mm_add_epi32(mm4, mm1);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
            mm0 = _mm_srli_si128(mm0, 3);      // mm0 = 000000ppoonnmmllkkjjiihhggffeedd SSE2
            mw1 = _mm_shuffle_epi32(mw0, 3 + (3 << 2) + (3 << 4) + (3 << 6));
            mm1 = _mm_cvtepu8_epi32(mm0);      // mm1 = 000000dd000000cc000000bb000000aa SSE4.1
            mm1 = _mm_mullo_epi32(mm1, mw1);   // mm1 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
            mm4 = _mm_add_epi32(mm4, mm1);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        }
        else
        {
            int off = j & 3;
            w -= (4 - j) & 3;
            s += (off ? off : 4) * 3;
            mw0 = _mm_loadu_si128((const __m128i *)w);
            w += 4;
            /* This is a use of Duff's Device. I'm very sorry, but on the other hand, Yay! */
            switch (off)
            {
                do
                {
                    mm0 = _mm_loadu_si128((const __m128i *)s);
                    // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
                    s += 4 * 3;
                    mw0 = _mm_loadu_si128((const __m128i *)w);
                    w += 4;
            case 0:
                    mw1 = _mm_shuffle_epi32(mw0, 0 + (0 << 2) + (0 << 4) + (0 << 6));
                    mm1 = _mm_cvtepu8_epi32(mm0);      // mm1 = 000000dd000000cc000000bb000000aa SSE4.1
                    mm1 = _mm_mullo_epi32(mm1, mw1);   // mm1 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
                    mm4 = _mm_add_epi32(mm4, mm1);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
                    mm0 = _mm_srli_si128(mm0, 3);      // mm0 = 000000ppoonnmmllkkjjiihhggffeedd SSE2
            case 3:
                    mw1 = _mm_shuffle_epi32(mw0, 1 + (1 << 2) + (1 << 4) + (1 << 6));
                    mm1 = _mm_cvtepu8_epi32(mm0);      // mm1 = 000000dd000000cc000000bb000000aa SSE4.1
                    mm1 = _mm_mullo_epi32(mm1, mw1);   // mm1 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
                    mm4 = _mm_add_epi32(mm4, mm1);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
                    mm0 = _mm_srli_si128(mm0, 3);      // mm0 = 000000ppoonnmmllkkjjiihhggffeedd SSE2
            case 2:
                    mw1 = _mm_shuffle_epi32(mw0, 2 + (2 << 2) + (2 << 4) + (2 << 6));
                    mm1 = _mm_cvtepu8_epi32(mm0);      // mm1 = 000000dd000000cc000000bb000000aa SSE4.1
                    mm1 = _mm_mullo_epi32(mm1, mw1);   // mm1 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
                    mm4 = _mm_add_epi32(mm4, mm1);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
                    mm0 = _mm_srli_si128(mm0, 3);      // mm0 = 000000ppoonnmmllkkjjiihhggffeedd SSE2
            case 1:
                    mw1 = _mm_shuffle_epi32(mw0, 3 + (3 << 2) + (3 << 4) + (3 << 6));
                    mm1 = _mm_cvtepu8_epi32(mm0);      // mm1 = 000000dd000000cc000000bb000000aa SSE4.1
                    mm1 = _mm_mullo_epi32(mm1, mw1);   // mm1 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
                    mm4 = _mm_add_epi32(mm4, mm1);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
                    j -= 4;
                } while (j > 0);
            }
        }
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT); // Shift down
        mm4 = _mm_packus_epi32(mm4, mm4);  // Clamp to 0 to 65535 range.
        mm4 = _mm_packus_epi16(mm4, mm4);  // Clamp to 0 to 65535 range.
        j = _mm_extract_epi32(mm4, 0);
        *(int16_t *)tmp = j;
        ((int8_t *)tmp)[2] = j>>16;
        tmp += 3;
        index++;
        dst_w--;
    }

    while (dst_w > 0)
    {
        const uint8_t *s;

        /* Jump out of band to do the (rare) slow (edge) pixels */
        if (index->slow)
            goto slow;

        s = &src[index->first_pixel * 3];

        {
            const int32_t *w = &weights[index->index];
            uint32_t       j = index->n;
            int32_t        pixel0 = WEIGHT_ROUND;
            int32_t        pixel1 = WEIGHT_ROUND;
            int32_t        pixel2 = WEIGHT_ROUND;
            for (j = index->n; j > 0; j--)
            {
                int32_t wt = *w++;
                pixel0 += *s++ * wt;
                pixel1 += *s++ * wt;
                pixel2 += *s++ * wt;
            }
            pixel0 >>= WEIGHT_SHIFT;
            pixel1 >>= WEIGHT_SHIFT;
            pixel2 >>= WEIGHT_SHIFT;
            *tmp++ = CLAMP(pixel0, 0, 255);
            *tmp++ = CLAMP(pixel1, 0, 255);
            *tmp++ = CLAMP(pixel2, 0, 255);
        }
        index++;
        dst_w--;
    }
}

static void
zoom_x4_sse(uint8_t       * ipa_restrict tmp,
            const uint8_t * ipa_restrict src,
            const index_t * ipa_restrict index,
            const int32_t * ipa_restrict weights,
            uint32_t                     dst_w,
            uint32_t                     src_w,
            uint32_t                     channels,
            const uint8_t * ipa_restrict bg)
{
    __m128i round = _mm_set1_epi32(WEIGHT_ROUND);

    if (0)
slow:
    {
        /* Do any where we might index off the edge of the source */
        int            pn = index->first_pixel;
        const uint8_t *s = &src[pn * 4];
        const int32_t *w = &weights[index->index];
        uint32_t       j = index->n;
        int32_t        pixel0 = WEIGHT_ROUND;
        int32_t        pixel1 = WEIGHT_ROUND;
        int32_t        pixel2 = WEIGHT_ROUND;
        int32_t        pixel3 = WEIGHT_ROUND;
        int            pix_num = pn;
        if (pix_num < 0)
        {
            int32_t wt = *w++;
            assert(pix_num == -1);
            pixel0 += bg[0] * wt;
            pixel1 += bg[1] * wt;
            pixel2 += bg[2] * wt;
            pixel3 += bg[3] * wt;
            s += 4;
            j--;
            pix_num = 0;
        }
        pix_num = (int)src_w - pix_num;
        if (pix_num > (int)j)
            pix_num = j;
        j -= pix_num;
        while (pix_num > 0)
        {
            int32_t wt = *w++;
            pixel0 += *s++ * wt;
            pixel1 += *s++ * wt;
            pixel2 += *s++ * wt;
            pixel3 += *s++ * wt;
            pix_num--;
        }
        if (j > 0)
        {
            int32_t wt = *w;
            assert(j == 1);
            pixel0 += bg[0] * wt;
            pixel1 += bg[1] * wt;
            pixel2 += bg[2] * wt;
            pixel3 += bg[3] * wt;
        }
        pixel0 >>= WEIGHT_SHIFT;
        pixel1 >>= WEIGHT_SHIFT;
        pixel2 >>= WEIGHT_SHIFT;
        pixel3 >>= WEIGHT_SHIFT;
        *tmp++ = CLAMP(pixel0, 0, 255);
        *tmp++ = CLAMP(pixel1, 0, 255);
        *tmp++ = CLAMP(pixel2, 0, 255);
        *tmp++ = CLAMP(pixel3, 0, 255);
        index++;
        dst_w--;
    }

    while (dst_w > 0)
    {
        const uint8_t *s;
        int            j;
        const int32_t *w;
        __m128i mm0, mm1, mm4, mw0, mw1;

        /* Jump out of band to do the (rare) slow (edge) pixels */
        if (index->slow)
            goto slow;

        s = &src[index->first_pixel * 4];
        j = (int)index->n;
        w = &weights[index->index];

        mm4 = round;
        mm0 = _mm_loadu_si128((const __m128i *)s); // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        if (j == 4)
        {
            mw0 = _mm_loadu_si128((const __m128i *)w);
            mw1 = _mm_shuffle_epi32(mw0, 0 + (0 << 2) + (0 << 4) + (0 << 6));
            mm1 = _mm_cvtepu8_epi32(mm0);      // mm1 = 000000dd000000cc000000bb000000aa SSE4.1
            mm1 = _mm_mullo_epi32(mm1, mw1);   // mm1 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
            mm4 = _mm_add_epi32(mm4, mm1);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
            mm0 = _mm_srli_si128(mm0, 4);      // mm0 = 000000ppoonnmmllkkjjiihhggffeedd SSE2
            mw1 = _mm_shuffle_epi32(mw0, 1 + (1 << 2) + (1 << 4) + (1 << 6));
            mm1 = _mm_cvtepu8_epi32(mm0);      // mm1 = 000000dd000000cc000000bb000000aa SSE4.1
            mm1 = _mm_mullo_epi32(mm1, mw1);   // mm1 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
            mm4 = _mm_add_epi32(mm4, mm1);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
            mm0 = _mm_srli_si128(mm0, 4);      // mm0 = 000000ppoonnmmllkkjjiihhggffeedd SSE2
            mw1 = _mm_shuffle_epi32(mw0, 2 + (2 << 2) + (2 << 4) + (2 << 6));
            mm1 = _mm_cvtepu8_epi32(mm0);      // mm1 = 000000dd000000cc000000bb000000aa SSE4.1
            mm1 = _mm_mullo_epi32(mm1, mw1);   // mm1 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
            mm4 = _mm_add_epi32(mm4, mm1);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
            mm0 = _mm_srli_si128(mm0, 4);      // mm0 = 000000ppoonnmmllkkjjiihhggffeedd SSE2
            mw1 = _mm_shuffle_epi32(mw0, 3 + (3 << 2) + (3 << 4) + (3 << 6));
            mm1 = _mm_cvtepu8_epi32(mm0);      // mm1 = 000000dd000000cc000000bb000000aa SSE4.1
            mm1 = _mm_mullo_epi32(mm1, mw1);   // mm1 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
            mm4 = _mm_add_epi32(mm4, mm1);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        }
        else
        {
            int off = j & 3;
            w -= (4 - j) & 3;
            s += (off ? off : 4) * 4;
            mw0 = _mm_loadu_si128((const __m128i *)w);
            w += 4;
            /* This is a use of Duff's Device. I'm very sorry, but on the other hand, Yay! */
            switch (off)
            {
                do
                {
                    mm0 = _mm_loadu_si128((const __m128i *)s);
                    // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
                    s += 4 * 4;
                    mw0 = _mm_loadu_si128((const __m128i *)w);
                    w += 4;
            case 0:
                    mw1 = _mm_shuffle_epi32(mw0, 0 + (0 << 2) + (0 << 4) + (0 << 6));
                    mm1 = _mm_cvtepu8_epi32(mm0);      // mm1 = 000000dd000000cc000000bb000000aa SSE4.1
                    mm1 = _mm_mullo_epi32(mm1, mw1);   // mm1 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
                    mm4 = _mm_add_epi32(mm4, mm1);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
                    mm0 = _mm_srli_si128(mm0, 4);      // mm0 = 000000ppoonnmmllkkjjiihhggffeedd SSE2
            case 3:
                    mw1 = _mm_shuffle_epi32(mw0, 1 + (1 << 2) + (1 << 4) + (1 << 6));
                    mm1 = _mm_cvtepu8_epi32(mm0);      // mm1 = 000000dd000000cc000000bb000000aa SSE4.1
                    mm1 = _mm_mullo_epi32(mm1, mw1);   // mm1 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
                    mm4 = _mm_add_epi32(mm4, mm1);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
                    mm0 = _mm_srli_si128(mm0, 4);      // mm0 = 000000ppoonnmmllkkjjiihhggffeedd SSE2
            case 2:
                    mw1 = _mm_shuffle_epi32(mw0, 2 + (2 << 2) + (2 << 4) + (2 << 6));
                    mm1 = _mm_cvtepu8_epi32(mm0);      // mm1 = 000000dd000000cc000000bb000000aa SSE4.1
                    mm1 = _mm_mullo_epi32(mm1, mw1);   // mm1 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
                    mm4 = _mm_add_epi32(mm4, mm1);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
                    mm0 = _mm_srli_si128(mm0, 4);      // mm0 = 000000ppoonnmmllkkjjiihhggffeedd SSE2
            case 1:
                    mw1 = _mm_shuffle_epi32(mw0, 3 + (3 << 2) + (3 << 4) + (3 << 6));
                    mm1 = _mm_cvtepu8_epi32(mm0);      // mm1 = 000000dd000000cc000000bb000000aa SSE4.1
                    mm1 = _mm_mullo_epi32(mm1, mw1);   // mm1 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
                    mm4 = _mm_add_epi32(mm4, mm1);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
                    j -= 4;
                } while (j > 0);
            }
        }
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT);
        mm4 = _mm_packus_epi32(mm4,mm4);
        mm4 = _mm_packus_epi16(mm4,mm4);
        *(int32_t *)tmp = _mm_extract_epi32(mm4,0);
        tmp += 4;
        index++;
        dst_w--;
    }
}

/* Same as the C only version. */
static void
zoom_y1_sse(uint8_t       *              dst,
            const uint8_t * ipa_restrict tmp,
            const index_t * ipa_restrict index,
            const int32_t * ipa_restrict weights,
            uint32_t                     width,
            uint32_t                     channels,
            uint32_t                     mod,
            int32_t                      y)
{
    uint32_t stride = width;
    uint32_t offset = 0;

    while (width--)
    {
        uint32_t off = (index->first_pixel + y) * stride + offset;

        offset++;
        if (off >= mod)
            off -= mod;

        {
            const int32_t *w = &weights[index->index];
            uint32_t       j = index->n;
            int32_t        pixel0 = WEIGHT_ROUND;

            for (j = index->n; j > 0; j--)
            {
                pixel0 += tmp[off  ] * *w++;
                off += stride;
                if (off >= mod)
                    off -= mod;
            }
            pixel0 >>= WEIGHT_SHIFT;
            *dst++ = CLAMP(pixel0, 0, 255);
        }
        index++;
    }
}

static void
zoom_y3_sse(uint8_t       *              dst,
            const uint8_t * ipa_restrict tmp,
            const index_t * ipa_restrict index,
            const int32_t * ipa_restrict weights,
            uint32_t                     width,
            uint32_t                     channels,
            uint32_t                     mod,
            int32_t                      y)
{
    uint32_t stride = width * 3;
    uint32_t offset = 0;
    __m128i round = _mm_set1_epi32(WEIGHT_ROUND);

    while (width--)
    {
        uint32_t off = (index->first_pixel + y) * stride + offset;

        offset += 3;
        if (off >= mod)
            off -= mod;

        {
            const int32_t *w = &weights[index->index];
            int32_t        j = (int32_t)index->n;
            __m128i mm0, mm1, mm2, mw0, mw1;

            if (j == 4)
            {
                    mw0 = _mm_loadu_si128((const __m128i *)w);
                    mm0 = _mm_loadu_si128((const __m128i *)&tmp[off]);
                                                      // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
                    mw1 = _mm_shuffle_epi32(mw0, 0 + (0 << 2) + (0 << 4) + (0 << 6));
                    mm0 = _mm_cvtepu8_epi32(mm0);     // mm0 = 000000dd000000cc000000bb000000aa SSE4.1
                    mm0 = _mm_mullo_epi32(mm0,mw1);   // mm0 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
                    mm1 = _mm_add_epi32(round, mm0);  // mm1 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
                    off += stride;
                    if (off >= mod)
                        off -= mod;
                    mm0 = _mm_loadu_si128((const __m128i *)&tmp[off]);
                                                      // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
                    mw1 = _mm_shuffle_epi32(mw0, 1 + (1 << 2) + (1 << 4) + (1 << 6));
                    mm0 = _mm_cvtepu8_epi32(mm0);     // mm0 = 000000dd000000cc000000bb000000aa SSE4.1
                    mm0 = _mm_mullo_epi32(mm0,mw1);   // mm0 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
                    mm1 = _mm_add_epi32(mm1, mm0);    // mm1 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
                    off += stride;
                    if (off >= mod)
                        off -= mod;
                    mm0 = _mm_loadu_si128((const __m128i *)&tmp[off]);
                                                      // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
                    mw1 = _mm_shuffle_epi32(mw0, 2 + (2 << 2) + (2 << 4) + (2 << 6));
                    mm0 = _mm_cvtepu8_epi32(mm0);     // mm0 = 000000dd000000cc000000bb000000aa SSE4.1
                    mm0 = _mm_mullo_epi32(mm0,mw1);   // mm0 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
                    mm1 = _mm_add_epi32(mm1, mm0);    // mm1 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
                    off += stride;
                    if (off >= mod)
                        off -= mod;
                    mm0 = _mm_loadu_si128((const __m128i *)&tmp[off]);
                                                      // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
                    mw1 = _mm_shuffle_epi32(mw0, 3 + (3 << 2) + (3 << 4) + (3 << 6));
                    mm0 = _mm_cvtepu8_epi32(mm0);     // mm0 = 000000dd000000cc000000bb000000aa SSE4.1
                    mm0 = _mm_mullo_epi32(mm0,mw1);   // mm0 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
                    mm1 = _mm_add_epi32(mm1, mm0);    // mm1 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
            }
            else
            {
                int duff = j & 3;
                w -= (4 - j) & 3;
                mw0 = _mm_loadu_si128((const __m128i *)w);
                w += 4;
                mm1 = round;
                /* This is a use of Duff's Device. I'm very sorry, but on the other hand, Yay! */
                switch (duff)
                {
                    do
                    {
                        off += stride;
                        if (off >= mod)
                            off -= mod;
                        mw0 = _mm_loadu_si128((const __m128i *)w);
                        w += 4;
                case 0:
                        mm0 = _mm_loadu_si128((const __m128i *)&tmp[off]);
                                                           // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
                        mw1 = _mm_shuffle_epi32(mw0, 0 + (0 << 2) + (0 << 4) + (0 << 6));
                        off += stride;
                        if (off >= mod)
                            off -= mod;
                        mm2 = _mm_cvtepu8_epi32(mm0);      // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
                        mm2 = _mm_mullo_epi32(mm2, mw1);   // mm2 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
                        mm1 = _mm_add_epi32(mm1, mm2);     // mm1 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
                case 3:
                        mm0 = _mm_loadu_si128((const __m128i *)&tmp[off]);
                                                           // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
                        mw1 = _mm_shuffle_epi32(mw0, 1 + (1 << 2) + (1 << 4) + (1 << 6));
                        off += stride;
                        if (off >= mod)
                            off -= mod;
                        mm2 = _mm_cvtepu8_epi32(mm0);      // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
                        mm2 = _mm_mullo_epi32(mm2, mw1);   // mm2 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
                        mm1 = _mm_add_epi32(mm1, mm2);     // mm1 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
                case 2:
                        mm0 = _mm_loadu_si128((const __m128i *)&tmp[off]);
                                                           // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
                        mw1 = _mm_shuffle_epi32(mw0, 2 + (2 << 2) + (2 << 4) + (2 << 6));
                        off += stride;
                        if (off >= mod)
                            off -= mod;
                        mm2 = _mm_cvtepu8_epi32(mm0);      // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
                        mm2 = _mm_mullo_epi32(mm2, mw1);   // mm2 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
                        mm1 = _mm_add_epi32(mm1, mm2);     // mm1 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
                case 1:
                        mm0 = _mm_loadu_si128((const __m128i *)&tmp[off]);
                                                           // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
                        mw1 = _mm_shuffle_epi32(mw0, 3 + (3 << 2) + (3 << 4) + (3 << 6));
                        mm2 = _mm_cvtepu8_epi32(mm0);      // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
                        mm2 = _mm_mullo_epi32(mm2, mw1);   // mm2 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
                        mm1 = _mm_add_epi32(mm1, mm2);     // mm1 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
                        j -= 4;
                    } while (j > 0);
                }
            }
            mm1 = _mm_srai_epi32(mm1, WEIGHT_SHIFT); // Shift down
            mm1 = _mm_packus_epi32(mm1,mm1);  // Clamp to 0 to 65535 range.
            mm1 = _mm_packus_epi16(mm1,mm1);  // Clamp to 0 to 255 range.
            j = _mm_extract_epi32(mm1, 0);
            *(int16_t *)dst = j;
            ((uint8_t *)dst)[2] = j>>16;
            dst += 3;
        }
        index++;
    }
}

static void
zoom_y4_sse(uint8_t       *              dst,
            const uint8_t * ipa_restrict tmp,
            const index_t * ipa_restrict index,
            const int32_t * ipa_restrict weights,
            uint32_t                     width,
            uint32_t                     channels,
            uint32_t                     mod,
            int32_t                      y)
{
    uint32_t stride = width * 4;
    uint32_t offset = 0;
    __m128i round = _mm_set1_epi32(WEIGHT_ROUND);

    while (width--)
    {
        uint32_t off = (index->first_pixel + y) * stride + offset;

        offset += 4;
        if (off >= mod)
            off -= mod;

        {
            const int32_t *w = &weights[index->index];
            int32_t        j = (int32_t)index->n;
            __m128i mm0, mm1, mm2, mw0, mw1;

            if (j == 4)
            {
                    mw0 = _mm_loadu_si128((const __m128i *)w);
                    mm0 = _mm_loadu_si128((const __m128i *)&tmp[off]);
                                                      // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
                    mw1 = _mm_shuffle_epi32(mw0, 0 + (0 << 2) + (0 << 4) + (0 << 6));
                    mm0 = _mm_cvtepu8_epi32(mm0);     // mm0 = 000000dd000000cc000000bb000000aa SSE4.1
                    mm0 = _mm_mullo_epi32(mm0,mw1);   // mm0 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
                    mm1 = _mm_add_epi32(round, mm0);  // mm1 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
                    off += stride;
                    if (off >= mod)
                        off -= mod;
                    mm0 = _mm_loadu_si128((const __m128i *)&tmp[off]);
                                                      // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
                    mw1 = _mm_shuffle_epi32(mw0, 1 + (1 << 2) + (1 << 4) + (1 << 6));
                    mm0 = _mm_cvtepu8_epi32(mm0);     // mm0 = 000000dd000000cc000000bb000000aa SSE4.1
                    mm0 = _mm_mullo_epi32(mm0,mw1);   // mm0 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
                    mm1 = _mm_add_epi32(mm1, mm0);    // mm1 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
                    off += stride;
                    if (off >= mod)
                        off -= mod;
                    mm0 = _mm_loadu_si128((const __m128i *)&tmp[off]);
                                                      // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
                    mw1 = _mm_shuffle_epi32(mw0, 2 + (2 << 2) + (2 << 4) + (2 << 6));
                    mm0 = _mm_cvtepu8_epi32(mm0);     // mm0 = 000000dd000000cc000000bb000000aa SSE4.1
                    mm0 = _mm_mullo_epi32(mm0,mw1);   // mm0 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
                    mm1 = _mm_add_epi32(mm1, mm0);    // mm1 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
                    off += stride;
                    if (off >= mod)
                        off -= mod;
                    mm0 = _mm_loadu_si128((const __m128i *)&tmp[off]);
                                                      // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
                    mw1 = _mm_shuffle_epi32(mw0, 3 + (3 << 2) + (3 << 4) + (3 << 6));
                    mm0 = _mm_cvtepu8_epi32(mm0);     // mm0 = 000000dd000000cc000000bb000000aa SSE4.1
                    mm0 = _mm_mullo_epi32(mm0,mw1);   // mm0 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
                    mm1 = _mm_add_epi32(mm1, mm0);    // mm1 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
            }
            else
            {
                int duff = j & 3;
                w -= (4 - j) & 3;
                mw0 = _mm_loadu_si128((const __m128i *)w);
                w += 4;
                mm1 = round;
                /* This is a use of Duff's Device. I'm very sorry, but on the other hand, Yay! */
                switch (duff)
                {
                    do
                    {
                        off += stride;
                        if (off >= mod)
                            off -= mod;
                        mw0 = _mm_loadu_si128((const __m128i *)w);
                        w += 4;
                case 0:
                        mm0 = _mm_loadu_si128((const __m128i *)&tmp[off]);
                                                           // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
                        mw1 = _mm_shuffle_epi32(mw0, 0 + (0 << 2) + (0 << 4) + (0 << 6));
                        off += stride;
                        if (off >= mod)
                            off -= mod;
                        mm2 = _mm_cvtepu8_epi32(mm0);      // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
                        mm2 = _mm_mullo_epi32(mm2, mw1);   // mm2 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
                        mm1 = _mm_add_epi32(mm1, mm2);     // mm1 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
                case 3:
                        mm0 = _mm_loadu_si128((const __m128i *)&tmp[off]);
                                                           // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
                        mw1 = _mm_shuffle_epi32(mw0, 1 + (1 << 2) + (1 << 4) + (1 << 6));
                        off += stride;
                        if (off >= mod)
                            off -= mod;
                        mm2 = _mm_cvtepu8_epi32(mm0);      // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
                        mm2 = _mm_mullo_epi32(mm2, mw1);   // mm2 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
                        mm1 = _mm_add_epi32(mm1, mm2);     // mm1 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
                case 2:
                        mm0 = _mm_loadu_si128((const __m128i *)&tmp[off]);
                                                           // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
                        mw1 = _mm_shuffle_epi32(mw0, 2 + (2 << 2) + (2 << 4) + (2 << 6));
                        off += stride;
                        if (off >= mod)
                            off -= mod;
                        mm2 = _mm_cvtepu8_epi32(mm0);      // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
                        mm2 = _mm_mullo_epi32(mm2, mw1);   // mm2 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
                        mm1 = _mm_add_epi32(mm1, mm2);     // mm1 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
                case 1:
                        mm0 = _mm_loadu_si128((const __m128i *)&tmp[off]);
                                                           // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
                        mw1 = _mm_shuffle_epi32(mw0, 3 + (3 << 2) + (3 << 4) + (3 << 6));
                        mm2 = _mm_cvtepu8_epi32(mm0);      // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
                        mm2 = _mm_mullo_epi32(mm2, mw1);   // mm2 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
                        mm1 = _mm_add_epi32(mm1, mm2);     // mm1 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
                        j -= 4;
                    } while (j > 0);
                }
            }
            mm1 = _mm_srai_epi32(mm1, WEIGHT_SHIFT); // Shift down
            mm1 = _mm_packus_epi32(mm1,mm1);  // Clamp to 0 to 65535 range.
            mm1 = _mm_packus_epi16(mm1,mm1);  // Clamp to 0 to 255 range.
            *(int32_t *)dst = _mm_extract_epi32(mm1, 0);
            dst += 4;
        }
        index++;
    }
}
