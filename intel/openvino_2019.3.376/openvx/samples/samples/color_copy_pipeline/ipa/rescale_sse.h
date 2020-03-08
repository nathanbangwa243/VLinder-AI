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
#include <string.h>

static void
zoom_x1to1_1_sse(uint8_t       * ipa_restrict tmp,
                 const void    * ipa_restrict src,
                 int                          tmp_width,
                 int                          num_colors,
                 const index_t * ipa_restrict index,
                 const int32_t * ipa_restrict weights)
{
    __m128i round = _mm_set1_epi32(WEIGHT_ROUND);

    /* The SSE reads 16 pixels, so to avoid overruns, we have to do the last few slowly */
    for ( ; tmp_width != 0; --tmp_width )
    {
        int j = index->n;
        int slow = index->slow;
        const uint8_t *ipa_restrict pp = ((const uint8_t *)src) + index->first_pixel;
        const int32_t *ipa_restrict wp = weights + (index++)->index;

        if (j <= 4 && !slow)
        {
            /* This reads 16 uint8_t's rather than the 4 it needs. The !slow condition
             * above prevents this overrrunning the end. */
            __m128i mw0, mm0;
            mw0 = _mm_loadu_si128((const __m128i *)wp);
            mm0 = _mm_loadu_si128((const __m128i *)pp);
                                              // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
            mm0 = _mm_cvtepu8_epi32(mm0);     // mm0 = 000000dd000000cc000000bb000000aa SSE4.1
            mm0 = _mm_mullo_epi32(mm0,mw0);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
            mm0 = _mm_hadd_epi32(mm0,mm0);
            mm0 = _mm_hadd_epi32(mm0,mm0);
            mm0 = _mm_add_epi32(mm0, round);  // Add round
            mm0 = _mm_srai_epi32(mm0, WEIGHT_SHIFT-8); // Shift down
            mm0 = _mm_packus_epi32(mm0,mm0);  // Clamp to 0 to 65535 range.
            *tmp++ = _mm_extract_epi8(mm0,1);
        }
        else
        {
            int weight0 = WEIGHT_ROUND;

            for ( ; j > 0; --j )
                weight0 += *pp++ * *wp++;
            weight0 >>= WEIGHT_SHIFT;
            *tmp++ = (uint8_t)CLAMP(weight0, 0, 255);
        }
    }
}

static void
zoom_x1to1_3_sse(uint8_t       * ipa_restrict tmp,
                 const void    * ipa_restrict src,
                 int                          tmp_width,
                 int                          num_colors,
                 const index_t * ipa_restrict index,
                 const int32_t * ipa_restrict weights)
{
    __m128i round = _mm_set1_epi32(WEIGHT_ROUND);
    uint8_t local[256];

    while (!index->slow && tmp_width > 0)
    {
        int j;
        const uint8_t *ipa_restrict pp = ((const uint8_t *)src) + index->first_pixel;
        const int32_t *ipa_restrict wp;
        __m128i mm0, mm1, mm4, mw0, mw1;

        if (0)
        {
loaded_local:
            pp = local;
        }
        j = index->n;
        wp = weights + (index++)->index;
        tmp_width--;
        mm4 = round; // mm4 will be our running sum of WEIGHT_ROUND + pixel values * weights
        mm0 = _mm_loadu_si128((const __m128i *)pp);
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        if (j == 4)
        {
            mw0 = _mm_loadu_si128((const __m128i *)wp);
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
            const int32_t *weights = wp;
            weights -= (4 - j) & 3;
            pp += (off ? off : 4) * 3;
            mw0 = _mm_loadu_si128((const __m128i *)weights);
            weights += 4;
            /* This is a use of Duff's Device. I'm very sorry, but on the other hand, Yay! */
            switch (off)
            {
                do
                {
                    mm0 = _mm_loadu_si128((const __m128i *)pp);
                    // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
                    pp += 4 * 3;
                    mw0 = _mm_loadu_si128((const __m128i *)weights);
                    weights += 4;
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
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT - 8); // Shift down
        mm4 = _mm_packus_epi32(mm4, mm4);  // Clamp to 0 to 65535 range.
        *tmp++ = _mm_extract_epi8(mm4, 1);
        *tmp++ = _mm_extract_epi8(mm4, 3);
        *tmp++ = _mm_extract_epi8(mm4, 5);
    }

    /* And finish any stragglers */
    if (tmp_width != 0)
    {
        memcpy(local, ((const uint8_t *)src) + index->first_pixel, index->n * 3);
        goto loaded_local;
    }
}

static void
zoom_x1to1_4_sse(uint8_t       * ipa_restrict tmp,
                 const void    * ipa_restrict src,
                 int                          tmp_width,
                 int                          num_colors,
                 const index_t * ipa_restrict index,
                 const int32_t * ipa_restrict weights)
{
    __m128i round = _mm_set1_epi32(WEIGHT_ROUND);
    uint8_t local[256];

    while (!index->slow && tmp_width > 0)
    {
        int j;
        const uint8_t *ipa_restrict pp = ((const uint8_t *)src) + index->first_pixel;
        const int32_t *ipa_restrict wp;
        __m128i mm0, mm1, mm4, mw0, mw1;

        if (0)
        {
loaded_local:
           pp = local;
        }
        j = index->n;
        wp = weights + (index++)->index;
        tmp_width--;
        mm4 = round; // mm4 will be our running sum of WEIGHT_ROUND + pixel values * weights
        mm0 = _mm_loadu_si128((const __m128i *)pp);
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        if (j == 4)
        {
            mw0 = _mm_loadu_si128((const __m128i *)wp);
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
            const int32_t *weights = wp;
            weights -= (4 - j) & 3;
            pp += (off ? off : 4) * 4;
            mw0 = _mm_loadu_si128((const __m128i *)weights);
            weights += 4;
            /* This is a use of Duff's Device. I'm very sorry, but on the other hand, Yay! */
            switch (off)
            {
                do
                {
                    mm0 = _mm_loadu_si128((const __m128i *)pp);
                    // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
                    pp += 4 * 4;
                    mw0 = _mm_loadu_si128((const __m128i *)weights);
                    weights += 4;
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
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT - 8); // Shift down
        mm4 = _mm_packus_epi32(mm4,mm4);  // Clamp to 0 to 65535 range.
        *tmp++ = _mm_extract_epi8(mm4,1);
        *tmp++ = _mm_extract_epi8(mm4,3);
        *tmp++ = _mm_extract_epi8(mm4,5);
        *tmp++ = _mm_extract_epi8(mm4,7);
    }

    /* And finish any stragglers */
    if (tmp_width)
    {
        memcpy(local, ((const uint8_t *)src) + index->first_pixel, index->n * 4);
        goto loaded_local;
    }
}

static void
zoom_x2to2_1_sse(uint8_t       * ipa_restrict tmp_,
                 const void    * ipa_restrict src,
                 int                          tmp_width,
                 int                          num_colors,
                 const index_t * ipa_restrict index,
                 const int32_t * ipa_restrict weights)
{
    uint16_t *ipa_restrict tmp = (uint16_t *)tmp_;
    __m128i round = _mm_set1_epi32(WEIGHT_ROUND);

    /* The SSE reads 16 pixels, so to avoid overruns, we have to do the last few slowly */
    for ( ; tmp_width != 0; --tmp_width )
    {
        int j = index->n;
        int slow = index->slow;
        const uint16_t *ipa_restrict pp = ((const uint16_t *)src) + index->first_pixel;
        const int32_t *ipa_restrict wp = weights + (index++)->index;

        if (j <= 4 && !slow)
        {
            /* This reads 8 uint16_t's rather than the 4 it needs. The !slow condition
             * above prevents this overrrunning the end. */
            __m128i mw0, mm0;
            mw0 = _mm_loadu_si128((const __m128i *)wp);
            mm0 = _mm_loadu_si128((const __m128i *)pp);
                                              // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
            mm0 = _mm_cvtepu16_epi32(mm0);    // mm0 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
            mm0 = _mm_mullo_epi32(mm0,mw0);   // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
            mm0 = _mm_hadd_epi32(mm0,mm0);
            mm0 = _mm_hadd_epi32(mm0,mm0);
            mm0 = _mm_add_epi32(mm0, round);  // Add round
            mm0 = _mm_srai_epi32(mm0, WEIGHT_SHIFT); // Shift down
            mm0 = _mm_packus_epi32(mm0,mm0);  // Clamp to 0 to 65535 range.
            *tmp++ = _mm_extract_epi16(mm0,0);
        }
        else
        {
            int weight0 = WEIGHT_ROUND;

            for ( ; j > 0; --j )
                weight0 += *pp++ * *wp++;
            weight0 >>= WEIGHT_SHIFT;
            *tmp++ = (uint16_t)CLAMP(weight0, 0, 65535);
        }
    }
}

static void
zoom_x2to2_3_sse(uint8_t       * ipa_restrict tmp_,
                 const void    * ipa_restrict src,
                 int                          tmp_width,
                 int                          num_colors,
                 const index_t * ipa_restrict index,
                 const int32_t * ipa_restrict weights)
{
    uint16_t *ipa_restrict tmp = (uint16_t *)tmp_;
    uint16_t local[256];
    __m128i round = _mm_set1_epi32(WEIGHT_ROUND);

    while (!index->slow && tmp_width > 0)
    {
        int j;
        const uint16_t *ipa_restrict pp = ((const uint16_t *)src) + index->first_pixel;
        const int32_t *ipa_restrict wp;
        __m128i mm0, mm1, mm4, mw0, mw1;

        if (0)
        {
loaded_local:
            pp = local;
        }
        j = index->n;
        wp = weights + (index++)->index;
        tmp_width--;
        mm4 = round; // mm4 will be our running sum of WEIGHT_ROUND + pixel values * weights
        mm0 = _mm_loadu_si128((const __m128i *)pp);         // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        if (j == 4)
        {
            mw0 = _mm_loadu_si128((const __m128i *)wp);
            mw1 = _mm_shuffle_epi32(mw0, 0 + (0 << 2) + (0 << 4) + (0 << 6));
            mm1 = _mm_cvtepu16_epi32(mm0);                  // mm1 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
            mm1 = _mm_mullo_epi32(mm1, mw1);                // mm1 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
            mm4 = _mm_add_epi32(mm4, mm1);                  // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
            mm0 = _mm_srli_si128(mm0, 6);                   // mm0 = 000000000000hhhhggggffffeeeedddd SSE2
            mw1 = _mm_shuffle_epi32(mw0, 1 + (1 << 2) + (1 << 4) + (1 << 6));
            mm1 = _mm_cvtepu16_epi32(mm0);                  // mm1 = 0000gggg0000ffff0000eeee0000dddd SSE4.1
            mm1 = _mm_mullo_epi32(mm1, mw1);                // mm1 = 00wgwgwg00wfwfwf00wewewe00wdwdwd SSE4.1
            mm4 = _mm_add_epi32(mm4, mm1);                  // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
            mm0 = _mm_loadu_si128((const __m128i *)(pp+6)); // mm0 = nnnnmmmmllllkkkkjjjjiiiihhhhgggg SSE2
            mw1 = _mm_shuffle_epi32(mw0, 2 + (2 << 2) + (2 << 4) + (2 << 6));
            mm1 = _mm_cvtepu16_epi32(mm0);                  // mm1 = 0000jjjj0000iiii0000hhhh0000gggg SSE4.1
            mm1 = _mm_mullo_epi32(mm1, mw1);                // mm1 = 00wjwjwj00wiwiwi00whwhwh00wgwgwg SSE4.1
            mm4 = _mm_add_epi32(mm4, mm1);                  // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
            mm0 = _mm_srli_si128(mm0, 6);                   // mm0 = 000000000000nnnnmmmmllllkkkkjjjj SSE2
            mw1 = _mm_shuffle_epi32(mw0, 3 + (3 << 2) + (3 << 4) + (3 << 6));
            mm1 = _mm_cvtepu16_epi32(mm0);                  // mm1 = 0000mmmm0000llll0000kkkk0000jjjj SSE4.1
            mm1 = _mm_mullo_epi32(mm1, mw1);                // mm1 = 00wmwmwm00wlwlwl00wkwkwk00wjwjwj SSE4.1
            mm4 = _mm_add_epi32(mm4, mm1);                  // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        }
        else
        {
            int off = j & 3;
            const int32_t *weights = wp;
            weights -= (4 - j) & 3;
            pp += (j & 1 ? 3 : 6);
            mw0 = _mm_loadu_si128((const __m128i *)weights);
            weights += 4;
            /* This is a use of Duff's Device. I'm very sorry, but on the other hand, Yay! */
            switch (off)
            {
                do
                {
                    mm0 = _mm_loadu_si128((const __m128i *)pp); // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
                    pp += 2 * 3;
                    mw0 = _mm_loadu_si128((const __m128i *)weights);
                    weights += 4;
            case 0:
                    mw1 = _mm_shuffle_epi32(mw0, 0 + (0 << 2) + (0 << 4) + (0 << 6));
                    mm1 = _mm_cvtepu16_epi32(mm0);              // mm1 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
                    mm1 = _mm_mullo_epi32(mm1, mw1);            // mm1 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
                    mm4 = _mm_add_epi32(mm4, mm1);              // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
                    mm0 = _mm_srli_si128(mm0, 6);               // mm0 = 000000000000hhhhggggffffeeeedddd SSE2
            case 3:
                    mw1 = _mm_shuffle_epi32(mw0, 1 + (1 << 2) + (1 << 4) + (1 << 6));
                    mm1 = _mm_cvtepu16_epi32(mm0);              // mm1 = 0000gggg0000ffff0000eeee0000dddd SSE4.1
                    mm1 = _mm_mullo_epi32(mm1, mw1);            // mm1 = 00wgwgwg00wfwfwf00wewewe00wdwdwd SSE4.1
                    mm4 = _mm_add_epi32(mm4, mm1);              // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
                    mm0 = _mm_loadu_si128((const __m128i *)pp); // mm0 = nnnnmmmmllllkkkkjjjjiiiihhhhgggg SSE2
                    pp += 2 * 3;
            case 2:
                    mw1 = _mm_shuffle_epi32(mw0, 2 + (2 << 2) + (2 << 4) + (2 << 6));
                    mm1 = _mm_cvtepu16_epi32(mm0);             // mm1 = 0000jjjj0000iiii0000hhhh0000gggg SSE4.1
                    mm1 = _mm_mullo_epi32(mm1, mw1);           // mm1 = 00wjwjwj00wiwiwi00whwhwh00wgwgwg SSE4.1
                    mm4 = _mm_add_epi32(mm4, mm1);             // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
                    mm0 = _mm_srli_si128(mm0, 6);              // mm0 = 000000000000nnnnmmmmllllkkkkjjjj SSE2
            case 1:
                    mw1 = _mm_shuffle_epi32(mw0, 3 + (3 << 2) + (3 << 4) + (3 << 6));
                    mm1 = _mm_cvtepu16_epi32(mm0);             // mm1 = 0000mmmm0000llll0000kkkk0000jjjj SSE4.1
                    mm1 = _mm_mullo_epi32(mm1, mw1);           // mm1 = 00wmwmwm00wlwlwl00wkwkwk00wjwjwj SSE4.1
                    mm4 = _mm_add_epi32(mm4, mm1);             // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
                    j -= 4;
                } while (j > 0);
            }
        }
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT); // Shift down
        mm4 = _mm_packus_epi32(mm4, mm4);  // Clamp to 0 to 65535 range.
        *(uint32_t *)tmp = _mm_extract_epi32(mm4, 0);
        tmp[2] = _mm_extract_epi16(mm4, 2);
        tmp += 3;
    }


    /* And finish any stragglers */
    if (tmp_width)
    {
        memcpy(local, ((const uint16_t *)src) + index->first_pixel, index->n * 3 * sizeof(uint16_t));
        goto loaded_local;
    }
}

static void
zoom_x2to2_4_sse(uint8_t       * ipa_restrict tmp_,
                 const void    * ipa_restrict src,
                 int                          tmp_width,
                 int                          num_colors,
                 const index_t * ipa_restrict index,
                 const int32_t * ipa_restrict weights)
{
    uint16_t *ipa_restrict tmp = (uint16_t *)tmp_;
    uint16_t local[256];
    __m128i round = _mm_set1_epi32(WEIGHT_ROUND);

    while (!index->slow && tmp_width > 0)
    {
        int j;
        const uint16_t *ipa_restrict pp = ((const uint16_t *)src) + index->first_pixel;
        const int32_t *ipa_restrict wp;
        __m128i mm0, mm1, mm4, mw0, mw1;

        if (0)
        {
loaded_local:
            pp = local;
        }
        j = index->n;
        wp = weights + (index++)->index;
        tmp_width--;
        mm4 = round; // mm4 will be our running sum of WEIGHT_ROUND + pixel values * weights
        mm0 = _mm_loadu_si128((const __m128i *)pp);         // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        if (j == 4)
        {
            mw0 = _mm_loadu_si128((const __m128i *)wp);
            mw1 = _mm_shuffle_epi32(mw0, 0 + (0 << 2) + (0 << 4) + (0 << 6));
            mm1 = _mm_cvtepu16_epi32(mm0);                  // mm1 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
            mm1 = _mm_mullo_epi32(mm1, mw1);                // mm1 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
            mm4 = _mm_add_epi32(mm4, mm1);                  // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
            mm0 = _mm_srli_si128(mm0, 8);                   // mm0 = 000000000000hhhhggggffffeeeedddd SSE2
            mw1 = _mm_shuffle_epi32(mw0, 1 + (1 << 2) + (1 << 4) + (1 << 6));
            mm1 = _mm_cvtepu16_epi32(mm0);                  // mm1 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
            mm1 = _mm_mullo_epi32(mm1, mw1);                // mm1 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
            mm4 = _mm_add_epi32(mm4, mm1);                  // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
            mm0 = _mm_loadu_si128((const __m128i *)(pp+8)); // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
            mw1 = _mm_shuffle_epi32(mw0, 2 + (2 << 2) + (2 << 4) + (2 << 6));
            mm1 = _mm_cvtepu16_epi32(mm0);                  // mm1 = 0000llll0000kkkk0000jjjj0000iiii SSE4.1
            mm1 = _mm_mullo_epi32(mm1, mw1);                // mm1 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
            mm4 = _mm_add_epi32(mm4, mm1);                  // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
            mm0 = _mm_srli_si128(mm0, 8);                   // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
            mw1 = _mm_shuffle_epi32(mw0, 3 + (3 << 2) + (3 << 4) + (3 << 6));
            mm1 = _mm_cvtepu16_epi32(mm0);                  // mm1 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
            mm1 = _mm_mullo_epi32(mm1, mw1);                // mm1 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
            mm4 = _mm_add_epi32(mm4, mm1);                  // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        }
        else
        {
            int off = j & 3;
            const int32_t *weights = wp;
            weights -= (4 - j) & 3;
            pp += (j & 1 ? 4 : 8);
            mw0 = _mm_loadu_si128((const __m128i *)weights);
            weights += 4;
            /* This is a use of Duff's Device. I'm very sorry, but on the other hand, Yay! */
            switch (off)
            {
                do
                {
                    mm0 = _mm_loadu_si128((const __m128i *)pp); // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
                    pp += 2 * 4;
                    mw0 = _mm_loadu_si128((const __m128i *)weights);
                    weights += 4;
            case 0:
                    mw1 = _mm_shuffle_epi32(mw0, 0 + (0 << 2) + (0 << 4) + (0 << 6));
                    mm1 = _mm_cvtepu16_epi32(mm0);              // mm1 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
                    mm1 = _mm_mullo_epi32(mm1, mw1);            // mm1 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
                    mm4 = _mm_add_epi32(mm4, mm1);              // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
                    mm0 = _mm_srli_si128(mm0, 8);               // mm0 = 0000000000000000hhhhggggffffeeee SSE2
            case 3:
                    mw1 = _mm_shuffle_epi32(mw0, 1 + (1 << 2) + (1 << 4) + (1 << 6));
                    mm1 = _mm_cvtepu16_epi32(mm0);              // mm1 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
                    mm1 = _mm_mullo_epi32(mm1, mw1);            // mm1 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
                    mm4 = _mm_add_epi32(mm4, mm1);              // mm4 = 00xxxxxx00xxxxxx0000xxxx00xxxxxx SSE2
                    mm0 = _mm_loadu_si128((const __m128i *)pp); // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
                    pp += 2 * 4;
            case 2:
                    mw1 = _mm_shuffle_epi32(mw0, 2 + (2 << 2) + (2 << 4) + (2 << 6));
                    mm1 = _mm_cvtepu16_epi32(mm0);              // mm1 = 0000llll0000kkkk0000jjjj0000iiii SSE4.1
                    mm1 = _mm_mullo_epi32(mm1, mw1);            // mm1 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
                    mm4 = _mm_add_epi32(mm4, mm1);              // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
                    mm0 = _mm_srli_si128(mm0, 8);               // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
            case 1:
                    mw1 = _mm_shuffle_epi32(mw0, 3 + (3 << 2) + (3 << 4) + (3 << 6));
                    mm1 = _mm_cvtepu16_epi32(mm0);             // mm1 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
                    mm1 = _mm_mullo_epi32(mm1, mw1);           // mm1 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
                    mm4 = _mm_add_epi32(mm4, mm1);             // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
                    j -= 4;
                } while (j > 0);
            }
        }
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT); // Shift down
        mm4 = _mm_packus_epi32(mm4, mm4);  // Clamp to 0 to 65535 range.
#if defined (_M_X64)
        *(uint64_t *)tmp = _mm_extract_epi64(mm4, 0);
        tmp += 4;
#else
        *tmp++ = _mm_extract_epi16(mm4, 0);
        *tmp++ = _mm_extract_epi16(mm4, 1);
        *tmp++ = _mm_extract_epi16(mm4, 2);
        *tmp++ = _mm_extract_epi16(mm4, 3);
#endif
    }

    /* And finish any stragglers */
    if (tmp_width)
    {
        memcpy(local, ((const uint16_t *)src) + index->first_pixel, index->n * 4 * sizeof(uint16_t));
        goto loaded_local;
    }
}

static inline void
store_trailing_bytes(uint8_t *d, int width, __m128i mm4)
{
    /* Stores in reverse order. */
    switch (width)
    {
    case 15: d[14] = _mm_extract_epi8(mm4, 14);
    case 14: d[13] = _mm_extract_epi8(mm4, 13);
    case 13: d[12] = _mm_extract_epi8(mm4, 12);
    case 12: d[11] = _mm_extract_epi8(mm4, 11);
    case 11: d[10] = _mm_extract_epi8(mm4, 10);
    case 10: d[9]  = _mm_extract_epi8(mm4,  9);
    case  9: d[8]  = _mm_extract_epi8(mm4,  8);
    case  8: d[7]  = _mm_extract_epi8(mm4,  7);
    case  7: d[6]  = _mm_extract_epi8(mm4,  6);
    case  6: d[5]  = _mm_extract_epi8(mm4,  5);
    case  5: d[4]  = _mm_extract_epi8(mm4,  4);
    case  4: d[3]  = _mm_extract_epi8(mm4,  3);
    case  3: d[2]  = _mm_extract_epi8(mm4,  2);
    case  2: d[1]  = _mm_extract_epi8(mm4,  1);
    case  1: d[0]  = _mm_extract_epi8(mm4,  0);
    default: {/* Never happens */}
    }
}

static inline void
store_trailing_shorts(uint16_t *d, int width, __m128i mm4, __m128i mm6)
{
    /* Stores in reverse order. */
    switch (width)
    {
    case 15: d[14] = _mm_extract_epi16(mm6, 6);
    case 14: d[13] = _mm_extract_epi16(mm6, 5);
    case 13: d[12] = _mm_extract_epi16(mm6, 4);
    case 12: d[11] = _mm_extract_epi16(mm6, 3);
    case 11: d[10] = _mm_extract_epi16(mm6, 2);
    case 10: d[9]  = _mm_extract_epi16(mm6, 1);
    case  9: d[8]  = _mm_extract_epi16(mm6, 0);
    case  8: d[7]  = _mm_extract_epi16(mm4, 7);
    case  7: d[6]  = _mm_extract_epi16(mm4, 6);
    case  6: d[5]  = _mm_extract_epi16(mm4, 5);
    case  5: d[4]  = _mm_extract_epi16(mm4, 4);
    case  4: d[3]  = _mm_extract_epi16(mm4, 3);
    case  3: d[2]  = _mm_extract_epi16(mm4, 2);
    case  2: d[1]  = _mm_extract_epi16(mm4, 1);
    case  1: d[0]  = _mm_extract_epi16(mm4, 0);
    default: {/* Never happens */}
    }
}

static inline void
zoom_y1to1_2_sse(void          * ipa_restrict dst,
                 const uint8_t * ipa_restrict tmp,
                 int                          width,
                 int                          byte_stride,
                 const index_t * ipa_restrict index,
                 const int32_t * ipa_restrict weights)
{
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint8_t *ipa_restrict d = (uint8_t *)dst;
    __m128i mw0 = _mm_set1_epi32(cbp[0]*256);
    __m128i mw1 = _mm_set1_epi32(cbp[1]*256);
    __m128i round = _mm_set1_epi32(WEIGHT_ROUND);

    tmp += index->first_pixel;
    width -= 16;
    while (width >= 0)
    {
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        // Weight 0
        mm0 = _mm_loadu_si128((const __m128i *)tmp);
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm4 = _mm_mullo_epi32(mm2,mw0);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm5 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm6 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm7 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        // Weight 1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride));
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Add rounding
        mm4 = _mm_add_epi32(mm4, round);
        mm5 = _mm_add_epi32(mm5, round);
        mm6 = _mm_add_epi32(mm6, round);
        mm7 = _mm_add_epi32(mm7, round);
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4,mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6,mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        mm4 = _mm_srli_epi16(mm4,8);      // mm4 = 00HH00GG00FF00EE00DD00CC00BB00AA
        mm6 = _mm_srli_epi16(mm6,8);      // mm6 = 00PP00OO00NN00MM00LL00KK00JJ00II
        mm4 = _mm_packus_epi16(mm4,mm6);  // mm4 = PPOONNMMLLKKJJIIHHGGFFEEDDCCBBAA
        _mm_storeu_si128((__m128i *)d, mm4);
        d += 16;
        tmp += 16;
        width -= 16;
    }

    width += 16;
    if (width)
    {
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        // Weight 0
        mm0 = _mm_loadu_si128((const __m128i *)tmp);
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm4 = _mm_mullo_epi32(mm2,mw0);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm5 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm6 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm7 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        // Weight 1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride));
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Add rounding
        mm4 = _mm_add_epi32(mm4, round);
        mm5 = _mm_add_epi32(mm5, round);
        mm6 = _mm_add_epi32(mm6, round);
        mm7 = _mm_add_epi32(mm7, round);
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4,mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6,mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        mm4 = _mm_srli_epi16(mm4,8);      // mm4 = 00HH00GG00FF00EE00DD00CC00BB00AA
        mm6 = _mm_srli_epi16(mm6,8);      // mm6 = 00PP00OO00NN00MM00LL00KK00JJ00II
        mm4 = _mm_packus_epi16(mm4,mm6);  // mm4 = PPOONNMMLLKKJJIIHHGGFFEEDDCCBBAA
        store_trailing_bytes(d, width, mm4);
    }
}

static inline void
zoom_y1to1_3_sse(void          * ipa_restrict dst,
                 const uint8_t * ipa_restrict tmp,
                 int                          width,
                 int                          byte_stride,
                 const index_t * ipa_restrict index,
                 const int32_t * ipa_restrict weights)
{
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint8_t *ipa_restrict d = (uint8_t *)dst;
    __m128i mw0 = _mm_set1_epi32(cbp[0]*256);
    __m128i mw1 = _mm_set1_epi32(cbp[1]*256);
    __m128i mw2 = _mm_set1_epi32(cbp[2]*256);
    __m128i round = _mm_set1_epi32(WEIGHT_ROUND);

    tmp += index->first_pixel;
    width -= 16;
    while (width >= 0)
    {
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        // Weight 0
        mm0 = _mm_loadu_si128((const __m128i *)tmp);
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm4 = _mm_mullo_epi32(mm2,mw0);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm5 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm6 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm7 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        // Weight 1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride));
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Weight 2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + 2 * byte_stride));
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Add rounding
        mm4 = _mm_add_epi32(mm4, round);
        mm5 = _mm_add_epi32(mm5, round);
        mm6 = _mm_add_epi32(mm6, round);
        mm7 = _mm_add_epi32(mm7, round);
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4,mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6,mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        mm4 = _mm_srli_epi16(mm4,8);      // mm4 = 00HH00GG00FF00EE00DD00CC00BB00AA
        mm6 = _mm_srli_epi16(mm6,8);      // mm6 = 00PP00OO00NN00MM00LL00KK00JJ00II
        mm4 = _mm_packus_epi16(mm4,mm6);  // mm4 = PPOONNMMLLKKJJIIHHGGFFEEDDCCBBAA
        _mm_storeu_si128((__m128i *)d, mm4);
        d += 16;
        tmp += 16;
        width -= 16;
    }

    width += 16;
    if (width)
    {
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        // Weight 0
        mm0 = _mm_loadu_si128((const __m128i *)tmp);
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm4 = _mm_mullo_epi32(mm2,mw0);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm5 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm6 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm7 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        // Weight 1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride));
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Weight 2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + 2 * byte_stride));
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Add rounding
        mm4 = _mm_add_epi32(mm4, round);
        mm5 = _mm_add_epi32(mm5, round);
        mm6 = _mm_add_epi32(mm6, round);
        mm7 = _mm_add_epi32(mm7, round);
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4,mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6,mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        mm4 = _mm_srli_epi16(mm4,8);      // mm4 = 00HH00GG00FF00EE00DD00CC00BB00AA
        mm6 = _mm_srli_epi16(mm6,8);      // mm6 = 00PP00OO00NN00MM00LL00KK00JJ00II
        mm4 = _mm_packus_epi16(mm4,mm6);  // mm4 = PPOONNMMLLKKJJIIHHGGFFEEDDCCBBAA
        store_trailing_bytes(d, width, mm4);
    }
}

static inline void
zoom_y1to1_4_sse(void          * ipa_restrict dst,
                 const uint8_t * ipa_restrict tmp,
                 int             width,
                 int             byte_stride,
                 const index_t * ipa_restrict index,
                 const int32_t * ipa_restrict weights)
{
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint8_t *ipa_restrict d = (uint8_t *)dst;
    __m128i mw0   = _mm_set1_epi32(cbp[0]*256);
    __m128i mw1   = _mm_set1_epi32(cbp[1]*256);
    __m128i mw2   = _mm_set1_epi32(cbp[2]*256);
    __m128i mw3   = _mm_set1_epi32(cbp[3]*256);
    __m128i round = _mm_set1_epi32(WEIGHT_ROUND);

    tmp += index->first_pixel;
    width -= 16;
    while (width >= 0)
    {
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        // Weight 0
        mm0 = _mm_loadu_si128((const __m128i *)tmp);
                                            // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm4 = _mm_mullo_epi32(mm2,mw0);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm5 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm6 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm7 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        // Weight 1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride));
                                            // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Weight 2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + 2*byte_stride));
                                            // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Weight 3
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + 3*byte_stride));
                                            // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Add rounding
        mm4 = _mm_add_epi32(mm4, round);
        mm5 = _mm_add_epi32(mm5, round);
        mm6 = _mm_add_epi32(mm6, round);
        mm7 = _mm_add_epi32(mm7, round);
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4,mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6,mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        mm4 = _mm_srli_epi16(mm4,8);      // mm4 = 00HH00GG00FF00EE00DD00CC00BB00AA
        mm6 = _mm_srli_epi16(mm6,8);      // mm6 = 00PP00OO00NN00MM00LL00KK00JJ00II
        mm4 = _mm_packus_epi16(mm4,mm6);  // mm4 = PPOONNMMLLKKJJIIHHGGFFEEDDCCBBAA
        _mm_storeu_si128((__m128i *)d, mm4);
        d += 16;
        tmp += 16;
        width -= 16;
    }

    width += 16;
    if (width)
    {
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        // Weight 0
        mm0 = _mm_loadu_si128((const __m128i *)tmp);
                                            // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm4 = _mm_mullo_epi32(mm2,mw0);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm5 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm6 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm7 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        // Weight 1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride));
                                            // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Weight 2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + 2*byte_stride));
                                            // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Weight 3
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + 3*byte_stride));
                                            // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Add rounding
        mm4 = _mm_add_epi32(mm4, round);
        mm5 = _mm_add_epi32(mm5, round);
        mm6 = _mm_add_epi32(mm6, round);
        mm7 = _mm_add_epi32(mm7, round);
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4,mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6,mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        mm4 = _mm_srli_epi16(mm4,8);      // mm4 = 00HH00GG00FF00EE00DD00CC00BB00AA
        mm6 = _mm_srli_epi16(mm6,8);      // mm6 = 00PP00OO00NN00MM00LL00KK00JJ00II
        mm4 = _mm_packus_epi16(mm4,mm6);  // mm4 = PPOONNMMLLKKJJIIHHGGFFEEDDCCBBAA
        store_trailing_bytes(d, width, mm4);
    }
}

static inline void
zoom_y1to1_5_sse(void          * ipa_restrict dst,
                 const uint8_t * ipa_restrict tmp,
                 int                          width,
                 int                          byte_stride,
                 const index_t * ipa_restrict index,
                 const int32_t * ipa_restrict weights)
{
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint8_t *ipa_restrict d = (uint8_t *)dst;
    __m128i mw0 = _mm_set1_epi32(cbp[0]*256);
    __m128i mw1 = _mm_set1_epi32(cbp[1]*256);
    __m128i mw2 = _mm_set1_epi32(cbp[2]*256);
    __m128i mw3 = _mm_set1_epi32(cbp[3]*256);
    __m128i mw4 = _mm_set1_epi32(cbp[4]*256);
    __m128i round = _mm_set1_epi32(WEIGHT_ROUND);

    tmp += index->first_pixel;
    width -= 16;
    while (width >= 0)
    {
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        // Weight 0
        mm0 = _mm_loadu_si128((const __m128i *)tmp);
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm4 = _mm_mullo_epi32(mm2, mw0);  // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm5 = _mm_mullo_epi32(mm2, mw0);  // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm6 = _mm_mullo_epi32(mm2, mw0);  // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm7 = _mm_mullo_epi32(mm2, mw0);  // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        // Weight 1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride));
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw1);  // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw1);  // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw1);  // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw1);  // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Weight 2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + 2 * byte_stride));
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw2);  // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw2);  // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw2);  // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw2);  // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Weight 3
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + 3 * byte_stride));
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw3);  // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw3);  // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw3);  // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw3);  // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Weight 4
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + 4 * byte_stride));
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw4);  // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw4);  // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw4);  // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw4);  // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Add rounding
        mm4 = _mm_add_epi32(mm4, round);
        mm5 = _mm_add_epi32(mm5, round);
        mm6 = _mm_add_epi32(mm6, round);
        mm7 = _mm_add_epi32(mm7, round);
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4, mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6, mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        mm4 = _mm_srli_epi16(mm4,8);      // mm4 = 00HH00GG00FF00EE00DD00CC00BB00AA
        mm6 = _mm_srli_epi16(mm6,8);      // mm6 = 00PP00OO00NN00MM00LL00KK00JJ00II
        mm4 = _mm_packus_epi16(mm4,mm6);  // mm4 = PPOONNMMLLKKJJIIHHGGFFEEDDCCBBAA
        _mm_storeu_si128((__m128i *)d, mm4);
        d += 16;
        tmp += 16;
        width -= 16;
    }

    width += 16;
    if (width)
    {
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        // Weight 0
        mm0 = _mm_loadu_si128((const __m128i *)tmp);
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm4 = _mm_mullo_epi32(mm2, mw0);  // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm5 = _mm_mullo_epi32(mm2, mw0);  // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm6 = _mm_mullo_epi32(mm2, mw0);  // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm7 = _mm_mullo_epi32(mm2, mw0);  // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        // Weight 1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride));
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw1);  // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw1);  // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw1);  // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw1);  // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Weight 2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + 2 * byte_stride));
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw2);  // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw2);  // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw2);  // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw2);  // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Weight 3
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + 3 * byte_stride));
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw3);  // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw3);  // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw3);  // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw3);  // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Weight 4
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + 4 * byte_stride));
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw4);  // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw4);  // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw4);  // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw4);  // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Add rounding
        mm4 = _mm_add_epi32(mm4, round);
        mm5 = _mm_add_epi32(mm5, round);
        mm6 = _mm_add_epi32(mm6, round);
        mm7 = _mm_add_epi32(mm7, round);
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4, mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6, mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        mm4 = _mm_srli_epi16(mm4,8);      // mm4 = 00HH00GG00FF00EE00DD00CC00BB00AA
        mm6 = _mm_srli_epi16(mm6,8);      // mm6 = 00PP00OO00NN00MM00LL00KK00JJ00II
        mm4 = _mm_packus_epi16(mm4,mm6);  // mm4 = PPOONNMMLLKKJJIIHHGGFFEEDDCCBBAA
        store_trailing_bytes(d, width, mm4);
    }
}

static inline void
zoom_y1to1_n_sse(void          * ipa_restrict dst,
                 const uint8_t * ipa_restrict tmp,
                 int                          width,
                 int                          byte_stride,
                 const index_t * ipa_restrict index,
                 const int32_t * ipa_restrict weights)
{
    int cn = index->n;
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint8_t *ipa_restrict d = (uint8_t *)dst;
    __m128i round = _mm_set1_epi32(WEIGHT_ROUND);

    tmp += index->first_pixel;
    width -= 16;
    while (width >= 0)
    {
        int j;
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        mm4 = round; // running sum of WEIGHT_ROUND + pixel values * weights
        mm5 = round; // running sum of WEIGHT_ROUND + pixel values * weights
        mm6 = round; // running sum of WEIGHT_ROUND + pixel values * weights
        mm7 = round; // running sum of WEIGHT_ROUND + pixel values * weights
        for (j = 0; j < cn; j++)
        {
            __m128i mw;
            mw  = _mm_set1_epi32(cbp[j]*256);
            mm0 = _mm_loadu_si128((const __m128i *)(tmp + j * byte_stride));
                                                // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
            mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
            mm2 = _mm_mullo_epi32(mm2,mw);    // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
            mm4 = _mm_add_epi32(mm4,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
            mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
            mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
            mm2 = _mm_mullo_epi32(mm2,mw);    // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
            mm5 = _mm_add_epi32(mm5,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
            mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
            mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
            mm2 = _mm_mullo_epi32(mm2,mw);    // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
            mm6 = _mm_add_epi32(mm6,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
            mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
            mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
            mm2 = _mm_mullo_epi32(mm2,mw);    // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
            mm7 = _mm_add_epi32(mm7,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        }
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4,mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6,mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        mm4 = _mm_srli_epi16(mm4,8);      // mm4 = 00HH00GG00FF00EE00DD00CC00BB00AA
        mm6 = _mm_srli_epi16(mm6,8);      // mm6 = 00PP00OO00NN00MM00LL00KK00JJ00II
        mm4 = _mm_packus_epi16(mm4,mm6);  // mm4 = PPOONNMMLLKKJJIIHHGGFFEEDDCCBBAA
        _mm_storeu_si128((__m128i *)d, mm4);
        d += 16;
        tmp += 16;
        width -= 16;
    }

    width += 16;
    if (width)
    {
        int j;
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        mm4 = round; // running sum of WEIGHT_ROUND + pixel values * weights
        mm5 = round; // running sum of WEIGHT_ROUND + pixel values * weights
        mm6 = round; // running sum of WEIGHT_ROUND + pixel values * weights
        mm7 = round; // running sum of WEIGHT_ROUND + pixel values * weights
        for (j = 0; j < cn; j++)
        {
            __m128i mw;
            mw  = _mm_set1_epi32(cbp[j]*256);
            mm0 = _mm_loadu_si128((const __m128i *)(tmp + j * byte_stride));
                                                // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
            mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
            mm2 = _mm_mullo_epi32(mm2,mw);    // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
            mm4 = _mm_add_epi32(mm4,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
            mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
            mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
            mm2 = _mm_mullo_epi32(mm2,mw);    // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
            mm5 = _mm_add_epi32(mm5,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
            mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
            mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
            mm2 = _mm_mullo_epi32(mm2,mw);    // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
            mm6 = _mm_add_epi32(mm6,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
            mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
            mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
            mm2 = _mm_mullo_epi32(mm2,mw);    // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
            mm7 = _mm_add_epi32(mm7,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        }
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4,mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6,mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        mm4 = _mm_srli_epi16(mm4,8);      // mm4 = 00HH00GG00FF00EE00DD00CC00BB00AA
        mm6 = _mm_srli_epi16(mm6,8);      // mm6 = 00PP00OO00NN00MM00LL00KK00JJ00II
        mm4 = _mm_packus_epi16(mm4,mm6);  // mm4 = PPOONNMMLLKKJJIIHHGGFFEEDDCCBBAA
        store_trailing_bytes(d, width, mm4);
    }
}

static void
zoom_y1to1_sse(void          * ipa_restrict dst,
               const uint8_t * ipa_restrict tmp,
               int                          width,
               int                          byte_stride,
               const index_t * ipa_restrict index,
               const int32_t * ipa_restrict weights)
{
    switch (index->n)
    {
        case 2:
            zoom_y1to1_2_sse(dst, tmp, width, byte_stride, index, weights);
            break;
        case 3:
            zoom_y1to1_3_sse(dst, tmp, width, byte_stride, index, weights);
            break;
        case 4:
            zoom_y1to1_4_sse(dst, tmp, width, byte_stride, index, weights);
            break;
        case 5:
            zoom_y1to1_5_sse(dst, tmp, width, byte_stride, index, weights);
            break;
        default:
            zoom_y1to1_n_sse(dst, tmp, width, byte_stride, index, weights);
            break;
    }
}

/*
Notes on the following SIMD code for zoom_y2:

The core point of using SIMD is to bulk process data. As such
the key decision is what central multiplication operation we are
going to use.

The vanilla C routine has unsigned 8 bit input data (pointed to
by tmp), which it multiplies by 'just larger than 8 bit' signed
weights. (-64 to 320). It multiplies these together, adds several
(up to 5) such sums together, then clamps to 16 bit.

As such, the intermediate sums are larger than 16 bit.

If we reduced the weights to be 'just larger than 5 bits' (i.e.
-8 to 40) then after multiplying by 8bit unsigned we'd be in the
range -2040 to 10200. Summing 5 of those we'd be in the range
-10200 to 51000 - still too big to fit in a 16bit value without
the sign bits getting confused.

If we reduce the weights to be 'just larger than 4 bits' (i.e.
-4 to 20) then after multiplying by 8bit unsigned we'd be in the
range -1020 to 5100. Summing 5 of those we'd be in the range
-5100 to 25500. That would fit in a signed 16 bit region, but
really that sounds too low accuracy for our needs.

If we swapped to using an filter function other than Mitchell
(one that stays in the 0..1 range rather than spilling out of it)
we could get away with 6 bits of accuracy.

But let's ignore that for now, and work with what we've got.
We'll therefore use 32bits for the intermediate sums. We load
the initial vectors into mm0, expand them up to 32bits 4 at
a time, multiply/sum them all, and then collapse it all back
down at the end.
*/

static inline void
zoom_y1to2_2_sse(void          * ipa_restrict dst,
                 const uint8_t * ipa_restrict tmp,
                 int                          width,
                 int                          byte_stride,
                 const index_t * ipa_restrict index,
                 const int32_t * ipa_restrict weights)
{
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint16_t *ipa_restrict d = (uint16_t *)dst;
    __m128i mw0 = _mm_set1_epi32(cbp[0]);
    __m128i mw1 = _mm_set1_epi32(cbp[1]);
    __m128i round = _mm_set1_epi32(WEIGHT_ROUND>>8);

    tmp += index->first_pixel;
    width -= 16;
    while (width >= 0)
    {
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        // Weight 0
        mm0 = _mm_loadu_si128((const __m128i *)tmp);
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm4 = _mm_mullo_epi32(mm2,mw0);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm5 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm6 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm7 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        // Weight 1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride));
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Add rounding
        mm4 = _mm_add_epi32(mm4, round);
        mm5 = _mm_add_epi32(mm5, round);
        mm6 = _mm_add_epi32(mm6, round);
        mm7 = _mm_add_epi32(mm7, round);
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT-8);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT-8);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT-8);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT-8);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4,mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6,mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        _mm_storeu_si128((__m128i *)d, mm4);
        _mm_storeu_si128((__m128i *)(d + 8), mm6);
        d += 16;
        tmp += 16;
        width -= 16;
    }

    width += 16;
    if (width)
    {
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        // Weight 0
        mm0 = _mm_loadu_si128((const __m128i *)tmp);
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm4 = _mm_mullo_epi32(mm2,mw0);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm5 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm6 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm7 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        // Weight 1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride));
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Add rounding
        mm4 = _mm_add_epi32(mm4, round);
        mm5 = _mm_add_epi32(mm5, round);
        mm6 = _mm_add_epi32(mm6, round);
        mm7 = _mm_add_epi32(mm7, round);
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT-8);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT-8);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT-8);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT-8);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4,mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6,mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        store_trailing_shorts(d, width, mm4, mm6);
    }
}

static inline void
zoom_y1to2_3_sse(void          * ipa_restrict dst,
                 const uint8_t * ipa_restrict tmp,
                 int                          width,
                 int                          byte_stride,
                 const index_t * ipa_restrict index,
                 const int32_t * ipa_restrict weights)
{
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint16_t *ipa_restrict d = (uint16_t *)dst;
    __m128i mw0 = _mm_set1_epi32(cbp[0]);
    __m128i mw1 = _mm_set1_epi32(cbp[1]);
    __m128i mw2 = _mm_set1_epi32(cbp[2]);
    __m128i round = _mm_set1_epi32(WEIGHT_ROUND>>8);

    tmp += index->first_pixel;
    width -= 16;
    while (width >= 0)
    {
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        // Weight 0
        mm0 = _mm_loadu_si128((const __m128i *)tmp);
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm4 = _mm_mullo_epi32(mm2,mw0);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm5 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm6 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm7 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        // Weight 1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride));
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Weight 2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + 2 * byte_stride));
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Add rounding
        mm4 = _mm_add_epi32(mm4, round);
        mm5 = _mm_add_epi32(mm5, round);
        mm6 = _mm_add_epi32(mm6, round);
        mm7 = _mm_add_epi32(mm7, round);
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT-8);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT-8);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT-8);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT-8);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4,mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6,mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        _mm_storeu_si128((__m128i *)d, mm4);
        _mm_storeu_si128((__m128i *)(d + 8), mm6);
        d += 16;
        tmp += 16;
        width -= 16;
    }

    width += 16;
    if (width)
    {
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        // Weight 0
        mm0 = _mm_loadu_si128((const __m128i *)tmp);
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm4 = _mm_mullo_epi32(mm2,mw0);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm5 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm6 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm7 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        // Weight 1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride));
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Weight 2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + 2 * byte_stride));
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Add rounding
        mm4 = _mm_add_epi32(mm4, round);
        mm5 = _mm_add_epi32(mm5, round);
        mm6 = _mm_add_epi32(mm6, round);
        mm7 = _mm_add_epi32(mm7, round);
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT-8);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT-8);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT-8);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT-8);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4,mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6,mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        store_trailing_shorts(d, width, mm4, mm6);
    }
}

static inline void
zoom_y1to2_4_sse(void          * ipa_restrict dst,
                 const uint8_t * ipa_restrict tmp,
                 int             width,
                 int             byte_stride,
                 const index_t * ipa_restrict index,
                 const int32_t * ipa_restrict weights)
{
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint16_t *ipa_restrict d = (uint16_t *)dst;
    __m128i mw0   = _mm_set1_epi32(cbp[0]);
    __m128i mw1   = _mm_set1_epi32(cbp[1]);
    __m128i mw2   = _mm_set1_epi32(cbp[2]);
    __m128i mw3   = _mm_set1_epi32(cbp[3]);
    __m128i round = _mm_set1_epi32(WEIGHT_ROUND>>8);

    tmp += index->first_pixel;
    width -= 16;
    while (width >= 0)
    {
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        // Weight 0
        mm0 = _mm_loadu_si128((const __m128i *)tmp);
                                            // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm4 = _mm_mullo_epi32(mm2,mw0);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm5 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm6 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm7 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        // Weight 1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride));
                                            // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Weight 2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + 2*byte_stride));
                                            // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Weight 3
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + 3*byte_stride));
                                            // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Add rounding
        mm4 = _mm_add_epi32(mm4, round);
        mm5 = _mm_add_epi32(mm5, round);
        mm6 = _mm_add_epi32(mm6, round);
        mm7 = _mm_add_epi32(mm7, round);
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT-8);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT-8);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT-8);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT-8);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4,mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6,mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        _mm_storeu_si128((__m128i *)d, mm4);
        _mm_storeu_si128((__m128i *)(d + 8), mm6);
        d += 16;
        tmp += 16;
        width -= 16;
    }

    width += 16;
    if (width)
    {
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        // Weight 0
        mm0 = _mm_loadu_si128((const __m128i *)tmp);
                                            // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm4 = _mm_mullo_epi32(mm2,mw0);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm5 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm6 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm7 = _mm_mullo_epi32(mm2,mw0);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        // Weight 1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride));
                                            // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Weight 2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + 2*byte_stride));
                                            // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Weight 3
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + 3*byte_stride));
                                            // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);   // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);   // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);   // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);   // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Add rounding
        mm4 = _mm_add_epi32(mm4, round);
        mm5 = _mm_add_epi32(mm5, round);
        mm6 = _mm_add_epi32(mm6, round);
        mm7 = _mm_add_epi32(mm7, round);
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT-8);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT-8);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT-8);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT-8);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4,mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6,mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        store_trailing_shorts(d, width, mm4, mm6);
    }
}

static inline void
zoom_y1to2_5_sse(void          * ipa_restrict dst,
                 const uint8_t * ipa_restrict tmp,
                 int                          width,
                 int                          byte_stride,
                 const index_t * ipa_restrict index,
                 const int32_t * ipa_restrict weights)
{
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint16_t *ipa_restrict d = (uint16_t *)dst;
    __m128i mw0 = _mm_set1_epi32(cbp[0]);
    __m128i mw1 = _mm_set1_epi32(cbp[1]);
    __m128i mw2 = _mm_set1_epi32(cbp[2]);
    __m128i mw3 = _mm_set1_epi32(cbp[3]);
    __m128i mw4 = _mm_set1_epi32(cbp[4]);
    __m128i round = _mm_set1_epi32(WEIGHT_ROUND>>8);

    tmp += index->first_pixel;
    width -= 16;
    while (width >= 0)
    {
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        // Weight 0
        mm0 = _mm_loadu_si128((const __m128i *)tmp);
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm4 = _mm_mullo_epi32(mm2, mw0);  // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm5 = _mm_mullo_epi32(mm2, mw0);  // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm6 = _mm_mullo_epi32(mm2, mw0);  // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm7 = _mm_mullo_epi32(mm2, mw0);  // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        // Weight 1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride));
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw1);  // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw1);  // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw1);  // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw1);  // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Weight 2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + 2 * byte_stride));
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw2);  // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw2);  // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw2);  // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw2);  // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Weight 3
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + 3 * byte_stride));
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw3);  // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw3);  // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw3);  // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw3);  // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Weight 4
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + 4 * byte_stride));
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw4);  // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw4);  // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw4);  // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw4);  // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Add rounding
        mm4 = _mm_add_epi32(mm4, round);
        mm5 = _mm_add_epi32(mm5, round);
        mm6 = _mm_add_epi32(mm6, round);
        mm7 = _mm_add_epi32(mm7, round);
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT-8);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT-8);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT-8);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT-8);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4, mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6, mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        _mm_storeu_si128((__m128i *)d, mm4);
        _mm_storeu_si128((__m128i *)(d + 8), mm6);
        d += 16;
        tmp += 16;
        width -= 16;
    }

    width += 16;
    if (width)
    {
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        // Weight 0
        mm0 = _mm_loadu_si128((const __m128i *)tmp);
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm4 = _mm_mullo_epi32(mm2, mw0);  // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm5 = _mm_mullo_epi32(mm2, mw0);  // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm6 = _mm_mullo_epi32(mm2, mw0);  // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm7 = _mm_mullo_epi32(mm2, mw0);  // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        // Weight 1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride));
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw1);  // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw1);  // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw1);  // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw1);  // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Weight 2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + 2 * byte_stride));
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw2);  // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw2);  // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw2);  // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw2);  // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Weight 3
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + 3 * byte_stride));
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw3);  // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw3);  // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw3);  // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw3);  // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Weight 4
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + 4 * byte_stride));
        // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw4);  // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw4);  // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw4);  // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        mm0 = _mm_srli_si128(mm0, 4);     // mm0 = 000000000000000000000000ppoonnmm SSE2
        mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
        mm2 = _mm_mullo_epi32(mm2, mw4);  // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);    // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        // Add rounding
        mm4 = _mm_add_epi32(mm4, round);
        mm5 = _mm_add_epi32(mm5, round);
        mm6 = _mm_add_epi32(mm6, round);
        mm7 = _mm_add_epi32(mm7, round);
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT-8);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT-8);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT-8);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT-8);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4, mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6, mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        store_trailing_shorts(d, width, mm4, mm6);
    }
}

static inline void
zoom_y1to2_n_sse(void          * ipa_restrict dst,
                 const uint8_t * ipa_restrict tmp,
                 int                          width,
                 int                          byte_stride,
                 const index_t * ipa_restrict index,
                 const int32_t * ipa_restrict weights)
{
    int cn = index->n;
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint16_t *ipa_restrict d = (uint16_t *)dst;
    __m128i round = _mm_set1_epi32(WEIGHT_ROUND>>8);

    tmp += index->first_pixel;
    width -= 16;
    while (width >= 0)
    {
        int j;
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        mm4 = round;
        mm5 = round;
        mm6 = round;
        mm7 = round;
        for (j = 0; j < cn; j++)
        {
            __m128i mw;
            mw  = _mm_set1_epi32(cbp[j]);
            mm0 = _mm_loadu_si128((const __m128i *)(tmp + j * byte_stride));
                                                // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
            mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
            mm2 = _mm_mullo_epi32(mm2,mw);    // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
            mm4 = _mm_add_epi32(mm4,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
            mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
            mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
            mm2 = _mm_mullo_epi32(mm2,mw);    // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
            mm5 = _mm_add_epi32(mm5,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
            mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
            mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
            mm2 = _mm_mullo_epi32(mm2,mw);    // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
            mm6 = _mm_add_epi32(mm6,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
            mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
            mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
            mm2 = _mm_mullo_epi32(mm2,mw);    // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
            mm7 = _mm_add_epi32(mm7,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        }
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT-8);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT-8);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT-8);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT-8);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4,mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6,mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        _mm_storeu_si128((__m128i *)d, mm4);
        _mm_storeu_si128((__m128i *)(d + 8), mm6);
        d += 16;
        tmp += 16;
        width -= 16;
    }

    width += 16;
    if (width)
    {
        int j;
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        mm4 = round;
        mm5 = round;
        mm6 = round;
        mm7 = round;
        for (j = 0; j < cn; j++)
        {
            __m128i mw;
            mw  = _mm_set1_epi32(cbp[j]);
            mm0 = _mm_loadu_si128((const __m128i *)(tmp + j * byte_stride));
                                                // mm0 = ppoonnmmllkkjjiihhggffeeddccbbaa SSE2
            mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000dd000000cc000000bb000000aa SSE4.1
            mm2 = _mm_mullo_epi32(mm2,mw);    // mm4 = 0000wdwd0000wcwc0000wbwb0000wawa SSE4.1
            mm4 = _mm_add_epi32(mm4,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
            mm0 = _mm_srli_si128(mm0,4);      // mm0 = 00000000ppoonnmmllkkjjiihhggffee SSE2
            mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000hh000000gg000000ff000000ee SSE4.1
            mm2 = _mm_mullo_epi32(mm2,mw);    // mm5 = 0000whwh0000wgwg0000wfwf0000wewe SSE4.1
            mm5 = _mm_add_epi32(mm5,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
            mm0 = _mm_srli_si128(mm0,4);      // mm0 = 0000000000000000ppoonnmmllkkjjii SSE2
            mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
            mm2 = _mm_mullo_epi32(mm2,mw);    // mm5 = 0000wlwl0000wkwk0000wjwj0000wiwi SSE4.1
            mm6 = _mm_add_epi32(mm6,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
            mm0 = _mm_srli_si128(mm0,4);      // mm0 = 000000000000000000000000ppoonnmm SSE2
            mm2 = _mm_cvtepu8_epi32(mm0);     // mm2 = 000000pp000000oo000000nn000000mm SSE4.1
            mm2 = _mm_mullo_epi32(mm2,mw);    // mm5 = 0000wpwp0000wowo0000wnwn0000wmwm SSE4.1
            mm7 = _mm_add_epi32(mm7,mm2);     // mm4 = 0000xxxx0000xxxx0000xxxx0000xxxx SSE2
        }
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT-8);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT-8);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT-8);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT-8);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4,mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6,mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        store_trailing_shorts(d, width, mm4, mm6);
    }
}

static void
zoom_y1to2_sse(void          * ipa_restrict dst,
               const uint8_t * ipa_restrict tmp,
               int                          width,
               int                          byte_stride,
               const index_t * ipa_restrict index,
               const int32_t * ipa_restrict weights)
{
    switch (index->n)
    {
        case 2:
            zoom_y1to2_2_sse(dst, tmp, width, byte_stride, index, weights);
            break;
        case 3:
            zoom_y1to2_3_sse(dst, tmp, width, byte_stride, index, weights);
            break;
        case 4:
            zoom_y1to2_4_sse(dst, tmp, width, byte_stride, index, weights);
            break;
        case 5:
            zoom_y1to2_5_sse(dst, tmp, width, byte_stride, index, weights);
            break;
        default:
            zoom_y1to2_n_sse(dst, tmp, width, byte_stride, index, weights);
            break;
    }
}

static inline void
zoom_y2to1_2_sse(void          * ipa_restrict dst,
                 const uint8_t * ipa_restrict tmp_,
                 int                          width,
                 int                          byte_stride,
                 const index_t * ipa_restrict index,
                 const int32_t * ipa_restrict weights)
{
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint8_t *ipa_restrict d = (uint8_t *)dst;
    const uint16_t * ipa_restrict tmp = (const uint16_t *)tmp_;
    __m128i mw0 = _mm_set1_epi32(cbp[0]);
    __m128i mw1 = _mm_set1_epi32(cbp[1]);
    __m128i round = _mm_set1_epi32(WEIGHT_ROUND);

    tmp += index->first_pixel;
    width -= 16;
    while (width >= 0)
    {
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        // Weight 0
        mm0 = _mm_loadu_si128((const __m128i *)tmp);     // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm4 = _mm_mullo_epi32(mm2,mw0);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm5 = _mm_mullo_epi32(mm2,mw0);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp+8)); // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm6 = _mm_mullo_epi32(mm2,mw0);                  // mm6 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm7 = _mm_mullo_epi32(mm2,mw0);                  // mm7 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        // Weight 1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);                   // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);                   // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);                   // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);                   // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Add rounding
        mm4 = _mm_add_epi32(mm4, round);
        mm5 = _mm_add_epi32(mm5, round);
        mm6 = _mm_add_epi32(mm6, round);
        mm7 = _mm_add_epi32(mm7, round);
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4,mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6,mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        mm4 = _mm_srli_epi16(mm4,8);      // mm4 = 00HH00GG00FF00EE00DD00CC00BB00AA
        mm6 = _mm_srli_epi16(mm6,8);      // mm6 = 00PP00OO00NN00MM00LL00KK00JJ00II
        mm4 = _mm_packus_epi16(mm4,mm6);  // mm4 = PPOONNMMLLKKJJIIHHGGFFEEDDCCBBAA
        _mm_storeu_si128((__m128i *)d, mm4);
        d += 16;
        tmp += 16;
        width -= 16;
    }

    width += 16;
    if (width)
    {
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        // Weight 0
        mm0 = _mm_loadu_si128((const __m128i *)tmp);     // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm4 = _mm_mullo_epi32(mm2,mw0);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm5 = _mm_mullo_epi32(mm2,mw0);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp+8)); // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm6 = _mm_mullo_epi32(mm2,mw0);                  // mm6 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm7 = _mm_mullo_epi32(mm2,mw0);                  // mm7 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        // Weight 1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);                   // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);                   // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);                   // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);                   // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Add rounding
        mm4 = _mm_add_epi32(mm4, round);
        mm5 = _mm_add_epi32(mm5, round);
        mm6 = _mm_add_epi32(mm6, round);
        mm7 = _mm_add_epi32(mm7, round);
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4,mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6,mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        mm4 = _mm_srli_epi16(mm4,8);      // mm4 = 00HH00GG00FF00EE00DD00CC00BB00AA
        mm6 = _mm_srli_epi16(mm6,8);      // mm6 = 00PP00OO00NN00MM00LL00KK00JJ00II
        mm4 = _mm_packus_epi16(mm4,mm6);  // mm4 = PPOONNMMLLKKJJIIHHGGFFEEDDCCBBAA
        store_trailing_bytes(d, width, mm4);
    }
}

static inline void
zoom_y2to1_3_sse(void          * ipa_restrict dst,
                 const uint8_t * ipa_restrict tmp_,
                 int                          width,
                 int                          byte_stride,
                 const index_t * ipa_restrict index,
                 const int32_t * ipa_restrict weights)
{
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint8_t *ipa_restrict d = (uint8_t *)dst;
    const uint16_t * ipa_restrict tmp = (const uint16_t *)tmp_;
    __m128i mw0 = _mm_set1_epi32(cbp[0]);
    __m128i mw1 = _mm_set1_epi32(cbp[1]);
    __m128i mw2 = _mm_set1_epi32(cbp[2]);
    __m128i round = _mm_set1_epi32(WEIGHT_ROUND);

    tmp += index->first_pixel;
    width -= 16;
    while (width >= 0)
    {
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        // Weight 0
        mm0 = _mm_loadu_si128((const __m128i *)tmp);     // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm4 = _mm_mullo_epi32(mm2,mw0);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm5 = _mm_mullo_epi32(mm2,mw0);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp+8)); // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm6 = _mm_mullo_epi32(mm2,mw0);                  // mm6 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm7 = _mm_mullo_epi32(mm2,mw0);                  // mm7 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        // Weight 1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);                   // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);                   // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);                   // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);                   // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Weight 2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*2));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);                   // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);                   // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*2 + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);                   // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);                   // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Add rounding
        mm4 = _mm_add_epi32(mm4, round);
        mm5 = _mm_add_epi32(mm5, round);
        mm6 = _mm_add_epi32(mm6, round);
        mm7 = _mm_add_epi32(mm7, round);
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4,mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6,mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        mm4 = _mm_srli_epi16(mm4,8);      // mm4 = 00HH00GG00FF00EE00DD00CC00BB00AA
        mm6 = _mm_srli_epi16(mm6,8);      // mm6 = 00PP00OO00NN00MM00LL00KK00JJ00II
        mm4 = _mm_packus_epi16(mm4,mm6);  // mm4 = PPOONNMMLLKKJJIIHHGGFFEEDDCCBBAA
        _mm_storeu_si128((__m128i *)d, mm4);
        d += 16;
        tmp += 16;
        width -= 16;
    }

    width += 16;
    if (width)
    {
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        // Weight 0
        mm0 = _mm_loadu_si128((const __m128i *)tmp);     // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm4 = _mm_mullo_epi32(mm2,mw0);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm5 = _mm_mullo_epi32(mm2,mw0);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp+8)); // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm6 = _mm_mullo_epi32(mm2,mw0);                  // mm6 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm7 = _mm_mullo_epi32(mm2,mw0);                  // mm7 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        // Weight 1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);                   // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);                   // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);                   // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);                   // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Weight 2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*2));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);                   // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);                   // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*2 + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);                   // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);                   // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Add rounding
        mm4 = _mm_add_epi32(mm4, round);
        mm5 = _mm_add_epi32(mm5, round);
        mm6 = _mm_add_epi32(mm6, round);
        mm7 = _mm_add_epi32(mm7, round);
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4,mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6,mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        mm4 = _mm_srli_epi16(mm4,8);      // mm4 = 00HH00GG00FF00EE00DD00CC00BB00AA
        mm6 = _mm_srli_epi16(mm6,8);      // mm6 = 00PP00OO00NN00MM00LL00KK00JJ00II
        mm4 = _mm_packus_epi16(mm4,mm6);  // mm4 = PPOONNMMLLKKJJIIHHGGFFEEDDCCBBAA
        store_trailing_bytes(d, width, mm4);
    }
}

static inline void
zoom_y2to1_4_sse(void          * ipa_restrict dst,
                 const uint8_t * ipa_restrict tmp_,
                 int             width,
                 int             byte_stride,
                 const index_t * ipa_restrict index,
                 const int32_t * ipa_restrict weights)
{
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint8_t *ipa_restrict d = (uint8_t *)dst;
    const uint16_t * ipa_restrict tmp = (const uint16_t *)tmp_;
    __m128i mw0   = _mm_set1_epi32(cbp[0]);
    __m128i mw1   = _mm_set1_epi32(cbp[1]);
    __m128i mw2   = _mm_set1_epi32(cbp[2]);
    __m128i mw3   = _mm_set1_epi32(cbp[3]);
    __m128i round = _mm_set1_epi32(WEIGHT_ROUND);

    tmp += index->first_pixel;
    width -= 16;
    while (width >= 0)
    {
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        // Weight 0
        mm0 = _mm_loadu_si128((const __m128i *)tmp);     // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm4 = _mm_mullo_epi32(mm2,mw0);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm5 = _mm_mullo_epi32(mm2,mw0);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp+8)); // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm6 = _mm_mullo_epi32(mm2,mw0);                  // mm6 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm7 = _mm_mullo_epi32(mm2,mw0);                  // mm7 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        // Weight 1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);                   // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);                   // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);                   // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);                   // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Weight 2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*2));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);                   // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);                   // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*2 + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);                   // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);                   // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Weight 3
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*3));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);                   // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);                   // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*3 + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);                  // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);                   // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);                  // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);                   // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Add rounding
        mm4 = _mm_add_epi32(mm4, round);
        mm5 = _mm_add_epi32(mm5, round);
        mm6 = _mm_add_epi32(mm6, round);
        mm7 = _mm_add_epi32(mm7, round);
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4,mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6,mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        mm4 = _mm_srli_epi16(mm4,8);      // mm4 = 00HH00GG00FF00EE00DD00CC00BB00AA
        mm6 = _mm_srli_epi16(mm6,8);      // mm6 = 00PP00OO00NN00MM00LL00KK00JJ00II
        mm4 = _mm_packus_epi16(mm4,mm6);  // mm4 = PPOONNMMLLKKJJIIHHGGFFEEDDCCBBAA
        _mm_storeu_si128((__m128i *)d, mm4);
        d += 16;
        tmp += 16;
        width -= 16;
    }

    width += 16;
    if (width)
    {
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        // Weight 0
        mm0 = _mm_loadu_si128((const __m128i *)tmp);     // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm4 = _mm_mullo_epi32(mm2,mw0);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm5 = _mm_mullo_epi32(mm2,mw0);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp+8)); // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm6 = _mm_mullo_epi32(mm2,mw0);                  // mm6 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm7 = _mm_mullo_epi32(mm2,mw0);                  // mm7 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        // Weight 1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);                   // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);                   // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);                   // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);                   // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Weight 2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*2));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);                   // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);                   // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*2 + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);                   // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);                   // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Weight 3
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*3));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);                   // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);                   // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*3 + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);                  // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);                   // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);                  // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);                   // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Add rounding
        mm4 = _mm_add_epi32(mm4, round);
        mm5 = _mm_add_epi32(mm5, round);
        mm6 = _mm_add_epi32(mm6, round);
        mm7 = _mm_add_epi32(mm7, round);
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4,mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6,mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        mm4 = _mm_srli_epi16(mm4,8);      // mm4 = 00HH00GG00FF00EE00DD00CC00BB00AA
        mm6 = _mm_srli_epi16(mm6,8);      // mm6 = 00PP00OO00NN00MM00LL00KK00JJ00II
        mm4 = _mm_packus_epi16(mm4,mm6);  // mm4 = PPOONNMMLLKKJJIIHHGGFFEEDDCCBBAA
        store_trailing_bytes(d, width, mm4);
    }
}

static inline void
zoom_y2to1_5_sse(void          * ipa_restrict dst,
                 const uint8_t * ipa_restrict tmp_,
                 int                          width,
                 int                          byte_stride,
                 const index_t * ipa_restrict index,
                 const int32_t * ipa_restrict weights)
{
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint8_t *ipa_restrict d = (uint8_t *)dst;
    const uint16_t * ipa_restrict tmp = (const uint16_t *)tmp_;
    __m128i mw0 = _mm_set1_epi32(cbp[0]);
    __m128i mw1 = _mm_set1_epi32(cbp[1]);
    __m128i mw2 = _mm_set1_epi32(cbp[2]);
    __m128i mw3 = _mm_set1_epi32(cbp[3]);
    __m128i mw4 = _mm_set1_epi32(cbp[4]);
    __m128i round = _mm_set1_epi32(WEIGHT_ROUND);

    tmp += index->first_pixel;
    width -= 16;
    while (width >= 0)
    {
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        // Weight 0
        mm0 = _mm_loadu_si128((const __m128i *)tmp);     // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm4 = _mm_mullo_epi32(mm2,mw0);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm5 = _mm_mullo_epi32(mm2,mw0);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp+8)); // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm6 = _mm_mullo_epi32(mm2,mw0);                  // mm6 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm7 = _mm_mullo_epi32(mm2,mw0);                  // mm7 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        // Weight 1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);                   // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);                   // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);                   // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);                   // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Weight 2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*2));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);                   // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);                   // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*2 + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);                   // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);                   // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Weight 3
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*3));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);                   // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);                   // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*3 + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);                  // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);                   // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);                  // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);                   // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Weight 4
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*4));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw4);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);                   // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw4);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);                   // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*4 + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw4);                  // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);                   // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw4);                  // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);                   // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Add rounding
        mm4 = _mm_add_epi32(mm4, round);
        mm5 = _mm_add_epi32(mm5, round);
        mm6 = _mm_add_epi32(mm6, round);
        mm7 = _mm_add_epi32(mm7, round);
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4, mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6, mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        mm4 = _mm_srli_epi16(mm4,8);      // mm4 = 00HH00GG00FF00EE00DD00CC00BB00AA
        mm6 = _mm_srli_epi16(mm6,8);      // mm6 = 00PP00OO00NN00MM00LL00KK00JJ00II
        mm4 = _mm_packus_epi16(mm4,mm6);  // mm4 = PPOONNMMLLKKJJIIHHGGFFEEDDCCBBAA
        _mm_storeu_si128((__m128i *)d, mm4);
        d += 16;
        tmp += 16;
        width -= 16;
    }

    width += 16;
    if (width)
    {
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        // Weight 0
        mm0 = _mm_loadu_si128((const __m128i *)tmp);     // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm4 = _mm_mullo_epi32(mm2,mw0);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm5 = _mm_mullo_epi32(mm2,mw0);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp+8)); // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm6 = _mm_mullo_epi32(mm2,mw0);                  // mm6 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm7 = _mm_mullo_epi32(mm2,mw0);                  // mm7 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        // Weight 1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);                   // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);                   // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);                   // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);                   // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Weight 2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*2));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);                   // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);                   // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*2 + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);                   // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);                   // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Weight 3
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*3));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);                   // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);                   // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*3 + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);                  // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);                   // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);                  // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);                   // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Weight 4
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*4));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw4);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4, mm2);                   // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw4);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5, mm2);                   // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*4 + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw4);                  // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6, mm2);                   // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw4);                  // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7, mm2);                   // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Add rounding
        mm4 = _mm_add_epi32(mm4, round);
        mm5 = _mm_add_epi32(mm5, round);
        mm6 = _mm_add_epi32(mm6, round);
        mm7 = _mm_add_epi32(mm7, round);
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4, mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6, mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        mm4 = _mm_srli_epi16(mm4,8);      // mm4 = 00HH00GG00FF00EE00DD00CC00BB00AA
        mm6 = _mm_srli_epi16(mm6,8);      // mm6 = 00PP00OO00NN00MM00LL00KK00JJ00II
        mm4 = _mm_packus_epi16(mm4,mm6);  // mm4 = PPOONNMMLLKKJJIIHHGGFFEEDDCCBBAA
        store_trailing_bytes(d, width, mm4);
    }
}

static inline void
zoom_y2to1_n_sse(void          * ipa_restrict dst,
                 const uint8_t * ipa_restrict tmp_,
                 int                          width,
                 int                          byte_stride,
                 const index_t * ipa_restrict index,
                 const int32_t * ipa_restrict weights)
{
    int cn = index->n;
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint8_t *ipa_restrict d = (uint8_t *)dst;
    const uint16_t * ipa_restrict tmp = (const uint16_t *)tmp_;
    __m128i round = _mm_set1_epi32(WEIGHT_ROUND);

    tmp += index->first_pixel;
    width -= 16;
    while (width >= 0)
    {
        int j;
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        mm4 = round;
        mm5 = round;
        mm6 = round;
        mm7 = round;
        for (j = 0; j < cn; j++)
        {
            __m128i mw;
            mw  = _mm_set1_epi32(cbp[j]);
            mm0 = _mm_loadu_si128((const __m128i *)(tmp + j * byte_stride));
                                                // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
            mm2 = _mm_cvtepu16_epi32(mm0);    // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
            mm2 = _mm_mullo_epi32(mm2,mw);    // mm4 = 00wdwdwd0000wcwc00wbwbwb00wawawa SSE4.1
            mm4 = _mm_add_epi32(mm4,mm2);     // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
            mm0 = _mm_srli_si128(mm0,8);      // mm0 = 0000000000000000hhhhggggffffeeee SSE2
            mm2 = _mm_cvtepu16_epi32(mm0);    // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
            mm2 = _mm_mullo_epi32(mm2,mw);    // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
            mm5 = _mm_add_epi32(mm5,mm2);     // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
            mm0 = _mm_loadu_si128((const __m128i *)(tmp + j * byte_stride + 8));
                                                // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
            mm2 = _mm_cvtepu16_epi32(mm0);    // mm2 = 0000llll0000kkkk0000jjjj0000iiii SSE4.1
            mm2 = _mm_mullo_epi32(mm2,mw);    // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
            mm6 = _mm_add_epi32(mm6,mm2);     // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
            mm0 = _mm_srli_si128(mm0,8);      // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
            mm2 = _mm_cvtepu16_epi32(mm0);    // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
            mm2 = _mm_mullo_epi32(mm2,mw);    // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
            mm7 = _mm_add_epi32(mm7,mm2);     // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        }
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4,mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6,mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        mm4 = _mm_srli_epi16(mm4,8);      // mm4 = 00HH00GG00FF00EE00DD00CC00BB00AA
        mm6 = _mm_srli_epi16(mm6,8);      // mm6 = 00PP00OO00NN00MM00LL00KK00JJ00II
        mm4 = _mm_packus_epi16(mm4,mm6);  // mm4 = PPOONNMMLLKKJJIIHHGGFFEEDDCCBBAA
        _mm_storeu_si128((__m128i *)d, mm4);
        d += 16;
        tmp += 16;
        width -= 16;
    }

    width += 16;
    if (width)
    {
        int j;
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        mm4 = round;
        mm5 = round;
        mm6 = round;
        mm7 = round;
        for (j = 0; j < cn; j++)
        {
            __m128i mw;
            mw  = _mm_set1_epi32(cbp[j]);
            mm0 = _mm_loadu_si128((const __m128i *)(tmp + j * byte_stride));
                                                // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
            mm2 = _mm_cvtepu16_epi32(mm0);    // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
            mm2 = _mm_mullo_epi32(mm2,mw);    // mm4 = 00wdwdwd0000wcwc00wbwbwb00wawawa SSE4.1
            mm4 = _mm_add_epi32(mm4,mm2);     // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
            mm0 = _mm_srli_si128(mm0,8);      // mm0 = 0000000000000000hhhhggggffffeeee SSE2
            mm2 = _mm_cvtepu16_epi32(mm0);    // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
            mm2 = _mm_mullo_epi32(mm2,mw);    // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
            mm5 = _mm_add_epi32(mm5,mm2);     // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
            mm0 = _mm_loadu_si128((const __m128i *)(tmp + j * byte_stride + 8));
                                                // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
            mm2 = _mm_cvtepu16_epi32(mm0);    // mm2 = 0000llll0000kkkk0000jjjj0000iiii SSE4.1
            mm2 = _mm_mullo_epi32(mm2,mw);    // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
            mm6 = _mm_add_epi32(mm6,mm2);     // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
            mm0 = _mm_srli_si128(mm0,8);      // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
            mm2 = _mm_cvtepu16_epi32(mm0);    // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
            mm2 = _mm_mullo_epi32(mm2,mw);    // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
            mm7 = _mm_add_epi32(mm7,mm2);     // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        }
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4,mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6,mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        mm4 = _mm_srli_epi16(mm4,8);      // mm4 = 00HH00GG00FF00EE00DD00CC00BB00AA
        mm6 = _mm_srli_epi16(mm6,8);      // mm6 = 00PP00OO00NN00MM00LL00KK00JJ00II
        mm4 = _mm_packus_epi16(mm4,mm6);  // mm4 = PPOONNMMLLKKJJIIHHGGFFEEDDCCBBAA
        store_trailing_bytes(d, width, mm4);
    }
}

static void
zoom_y2to1_sse(void          * ipa_restrict dst,
               const uint8_t * ipa_restrict tmp,
               int                          width,
               int                          byte_stride,
               const index_t * ipa_restrict index,
               const int32_t * ipa_restrict weights)
{
    byte_stride >>= 1;
    switch (index->n)
    {
        case 2:
            zoom_y2to1_2_sse(dst, tmp, width, byte_stride, index, weights);
            break;
        case 3:
            zoom_y2to1_3_sse(dst, tmp, width, byte_stride, index, weights);
            break;
        case 4:
            zoom_y2to1_4_sse(dst, tmp, width, byte_stride, index, weights);
            break;
        case 5:
            zoom_y2to1_5_sse(dst, tmp, width, byte_stride, index, weights);
            break;
        default:
            zoom_y2to1_n_sse(dst, tmp, width, byte_stride, index, weights);
            break;
    }
}

static inline void
zoom_y2to2_2_sse(void          * ipa_restrict dst,
                 const uint8_t * ipa_restrict tmp_,
                 int                          width,
                 int                          byte_stride,
                 const index_t * ipa_restrict index,
                 const int32_t * ipa_restrict weights)
{
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint16_t *ipa_restrict d = (uint16_t *)dst;
    const uint16_t * ipa_restrict tmp = (const uint16_t *)tmp_;
    __m128i mw0 = _mm_set1_epi32(cbp[0]);
    __m128i mw1 = _mm_set1_epi32(cbp[1]);
    __m128i round = _mm_set1_epi32(WEIGHT_ROUND);

    tmp += index->first_pixel;
    width -= 16;
    while (width >= 0)
    {
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        // Weight 0
        mm0 = _mm_loadu_si128((const __m128i *)tmp);     // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm4 = _mm_mullo_epi32(mm2,mw0);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm5 = _mm_mullo_epi32(mm2,mw0);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp+8)); // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm6 = _mm_mullo_epi32(mm2,mw0);                  // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm7 = _mm_mullo_epi32(mm2,mw0);                  // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        // Weight 1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm2 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);                    // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);                    // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm2 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);                    // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm2 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);                    // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Add rounding
        mm4 = _mm_add_epi32(mm4, round);
        mm5 = _mm_add_epi32(mm5, round);
        mm6 = _mm_add_epi32(mm6, round);
        mm7 = _mm_add_epi32(mm7, round);
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4,mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6,mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        _mm_storeu_si128((__m128i *)d, mm4);
        _mm_storeu_si128((__m128i *)(d + 8), mm6);
        d += 16;
        tmp += 16;
        width -= 16;
    }

    width += 16;
    if (width)
    {
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        // Weight 0
        mm0 = _mm_loadu_si128((const __m128i *)tmp);     // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm4 = _mm_mullo_epi32(mm2,mw0);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm5 = _mm_mullo_epi32(mm2,mw0);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp+8)); // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm6 = _mm_mullo_epi32(mm2,mw0);                  // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm7 = _mm_mullo_epi32(mm2,mw0);                  // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        // Weight 1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm2 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);                    // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);                    // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm2 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);                    // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm2 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);                    // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Add rounding
        mm4 = _mm_add_epi32(mm4, round);
        mm5 = _mm_add_epi32(mm5, round);
        mm6 = _mm_add_epi32(mm6, round);
        mm7 = _mm_add_epi32(mm7, round);
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4,mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6,mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        store_trailing_shorts(d, width, mm4, mm6);
    }
}

static inline void
zoom_y2to2_3_sse(void          * ipa_restrict dst,
                 const uint8_t * ipa_restrict tmp_,
                 int                          width,
                 int                          byte_stride,
                 const index_t * ipa_restrict index,
                 const int32_t * ipa_restrict weights)
{
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint16_t *ipa_restrict d = (uint16_t *)dst;
    const uint16_t * ipa_restrict tmp = (const uint16_t *)tmp_;
    __m128i mw0 = _mm_set1_epi32(cbp[0]);
    __m128i mw1 = _mm_set1_epi32(cbp[1]);
    __m128i mw2 = _mm_set1_epi32(cbp[2]);
    __m128i round = _mm_set1_epi32(WEIGHT_ROUND);

    tmp += index->first_pixel;
    width -= 16;
    while (width >= 0)
    {
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        // Weight 0
        mm0 = _mm_loadu_si128((const __m128i *)tmp);     // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm4 = _mm_mullo_epi32(mm2,mw0);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm5 = _mm_mullo_epi32(mm2,mw0);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp+8)); // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm6 = _mm_mullo_epi32(mm2,mw0);                  // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm7 = _mm_mullo_epi32(mm2,mw0);                  // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        // Weight 1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm2 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);                    // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);                    // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm2 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);                    // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm2 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);                    // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Weight 2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*2));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm2 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);                    // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm2 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);                    // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*2 + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);                    // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);                    // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Add rounding
        mm4 = _mm_add_epi32(mm4, round);
        mm5 = _mm_add_epi32(mm5, round);
        mm6 = _mm_add_epi32(mm6, round);
        mm7 = _mm_add_epi32(mm7, round);
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4,mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6,mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        _mm_storeu_si128((__m128i *)d, mm4);
        _mm_storeu_si128((__m128i *)(d + 8), mm6);
        d += 16;
        tmp += 16;
        width -= 16;
    }

    width += 16;
    if (width)
    {
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        // Weight 0
        mm0 = _mm_loadu_si128((const __m128i *)tmp);     // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm4 = _mm_mullo_epi32(mm2,mw0);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm5 = _mm_mullo_epi32(mm2,mw0);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp+8)); // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm6 = _mm_mullo_epi32(mm2,mw0);                  // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm7 = _mm_mullo_epi32(mm2,mw0);                  // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        // Weight 1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm2 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);                    // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);                    // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm2 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);                    // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm2 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);                    // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Weight 2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*2));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm2 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);                    // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm2 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);                    // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*2 + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);                    // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);                    // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Add rounding
        mm4 = _mm_add_epi32(mm4, round);
        mm5 = _mm_add_epi32(mm5, round);
        mm6 = _mm_add_epi32(mm6, round);
        mm7 = _mm_add_epi32(mm7, round);
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4,mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6,mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        store_trailing_shorts(d, width, mm4, mm6);
    }
}

static inline void
zoom_y2to2_4_sse(void          * ipa_restrict dst,
                 const uint8_t * ipa_restrict tmp_,
                 int             width,
                 int             byte_stride,
                 const index_t * ipa_restrict index,
                 const int32_t * ipa_restrict weights)
{
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint16_t *ipa_restrict d = (uint16_t *)dst;
    const uint16_t * ipa_restrict tmp = (const uint16_t *)tmp_;
    __m128i mw0   = _mm_set1_epi32(cbp[0]);
    __m128i mw1   = _mm_set1_epi32(cbp[1]);
    __m128i mw2   = _mm_set1_epi32(cbp[2]);
    __m128i mw3   = _mm_set1_epi32(cbp[3]);
    __m128i round = _mm_set1_epi32(WEIGHT_ROUND);

    tmp += index->first_pixel;
    width -= 16;
    while (width >= 0)
    {
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        // Weight 0
        mm0 = _mm_loadu_si128((const __m128i *)tmp);     // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm4 = _mm_mullo_epi32(mm2,mw0);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm5 = _mm_mullo_epi32(mm2,mw0);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp+8)); // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm6 = _mm_mullo_epi32(mm2,mw0);                  // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm7 = _mm_mullo_epi32(mm2,mw0);                  // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        // Weight 1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm2 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);                    // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);                    // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm2 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);                    // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm2 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);                    // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Weight 2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*2));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm2 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);                    // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm2 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);                    // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*2 + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);                    // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);                    // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Weight 3
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*3));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);                  // mm2 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);                    // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);                    // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*3 + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);                  // mm2 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);                    // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);                  // mm2 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);                    // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Add rounding
        mm4 = _mm_add_epi32(mm4, round);
        mm5 = _mm_add_epi32(mm5, round);
        mm6 = _mm_add_epi32(mm6, round);
        mm7 = _mm_add_epi32(mm7, round);
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4,mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6,mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        _mm_storeu_si128((__m128i *)d, mm4);
        _mm_storeu_si128((__m128i *)(d + 8), mm6);
        d += 16;
        tmp += 16;
        width -= 16;
    }

    width += 16;
    if (width)
    {
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        // Weight 0
        mm0 = _mm_loadu_si128((const __m128i *)tmp);     // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm4 = _mm_mullo_epi32(mm2,mw0);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm5 = _mm_mullo_epi32(mm2,mw0);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp+8)); // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm6 = _mm_mullo_epi32(mm2,mw0);                  // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm7 = _mm_mullo_epi32(mm2,mw0);                  // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        // Weight 1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm2 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);                    // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);                    // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm2 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);                    // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm2 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);                    // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Weight 2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*2));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm2 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);                    // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm2 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);                    // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*2 + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);                    // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);                    // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Weight 3
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*3));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);                  // mm2 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);                    // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);                    // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*3 + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);                  // mm2 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);                    // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);                  // mm2 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);                    // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Add rounding
        mm4 = _mm_add_epi32(mm4, round);
        mm5 = _mm_add_epi32(mm5, round);
        mm6 = _mm_add_epi32(mm6, round);
        mm7 = _mm_add_epi32(mm7, round);
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4,mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6,mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        store_trailing_shorts(d, width, mm4, mm6);
    }
}

static inline void
zoom_y2to2_5_sse(void          * ipa_restrict dst,
                 const uint8_t * ipa_restrict tmp_,
                 int                          width,
                 int                          byte_stride,
                 const index_t * ipa_restrict index,
                 const int32_t * ipa_restrict weights)
{
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint16_t *ipa_restrict d = (uint16_t *)dst;
    const uint16_t * ipa_restrict tmp = (const uint16_t *)tmp_;
    __m128i mw0 = _mm_set1_epi32(cbp[0]);
    __m128i mw1 = _mm_set1_epi32(cbp[1]);
    __m128i mw2 = _mm_set1_epi32(cbp[2]);
    __m128i mw3 = _mm_set1_epi32(cbp[3]);
    __m128i mw4 = _mm_set1_epi32(cbp[4]);
    __m128i round = _mm_set1_epi32(WEIGHT_ROUND);

    tmp += index->first_pixel;
    width -= 16;
    while (width >= 0)
    {
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        // Weight 0
        mm0 = _mm_loadu_si128((const __m128i *)tmp);     // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm4 = _mm_mullo_epi32(mm2,mw0);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm5 = _mm_mullo_epi32(mm2,mw0);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp+8)); // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm6 = _mm_mullo_epi32(mm2,mw0);                  // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm7 = _mm_mullo_epi32(mm2,mw0);                  // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        // Weight 1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm2 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);                    // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);                    // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm2 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);                    // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm2 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);                    // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Weight 2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*2));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm2 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);                    // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm2 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);                    // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*2 + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);                    // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);                    // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Weight 3
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*3));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);                  // mm2 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);                    // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);                    // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*3 + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);                  // mm2 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);                    // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);                  // mm2 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);                    // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Weight 4
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*4));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw4);                  // mm2 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);                    // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw4);                  // mm2 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);                    // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*4 + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw4);                  // mm2 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);                    // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw4);                  // mm2 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);                    // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Add rounding
        mm4 = _mm_add_epi32(mm4, round);
        mm5 = _mm_add_epi32(mm5, round);
        mm6 = _mm_add_epi32(mm6, round);
        mm7 = _mm_add_epi32(mm7, round);
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4, mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6, mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        _mm_storeu_si128((__m128i *)d, mm4);
        _mm_storeu_si128((__m128i *)(d + 8), mm6);
        d += 16;
        tmp += 16;
        width -= 16;
    }

    width += 16;
    if (width)
    {
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        // Weight 0
        mm0 = _mm_loadu_si128((const __m128i *)tmp);     // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm4 = _mm_mullo_epi32(mm2,mw0);                  // mm4 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm5 = _mm_mullo_epi32(mm2,mw0);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp+8)); // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm6 = _mm_mullo_epi32(mm2,mw0);                  // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm7 = _mm_mullo_epi32(mm2,mw0);                  // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        // Weight 1
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm2 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);                    // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);                    // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm2 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);                    // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw1);                  // mm2 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);                    // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Weight 2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*2));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm2 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);                    // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm2 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);                    // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*2 + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);                    // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw2);                  // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);                    // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Weight 3
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*3));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);                  // mm2 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);                    // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);                  // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);                    // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*3 + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);                  // mm2 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);                    // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw3);                  // mm2 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);                    // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Weight 4
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*4));
                                                            // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw4);                  // mm2 = 00wdwdwd00wcwcwc00wbwbwb00wawawa SSE4.1
        mm4 = _mm_add_epi32(mm4,mm2);                    // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000hhhhggggffffeeee SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw4);                  // mm2 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
        mm5 = _mm_add_epi32(mm5,mm2);                    // mm5 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_loadu_si128((const __m128i *)(tmp + byte_stride*4 + 8));
                                                            // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 000000ll000000kk000000jj000000ii SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw4);                  // mm2 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
        mm6 = _mm_add_epi32(mm6,mm2);                    // mm6 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        mm0 = _mm_srli_si128(mm0,8);                     // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
        mm2 = _mm_cvtepu16_epi32(mm0);                   // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
        mm2 = _mm_mullo_epi32(mm2,mw4);                  // mm2 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
        mm7 = _mm_add_epi32(mm7,mm2);                    // mm7 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        // Add rounding
        mm4 = _mm_add_epi32(mm4, round);
        mm5 = _mm_add_epi32(mm5, round);
        mm6 = _mm_add_epi32(mm6, round);
        mm7 = _mm_add_epi32(mm7, round);
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4, mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6, mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        store_trailing_shorts(d, width, mm4, mm6);
    }
}

static inline void
zoom_y2to2_n_sse(void          * ipa_restrict dst,
                 const uint8_t * ipa_restrict tmp_,
                 int                          width,
                 int                          byte_stride,
                 const index_t * ipa_restrict index,
                 const int32_t * ipa_restrict weights)
{
    int cn = index->n;
    const int32_t *ipa_restrict cbp = weights + index->index;
    uint16_t *ipa_restrict d = (uint16_t *)dst;
    const uint16_t * ipa_restrict tmp = (const uint16_t *)tmp_;
    __m128i round = _mm_set1_epi32(WEIGHT_ROUND);

    tmp += index->first_pixel;
    width -= 16;
    while (width >= 0)
    {
        int j;
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        mm4 = round;
        mm5 = round;
        mm6 = round;
        mm7 = round;
        for (j = 0; j < cn; j++)
        {
            __m128i mw;
            mw  = _mm_set1_epi32(cbp[j]);
            mm0 = _mm_loadu_si128((const __m128i *)(tmp + j * byte_stride));
                                                // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
            mm2 = _mm_cvtepu16_epi32(mm0);    // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
            mm2 = _mm_mullo_epi32(mm2,mw);    // mm4 = 00wdwdwd0000wcwc00wbwbwb00wawawa SSE4.1
            mm4 = _mm_add_epi32(mm4,mm2);     // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
            mm0 = _mm_srli_si128(mm0,8);      // mm0 = 0000000000000000hhhhggggffffeeee SSE2
            mm2 = _mm_cvtepu16_epi32(mm0);    // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
            mm2 = _mm_mullo_epi32(mm2,mw);    // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
            mm5 = _mm_add_epi32(mm5,mm2);     // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
            mm0 = _mm_loadu_si128((const __m128i *)(tmp + j * byte_stride + 8));
                                                // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
            mm2 = _mm_cvtepu16_epi32(mm0);    // mm2 = 0000llll0000kkkk0000jjjj0000iiii SSE4.1
            mm2 = _mm_mullo_epi32(mm2,mw);    // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
            mm6 = _mm_add_epi32(mm6,mm2);     // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
            mm0 = _mm_srli_si128(mm0,8);      // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
            mm2 = _mm_cvtepu16_epi32(mm0);    // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
            mm2 = _mm_mullo_epi32(mm2,mw);    // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
            mm7 = _mm_add_epi32(mm7,mm2);     // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        }
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4,mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6,mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        _mm_storeu_si128((__m128i *)d, mm4);
        _mm_storeu_si128((__m128i *)(d + 8), mm6);
        d += 16;
        tmp += 16;
        width -= 16;
    }

    width += 16;
    if (width)
    {
        int j;
        __m128i mm0, mm2, mm4, mm5, mm6, mm7;
        mm4 = round;
        mm5 = round;
        mm6 = round;
        mm7 = round;
        for (j = 0; j < cn; j++)
        {
            __m128i mw;
            mw  = _mm_set1_epi32(cbp[j]);
            mm0 = _mm_loadu_si128((const __m128i *)(tmp + j * byte_stride));
                                                // mm0 = hhhhggggffffeeeeddddccccbbbbaaaa SSE2
            mm2 = _mm_cvtepu16_epi32(mm0);    // mm2 = 0000dddd0000cccc0000bbbb0000aaaa SSE4.1
            mm2 = _mm_mullo_epi32(mm2,mw);    // mm4 = 00wdwdwd0000wcwc00wbwbwb00wawawa SSE4.1
            mm4 = _mm_add_epi32(mm4,mm2);     // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
            mm0 = _mm_srli_si128(mm0,8);      // mm0 = 0000000000000000hhhhggggffffeeee SSE2
            mm2 = _mm_cvtepu16_epi32(mm0);    // mm2 = 0000hhhh0000gggg0000ffff0000eeee SSE4.1
            mm2 = _mm_mullo_epi32(mm2,mw);    // mm5 = 00whwhwh00wgwgwg00wfwfwf00wewewe SSE4.1
            mm5 = _mm_add_epi32(mm5,mm2);     // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
            mm0 = _mm_loadu_si128((const __m128i *)(tmp + j * byte_stride + 8));
                                                // mm0 = ppppoooonnnnmmmmllllkkkkjjjjiiii SSE2
            mm2 = _mm_cvtepu16_epi32(mm0);    // mm2 = 0000llll0000kkkk0000jjjj0000iiii SSE4.1
            mm2 = _mm_mullo_epi32(mm2,mw);    // mm5 = 00wlwlwl00wkwkwk00wjwjwj00wiwiwi SSE4.1
            mm6 = _mm_add_epi32(mm6,mm2);     // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
            mm0 = _mm_srli_si128(mm0,8);      // mm0 = 0000000000000000ppppoooonnnnmmmm SSE2
            mm2 = _mm_cvtepu16_epi32(mm0);    // mm2 = 0000pppp0000oooo0000nnnn0000mmmm SSE4.1
            mm2 = _mm_mullo_epi32(mm2,mw);    // mm5 = 00wpwpwp00wowowo00wnwnwn00wmwmwm SSE4.1
            mm7 = _mm_add_epi32(mm7,mm2);     // mm4 = 00xxxxxx00xxxxxx00xxxxxx00xxxxxx SSE2
        }
        // Shift down
        mm4 = _mm_srai_epi32(mm4, WEIGHT_SHIFT);
        mm5 = _mm_srai_epi32(mm5, WEIGHT_SHIFT);
        mm6 = _mm_srai_epi32(mm6, WEIGHT_SHIFT);
        mm7 = _mm_srai_epi32(mm7, WEIGHT_SHIFT);
        // Pack and Store
        mm4 = _mm_packus_epi32(mm4,mm5);  // mm4 = HHHHGGGGFFFFEEEEDDDDCCCCBBBBAAAA
        mm6 = _mm_packus_epi32(mm6,mm7);  // mm6 = PPPPOOOONNNNMMMMLLLLKKKKJJJJIIII
        store_trailing_shorts(d, width, mm4, mm6);
    }
}

static void
zoom_y2to2_sse(void          * ipa_restrict dst,
               const uint8_t * ipa_restrict tmp,
               int                          width,
               int                          byte_stride,
               const index_t * ipa_restrict index,
               const int32_t * ipa_restrict weights)
{
    byte_stride >>= 1;
    switch (index->n)
    {
        case 2:
            zoom_y2to2_2_sse(dst, tmp, width, byte_stride, index, weights);
            break;
        case 3:
            zoom_y2to2_3_sse(dst, tmp, width, byte_stride, index, weights);
            break;
        case 4:
            zoom_y2to2_4_sse(dst, tmp, width, byte_stride, index, weights);
            break;
        case 5:
            zoom_y2to2_5_sse(dst, tmp, width, byte_stride, index, weights);
            break;
        default:
            zoom_y2to2_n_sse(dst, tmp, width, byte_stride, index, weights);
            break;
    }
}
